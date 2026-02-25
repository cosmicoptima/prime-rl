import time
from contextlib import nullcontext
from datetime import timedelta

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.utils.act_offloading import maybe_activation_offloading
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from loguru import logger
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.weights import setup_weight_ckpt_manager
from prime_rl.trainer.distill.config import DistillTrainerConfig
from prime_rl.trainer.distill.data import setup_dataloader, setup_dataset
from prime_rl.trainer.distill.loss import compute_distill_loss
from prime_rl.utils.logger import setup_logger
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.model import (
    forward,
    setup_tokenizer,
    setup_model,
)
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.utils import (
    MemoryProfiler,
    setup_torch_distributed,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, to_col_format
import torch.distributed as dist


@clean_exit
@logger.catch(reraise=True)
def train(config: DistillTrainerConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "trainer" / f"rank_{world.rank}.log" if config.log.file else None,
    )
    logger.info(f"Starting context distillation trainer in {world}")

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    # Set precision
    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))
    torch.set_float32_matmul_precision("high")

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model, config.data.seq_len)

    # Initialize the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model})")
    model = setup_model(config.model, parallel_dims)
    tokenizer = setup_tokenizer(config.model)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, model, parallel_dims.world_mesh["dp_shard_cp"])

    # Set up the learning rate scheduler
    scheduler_steps = (
        config.max_steps - config.ckpt.resume_step
        if config.max_steps is not None
        and (config.ckpt and config.ckpt.skip_scheduler and config.ckpt.resume_step is not None)
        else config.max_steps
    )
    logger.info(f"Setting up {config.scheduler.type} scheduler with {scheduler_steps} steps ({config.scheduler})")
    scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)

    # Set up weight checkpoint manager
    logger.info(f"Initializing weight checkpoint manager ({config.weights})")
    weight_ckpt_manager = setup_weight_ckpt_manager(
        config.output_dir, config.weights, config.ckpt, 0, config.model.experimental.lora
    )

    # Set up checkpoint manager
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
    assert ckpt_manager is None or (ckpt_manager is not None and weight_ckpt_manager is not None), (
        "If ckpt_manager is set, weight_ckpt_manager must also be set"
    )

    # Set up the dataset and dataloader
    logger.info(f"Initializing data ({config.data})")
    dataset = setup_dataset(tokenizer, config.data, config.model.cp * config.model.tp)
    dataloader = setup_dataloader(dataset, config.data)
    dataiter = iter(dataloader)

    # Check world size / batch compatibility
    batch_size = config.data.batch_size
    if world.world_size > batch_size:
        raise ValueError(
            f"There must be at least one micro batch per rank, but only have {batch_size} for {world.world_size} ranks."
        )
    if batch_size % world.world_size != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by world size ({world.world_size})."
        )

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if ckpt_manager is not None and config.ckpt and config.ckpt.resume_step:
        logger.info(f"Resuming training from checkpoint step {config.ckpt.resume_step}")
        ckpt_manager.load(
            model,
            [optimizer],
            scheduler if not config.ckpt.skip_scheduler else None,
            progress if not config.ckpt.skip_progress else None,
            step=config.ckpt.resume_step,
            dataloader=dataloader if not config.ckpt.skip_dataloader else None,
        )
        if config.ckpt.skip_scheduler:
            scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)
    logger.info(f"Starting from step {progress.step}")

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    max_memory = torch.cuda.mem_get_info()[1] / 1024**3
    is_first_step = True
    maybe_record_function = nullcontext
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function

    while True:
        torch.cuda.reset_peak_memory_stats()
        is_last_step = config.max_steps is not None and progress.step == config.max_steps

        # Save weight checkpoint
        save_weights_time = 0
        if (
            weight_ckpt_manager is not None
            and (config.weights and config.weights.interval)
            and not (is_first_step or is_last_step)
            and progress.step % config.weights.interval == 0
        ):
            logger.info(f"Saving weight checkpoint at step {progress.step}")
            save_weights_start_time = time.time()
            weight_ckpt_manager.save(model, tokenizer, step=progress.step)
            save_weights_time = time.time() - save_weights_start_time

        # Save full checkpoint
        save_ckpt_time = 0
        if (
            ckpt_manager is not None
            and weight_ckpt_manager is not None
            and config.ckpt
            and config.ckpt.interval
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step, dataloader=dataloader)
            save_ckpt_time = time.time() - save_ckpt_start_time
            ckpt_manager.maybe_clean()

        # Break if max steps reached
        if config.max_steps is not None and progress.step >= config.max_steps:
            break

        memory_profiler = (
            MemoryProfiler(progress.step, config.memory_profiler_path) if config.memory_profiler_path else None
        )

        step_start_time = time.time()
        forward_backward_start_time = time.time()
        grad_accum_steps = config.data.batch_size * config.model.cp * config.model.tp // world.world_size

        batch_loss = torch.tensor(0.0).to("cuda")
        batch_kl_mean = torch.tensor(0.0).to("cuda")
        nan_loss_count = torch.tensor(0).to("cuda")

        for micro_step in range(grad_accum_steps):
            model.set_requires_all_reduce(micro_step == grad_accum_steps - 1)

            micro_batch = next(dataiter)

            # Unpack student tensors
            student_input_ids = micro_batch["student_input_ids"].to("cuda")
            student_position_ids = micro_batch["student_position_ids"].to("cuda")
            student_loss_mask = micro_batch["student_loss_mask"].to("cuda")

            # Unpack teacher tensors
            teacher_input_ids = micro_batch["teacher_input_ids"].to("cuda")
            teacher_position_ids = micro_batch["teacher_position_ids"].to("cuda")
            teacher_loss_mask = micro_batch["teacher_loss_mask"].to("cuda")

            # Teacher forward pass (no gradient — inference only)
            with maybe_record_function("teacher_forward"), torch.no_grad():
                teacher_logits = forward(model, teacher_input_ids, teacher_position_ids)

            # Extract teacher response logits and free the full tensor
            teacher_response_logits = teacher_logits[teacher_loss_mask].clone()
            del teacher_logits

            # Student forward pass (with gradient)
            with maybe_record_function("student_forward"), maybe_activation_offloading(config.model.ac_offloading):
                student_logits = forward(model, student_input_ids, student_position_ids)

            # Extract student response logits
            student_response_logits = student_logits[student_loss_mask]

            # Verify alignment
            assert teacher_response_logits.shape[0] == student_response_logits.shape[0], (
                f"Response token count mismatch: teacher={teacher_response_logits.shape[0]}, "
                f"student={student_response_logits.shape[0]}"
            )

            # Compute KL divergence loss
            loss, loss_metrics = compute_distill_loss(
                student_response_logits,
                teacher_response_logits,
                temperature=config.temperature,
            )

            # Free logits before backward
            del student_logits, teacher_response_logits, student_response_logits

            # Accumulate loss
            current_loss = loss.detach() / grad_accum_steps
            if not torch.isnan(current_loss):
                batch_loss += current_loss
                batch_kl_mean += loss_metrics["kl_mean"] / grad_accum_steps
            else:
                nan_loss_count += 1
                logger.warning("Loss is nan, skipping")

            # Backward pass (gradients flow only through student computation graph)
            with maybe_record_function("backward"):
                (loss / grad_accum_steps).backward()

            logger.debug(
                f"Micro Step {micro_step}/{grad_accum_steps} | "
                f"Loss: {loss.item():.4f} | KL: {loss_metrics['kl_mean']:.4f}"
            )

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optim.max_norm).full_tensor()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        forward_backward_time = time.time() - forward_backward_start_time

        if memory_profiler is not None:
            memory_profiler.step()

        # Synchronize metrics
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(nan_loss_count, op=dist.ReduceOp.SUM)

        # Compute step metrics
        num_tokens = config.data.batch_size * config.data.seq_len
        progress.total_tokens += num_tokens
        progress.total_samples = dataset.step
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3

        # Log step
        step_time = time.time() - step_start_time
        logger.success(
            f"Step {progress.step} | Time: {step_time:.2f}s | KL Loss: {batch_loss.item():.4f} | "
            f"Grad. Norm: {grad_norm:.4f} | LR: {current_lr:.2e} | "
            f"Throughput: {throughput:.0f} tokens/s | Peak Mem.: {peak_memory:.1f}/{max_memory:.1f} GiB"
        )

        # Log metrics
        monitor.log({
            "loss/kl_mean": batch_loss.item(),
            "loss/nan_count": nan_loss_count.item(),
            "step": progress.step,
        })
        monitor.log({
            "progress/num_samples": progress.total_samples,
            "progress/num_tokens": progress.total_tokens,
            "step": progress.step,
        })
        monitor.log({
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/peak_memory": peak_memory,
            "perf/mfu": mfu,
            "step": progress.step,
        })
        monitor.log({
            "optim/lr": current_lr,
            "optim/grad_norm": grad_norm.item(),
            "step": progress.step,
        })
        monitor.log({
            "time/step": step_time,
            "time/save_ckpt": save_ckpt_time,
            "time/save_weights": save_weights_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        })

        is_first_step = False
        progress.step += 1

    if config.trace_path:
        prof.__exit__(None, None, None)
        config.trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = str(config.trace_path / f"trace_{dist.get_rank()}.json.gz")
        logger.info(f"Saving trace to {trace_file}")
        prof.export_chrome_trace(trace_file)

    # Log final distributions
    monitor.log_final_distributions()

    # Write final weight checkpoint
    if weight_ckpt_manager is not None:
        assert config.weights is not None
        logger.info("Writing final weight checkpoint")
        weight_ckpt_manager.save(model, tokenizer, step=progress.step)

    # Write final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step, dataloader=dataloader)
        ckpt_manager.maybe_clean()

    logger.info(f"Peak memory: {max(to_col_format(monitor.history)['perf/peak_memory']):.1f} GiB")
    logger.success("Context distillation trainer finished!")


def main():
    train(parse_argv(DistillTrainerConfig))


if __name__ == "__main__":
    main()
