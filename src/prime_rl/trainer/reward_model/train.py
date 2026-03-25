import math
import time
from datetime import timedelta

# ruff: noqa: I001
from prime_rl.utils.act_offloading import maybe_activation_offloading
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.weights import setup_weight_ckpt_manager
from prime_rl.trainer.reward_model.config import RewardModelTrainerConfig
from prime_rl.trainer.reward_model.data import setup_dataloader, setup_dataset
from prime_rl.trainer.reward_model.loss import compute_bt_loss
from prime_rl.utils.logger import setup_logger
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.model import (
    get_model,
    setup_tokenizer,
    setup_fsdp,
    apply_ac,
    apply_compile,
    can_load_dcp_from_hf,
    load_dcp_from_hf,
    DTYPE_MAP,
)
from prime_rl.trainer.lora import apply_lora_to_model
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
from prime_rl.utils.tensor_hashing import get_module_signature


def setup_reward_model(config, parallel_dims):
    """Load a causal LM and replace its language modeling head with a scalar reward head."""
    logger = get_logger()

    # Load the base model (always to CPU first, not meta, to avoid DCP shape mismatch)
    model = get_model(
        config,
        device=torch.device("cpu"),
        dtype=DTYPE_MAP[config.optimization_dtype],
    )

    # Replace lm_head with scalar reward head
    hidden_size = model.config.hidden_size
    old_lm_head = model.lm_head
    model.lm_head = nn.Linear(hidden_size, 1, bias=False)
    nn.init.normal_(model.lm_head.weight, std=1.0 / math.sqrt(hidden_size))
    model.lm_head.weight.data = model.lm_head.weight.data.to(old_lm_head.weight.dtype)
    del old_lm_head

    # Break any embedding-head weight tying
    model.config.tie_word_embeddings = False

    logger.info(f"Replaced lm_head with reward head: Linear({hidden_size}, 1)")

    # If LoRA is configured, ensure lm_head is in modules_to_save so it stays trainable
    if config.experimental.lora is not None:
        if "lm_head" not in config.experimental.lora.modules_to_save:
            config.experimental.lora.modules_to_save.append("lm_head")
            logger.info("Added 'lm_head' (reward head) to LoRA modules_to_save")
        apply_lora_to_model(model, config.experimental.lora)

    # AC -> compile -> FSDP (same order as setup_model)
    if config.ac is not None:
        apply_ac(model, config.ac)
    if config.compile is not None:
        apply_compile(model, config.compile)

    setup_fsdp(model, config, parallel_dims)

    logger.debug(f"Reward model signature: {get_module_signature(model, compress=True)}")
    return model


def extract_rewards(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Extract the reward scalar from the last real token position.

    Args:
        scores: Model output of shape [B, L, 1] (the "logits" from the reward head).
        lengths: True sequence lengths of shape [B] (index of last real token + 1).

    Returns:
        Rewards of shape [B].
    """
    batch_size = scores.shape[0]
    # Last real token is at position (length - 1)
    last_indices = (lengths - 1).clamp(min=0).to(scores.device)
    rewards = scores[torch.arange(batch_size, device=scores.device), last_indices, 0]
    return rewards


@clean_exit
@logger.catch(reraise=True)
def train(config: RewardModelTrainerConfig):
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "reward_model" / f"rank_{world.rank}.log" if config.log.file else None,
    )
    logger.info(f"Starting reward model trainer in {world}")

    # Monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    # Distributed setup
    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))
    torch.set_float32_matmul_precision("high")

    # Parallel dims
    parallel_dims = get_parallel_dims(config.model, config.data.seq_len)

    # Model and tokenizer
    logger.info(f"Initializing reward model ({config.model})")
    model = setup_reward_model(config.model, parallel_dims)
    tokenizer = setup_tokenizer(config.model)

    # Optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, model, parallel_dims.world_mesh["dp_shard_cp"])

    # Scheduler
    scheduler_steps = (
        config.max_steps - config.ckpt.resume_step
        if config.max_steps is not None
        and (config.ckpt and config.ckpt.skip_scheduler and config.ckpt.resume_step is not None)
        else config.max_steps
    )
    logger.info(f"Setting up {config.scheduler.type} scheduler with {scheduler_steps} steps ({config.scheduler})")
    scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)

    # Weight checkpoint manager
    logger.info(f"Initializing weight checkpoint manager ({config.weights})")
    weight_ckpt_manager = setup_weight_ckpt_manager(
        config.output_dir, config.weights, config.ckpt, 0, config.model.experimental.lora
    )

    # Checkpoint manager
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)

    # Dataset and dataloader
    logger.info(f"Initializing data ({config.data})")
    dataset = setup_dataset(tokenizer, config.data)
    dataloader = setup_dataloader(dataset, config.data)
    dataiter = iter(dataloader)

    # Batch size checks
    batch_size = config.data.batch_size
    if world.world_size > batch_size:
        raise ValueError(
            f"Need at least one sample per rank, but only have {batch_size} for {world.world_size} ranks."
        )

    # Resume
    progress = Progress()
    if ckpt_manager is not None and config.ckpt and config.ckpt.resume_step:
        logger.info(f"Resuming from checkpoint step {config.ckpt.resume_step}")
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
            t0 = time.time()
            weight_ckpt_manager.save(model, tokenizer, step=progress.step)
            save_weights_time = time.time() - t0

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
            t0 = time.time()
            ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step, dataloader=dataloader)
            save_ckpt_time = time.time() - t0
            ckpt_manager.maybe_clean()

        # Done?
        if config.max_steps is not None and progress.step >= config.max_steps:
            break

        memory_profiler = (
            MemoryProfiler(progress.step, config.memory_profiler_path) if config.memory_profiler_path else None
        )

        step_start_time = time.time()
        fwd_bwd_start_time = time.time()

        # Gradient accumulation: each rank processes batch_size / world_size pairs
        grad_accum_steps = max(1, config.data.batch_size * config.model.cp * config.model.tp // world.world_size)

        batch_loss = torch.tensor(0.0, device="cuda")
        batch_accuracy = torch.tensor(0.0, device="cuda")
        batch_margin = torch.tensor(0.0, device="cuda")
        nan_loss_count = torch.tensor(0, device="cuda")

        for micro_step in range(grad_accum_steps):
            model.set_requires_all_reduce(micro_step == grad_accum_steps - 1)

            batch = next(dataiter)

            # Move to GPU
            chosen_ids = batch.chosen_input_ids.to("cuda")
            chosen_pos = batch.chosen_position_ids.to("cuda")
            chosen_len = batch.chosen_lengths
            rejected_ids = batch.rejected_input_ids.to("cuda")
            rejected_pos = batch.rejected_position_ids.to("cuda")
            rejected_len = batch.rejected_lengths

            # Forward pass on chosen
            with maybe_activation_offloading(config.model.ac_offloading):
                chosen_scores = model(input_ids=chosen_ids, position_ids=chosen_pos).logits  # [B, L, 1]
            chosen_rewards = extract_rewards(chosen_scores, chosen_len)
            del chosen_scores

            # Forward pass on rejected
            with maybe_activation_offloading(config.model.ac_offloading):
                rejected_scores = model(input_ids=rejected_ids, position_ids=rejected_pos).logits  # [B, L, 1]
            rejected_rewards = extract_rewards(rejected_scores, rejected_len)
            del rejected_scores

            # Bradley-Terry loss
            loss, loss_metrics = compute_bt_loss(chosen_rewards, rejected_rewards)

            # Accumulate
            current_loss = loss.detach() / grad_accum_steps
            if not torch.isnan(current_loss):
                batch_loss += current_loss
                batch_accuracy += loss_metrics["accuracy"] / grad_accum_steps
                batch_margin += loss_metrics["mean_margin"] / grad_accum_steps
            else:
                nan_loss_count += 1
                logger.warning("Loss is NaN, skipping")

            (loss / grad_accum_steps).backward()

            logger.debug(
                f"Micro step {micro_step}/{grad_accum_steps} | "
                f"Loss: {loss.item():.4f} | Acc: {loss_metrics['accuracy']:.3f} | "
                f"Margin: {loss_metrics['mean_margin']:.3f}"
            )

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optim.max_norm).full_tensor()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # LR scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        fwd_bwd_time = time.time() - fwd_bwd_start_time

        if memory_profiler is not None:
            memory_profiler.step()

        # Sync metrics
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_accuracy, op=dist.ReduceOp.AVG)
        dist.all_reduce(nan_loss_count, op=dist.ReduceOp.SUM)

        # Perf
        num_tokens = config.data.batch_size * config.data.seq_len * 2  # chosen + rejected
        progress.total_tokens += num_tokens
        progress.total_samples = dataset.step
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3

        step_time = time.time() - step_start_time
        logger.success(
            f"Step {progress.step} | Time: {step_time:.2f}s | BT Loss: {batch_loss.item():.4f} | "
            f"Acc: {batch_accuracy.item():.3f} | Margin: {batch_margin.item():.3f} | "
            f"Grad Norm: {grad_norm:.4f} | LR: {current_lr:.2e} | "
            f"Peak Mem: {peak_memory:.1f}/{max_memory:.1f} GiB"
        )

        # Log to wandb
        monitor.log({
            "loss/bt_loss": batch_loss.item(),
            "loss/accuracy": batch_accuracy.item(),
            "loss/mean_margin": batch_margin.item(),
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
            "time/forward_backward": fwd_bwd_time,
            "step": progress.step,
        })

        is_first_step = False
        progress.step += 1

    # Final distributions
    monitor.log_final_distributions()

    # Final weight checkpoint
    if weight_ckpt_manager is not None:
        assert config.weights is not None
        logger.info("Writing final weight checkpoint")
        weight_ckpt_manager.save(model, tokenizer, step=progress.step)

    # Final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step, dataloader=dataloader)
        ckpt_manager.maybe_clean()

    logger.info(f"Peak memory: {max(to_col_format(monitor.history)['perf/peak_memory']):.1f} GiB")
    logger.success("Reward model training finished!")


def main():
    train(parse_argv(RewardModelTrainerConfig))


if __name__ == "__main__":
    main()
