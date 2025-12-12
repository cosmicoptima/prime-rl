import asyncio
import html as html_module
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from openai import AsyncOpenAI
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import WandbMonitorConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pydantic_config import BaseSettings


class WandbMonitor:
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbMonitorConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseSettings | None = None,
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0
        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return
        assert config is not None
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")
        self._maybe_overwrite_wandb_command()
        self.wandb = wandb.init(
            project=config.project,
            name=config.name,
            id=config.id,
            dir=output_dir,
            resume="allow",
            config=run_config.model_dump() if run_config else None,
            mode="offline" if config.offline else None,
        )

        # Optionally, initialize sample logging attributes
        if config is not None and config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.samples_cols = [
                    "step",
                    "tag",
                    "problem_id",
                    "sample_id",
                    "num_input_tokens",
                    "num_output_tokens",
                    "input_tokens",
                    "output_tokens",
                    "prompt",
                    "completion",
                    "reward",
                    "advantage",
                ]
                self.samples_table = wandb.Table(
                    columns=self.samples_cols,
                    log_mode="INCREMENTAL",
                )
                self.tokenizer = tokenizer
                self.samples = []

            if config is not None and config.log_extras.distributions:
                self.last_log_distributions_step = -1
                # Incremental table is initialized dynamically in `log_distributions`
                self.distributions_table = None
                self.distributions = []

    def _maybe_overwrite_wandb_command(self) -> None:
        """Overwrites sys.argv with the start command if it is set in the environment variables."""
        wandb_args = os.environ.get("WANDB_ARGS", None)
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

    def log(self, metrics: dict[str, Any]) -> None:
        self.history.append(metrics)
        if not self.is_master:
            return
        if not self.enabled:
            return
        wandb.log(metrics, step=metrics.get("step", None))

    def log_samples(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        rollouts_per_problem: int,
        step: int,
    ) -> None:
        """Log prompt/response samples to W&B table.

        Args:
            input_tokens: List of input token sequences
            output_tokens: List of output token sequences
            rewards: List of rewards for each sample
            task_rewards: Optional list of task-specific rewards
            step: Current training step
        """
        if not self.is_master:
            return
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            # Do not log samples if not enabled or not log interval step
            return
        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for sample logging"
        self.logger.info(f"Logging samples to W&B table at step {step}")
        start_time = time.time()
        batch_size = len(input_tokens)
        num_problems = batch_size // rollouts_per_problem

        # Compute per-problem statistics
        per_problem_tokens = defaultdict(list)
        tokens = [input_tokens[i] + output_tokens[i] for i in range(batch_size)]
        for i, t in enumerate(tokens):
            problem_id = i // rollouts_per_problem
            per_problem_tokens[problem_id].append(t)
        assert len(per_problem_tokens) == num_problems
        assert list(per_problem_tokens.keys()) == list(range(num_problems))

        per_problem_seq_len = {
            problem_id: sum(len(t) for t in tokens) / len(tokens) for problem_id, tokens in per_problem_tokens.items()
        }
        self.logger.debug(f"Per-problem seq len: {per_problem_seq_len}")
        min_len_problem_id = min(per_problem_seq_len.items(), key=lambda kv: kv[1])[0]
        max_len_problem_id = max(per_problem_seq_len.items(), key=lambda kv: kv[1])[0]
        random_problem_id = random.choice(list(range(num_problems)))
        problem_ids = {
            "min_len": min_len_problem_id,
            "max_len": max_len_problem_id,
            "random": random_problem_id,
        }
        self.logger.debug(f"Logging samples for problems: {problem_ids}")

        # Randomly select and log samples
        for tag, problem_id in problem_ids.items():
            start_idx = problem_id * rollouts_per_problem
            for sample_id in range(start_idx, start_idx + rollouts_per_problem):
                sample = {
                    "step": step,
                    "tag": tag,
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "num_input_tokens": len(input_tokens[sample_id]),
                    "num_output_tokens": len(output_tokens[sample_id]),
                    "input_tokens": str(input_tokens[sample_id]),
                    "output_tokens": str(output_tokens[sample_id]),
                    "prompt": self.tokenizer.decode(input_tokens[sample_id]),
                    "completion": self.tokenizer.decode(output_tokens[sample_id]),
                    "reward": float(rewards[sample_id]),
                    "advantage": float(advantages[sample_id]),
                }
                assert list(sample.keys()) == self.samples_cols, (
                    "Order of columns in the table must be the same as order of the keys here"
                )
                self.samples_table.add_data(*sample.values())
                self.samples.append(sample)
        wandb.log({"samples": self.samples_table}, step=step)
        self.last_log_samples_step = step
        self.logger.debug(f"Logged samples at step {step} to W&B table in {time.time() - start_time:.2f}s")

        # Also log HTML view for easier reading
        self._log_samples_html(problem_ids, input_tokens, output_tokens, rewards, advantages, rollouts_per_problem, step)

    def _log_samples_html(
        self,
        problem_ids: dict[str, int],
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        rollouts_per_problem: int,
        step: int,
    ) -> None:
        """Log an HTML view of samples for easier reading in wandb."""
        css = """
        <style>
            .samples-container { font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; }
            .problem { margin-bottom: 32px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
            .problem-header { background: #f5f5f5; padding: 12px 16px; font-weight: 600; border-bottom: 1px solid #ddd; }
            .rollout { padding: 16px; border-bottom: 1px solid #eee; }
            .rollout:last-child { border-bottom: none; }
            .rollout-header { display: flex; gap: 16px; margin-bottom: 12px; font-size: 13px; color: #666; }
            .rollout-header span { background: #f0f0f0; padding: 2px 8px; border-radius: 4px; }
            .reward-positive { background: #d4edda !important; color: #155724; }
            .reward-negative { background: #f8d7da !important; color: #721c24; }
            .prompt { background: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 12px; white-space: pre-wrap; font-size: 14px; line-height: 1.5; }
            .prompt-label { font-size: 11px; text-transform: uppercase; color: #666; margin-bottom: 4px; font-weight: 600; }
            .completion { background: #fff; padding: 12px; border: 1px solid #e0e0e0; border-radius: 6px; white-space: pre-wrap; font-size: 14px; line-height: 1.5; }
            .completion-label { font-size: 11px; text-transform: uppercase; color: #666; margin-bottom: 4px; font-weight: 600; }
        </style>
        """

        html_parts = [css, '<div class="samples-container">']
        html_parts.append(f'<h2>Step {step} Samples</h2>')

        for tag, problem_id in problem_ids.items():
            html_parts.append(f'<div class="problem">')
            html_parts.append(f'<div class="problem-header">Problem {problem_id} ({tag})</div>')

            start_idx = problem_id * rollouts_per_problem
            for i, sample_id in enumerate(range(start_idx, start_idx + rollouts_per_problem)):
                reward = rewards[sample_id]
                advantage = advantages[sample_id]
                reward_class = "reward-positive" if advantage > 0 else "reward-negative" if advantage < 0 else ""

                prompt = html_module.escape(self.tokenizer.decode(input_tokens[sample_id]))
                completion = html_module.escape(self.tokenizer.decode(output_tokens[sample_id]))

                html_parts.append(f'<div class="rollout">')
                html_parts.append(f'<div class="rollout-header">')
                html_parts.append(f'<span>Rollout {i+1}/{rollouts_per_problem}</span>')
                html_parts.append(f'<span class="{reward_class}">Reward: {reward:.4f}</span>')
                html_parts.append(f'<span class="{reward_class}">Advantage: {advantage:.4f}</span>')
                html_parts.append(f'</div>')

                html_parts.append(f'<div class="prompt-label">Prompt</div>')
                html_parts.append(f'<div class="prompt">{prompt}</div>')
                html_parts.append(f'<div class="completion-label">Completion</div>')
                html_parts.append(f'<div class="completion">{completion}</div>')
                html_parts.append(f'</div>')

            html_parts.append('</div>')

        html_parts.append('</div>')
        html_content = '\n'.join(html_parts)

        wandb.log({f"samples_html": wandb.Html(html_content)}, step=step)

    def setup_fixed_prompts(self, dataset: Any) -> None:
        """Initialize fixed prompts by sampling from the dataset.

        Should be called once at startup if fixed_prompts is configured without explicit prompts.
        """
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.fixed_prompts
        ):
            return

        fp_config = self.config.log_extras.fixed_prompts

        # If explicit prompts provided, use those
        if fp_config.prompts:
            self._fixed_prompts = fp_config.prompts
            return

        # Sample prompts from dataset
        rng = random.Random(fp_config.seed)
        num_prompts = min(fp_config.num_prompts, len(dataset))
        indices = rng.sample(range(len(dataset)), num_prompts)

        self._fixed_prompts = []
        for idx in indices:
            example = dataset[idx]
            # Handle different dataset formats
            if "question" in example:
                prompt = example["question"]
            elif "prompt" in example:
                prompt = example["prompt"]
            elif "messages" in example:
                # Chat format - extract user message
                messages = example["messages"]
                prompt = next((m["content"] for m in messages if m.get("role") == "user"), None)
            else:
                # Fallback: try to find any string field
                prompt = None
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 10:
                        prompt = value
                        break

            if prompt:
                self._fixed_prompts.append(prompt)

        self.logger.info(f"Sampled {len(self._fixed_prompts)} fixed prompts from dataset")

    async def log_fixed_prompts(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: dict[str, Any],
        step: int,
    ) -> None:
        """Run fixed prompt evaluation and log results to wandb.

        This generates responses for a fixed set of prompts at regular intervals,
        allowing you to track how the model's responses change over training.
        """
        if not self.is_master:
            return
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.fixed_prompts
        ):
            return

        # Check if prompts have been set up
        if not hasattr(self, "_fixed_prompts") or not self._fixed_prompts:
            return

        fp_config = self.config.log_extras.fixed_prompts
        if step % fp_config.interval != 0:
            return

        self.logger.info(f"Running fixed prompt evaluation at step {step}")
        start_time = time.time()

        prompts = self._fixed_prompts
        num_samples = fp_config.num_samples

        # Generate responses for each prompt
        results: list[dict[str, Any]] = []
        for prompt_idx, prompt in enumerate(prompts):
            # Generate multiple samples for this prompt
            responses = await asyncio.gather(*[
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **sampling_args,
                )
                for _ in range(num_samples)
            ])

            for sample_idx, response in enumerate(responses):
                content = response.choices[0].message.content or ""
                results.append({
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "sample_idx": sample_idx,
                    "completion": content,
                })

        # Build HTML view
        css = """
        <style>
            .fixed-prompts-container { font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; }
            .prompt-section { margin-bottom: 32px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
            .prompt-header { background: #e8f4fc; padding: 12px 16px; font-weight: 600; border-bottom: 1px solid #ddd; }
            .prompt-text { background: #f8f9fa; padding: 12px 16px; border-bottom: 1px solid #eee; white-space: pre-wrap; font-size: 14px; }
            .sample { padding: 16px; border-bottom: 1px solid #eee; }
            .sample:last-child { border-bottom: none; }
            .sample-header { font-size: 12px; color: #666; margin-bottom: 8px; font-weight: 600; }
            .sample-content { background: #fff; padding: 12px; border: 1px solid #e0e0e0; border-radius: 6px; white-space: pre-wrap; font-size: 14px; line-height: 1.5; }
        </style>
        """

        html_parts = [css, '<div class="fixed-prompts-container">']
        html_parts.append(f'<h2>Fixed Prompts - Step {step}</h2>')

        for prompt_idx, prompt in enumerate(prompts):
            prompt_results = [r for r in results if r["prompt_idx"] == prompt_idx]

            html_parts.append('<div class="prompt-section">')
            html_parts.append(f'<div class="prompt-header">Prompt {prompt_idx + 1}</div>')
            html_parts.append(f'<div class="prompt-text">{html_module.escape(prompt)}</div>')

            for result in prompt_results:
                html_parts.append('<div class="sample">')
                html_parts.append(f'<div class="sample-header">Sample {result["sample_idx"] + 1}</div>')
                html_parts.append(f'<div class="sample-content">{html_module.escape(result["completion"])}</div>')
                html_parts.append('</div>')

            html_parts.append('</div>')

        html_parts.append('</div>')
        html_content = '\n'.join(html_parts)

        wandb.log({"fixed_prompts": wandb.Html(html_content)}, step=step)
        self.logger.debug(f"Logged fixed prompts at step {step} in {time.time() - start_time:.2f}s")

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if not self.is_master:
            return
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.distributions
            or step % self.config.log_extras.interval != 0
        ):
            return
        assert self.last_log_distributions_step <= step, "Step must be greater than last logged step"
        self.logger.info(f"Logging distributions for keys {list(distributions.keys())} to W&B table at step {step}")

        # Initialize incremental table if not already done
        if self.distributions_table is None:
            self.distributions_cols = list(distributions.keys())
            self.distributions_table = wandb.Table(
                columns=["step"] + self.distributions_cols,
                log_mode="INCREMENTAL",
            )
        assert self.distributions_cols == list(distributions.keys()), (
            "Columns in the table must be the same across all steps"
        )

        # Append to distributions
        start_time = time.time()
        row = {"step": step, **distributions}
        self.distributions.append(row)
        self.distributions_table.add_data(*row.values())
        wandb.log({"distributions": self.distributions_table}, step=step)
        self.last_log_distributions_step = step
        self.logger.debug(f"Logged distributions at step {step} to W&B table in {time.time() - start_time:.2f}s")

    def log_final_samples(self) -> None:
        """Log final samples to W&B table."""
        if not self.is_master:
            return
        if not self.config or not self.config.log_extras or not self.config.log_extras.samples:
            return
        self.logger.info("Logging final samples to W&B table")
        df = pd.DataFrame(self.samples)
        table = wandb.Table(dataframe=df)
        wandb.log({"final-samples": table})

    def log_final_distributions(self) -> None:
        """Log final distributions to W&B table."""
        if not self.is_master:
            return
        if not self.config or not self.config.log_extras or not self.config.log_extras.distributions:
            return
        self.logger.info("Logging final distributions to W&B table")
        df = pd.DataFrame(self.distributions)
        table = wandb.Table(dataframe=df)
        wandb.log({"final-distributions": table})

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to W&B table."""
        if not self.is_master or not self.enabled:
            return
        self.logger.info("Saving final summary to file")
        assert self.output_dir is not None, "Output directory is required for saving final summary"
        dir_path = self.output_dir / f"run-{self.wandb.id}"
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / filename, "w") as f:
            json.dump(wandb.summary._as_dict(), f)


_MONITOR: WandbMonitor | None = None


def get_monitor() -> WandbMonitor:
    """Returns the global monitor."""
    global _MONITOR
    if _MONITOR is None:
        raise RuntimeError("WandbMonitor not initialized. Please call `setup_monitor` first.")
    return _MONITOR


def setup_monitor(
    config: WandbMonitorConfig | None,
    output_dir: Path | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    run_config: BaseSettings | None = None,
) -> WandbMonitor:
    """Sets up a monitor to log metrics to W&B."""
    global _MONITOR
    if _MONITOR is not None:
        raise RuntimeError("WandbMonitor already initialized. Please call `setup_monitor` only once.")
    _MONITOR = WandbMonitor(config=config, output_dir=output_dir, tokenizer=tokenizer, run_config=run_config)
    return _MONITOR
