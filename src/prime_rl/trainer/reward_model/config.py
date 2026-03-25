from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from prime_rl.trainer.config import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    WeightCheckpointConfig,
)
from prime_rl.utils.config import LogConfig, WandbMonitorConfig
from prime_rl.utils.pydantic_config import BaseSettings


class RewardModelDataConfig(BaseModel):
    """Configures data for Bradley-Terry reward model training.

    Expects a HuggingFace dataset with columns:
      - prompt (str): the user prompt
      - chosen (str): the preferred response
      - rejected (str): the less preferred response
    """

    name: Annotated[str, Field(description="Name or path of the HF dataset.")] = ""
    split: Annotated[str, Field(description="Split to use.")] = "train"

    batch_size: Annotated[int, Field(ge=1, description="Number of preference pairs per batch.")] = 8
    seq_len: Annotated[int, Field(ge=1, description="Max sequence length (prompt + response).")] = 2048
    shuffle: Annotated[bool, Field(description="Whether to shuffle the dataset.")] = True
    seed: Annotated[int, Field(description="Random seed.")] = 0

    # Optional: column name overrides
    prompt_column: Annotated[str, Field(description="Column name for prompts.")] = "prompt"
    chosen_column: Annotated[str, Field(description="Column name for chosen responses.")] = "chosen"
    rejected_column: Annotated[str, Field(description="Column name for rejected responses.")] = "rejected"


class RewardModelTrainerConfig(BaseSettings):
    """Configures the Bradley-Terry reward model trainer."""

    model: ModelConfig = ModelConfig()
    data: RewardModelDataConfig = RewardModelDataConfig()
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()
    ckpt: CheckpointConfig | None = None
    weights: WeightCheckpointConfig | None = None
    log: LogConfig = LogConfig()
    wandb: WandbMonitorConfig | None = None

    output_dir: Annotated[Path, Field(description="Directory to write outputs to.")] = Path("outputs")
    max_steps: Annotated[int | None, Field(description="Maximum number of training steps.")] = None

    memory_profiler_path: Annotated[Path | None, Field(description="Path to write memory profile to.")] = None
    dist_timeout_seconds: Annotated[int, Field(description="Timeout for torch distributed ops.")] = 600

    @model_validator(mode="after")
    def validate_ckpt_managers(self):
        if self.ckpt is not None:
            if self.weights is None:
                self.weights = WeightCheckpointConfig()
            if self.ckpt.interval is not None and self.weights.interval is None:
                self.weights.interval = self.ckpt.interval
            if (
                self.ckpt.interval is not None
                and self.weights.interval is not None
                and self.ckpt.interval % self.weights.interval != 0
            ):
                raise ValueError(
                    "Use a weight checkpoint interval that ensures a weight checkpoint is saved with every full checkpoint"
                )
        return self
