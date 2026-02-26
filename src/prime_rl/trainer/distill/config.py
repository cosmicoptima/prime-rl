from pathlib import Path
from typing import Annotated, Literal

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
from prime_rl.trainer.sft.config import LossMaskConfig
from prime_rl.utils.config import LogConfig, WandbMonitorConfig
from prime_rl.utils.pydantic_config import BaseSettings


class DistillDataConfig(BaseModel):
    """Configures data for context distillation."""

    type: Literal["distill"] = "distill"

    # HF dataset with pre-generated responses (columns: prompt, response)
    name: Annotated[str, Field(description="Name or path of the HF dataset with pre-generated responses.")] = ""
    split: Annotated[str, Field(description="Split to use.")] = "train"

    # Soul document (the context to distill away)
    soul_doc: Annotated[str, Field(description="Inline soul document text.")] = ""
    soul_doc_path: Annotated[Path | None, Field(description="Path to a text file containing the soul document.")] = None

    # Standard settings
    batch_size: Annotated[int, Field(ge=1)] = 16
    seq_len: Annotated[int, Field(ge=1)] = 2048
    shuffle: Annotated[bool, Field(description="Whether to shuffle the dataset.")] = True
    seed: Annotated[int, Field(description="Random seed for shuffling.")] = 0
    loss_mask: LossMaskConfig = LossMaskConfig()

    def get_soul_doc(self) -> str:
        if self.soul_doc:
            return self.soul_doc
        if self.soul_doc_path is not None:
            return self.soul_doc_path.read_text().strip()
        raise ValueError("Must provide either soul_doc or soul_doc_path in data config")


class DistillTrainerConfig(BaseSettings):
    """Configures the context distillation trainer."""

    model: ModelConfig = ModelConfig()
    data: DistillDataConfig = DistillDataConfig()
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()
    ckpt: CheckpointConfig | None = None
    weights: WeightCheckpointConfig | None = None
    log: LogConfig = LogConfig()
    wandb: WandbMonitorConfig | None = None

    output_dir: Annotated[Path, Field(description="Directory to write outputs to.")] = Path("outputs")
    max_steps: Annotated[int | None, Field(description="Maximum number of training steps.")] = None

    # Distillation-specific
    temperature: Annotated[float, Field(description="Temperature for softening distributions.")] = 1.0

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
