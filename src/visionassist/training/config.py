"""Validated configuration for Phase 8 QLoRA training infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuantizationConfig(BaseModel):
    """4-bit loading policy for QLoRA."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"


class LoraTrainingConfig(BaseModel):
    """LoRA adapter policy."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_suffixes: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    train_multimodal_projector: bool = False


class DataTrainingConfig(BaseModel):
    """Dataset paths, subset controls, and processor limits."""

    model_config = ConfigDict(extra="forbid")

    train_path: Path = Path("data/processed/visa_instructions/train.jsonl")
    validation_path: Path = Path(
        "data/processed/visa_instructions/validation.jsonl"
    )
    max_sequence_length: int | None = Field(default=4096, ge=128)
    image_min_pixels: int | None = Field(default=None, ge=1)
    image_max_pixels: int | None = Field(default=None, ge=1)
    train_limit: int | None = Field(default=None, ge=1)
    validation_limit: int | None = Field(default=None, ge=1)
    subset_seed: int = 42
    train_task_quotas: dict[str, int] | None = None

    @model_validator(mode="after")
    def validate_pixels(self) -> DataTrainingConfig:
        if (
            self.image_min_pixels is not None
            and self.image_max_pixels is not None
            and self.image_min_pixels > self.image_max_pixels
        ):
            raise ValueError("image_min_pixels cannot exceed image_max_pixels.")
        if self.train_task_quotas is not None:
            if not self.train_task_quotas:
                raise ValueError("train_task_quotas cannot be empty.")
            invalid = {
                task: quota
                for task, quota in self.train_task_quotas.items()
                if not task.strip() or quota < 1
            }
            if invalid:
                raise ValueError(
                    "train_task_quotas requires non-empty tasks and positive "
                    f"quotas: {invalid}"
                )
            quota_total = sum(self.train_task_quotas.values())
            if self.train_limit is None:
                raise ValueError("train_limit is required with train_task_quotas.")
            if quota_total != self.train_limit:
                raise ValueError(
                    "train_task_quotas must sum exactly to train_limit: "
                    f"{quota_total} != {self.train_limit}."
                )
        return self


class TrainerRuntimeConfig(BaseModel):
    """Transformers Trainer runtime settings."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=-1, ge=-1)
    num_train_epochs: float = Field(default=1.0, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    gradient_checkpointing: bool = True
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    logging_steps: int = Field(default=5, ge=1)
    eval_steps: int = Field(default=25, ge=1)
    save_steps: int = Field(default=25, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    dataloader_num_workers: int = Field(default=0, ge=0)
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    report_to: list[str] = Field(default_factory=list)
    tf32: bool = True

    @model_validator(mode="after")
    def validate_save_eval_alignment(self) -> TrainerRuntimeConfig:
        if self.save_steps % self.eval_steps != 0:
            raise ValueError(
                "save_steps must be a multiple of eval_steps when loading the "
                "best model at the end."
            )
        return self


class CheckpointConfig(BaseModel):
    """Checkpoint retention and persistence settings."""

    model_config = ConfigDict(extra="forbid")

    resume: Literal["none", "latest", "best"] | Path = "latest"
    keep_latest: int = Field(default=2, ge=1)
    keep_best: int = Field(default=1, ge=1)
    persistent_output_dir: Path | None = None
    sync_every_save: bool = True


class Phase8TrainingConfig(BaseModel):
    """Complete Phase 8 configuration."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    initial_adapter_path: Path | None = None
    model_revision: str | None = None
    processor_revision: str | None = None
    output_dir: Path
    seed: int = 42
    trust_remote_code: bool = False
    attention_implementation: Literal[
        "auto", "sdpa", "flash_attention_2", "eager"
    ] = "auto"
    hardware_profile: Literal["auto", "low_vram", "standard_vram", "high_vram"] = (
        "auto"
    )
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraTrainingConfig = Field(default_factory=LoraTrainingConfig)
    data: DataTrainingConfig = Field(default_factory=DataTrainingConfig)
    training: TrainerRuntimeConfig = Field(default_factory=TrainerRuntimeConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def align_checkpoint_limit(self) -> Phase8TrainingConfig:
        minimum = self.checkpoints.keep_latest + self.checkpoints.keep_best
        if self.training.save_total_limit < minimum:
            raise ValueError(
                "training.save_total_limit must be at least "
                "checkpoints.keep_latest + checkpoints.keep_best."
            )
        return self


def load_training_config(path: Path) -> Phase8TrainingConfig:
    """Load a Phase 8 YAML configuration."""

    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return Phase8TrainingConfig.model_validate(payload)
