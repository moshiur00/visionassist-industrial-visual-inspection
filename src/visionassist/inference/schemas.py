"""Configuration models for resumable VisionAssist inference."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationConfig(BaseModel):
    """Deterministic text-generation settings."""

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    do_sample: bool = False
    num_beams: int = Field(default=1, ge=1, le=8)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5.0)
    temperature: float | None = Field(default=None, gt=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sampling(self) -> GenerationConfig:
        if not self.do_sample and (self.temperature is not None or self.top_p is not None):
            raise ValueError("temperature/top_p require do_sample=true.")
        return self

    def model_kwargs(self) -> dict[str, object]:
        """Return arguments accepted by ``model.generate``."""

        values: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            if self.temperature is not None:
                values["temperature"] = self.temperature
            if self.top_p is not None:
                values["top_p"] = self.top_p
        return values


class InferenceConfig(BaseModel):
    """Validated Phase 7C inference configuration."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_revision: str | None = None
    processor_revision: str | None = None
    adapter_path: Path | None = None
    benchmark_path: Path = Path(
        "data/benchmarks/visa_baseline_v1/benchmark.jsonl"
    )
    benchmark_manifest_path: Path = Path(
        "data/benchmarks/visa_baseline_v1/benchmark_manifest.json"
    )
    output_dir: Path = Path("outputs/baseline/qwen2_5_vl_3b_direct")
    partial_predictions_path: Path = Path(
        "outputs/baseline/qwen2_5_vl_3b_direct/predictions.partial.jsonl"
    )
    predictions_path: Path = Path(
        "outputs/baseline/qwen2_5_vl_3b_direct/predictions.jsonl"
    )
    errors_path: Path = Path(
        "outputs/baseline/qwen2_5_vl_3b_direct/inference_errors.jsonl"
    )
    run_manifest_path: Path = Path(
        "outputs/baseline/qwen2_5_vl_3b_direct/run_manifest.json"
    )
    evaluation_records_path: Path | None = None
    expected_dataset_split: Literal["train", "validation", "test"] = "test"
    subset_limit: int | None = Field(default=None, ge=1)
    subset_seed: int = 42
    allow_path_normalized_hash_mismatch: bool = False
    system_prompt: str | None = None
    precision: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    device_map: str = "auto"
    attention_implementation: Literal["auto", "sdpa", "flash_attention_2", "eager"] = (
        "auto"
    )
    load_in_4bit: bool = False
    image_min_pixels: int | None = Field(default=None, ge=1)
    image_max_pixels: int | None = Field(default=None, ge=1)
    trust_remote_code: bool = False
    seed: int = 42
    save_every: int = Field(default=1, ge=1)
    stop_after: int | None = Field(default=None, ge=1)
    max_errors: int = Field(default=20, ge=0)
    retry_failed: bool = True
    overwrite: bool = False
    persistent_output_dir: Path | None = None
    persistent_sync_every: int = Field(default=25, ge=1)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @model_validator(mode="after")
    def validate_image_pixels(self) -> InferenceConfig:
        if (
            self.image_min_pixels is not None
            and self.image_max_pixels is not None
            and self.image_min_pixels > self.image_max_pixels
        ):
            raise ValueError("image_min_pixels cannot exceed image_max_pixels.")
        return self

    @model_validator(mode="after")
    def validate_output_paths(self) -> InferenceConfig:
        managed = (
            self.partial_predictions_path,
            self.predictions_path,
            self.errors_path,
            self.run_manifest_path,
        )
        if self.evaluation_records_path is not None:
            managed += (self.evaluation_records_path,)
        for path in managed:
            if self.output_dir not in path.parents and path != self.output_dir:
                raise ValueError(f"Output path must be inside output_dir: {path}")
        return self


def load_inference_config(path: Path) -> InferenceConfig:
    """Load and validate an inference YAML file."""

    if not path.is_file():
        raise FileNotFoundError(f"Inference configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return InferenceConfig.model_validate(payload)
