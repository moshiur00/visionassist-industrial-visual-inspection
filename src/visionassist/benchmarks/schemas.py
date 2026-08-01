"""Schemas and configuration for frozen VisionAssist benchmarks."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkTaskTargets(BaseModel):
    """Requested number of benchmark records per task family."""

    model_config = ConfigDict(extra="forbid")

    binary_inspection: int = Field(default=400, ge=0)
    product_identification: int = Field(default=300, ge=0)
    defect_identification: int = Field(default=300, ge=0)
    localization: int = Field(default=300, ge=0)
    evidence_explanation: int = Field(default=200, ge=0)
    structured_report: int = Field(default=300, ge=0)
    technician_note: int = Field(default=150, ge=0)
    uncertainty: int = Field(default=150, ge=0)

    @property
    def total(self) -> int:
        """Return the total requested benchmark size."""

        return sum(self.model_dump().values())


class BenchmarkConfig(BaseModel):
    """Validated Phase 7A benchmark configuration."""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str = Field(default="visa_baseline_v1", min_length=1)
    schema_version: str = "1.0"
    source_test_path: Path = Path("data/processed/visa_instructions/test.jsonl")
    output_root: Path = Path("data/benchmarks/visa_baseline_v1")
    benchmark_path: Path = Path("data/benchmarks/visa_baseline_v1/benchmark.jsonl")
    manifest_path: Path = Path(
        "data/benchmarks/visa_baseline_v1/benchmark_manifest.json"
    )
    distribution_path: Path = Path(
        "data/benchmarks/visa_baseline_v1/benchmark_distribution.json"
    )
    sha256_path: Path = Path("data/benchmarks/visa_baseline_v1/benchmark_sha256.txt")
    report_root: Path = Path("reports/baseline")
    validation_report_path: Path = Path(
        "reports/baseline/visa_baseline_v1_validation.json"
    )
    validation_error_path: Path = Path(
        "reports/baseline/visa_baseline_v1_errors.jsonl"
    )
    statistics_path: Path = Path("reports/baseline/visa_baseline_v1_statistics.json")
    seed: int = 42
    strict: bool = True
    verify_images: bool = True
    allowed_image_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png"]
    )
    task_targets: BenchmarkTaskTargets = Field(default_factory=BenchmarkTaskTargets)

    @model_validator(mode="after")
    def validate_paths(self) -> BenchmarkConfig:
        if self.task_targets.total <= 0:
            raise ValueError("At least one benchmark task target must be positive.")
        return self


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load a benchmark YAML configuration."""

    if not path.is_file():
        raise FileNotFoundError(f"Benchmark configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return BenchmarkConfig.model_validate(payload)
