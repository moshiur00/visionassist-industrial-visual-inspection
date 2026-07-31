"""Configuration loading for VisA acquisition and audit commands."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class VisaConfig(BaseModel):
    """Validated configuration for Phase 1 of the VisA pipeline."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "visa"
    version: str
    download_url: HttpUrl
    archive_path: Path
    raw_root: Path
    report_root: Path
    manifest_path: Path
    archive_receipt_path: Path
    license_report_path: Path
    dataset_card_path: Path
    expected_total_images: int = Field(gt=0)
    expected_normal_images: int = Field(gt=0)
    expected_anomalous_images: int = Field(gt=0)
    expected_categories: list[str] = Field(min_length=1)
    allowed_image_extensions: list[str] = Field(min_length=1)
    verify_images: bool = True
    compute_sha256: bool = True
    chunk_size_bytes: int = Field(default=1024 * 1024, ge=64 * 1024)
    request_timeout_seconds: int = Field(default=60, ge=1, le=600)
    canonical_manifest_path: Path = Path("data/interim/visa_canonical.jsonl")
    phase2_report_path: Path = Path("reports/dataset_audit/visa_phase2_validation.json")
    phase2_error_path: Path = Path("reports/dataset_audit/visa_phase2_errors.jsonl")
    strict_phase2: bool = True
    require_binary_masks: bool = True
    phase3_manifest_path: Path = Path("data/interim/visa_features.jsonl")
    phase3_report_path: Path = Path(
        "reports/dataset_audit/visa_phase3_validation.json"
    )
    phase3_error_path: Path = Path(
        "reports/dataset_audit/visa_phase3_errors.jsonl"
    )
    strict_phase3: bool = True
    phase3_minor_max_area_ratio: float = Field(default=0.005, gt=0.0, lt=1.0)
    phase3_moderate_max_area_ratio: float = Field(default=0.02, gt=0.0, lt=1.0)
    phase3_major_keywords: list[str] = Field(
        default_factory=lambda: [
            "missing",
            "misplaced",
            "damaged",
            "crack",
            "broken",
        ]
    )
    phase4_split_root: Path = Path("data/splits/vlm_supervised")
    phase4_assignment_path: Path = Path(
        "data/splits/vlm_supervised/split_assignments.csv"
    )
    phase4_report_path: Path = Path(
        "reports/dataset_audit/visa_phase4_validation.json"
    )
    phase4_error_path: Path = Path(
        "reports/dataset_audit/visa_phase4_errors.jsonl"
    )
    strict_phase4: bool = True
    phase4_seed: int = 42
    phase4_train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    phase4_validation_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    phase5_output_root: Path = Path("data/processed/visa_instructions")
    phase5_report_path: Path = Path(
        "reports/dataset_audit/visa_phase5_validation.json"
    )
    phase5_error_path: Path = Path(
        "reports/dataset_audit/visa_phase5_errors.jsonl"
    )
    strict_phase5: bool = True
    phase5_normal_instructions_per_image: int = Field(default=3, ge=1, le=20)
    phase5_anomalous_instructions_per_image: int = Field(default=20, ge=1, le=20)
    phase6_report_root: Path = Path("reports/training_readiness")
    phase6_report_path: Path = Path(
        "reports/training_readiness/visa_phase6_validation.json"
    )
    phase6_error_path: Path = Path(
        "reports/training_readiness/visa_phase6_errors.jsonl"
    )
    phase6_statistics_path: Path = Path(
        "reports/training_readiness/visa_phase6_statistics.json"
    )
    phase6_gallery_path: Path = Path(
        "reports/training_readiness/visa_phase6_sample_gallery.html"
    )
    phase6_processor_report_path: Path = Path(
        "reports/training_readiness/visa_phase6_processor.json"
    )
    phase6_sequence_statistics_path: Path = Path(
        "reports/training_readiness/visa_phase6_sequence_statistics.json"
    )
    phase6_gallery_review_path: Path = Path(
        "reports/training_readiness/visa_phase6_gallery_review.json"
    )
    strict_phase6: bool = True
    phase6_expected_instructions: int = Field(default=52863, gt=0)
    phase6_gallery_samples_per_family: int = Field(default=6, ge=1, le=50)
    phase6_seed: int = 42
    phase6_processor_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    phase6_processor_sample_size: int = Field(default=256, ge=1, le=5000)
    phase6_analysis_sequence_limit: int = Field(default=4096, ge=128)
    phase6_max_sequence_length: int | None = Field(default=None, ge=128)
    phase6_trust_remote_code: bool = False

    @model_validator(mode="after")
    def validate_phase3_thresholds(self) -> VisaConfig:
        if self.phase3_minor_max_area_ratio >= self.phase3_moderate_max_area_ratio:
            raise ValueError(
                "phase3_minor_max_area_ratio must be below "
                "phase3_moderate_max_area_ratio."
            )
        if self.phase4_train_ratio + self.phase4_validation_ratio >= 1.0:
            raise ValueError(
                "phase4_train_ratio + phase4_validation_ratio must be below 1.0."
            )
        return self

    @property
    def phase4_test_ratio(self) -> float:
        """Return the remaining Phase 4 ratio assigned to the test split."""

        return 1.0 - self.phase4_train_ratio - self.phase4_validation_ratio


# Backward-compatible alias used by the existing audit module.
VisaAuditConfig = VisaConfig


def load_visa_config(path: Path) -> VisaConfig:
    """Read and validate a YAML configuration file."""

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return VisaConfig.model_validate(payload)


def load_visa_audit_config(path: Path) -> VisaConfig:
    """Compatibility wrapper for older callers."""

    return load_visa_config(path)
