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
