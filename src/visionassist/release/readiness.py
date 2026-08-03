"""Deterministic Phase 12 release-readiness checks."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from visionassist.data.checksum import sha256_file

SHA256_PATTERN = r"^[a-f0-9]{64}$"
REVISION_PATTERN = r"^[a-f0-9]{40}$"


class AdapterArtifactConfig(BaseModel):
    """Immutable adapter directory and expected files."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    base_model_id: str = Field(min_length=1)
    files: dict[str, str]

    @model_validator(mode="after")
    def validate_files(self) -> AdapterArtifactConfig:
        required = {"adapter_config.json", "adapter_model.safetensors"}
        missing = required - set(self.files)
        if missing:
            raise ValueError(f"Required adapter hashes are missing: {sorted(missing)}")
        invalid = {
            name: digest
            for name, digest in self.files.items()
            if (
                not name
                or Path(name).name != name
                or not re.fullmatch(SHA256_PATTERN, digest)
            )
        }
        if invalid:
            raise ValueError(f"Invalid adapter file hashes: {invalid}")
        return self


class MetricThresholds(BaseModel):
    """Minimum or maximum metrics required by a release gate."""

    model_config = ConfigDict(extra="forbid")

    max_failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_binary_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    min_defect_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    min_evidence_fact_coverage: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    min_localization_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    min_adjacent_tolerance_accuracy: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    min_product_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    min_structured_json_validity: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    min_structured_schema_validity: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    min_structured_defect_f1: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    min_technician_note_fact_coverage: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    min_appropriate_abstention_accuracy: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    max_unsupported_root_cause_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    max_unsupported_safety_claim_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )


class AcceptanceConfig(BaseModel):
    """Expected clean-runtime acceptance artifacts."""

    model_config = ConfigDict(extra="forbid")

    run_manifest_path: Path
    assessment_summary_path: Path
    metrics_path: Path
    records: int = Field(ge=1)
    instruction_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    task_quotas: dict[str, int]
    thresholds: MetricThresholds

    @model_validator(mode="after")
    def validate_task_quotas(self) -> AcceptanceConfig:
        if not self.task_quotas or any(
            not task.strip() or quota < 1 for task, quota in self.task_quotas.items()
        ):
            raise ValueError("Acceptance task quotas must be positive.")
        if sum(self.task_quotas.values()) != self.records:
            raise ValueError("Acceptance task quotas must sum to records.")
        return self


class ReleaseReadinessConfig(BaseModel):
    """Complete Phase 12 release contract."""

    model_config = ConfigDict(extra="forbid")

    release_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=REVISION_PATTERN)
    processor_revision: str = Field(pattern=REVISION_PATTERN)
    promoted: AdapterArtifactConfig
    rollback: AdapterArtifactConfig
    promotion_evidence_path: Path
    promotion_thresholds: MetricThresholds
    acceptance: AcceptanceConfig
    model_card_path: Path
    rollback_runbook_path: Path
    report_path: Path

    @model_validator(mode="after")
    def validate_model_alignment(self) -> ReleaseReadinessConfig:
        if self.promoted.base_model_id != self.model_id:
            raise ValueError("Promoted adapter base model does not match model_id.")
        if self.rollback.base_model_id != self.model_id:
            raise ValueError("Rollback adapter base model does not match model_id.")
        return self


def load_release_readiness_config(path: Path) -> ReleaseReadinessConfig:
    """Load a Phase 12 YAML contract."""

    if not path.is_file():
        raise FileNotFoundError(f"Release configuration not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return ReleaseReadinessConfig.model_validate(payload)


def _resolved(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _check(
    checks: list[dict[str, Any]],
    name: str,
    status: Literal["pass", "fail", "pending"],
    detail: str,
) -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def _verify_adapter(
    root: Path,
    label: str,
    artifact: AdapterArtifactConfig,
    checks: list[dict[str, Any]],
) -> None:
    directory = _resolved(root, artifact.path)
    if not directory.is_dir():
        _check(checks, f"{label}_adapter_directory", "fail", f"Missing: {directory}")
        return
    _check(checks, f"{label}_adapter_directory", "pass", str(directory))
    for name, expected in sorted(artifact.files.items()):
        path = directory / name
        if not path.is_file():
            _check(checks, f"{label}_{name}", "fail", f"Missing: {path}")
            continue
        actual = sha256_file(path)
        status: Literal["pass", "fail"] = "pass" if actual == expected else "fail"
        _check(
            checks,
            f"{label}_{name}",
            status,
            f"sha256={actual}; expected={expected}",
        )
    adapter_config_path = directory / "adapter_config.json"
    if adapter_config_path.is_file():
        try:
            payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            actual_model = payload.get("base_model_name_or_path")
            status = "pass" if actual_model == artifact.base_model_id else "fail"
            _check(
                checks,
                f"{label}_base_model",
                status,
                f"actual={actual_model}; expected={artifact.base_model_id}",
            )
        except (OSError, json.JSONDecodeError) as exc:
            _check(checks, f"{label}_adapter_config", "fail", str(exc))


def _metric_values(evidence: dict[str, Any]) -> dict[str, float | None]:
    test = evidence.get("test", {})
    return {
        "failure_rate": test.get("failure_rate"),
        "binary_accuracy": test.get("binary_accuracy"),
        "defect_f1": test.get("defect_f1"),
        "evidence_fact_coverage": test.get("evidence_fact_coverage"),
        "localization_accuracy": test.get("localization_accuracy"),
        "adjacent_tolerance_accuracy": test.get("adjacent_tolerance_accuracy"),
        "product_accuracy": test.get("product_accuracy"),
        "structured_json_validity": test.get("structured_json_validity"),
        "structured_schema_validity": test.get("structured_schema_validity"),
        "structured_defect_f1": test.get("structured_defect_f1"),
        "technician_note_fact_coverage": test.get(
            "technician_note_fact_coverage"
        ),
        "appropriate_abstention_accuracy": test.get(
            "appropriate_abstention_accuracy"
        ),
        "unsupported_root_cause_rate": test.get("unsupported_root_cause_rate"),
        "unsupported_safety_claim_rate": test.get(
            "unsupported_safety_claim_rate"
        ),
    }


def _acceptance_metric_values(metrics: dict[str, Any]) -> dict[str, float | None]:
    tasks = metrics.get("per_task", {})
    return {
        "failure_rate": metrics.get("failure_rate"),
        "binary_accuracy": tasks.get("binary_inspection", {}).get("accuracy"),
        "defect_f1": tasks.get("defect_identification", {}).get("f1"),
        "evidence_fact_coverage": tasks.get("evidence_explanation", {}).get(
            "mean_fact_coverage"
        ),
        "localization_accuracy": tasks.get("localization", {}).get("accuracy"),
        "adjacent_tolerance_accuracy": tasks.get("localization", {}).get(
            "adjacent_tolerance_accuracy"
        ),
        "product_accuracy": tasks.get("product_identification", {}).get(
            "accuracy"
        ),
        "structured_json_validity": tasks.get("structured_report", {}).get(
            "json_valid_rate"
        ),
        "structured_schema_validity": tasks.get("structured_report", {}).get(
            "schema_valid_rate"
        ),
        "structured_defect_f1": tasks.get("structured_report", {}).get(
            "defect_f1"
        ),
        "technician_note_fact_coverage": tasks.get("technician_note", {}).get(
            "mean_fact_coverage"
        ),
        "appropriate_abstention_accuracy": tasks.get("uncertainty", {}).get(
            "appropriate_abstention_accuracy"
        ),
        "unsupported_root_cause_rate": max(
            (
                values.get("unsupported_root_cause_rate") or 0.0
                for values in tasks.values()
            ),
            default=0.0,
        ),
        "unsupported_safety_claim_rate": max(
            (
                values.get("unsupported_safety_claim_rate") or 0.0
                for values in tasks.values()
            ),
            default=0.0,
        ),
    }


def _verify_thresholds(
    label: str,
    values: dict[str, float | None],
    thresholds: MetricThresholds,
    checks: list[dict[str, Any]],
) -> None:
    for field, threshold in thresholds.model_dump(exclude_none=True).items():
        if field.startswith("max_"):
            metric = field[4:]
            relation = "<="
            value = values.get(metric)
            passed = value is not None and value <= threshold
        else:
            metric = field[4:]
            relation = ">="
            value = values.get(metric)
            passed = value is not None and value >= threshold
        _check(
            checks,
            f"{label}_{metric}",
            "pass" if passed else "fail",
            f"actual={value}; required {relation} {threshold}",
        )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def evaluate_release_readiness(
    config: ReleaseReadinessConfig,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate immutable artifacts, promotion evidence, and acceptance output."""

    root = (project_root or Path.cwd()).resolve()
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "model_revision_pinned",
        "pass",
        f"{config.model_id}@{config.model_revision}",
    )
    _check(
        checks,
        "processor_revision_pinned",
        "pass",
        config.processor_revision,
    )
    _verify_adapter(root, "promoted", config.promoted, checks)
    _verify_adapter(root, "rollback", config.rollback, checks)

    for label, document in (
        ("model_card", config.model_card_path),
        ("rollback_runbook", config.rollback_runbook_path),
    ):
        path = _resolved(root, document)
        _check(
            checks,
            label,
            "pass" if path.is_file() else "fail",
            str(path),
        )

    evidence_path = _resolved(root, config.promotion_evidence_path)
    if evidence_path.is_file():
        try:
            evidence = _load_json(evidence_path)
            decision = evidence.get("decision", {}).get("status")
            _check(
                checks,
                "promotion_decision",
                "pass" if decision == "promoted" else "fail",
                f"decision={decision}",
            )
            errors = evidence.get("test", {}).get("inference_errors")
            _check(
                checks,
                "promotion_inference_errors",
                "pass" if errors == 0 else "fail",
                f"inference_errors={errors}",
            )
            _verify_thresholds(
                "promotion",
                _metric_values(evidence),
                config.promotion_thresholds,
                checks,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _check(checks, "promotion_evidence", "fail", str(exc))
    else:
        _check(checks, "promotion_evidence", "fail", f"Missing: {evidence_path}")

    acceptance_paths = {
        "manifest": _resolved(root, config.acceptance.run_manifest_path),
        "summary": _resolved(root, config.acceptance.assessment_summary_path),
        "metrics": _resolved(root, config.acceptance.metrics_path),
    }
    missing_acceptance = [
        label for label, path in acceptance_paths.items() if not path.is_file()
    ]
    if missing_acceptance:
        _check(
            checks,
            "clean_runtime_acceptance",
            "pending",
            f"Missing acceptance artifacts: {missing_acceptance}",
        )
    else:
        try:
            manifest = _load_json(acceptance_paths["manifest"])
            summary = _load_json(acceptance_paths["summary"])
            metrics = _load_json(acceptance_paths["metrics"])
            manifest_ok = all(
                (
                    manifest.get("status") == "complete",
                    manifest.get("model_id") == config.model_id,
                    manifest.get("model_revision") == config.model_revision,
                    manifest.get("processor_revision") == config.processor_revision,
                    manifest.get("benchmark_records") == config.acceptance.records,
                    manifest.get("completed_predictions") == config.acceptance.records,
                    manifest.get("errors") == 0,
                    manifest.get("instruction_ids_sha256")
                    == config.acceptance.instruction_ids_sha256,
                    manifest.get("subset_task_quotas")
                    == config.acceptance.task_quotas,
                    manifest.get("adapter_file_sha256", {}).get(
                        "adapter_model.safetensors"
                    )
                    == config.promoted.files["adapter_model.safetensors"],
                )
            )
            summary_ok = all(
                (
                    summary.get("records") == config.acceptance.records,
                    summary.get("predictions") == config.acceptance.records,
                    summary.get("inference_errors") == 0,
                )
            )
            _check(
                checks,
                "clean_runtime_manifest",
                "pass" if manifest_ok else "fail",
                "Pinned identities, task quotas, adapter hash, and counts",
            )
            _check(
                checks,
                "clean_runtime_summary",
                "pass" if summary_ok else "fail",
                "records="
                f"{summary.get('records')}; "
                f"errors={summary.get('inference_errors')}",
            )
            _verify_thresholds(
                "acceptance",
                _acceptance_metric_values(metrics),
                config.acceptance.thresholds,
                checks,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _check(checks, "clean_runtime_acceptance", "fail", str(exc))

    statuses = {check["status"] for check in checks}
    if "fail" in statuses:
        status = "blocked"
    elif "pending" in statuses:
        status = "pending"
    else:
        status = "ready"
    return {
        "schema_version": "1.0",
        "release_id": config.release_id,
        "status": status,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "processor_revision": config.processor_revision,
        "promoted_adapter": config.promoted.path.as_posix(),
        "rollback_adapter": config.rollback.path.as_posix(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "counts": {
            state: sum(check["status"] == state for check in checks)
            for state in ("pass", "fail", "pending")
        },
    }


def write_release_readiness_report(
    config: ReleaseReadinessConfig,
    *,
    project_root: Path | None = None,
) -> Path:
    """Write the Phase 12 readiness report atomically."""

    root = (project_root or Path.cwd()).resolve()
    report = evaluate_release_readiness(config, project_root=root)
    path = _resolved(root, config.report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
