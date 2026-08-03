from __future__ import annotations

import json
from pathlib import Path

from visionassist.data.checksum import sha256_file
from visionassist.inference.schemas import InferenceConfig
from visionassist.release.readiness import (
    AcceptanceConfig,
    AdapterArtifactConfig,
    MetricThresholds,
    ReleaseReadinessConfig,
    evaluate_release_readiness,
    write_release_readiness_report,
)

REVISION = "a" * 40


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _adapter(root: Path, name: str) -> AdapterArtifactConfig:
    directory = root / name
    directory.mkdir()
    _write_json(
        directory / "adapter_config.json",
        {"base_model_name_or_path": "model/base"},
    )
    (directory / "adapter_model.safetensors").write_bytes(name.encode())
    return AdapterArtifactConfig(
        path=Path(name),
        base_model_id="model/base",
        files={
            file.name: sha256_file(file)
            for file in directory.iterdir()
            if file.is_file()
        },
    )


def _release_config(root: Path) -> ReleaseReadinessConfig:
    promoted = _adapter(root, "promoted")
    rollback = _adapter(root, "rollback")
    (root / "MODEL_CARD.md").write_text("model card", encoding="utf-8")
    (root / "ROLLBACK.md").write_text("rollback", encoding="utf-8")
    evidence = {
        "decision": {"status": "promoted"},
        "test": {
            "inference_errors": 0,
            "failure_rate": 0.4,
            "binary_accuracy": 0.9,
            "defect_f1": 0.5,
            "evidence_fact_coverage": 0.6,
            "localization_accuracy": 0.5,
            "adjacent_tolerance_accuracy": 0.9,
            "product_accuracy": 0.9,
            "structured_json_validity": 1.0,
            "structured_schema_validity": 1.0,
            "structured_defect_f1": 0.6,
            "technician_note_fact_coverage": 0.7,
            "appropriate_abstention_accuracy": 1.0,
            "unsupported_root_cause_rate": 0.0,
            "unsupported_safety_claim_rate": 0.0,
        },
    }
    _write_json(root / "evidence.json", evidence)
    thresholds = MetricThresholds(
        max_failure_rate=0.5,
        min_defect_f1=0.4,
        max_unsupported_root_cause_rate=0.0,
    )
    return ReleaseReadinessConfig(
        release_id="release-v1",
        model_id="model/base",
        model_revision=REVISION,
        processor_revision=REVISION,
        promoted=promoted,
        rollback=rollback,
        promotion_evidence_path=Path("evidence.json"),
        promotion_thresholds=thresholds,
        acceptance=AcceptanceConfig(
            run_manifest_path=Path("acceptance/run_manifest.json"),
            assessment_summary_path=Path("acceptance/assessment_summary.json"),
            metrics_path=Path("acceptance/evaluation/metrics.json"),
            records=8,
            instruction_ids_sha256="b" * 64,
            task_quotas={"binary_inspection": 4, "structured_report": 4},
            thresholds=MetricThresholds(
                max_failure_rate=0.7,
                min_binary_accuracy=0.7,
                min_structured_json_validity=0.9,
            ),
        ),
        model_card_path=Path("MODEL_CARD.md"),
        rollback_runbook_path=Path("ROLLBACK.md"),
        report_path=Path("release_readiness.json"),
    )


def _write_acceptance(root: Path, config: ReleaseReadinessConfig) -> None:
    _write_json(
        root / config.acceptance.run_manifest_path,
        {
            "status": "complete",
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "processor_revision": config.processor_revision,
            "benchmark_records": 8,
            "completed_predictions": 8,
            "errors": 0,
            "instruction_ids_sha256": config.acceptance.instruction_ids_sha256,
            "subset_task_quotas": config.acceptance.task_quotas,
            "adapter_file_sha256": {
                "adapter_model.safetensors": config.promoted.files[
                    "adapter_model.safetensors"
                ]
            },
        },
    )
    _write_json(
        root / config.acceptance.assessment_summary_path,
        {"records": 8, "predictions": 8, "inference_errors": 0},
    )
    _write_json(
        root / config.acceptance.metrics_path,
        {
            "failure_rate": 0.25,
            "per_task": {
                "binary_inspection": {"accuracy": 0.75},
                "structured_report": {
                    "json_valid_rate": 1.0,
                    "schema_valid_rate": 1.0,
                },
            },
        },
    )


def test_release_is_pending_until_clean_runtime_acceptance(tmp_path: Path) -> None:
    config = _release_config(tmp_path)

    report = evaluate_release_readiness(config, project_root=tmp_path)

    assert report["status"] == "pending"
    assert report["counts"]["fail"] == 0
    assert report["counts"]["pending"] == 1


def test_release_becomes_ready_with_matching_acceptance(tmp_path: Path) -> None:
    config = _release_config(tmp_path)
    _write_acceptance(tmp_path, config)

    report = evaluate_release_readiness(config, project_root=tmp_path)
    output = write_release_readiness_report(config, project_root=tmp_path)

    assert report["status"] == "ready"
    assert report["counts"]["fail"] == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ready"


def test_release_blocks_tampered_promoted_adapter(tmp_path: Path) -> None:
    config = _release_config(tmp_path)
    _write_acceptance(tmp_path, config)
    (tmp_path / "promoted/adapter_model.safetensors").write_bytes(b"tampered")

    report = evaluate_release_readiness(config, project_root=tmp_path)

    assert report["status"] == "blocked"
    assert any(
        check["name"] == "promoted_adapter_model.safetensors"
        and check["status"] == "fail"
        for check in report["checks"]
    )


def test_inference_task_quotas_require_exact_subset_limit() -> None:
    quotas = {"binary_inspection": 4, "structured_report": 4}
    config = InferenceConfig(
        run_id="acceptance",
        output_dir=Path("outputs/acceptance"),
        partial_predictions_path=Path("outputs/acceptance/partial.jsonl"),
        predictions_path=Path("outputs/acceptance/predictions.jsonl"),
        errors_path=Path("outputs/acceptance/errors.jsonl"),
        run_manifest_path=Path("outputs/acceptance/manifest.json"),
        subset_limit=8,
        subset_task_quotas=quotas,
    )

    assert config.subset_task_quotas == quotas
