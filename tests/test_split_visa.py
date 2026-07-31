"""Tests for deterministic Phase 4 split generation."""

from __future__ import annotations

import json
from pathlib import Path

from visionassist.data.config import load_visa_config
from visionassist.data.split_visa import split_visa
from visionassist.schemas.dataset import (
    Condition,
    DerivedImageRecord,
    SourceSplit,
    VisualSeverity,
)


def _record(index: int, category: str, condition: Condition, sha256: str) -> DerivedImageRecord:
    anomalous = condition is Condition.ANOMALOUS
    payload = {
        "schema_version": "1.1",
        "image_id": f"visa_{category}_{index:04d}",
        "source": "visa",
        "source_version": "2022-09-22",
        "category": category,
        "condition": condition.value,
        "source_split": SourceSplit.UNKNOWN.value,
        "defect_type": "scratch" if anomalous else None,
        "image_path": f"data/raw/visa/{category}/{index:04d}.JPG",
        "mask_path": f"data/raw/visa/{category}/{index:04d}.png" if anomalous else None,
        "annotation_path": f"data/raw/visa/{category}/image_anno.csv",
        "annotation_row": index + 2,
        "width": 100,
        "height": 100,
        "file_size_bytes": 1000,
        "sha256": sha256,
        "mask": (
            {
                "path": f"data/raw/visa/{category}/{index:04d}.png",
                "width": 100,
                "height": 100,
                "foreground_pixels": 100,
                "foreground_ratio": 0.01,
                "is_binary": True,
                "unique_values": [0, 1],
            }
            if anomalous
            else None
        ),
        "anomaly_area_pixels": 100 if anomalous else 0,
        "anomaly_area_ratio": 0.01 if anomalous else 0.0,
        "bounding_box": (
            {
                "x_min": 10,
                "y_min": 10,
                "x_max": 19,
                "y_max": 19,
                "width": 10,
                "height": 10,
                "x_min_normalized": 0.1,
                "y_min_normalized": 0.1,
                "x_max_normalized": 0.2,
                "y_max_normalized": 0.2,
            }
            if anomalous
            else None
        ),
        "centroid": (
            {
                "x": 14.5,
                "y": 14.5,
                "x_normalized": 0.145,
                "y_normalized": 0.145,
            }
            if anomalous
            else None
        ),
        "nine_grid_location": "top_left" if anomalous else None,
        "visual_severity": (
            VisualSeverity.MODERATE.value if anomalous else VisualSeverity.NONE.value
        ),
        "severity_basis": "area_ratio" if anomalous else "normal_sample",
    }
    return DerivedImageRecord.model_validate(payload)


def _config(tmp_path: Path, total: int):
    config = load_visa_config(Path("configs/data/visa.yaml"))
    return config.model_copy(
        update={
            "expected_total_images": total,
            "phase3_manifest_path": tmp_path / "features.jsonl",
            "phase4_split_root": tmp_path / "splits",
            "phase4_assignment_path": tmp_path / "splits" / "assignments.csv",
            "phase4_report_path": tmp_path / "report.json",
            "phase4_error_path": tmp_path / "errors.jsonl",
        }
    )


def _write(path: Path, records: list[DerivedImageRecord]) -> None:
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )


def test_phase4_assigns_every_record_without_leakage(tmp_path: Path) -> None:
    records = [
        _record(i, "pcb1", Condition.NORMAL if i < 20 else Condition.ANOMALOUS, f"{i:064x}")
        for i in range(30)
    ]
    config = _config(tmp_path, len(records))
    _write(config.phase3_manifest_path, records)

    result = split_visa(config)

    assert result.records == 30
    assert result.train_records + result.validation_records + result.test_records == 30
    report = json.loads(config.phase4_report_path.read_text(encoding="utf-8"))
    assert report["leakage_checks"]["passed"] is True
    assert all(report["checks"].values())


def test_phase4_is_deterministic(tmp_path: Path) -> None:
    records = [_record(i, "candle", Condition.NORMAL, f"{i:064x}") for i in range(20)]
    config = _config(tmp_path, len(records))
    _write(config.phase3_manifest_path, records)

    split_visa(config)
    first = config.phase4_assignment_path.read_text(encoding="utf-8")
    split_visa(config)
    second = config.phase4_assignment_path.read_text(encoding="utf-8")

    assert first == second


def test_exact_duplicate_hashes_stay_in_one_split(tmp_path: Path) -> None:
    records = [_record(i, "pcb2", Condition.NORMAL, f"{i:064x}") for i in range(18)]
    records[1] = records[1].model_copy(update={"sha256": records[0].sha256})
    config = _config(tmp_path, len(records))
    _write(config.phase3_manifest_path, records)

    split_visa(config)
    rows = config.phase4_assignment_path.read_text(encoding="utf-8").splitlines()
    selected = [row for row in rows if records[0].image_id in row or records[1].image_id in row]

    assert len(selected) == 2
    assert selected[0].split(",")[1] == selected[1].split(",")[1]
