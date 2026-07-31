"""Tests for Phase 5 grounded instruction generation."""

from __future__ import annotations

import json
from pathlib import Path

from visionassist.data.generate_instructions import (
    _json_answer,
    _record_to_instruction,
    _selected_templates,
    _templates,
)
from visionassist.data.config import VisaConfig
from visionassist.schemas.dataset import DatasetSplit, DerivedImageRecord


def _config(tmp_path: Path) -> VisaConfig:
    return VisaConfig.model_validate(
        {
            "version": "2022-09-22",
            "download_url": "https://example.com/visa.tar",
            "archive_path": tmp_path / "visa.tar",
            "raw_root": tmp_path / "raw",
            "report_root": tmp_path / "reports",
            "manifest_path": tmp_path / "raw.jsonl",
            "archive_receipt_path": tmp_path / "receipt.json",
            "license_report_path": tmp_path / "license.md",
            "dataset_card_path": tmp_path / "card.md",
            "expected_total_images": 2,
            "expected_normal_images": 1,
            "expected_anomalous_images": 1,
            "expected_categories": ["pcb1"],
            "allowed_image_extensions": [".png"],
            "phase5_normal_instructions_per_image": 3,
            "phase5_anomalous_instructions_per_image": 20,
        }
    )


def _record(condition: str) -> DerivedImageRecord:
    anomalous = condition == "anomalous"
    payload = {
        "image_id": f"visa_pcb1_{condition}_001",
        "source_version": "2022-09-22",
        "category": "pcb1",
        "condition": condition,
        "source_split": "unknown",
        "defect_type": "missing_component" if anomalous else None,
        "image_path": f"data/raw/visa/pcb1/{condition}.png",
        "mask_path": "data/raw/visa/pcb1/mask.png" if anomalous else None,
        "annotation_path": "data/raw/visa/pcb1/image_anno.csv",
        "annotation_row": 2,
        "width": 10,
        "height": 10,
        "file_size_bytes": 100,
        "sha256": "a" * 64,
        "mask": {
            "path": "data/raw/visa/pcb1/mask.png",
            "width": 10,
            "height": 10,
            "foreground_pixels": 4,
            "foreground_ratio": 0.04,
            "is_binary": True,
            "unique_values": [0, 1],
        } if anomalous else None,
        "anomaly_area_pixels": 4 if anomalous else 0,
        "anomaly_area_ratio": 0.04 if anomalous else 0.0,
        "bounding_box": {
            "x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2,
            "width": 2, "height": 2,
            "x_min_normalized": 0.1, "y_min_normalized": 0.1,
            "x_max_normalized": 0.3, "y_max_normalized": 0.3,
        } if anomalous else None,
        "centroid": {
            "x": 1.5, "y": 1.5, "x_normalized": 0.15, "y_normalized": 0.15,
        } if anomalous else None,
        "nine_grid_location": "top_left" if anomalous else None,
        "visual_severity": "major" if anomalous else "none",
        "severity_basis": "test",
    }
    return DerivedImageRecord.model_validate(payload)


def test_template_counts_follow_configuration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert len(_selected_templates(_record("normal"), config)) == 3
    assert len(_selected_templates(_record("anomalous"), config)) == 20


def test_instruction_is_qwen_compatible_and_split_bound(tmp_path: Path) -> None:
    record = _record("anomalous")
    instruction = _record_to_instruction(record, DatasetSplit.TRAIN, _templates()[0])
    assert instruction.dataset_split is DatasetSplit.TRAIN
    assert instruction.messages[0].content[0].type == "image"
    assert instruction.messages[0].content[1].type == "text"
    assert instruction.messages[1].role == "assistant"
    assert instruction.metadata.location == "top_left"


def test_structured_answer_is_valid_json() -> None:
    payload = json.loads(_json_answer(_record("anomalous")))
    assert payload["condition"] == "defective"
    assert payload["defect_type"] == "missing component"
    assert "safety_note" in payload


def test_all_required_task_families_exist() -> None:
    families = {template.family for template in _templates()}
    assert families == {
        "binary_inspection",
        "product_identification",
        "defect_identification",
        "localization",
        "evidence_explanation",
        "structured_report",
        "technician_note",
        "uncertainty",
    }
