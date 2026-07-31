"""Tests for Phase 6 training-readiness validation."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from visionassist.data.config import VisaConfig
from visionassist.training.formatting import qwen_messages, resolve_image_path
from visionassist.training.readiness import validate_training_readiness


def _config(tmp_path: Path) -> VisaConfig:
    return VisaConfig.model_validate(
        {
            "version": "test",
            "download_url": "https://example.com/visa.tar",
            "archive_path": tmp_path / "archive.tar",
            "raw_root": tmp_path / "data/raw/visa",
            "report_root": tmp_path / "reports/dataset_audit",
            "manifest_path": tmp_path / "data/manifests/raw.jsonl",
            "archive_receipt_path": tmp_path / "receipt.json",
            "license_report_path": tmp_path / "license.md",
            "dataset_card_path": tmp_path / "DATASET_CARD.md",
            "expected_total_images": 1,
            "expected_normal_images": 1,
            "expected_anomalous_images": 1,
            "expected_categories": ["pcb1"],
            "allowed_image_extensions": [".png"],
            "phase5_output_root": tmp_path / "data/processed/visa_instructions",
            "phase6_report_root": tmp_path / "reports/training_readiness",
            "phase6_report_path": tmp_path / "reports/training_readiness/report.json",
            "phase6_error_path": tmp_path / "reports/training_readiness/errors.jsonl",
            "phase6_statistics_path": tmp_path / "reports/training_readiness/stats.json",
            "phase6_gallery_path": tmp_path / "reports/training_readiness/gallery.html",
            "phase6_processor_report_path": tmp_path / "reports/training_readiness/processor.json",
            "phase6_expected_instructions": 1,
            "phase6_gallery_samples_per_family": 1,
            "strict_phase6": False,
        }
    )


def _instruction(image_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "instruction_id": "visa_pcb1_normal_001__binary_01",
        "image_id": "visa_pcb1_normal_001",
        "dataset_split": "train",
        "task_family": "binary_inspection",
        "template_id": "binary_01",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.as_posix()},
                    {"type": "text", "text": "Is this item normal?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The item is normal."}],
            },
        ],
        "metadata": {
            "source": "visa",
            "category": "pcb1",
            "condition": "normal",
            "defect_type": None,
            "location": None,
            "visual_severity": "none",
            "answer_format": "text",
            "grounded_by": ["image_level_label"],
        },
    }


def test_resolve_image_path_and_qwen_messages(tmp_path: Path) -> None:
    config = _config(tmp_path)
    relative = Path("data/raw/visa/pcb1/001.png")
    image_path = tmp_path / relative
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(image_path)
    from visionassist.schemas.instruction import InstructionRecord

    record = InstructionRecord.model_validate(_instruction(relative))
    assert resolve_image_path(record, tmp_path) == image_path.resolve()
    messages = qwen_messages(record, tmp_path)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_phase6_writes_reports_for_valid_record(tmp_path: Path) -> None:
    config = _config(tmp_path)
    relative = Path("data/raw/visa/pcb1/001.png")
    image_path = tmp_path / relative
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(image_path)
    config.phase5_output_root.mkdir(parents=True)
    (config.phase5_output_root / "train.jsonl").write_text(
        json.dumps(_instruction(relative)) + "\n", encoding="utf-8"
    )
    for split in ("validation", "test"):
        (config.phase5_output_root / f"{split}.jsonl").write_text("", encoding="utf-8")

    result = validate_training_readiness(config, project_root=tmp_path)
    assert result.instructions == 1
    assert result.unique_images == 1
    assert result.errors == 0
    assert result.report_path.is_file()
    assert result.statistics_path.is_file()
    assert result.gallery_path.is_file()


def test_phase6_detects_missing_image(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.phase5_output_root.mkdir(parents=True)
    (config.phase5_output_root / "train.jsonl").write_text(
        json.dumps(_instruction(Path("data/raw/visa/missing.png"))) + "\n",
        encoding="utf-8",
    )
    for split in ("validation", "test"):
        (config.phase5_output_root / f"{split}.jsonl").write_text("", encoding="utf-8")

    result = validate_training_readiness(config, project_root=tmp_path)
    assert result.errors == 1
    assert "image does not exist" in result.error_path.read_text(encoding="utf-8")


def test_phase6_extended_statistics_and_review_record(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.phase6_gallery_review_path = tmp_path / "reports/training_readiness/review.json"
    relative = Path("data/raw/visa/pcb1/001.png")
    image_path = tmp_path / relative
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(image_path)
    config.phase5_output_root.mkdir(parents=True)
    (config.phase5_output_root / "train.jsonl").write_text(
        json.dumps(_instruction(relative)) + "\n", encoding="utf-8"
    )
    for split in ("validation", "test"):
        (config.phase5_output_root / f"{split}.jsonl").write_text("", encoding="utf-8")

    result = validate_training_readiness(
        config,
        project_root=tmp_path,
        approve_gallery=True,
        reviewer="Unit Tester",
    )
    statistics = json.loads(result.statistics_path.read_text(encoding="utf-8"))
    for key in (
        "defect_label_counts",
        "location_counts",
        "severity_counts",
        "repeated_prompts",
        "repeated_answers",
        "instructions_per_image",
        "unique_prompt_template_combinations",
    ):
        assert key in statistics
    review = json.loads(result.gallery_review_path.read_text(encoding="utf-8"))
    assert review["review_status"] == "approved"
    assert review["reviewed_by"] == "Unit Tester"


def test_structured_json_defect_spacing_is_semantically_equal() -> None:
    from visionassist.training.readiness import _json_values_match

    assert _json_values_match(
        "defect_type",
        "chunk of wax missing, foreign particals on candle",
        "chunk of wax missing,foreign particals on candle",
    )
    assert _json_values_match(
        "defect_type",
        "bubble, discolor, scratch",
        "bubble,discolor,scratch",
    )


def test_gallery_sample_contains_every_supported_task_family() -> None:
    from visionassist.schemas.instruction import InstructionRecord
    from visionassist.training.readiness import SUPPORTED_TASKS, _gallery_sample

    records = []
    for index, family in enumerate(sorted(SUPPORTED_TASKS)):
        payload = _instruction(Path(f"data/raw/visa/pcb1/{index:03d}.png"))
        payload["instruction_id"] = f"example-{family}"
        payload["image_id"] = f"image-{family}"
        payload["task_family"] = family
        payload["template_id"] = f"{family}-template"
        records.append(InstructionRecord.model_validate(payload))

    selected = _gallery_sample(records, per_family=1, seed=42)

    assert {record.task_family for record in selected} == SUPPORTED_TASKS
    assert len(selected) == len(SUPPORTED_TASKS)
