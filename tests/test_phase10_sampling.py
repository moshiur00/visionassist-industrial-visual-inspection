from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionassist.training.config import Phase8TrainingConfig
from visionassist.training.dataset import VisionAssistJsonlDataset
from visionassist.training.experiment import selection_summary, subset_dataset


def _record(index: int, task: str, condition: str = "anomalous") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "instruction_id": f"instruction-{index}",
        "image_id": f"image-{index}",
        "dataset_split": "train",
        "task_family": task,
        "template_id": "template-01",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data/raw/visa/pcb1/{index}.jpg"},
                    {"type": "text", "text": "Inspect this item."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The item is defective."}],
            },
        ],
        "metadata": {
            "source": "visa",
            "category": "pcb1" if index % 2 else "candle",
            "condition": condition,
            "defect_type": "missing" if condition == "anomalous" else None,
            "location": "center" if condition == "anomalous" else None,
            "visual_severity": "major" if condition == "anomalous" else "none",
            "answer_format": "text",
            "grounded_by": ["image_level_label"],
        },
    }


def _dataset(tmp_path: Path) -> VisionAssistJsonlDataset:
    path = tmp_path / "train.jsonl"
    rows = [
        *[_record(index, "binary_inspection") for index in range(6)],
        *[_record(index, "defect_identification") for index in range(6, 12)],
        *[_record(index, "uncertainty", "normal") for index in range(12, 18)],
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return VisionAssistJsonlDataset(path)


def test_task_quota_selection_is_exact_unique_and_reproducible(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    quotas = {
        "binary_inspection": 3,
        "defect_identification": 4,
        "uncertainty": 2,
    }

    first = subset_dataset(dataset, 9, 42, quotas)
    second = subset_dataset(dataset, 9, 42, quotas)
    first_summary = selection_summary(first)
    second_summary = selection_summary(second)

    assert len(first) == 9
    assert first_summary["unique_instruction_ids"] == 9
    assert first_summary["task_families"] == quotas
    assert first_summary["instruction_ids_sha256"] == second_summary[
        "instruction_ids_sha256"
    ]
    assert set(first_summary["categories"]) == {"candle", "pcb1"}
    assert set(first_summary["conditions"]) == {"anomalous", "normal"}


def test_task_quota_selection_rejects_unavailable_records(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with pytest.raises(ValueError, match="exceeds available"):
        subset_dataset(dataset, 7, 42, {"uncertainty": 7})


def test_task_quotas_must_sum_to_train_limit(tmp_path: Path) -> None:
    payload = {
        "run_id": "pilot",
        "output_dir": str(tmp_path / "output"),
        "data": {
            "train_limit": 10,
            "train_task_quotas": {
                "binary_inspection": 4,
                "defect_identification": 5,
            },
        },
    }
    with pytest.raises(ValueError, match="sum exactly"):
        Phase8TrainingConfig.model_validate(payload)
