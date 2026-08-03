from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionassist.training.hard_examples import (
    HardExampleConfig,
    select_hard_examples,
    write_hard_examples,
)


def _instruction(
    index: int,
    split: str,
    task: str,
    *,
    category: str = "pcb1",
    defect: str = "melt",
    location: str = "center_left",
) -> dict[str, object]:
    return {
        "instruction_id": f"{split}-{index}-{task}",
        "image_id": f"{split}-image-{index}",
        "dataset_split": split,
        "task_family": task,
        "template_id": "template",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data/{split}/{index}.jpg"},
                    {"type": "text", "text": "Inspect."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Defective."}],
            },
        ],
        "metadata": {
            "category": category,
            "condition": "anomalous",
            "defect_type": defect,
            "location": location,
            "visual_severity": "major",
            "answer_format": "text",
            "grounded_by": ["defect_label"],
        },
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _config(tmp_path: Path) -> HardExampleConfig:
    tasks = ("defect_identification", "localization", "binary_inspection")
    train = [
        _instruction(index, "train", task)
        for task in tasks
        for index in range(5)
    ]
    validation = [
        _instruction(100, "validation", "defect_identification"),
        _instruction(101, "validation", "localization"),
    ]
    test = [_instruction(200, "test", "binary_inspection")]
    _write(tmp_path / "train.jsonl", train)
    _write(tmp_path / "validation.jsonl", validation)
    _write(tmp_path / "test.jsonl", test)
    predictions = [
        {
            "instruction_id": validation[0]["instruction_id"],
            "task_family": "defect_identification",
            "category": "pcb1",
            "condition": "anomalous",
            "defect_type": "melt",
            "location": "center_left",
            "prediction": "The annotated defect is missing.",
        },
        {
            "instruction_id": validation[1]["instruction_id"],
            "task_family": "localization",
            "category": "pcb1",
            "condition": "anomalous",
            "defect_type": "melt",
            "location": "center_left",
            "prediction": "The defect is in the center.",
        },
    ]
    _write(tmp_path / "predictions.jsonl", predictions)
    return HardExampleConfig(
        run_id="hard-v1",
        seed=43,
        train_path=tmp_path / "train.jsonl",
        validation_path=tmp_path / "validation.jsonl",
        test_path=tmp_path / "test.jsonl",
        validation_predictions_path=tmp_path / "predictions.jsonl",
        output_path=tmp_path / "selected.jsonl",
        manifest_path=tmp_path / "manifest.json",
        min_per_category_per_task=0,
        task_quotas={
            "binary_inspection": 2,
            "defect_identification": 3,
            "localization": 3,
        },
    )


def test_selection_is_exact_reproducible_and_leakage_safe(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first, first_manifest = select_hard_examples(config)
    second, second_manifest = select_hard_examples(config)

    assert [row.instruction_id for row in first] == [
        row.instruction_id for row in second
    ]
    assert first_manifest == second_manifest
    assert first_manifest["records"] == 8
    assert first_manifest["unique_instruction_ids"] == 8
    assert first_manifest["task_counts"] == config.task_quotas
    assert first_manifest["leakage"] == {
        "validation_image_overlap": 0,
        "test_image_overlap": 0,
    }
    output, manifest = write_hard_examples(config)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 8
    assert json.loads(manifest.read_text(encoding="utf-8"))["records"] == 8


def test_selection_enforces_category_floor(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.min_per_category_per_task = 1
    selected, manifest = select_hard_examples(config)

    assert len(selected) == 8
    for counts in manifest["task_category_counts"].values():
        assert counts == {"pcb1": sum(counts.values())}


def test_selection_enforces_task_condition_quotas(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.task_condition_quotas = {
        task: {"anomalous": quota} for task, quota in config.task_quotas.items()
    }
    _, manifest = select_hard_examples(config)

    assert manifest["task_condition_counts"] == config.task_condition_quotas


def test_selection_rejects_train_held_out_image_overlap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    validation = [
        _instruction(0, "validation", "defect_identification")
    ]
    validation[0]["image_id"] = "train-image-0"
    _write(config.validation_path, validation)
    _write(
        config.validation_predictions_path,
        [
            {
                "instruction_id": validation[0]["instruction_id"],
                "task_family": "defect_identification",
                "category": "pcb1",
                "condition": "anomalous",
                "defect_type": "melt",
                "location": "center_left",
                "prediction": "missing",
            }
        ],
    )

    with pytest.raises(ValueError, match="image leakage"):
        select_hard_examples(config)
