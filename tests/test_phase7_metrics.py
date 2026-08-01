from __future__ import annotations

import json
from pathlib import Path

from visionassist.evaluation.metrics import adjacent_location, classification_metrics, set_metrics
from visionassist.evaluation.parsers import (
    parse_condition,
    parse_defects,
    parse_location,
    parse_product,
    parse_uncertainty,
)
from visionassist.evaluation.task_metrics import EvaluationConfig, evaluate_baseline_predictions
from visionassist.schemas.instruction import InstructionRecord


def record(instruction_id: str, task: str, target: str, metadata: dict[str, object]) -> InstructionRecord:
    return InstructionRecord.model_validate(
        {
            "instruction_id": instruction_id,
            "image_id": instruction_id.replace("instruction", "image"),
            "dataset_split": "test",
            "task_family": task,
            "template_id": "test_01",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "data/raw/test.jpg"},
                        {"type": "text", "text": "Inspect."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ],
            "metadata": {
                "source": "visa",
                "answer_format": "text",
                "grounded_by": ["test"],
                **metadata,
            },
        }
    )


def test_prediction_parsers() -> None:
    assert parse_condition('{"condition":"defective"}') == "anomalous"
    assert parse_condition("No anomaly is visible; the item is normal.") == "normal"
    assert parse_product("This is PCB 2.") == "pcb2"
    assert parse_product('{"product":"pipe fryum"}') == "pipe_fryum"
    assert parse_location("The anomaly is in the lower-right region.") == "bottom_right"
    assert parse_defects("The defects are bubble, scratch.", {"bubble,scratch"}) == {
        "bubble",
        "scratch",
    }
    uncertainty = parse_uncertainty(
        "The root cause and safety impact cannot be determined from this image alone."
    )
    assert uncertainty["abstains"]
    assert not uncertainty["unsupported_root_cause_claim"]


def test_dependency_free_metrics() -> None:
    metrics = classification_metrics(["normal", "anomalous"], ["normal", None])
    assert metrics["accuracy"] == 0.5
    assert metrics["unparseable_count"] == 1
    score = set_metrics({"scratch", "bubble"}, {"scratch"})
    assert score["precision"] == 1.0
    assert score["recall"] == 0.5
    assert adjacent_location("center", "top_left")
    assert not adjacent_location("top_left", "bottom_right")


def test_evaluate_predictions_end_to_end(tmp_path: Path) -> None:
    benchmark_records = [
        record(
            "instruction_1",
            "binary_inspection",
            "The item is defective.",
            {
                "category": "pcb1",
                "condition": "anomalous",
                "defect_type": "scratch",
                "location": "center",
                "visual_severity": "minor",
            },
        ),
        record(
            "instruction_2",
            "uncertainty",
            "The root cause cannot be determined from this image alone.",
            {
                "category": "pcb1",
                "condition": "anomalous",
                "defect_type": "scratch",
                "location": "center",
                "visual_severity": "minor",
            },
        ),
    ]
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        "\n".join(item.model_dump_json() for item in benchmark_records) + "\n"
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps(
                    {"instruction_id": "instruction_1", "prediction": "The item is defective."}
                ),
                json.dumps(
                    {
                        "instruction_id": "instruction_2",
                        "prediction": "The root cause cannot be determined from this image alone.",
                    }
                ),
            ]
        )
        + "\n"
    )
    output = tmp_path / "output"
    config = EvaluationConfig(
        output_root=output,
        metrics_path=output / "metrics.json",
        per_task_path=output / "tasks.csv",
        per_category_path=output / "categories.csv",
        failures_path=output / "failures.jsonl",
        parsing_errors_path=output / "parsing.jsonl",
    )
    result = evaluate_baseline_predictions(benchmark, predictions, config)
    assert result.failures == 0
    metrics = json.loads(config.metrics_path.read_text())
    assert metrics["per_task"]["binary_inspection"]["accuracy"] == 1.0
    assert metrics["per_task"]["uncertainty"]["appropriate_abstention_accuracy"] == 1.0
