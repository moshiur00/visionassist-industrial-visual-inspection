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


def test_hardened_condition_parser_handles_yes_no_annotation_phrases() -> None:
    assert parse_condition("No, there are no annotated anomalies present in the image.") == "normal"
    assert parse_condition("Yes, there is an annotated anomaly in the image.") == "anomalous"
    assert parse_condition('{"status":"pass"}') == "normal"
    assert parse_condition('{"results":[{"pass":"FAIL"}]}') is None


def test_hardened_product_parser_is_conservative() -> None:
    assert parse_product("The product category is printed circuit board 2.") == "pcb2"
    assert parse_product("The image shows buttons or poker chips.") is None


def test_hardened_location_parser_handles_row_column() -> None:
    assert parse_location("The anomaly is in the second row, third column.") == "center_right"
    assert parse_location("The anomaly is in the first row, second column.") == "top_center"


def test_semantic_defect_parser_preserves_strict_and_normalizes_variants() -> None:
    vocabulary = {"small_scratches", "different_colour_spot"}
    assert parse_defects("There are small scratches.", vocabulary) == {"small_scratches"}
    assert parse_defects("There is a small scratch.", vocabulary, semantic=True) == {"small_scratch"}

def test_structured_report_invalid_json_does_not_break_aggregation(
    tmp_path: Path,
) -> None:
    benchmark_record = record(
        "instruction_structured_invalid",
        "structured_report",
        json.dumps(
            {
                "product": "pcb1",
                "condition": "defective",
                "defect_type": "scratch",
                "location": "center",
                "visual_severity": "minor",
                "recommended_action": "Send for manual review.",
                "safety_note": (
                    "Root cause cannot be determined from the image alone."
                ),
            }
        ),
        {
            "category": "pcb1",
            "condition": "anomalous",
            "defect_type": "scratch",
            "location": "center",
            "visual_severity": "minor",
            "answer_format": "json",
        },
    )

    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        benchmark_record.model_dump_json() + "\n",
        encoding="utf-8",
    )

    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "instruction_id": benchmark_record.instruction_id,
                "prediction": "This is not valid JSON.",
            }
        )
        + "\n",
        encoding="utf-8",
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

    result = evaluate_baseline_predictions(
        benchmark,
        predictions,
        config,
    )

    assert result.failures == 1

    metrics = json.loads(
        config.metrics_path.read_text(encoding="utf-8")
    )
    structured = metrics["per_task"]["structured_report"]

    assert structured["count"] == 1
    assert structured["json_valid_rate"] == 0.0
    assert structured["defect_f1"] == 0.0
    assert structured["strict_defect_f1"] == 0.0