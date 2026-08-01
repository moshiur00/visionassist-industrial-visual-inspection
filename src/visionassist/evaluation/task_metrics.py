"""Evaluate VisionAssist baseline predictions with task-specific parsers."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from visionassist.evaluation.metrics import (
    adjacent_location,
    aggregate_set_metrics,
    classification_metrics,
    safe_divide,
    set_metrics,
)
from visionassist.evaluation.normalize import parse_json_object, split_compound_label
from visionassist.evaluation.parsers import (
    canonicalize_defect_set,
    parse_condition,
    parse_defects,
    parse_location,
    parse_product,
    parse_severity,
    parse_uncertainty,
)
from visionassist.schemas.instruction import InstructionRecord


class EvaluationConfig(BaseModel):
    """Output and policy configuration for Phase 7B evaluation."""

    model_config = ConfigDict(extra="forbid")

    output_root: Path = Path("outputs/baseline/evaluation")
    metrics_path: Path = Path("outputs/baseline/evaluation/metrics.json")
    per_task_path: Path = Path("outputs/baseline/evaluation/per_task_metrics.csv")
    per_category_path: Path = Path(
        "outputs/baseline/evaluation/per_category_metrics.csv"
    )
    failures_path: Path = Path("outputs/baseline/evaluation/failures.jsonl")
    parsing_errors_path: Path = Path(
        "outputs/baseline/evaluation/parsing_errors.jsonl"
    )
    strict_prediction_coverage: bool = True
    required_structured_fields: list[str] = Field(
        default_factory=lambda: [
            "product",
            "condition",
            "defect_type",
            "location",
            "visual_severity",
            "recommended_action",
            "safety_note",
        ]
    )


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load Phase 7B YAML configuration."""

    if not path.is_file():
        raise FileNotFoundError(f"Evaluation configuration not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return EvaluationConfig.model_validate(payload)


@dataclass(frozen=True)
class EvaluationResult:
    """Summary returned by baseline evaluation."""

    benchmark_records: int
    predictions: int
    failures: int
    metrics_path: Path
    per_task_path: Path
    per_category_path: Path
    failures_path: Path


def _assistant_target(record: InstructionRecord) -> str:
    item = record.messages[1].content[0]
    if item.text is None:
        raise ValueError(f"Missing assistant target: {record.instruction_id}")
    return item.text


def _read_benchmark(path: Path) -> dict[str, InstructionRecord]:
    records: dict[str, InstructionRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = InstructionRecord.model_validate_json(line)
            if record.instruction_id in records:
                raise ValueError(
                    f"Duplicate benchmark instruction ID at line {line_number}: "
                    f"{record.instruction_id}"
                )
            records[record.instruction_id] = record
    return records


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Prediction line {line_number} is not an object.")
            instruction_id = payload.get("instruction_id")
            prediction = payload.get("prediction")
            if not isinstance(instruction_id, str) or not isinstance(prediction, str):
                raise ValueError(
                    f"Prediction line {line_number} requires string instruction_id "
                    "and prediction fields."
                )
            if instruction_id in predictions:
                raise ValueError(f"Duplicate prediction ID: {instruction_id}")
            predictions[instruction_id] = payload
    return predictions


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _structured_score(
    record: InstructionRecord,
    prediction: str,
    defect_vocabulary: set[str],
    required_fields: list[str],
) -> tuple[dict[str, Any], list[str]]:
    payload = parse_json_object(prediction)
    failure_tags: list[str] = []
    if payload is None:
        return {
            "json_valid": False,
            "schema_valid": False,
            "field_completeness": 0.0,
            "unsupported_field_count": 0,
            "condition_correct": False,
            "product_correct": False,
            "strict_defect_exact_match": False,
            "strict_defect_f1": 0.0,
            "defect_f1": 0.0,
            "location_correct": False,
            "severity_correct": False,
        }, ["invalid_json"]

    expected_fields = set(required_fields)
    present = {key for key in required_fields if key in payload}
    unsupported = set(payload) - expected_fields
    if present != expected_fields:
        failure_tags.append("missing_required_field")
    if unsupported:
        failure_tags.append("unsupported_field")

    condition = parse_condition(prediction)
    product = parse_product(prediction)
    location = parse_location(prediction)
    severity = parse_severity(prediction)
    true_defects = split_compound_label(record.metadata.defect_type)
    pred_defects = parse_defects(prediction, defect_vocabulary)
    strict_defect_result = set_metrics(true_defects, pred_defects)
    semantic_true_defects = canonicalize_defect_set(true_defects)
    semantic_pred_defects = parse_defects(prediction, defect_vocabulary, semantic=True)
    defect_result = set_metrics(semantic_true_defects, semantic_pred_defects)
    condition_correct = condition == record.metadata.condition
    product_correct = product == record.metadata.category
    location_correct = location == record.metadata.location
    severity_correct = severity == record.metadata.visual_severity
    if not condition_correct:
        failure_tags.append("wrong_condition")
    if not product_correct:
        failure_tags.append("wrong_product")
    if not bool(defect_result["exact_match"]):
        failure_tags.append("wrong_defect")
    if not location_correct:
        failure_tags.append("wrong_location")
    if not severity_correct:
        failure_tags.append("wrong_severity")

    return {
        "json_valid": True,
        "schema_valid": present == expected_fields,
        "field_completeness": safe_divide(len(present), len(expected_fields)),
        "unsupported_field_count": len(unsupported),
        "condition_correct": condition_correct,
        "product_correct": product_correct,
        "strict_defect_exact_match": bool(strict_defect_result["exact_match"]),
        "strict_defect_f1": float(strict_defect_result["f1"]),
        "defect_f1": float(defect_result["f1"]),
        "location_correct": location_correct,
        "severity_correct": severity_correct,
    }, failure_tags


def _evaluate_one(
    record: InstructionRecord,
    prediction: str,
    defect_vocabulary: set[str],
    config: EvaluationConfig,
) -> tuple[dict[str, Any], list[str]]:
    task = record.task_family
    failures: list[str] = []
    result: dict[str, Any] = {
        "instruction_id": record.instruction_id,
        "task_family": task,
        "category": record.metadata.category,
        "condition": record.metadata.condition,
    }

    uncertainty = parse_uncertainty(prediction)
    if uncertainty["unsupported_root_cause_claim"]:
        failures.append("unsupported_root_cause")
    if uncertainty["unsupported_safety_claim"]:
        failures.append("unsupported_safety_claim")
    result.update(uncertainty)

    if task == "binary_inspection":
        parsed = parse_condition(prediction)
        correct = parsed == record.metadata.condition
        result.update({"truth": record.metadata.condition, "parsed": parsed, "correct": correct})
        if not correct:
            failures.append(
                "false_negative" if record.metadata.condition == "anomalous" else "false_positive"
            )
    elif task == "product_identification":
        parsed = parse_product(prediction)
        correct = parsed == record.metadata.category
        result.update({"truth": record.metadata.category, "parsed": parsed, "correct": correct})
        if not correct:
            failures.append("wrong_product")
    elif task == "defect_identification":
        strict_truth = split_compound_label(record.metadata.defect_type)
        strict_parsed = parse_defects(prediction, defect_vocabulary)
        strict_scores = set_metrics(strict_truth, strict_parsed)
        semantic_truth = canonicalize_defect_set(strict_truth)
        semantic_parsed = parse_defects(prediction, defect_vocabulary, semantic=True)
        scores = set_metrics(semantic_truth, semantic_parsed)
        result.update(
            {
                "truth_set": sorted(semantic_truth),
                "parsed_set": sorted(semantic_parsed),
                "strict_truth_set": sorted(strict_truth),
                "strict_parsed_set": sorted(strict_parsed),
                "strict_exact_match": strict_scores["exact_match"],
                "strict_precision": strict_scores["precision"],
                "strict_recall": strict_scores["recall"],
                "strict_f1": strict_scores["f1"],
                **scores,
            }
        )
        if not bool(scores["exact_match"]):
            failures.append(
                "partial_compound_defect" if float(scores["f1"]) > 0 else "wrong_defect"
            )
    elif task == "localization":
        parsed = parse_location(prediction)
        truth = record.metadata.location
        correct = parsed == truth
        adjacent = adjacent_location(truth or "", parsed)
        result.update({"truth": truth, "parsed": parsed, "correct": correct, "adjacent": adjacent})
        if not correct:
            failures.append("adjacent_location" if adjacent else "wrong_location")
    elif task == "structured_report":
        structured, tags = _structured_score(
            record, prediction, defect_vocabulary, config.required_structured_fields
        )
        result.update(structured)
        failures.extend(tags)
    elif task == "uncertainty":
        correct = (
            uncertainty["abstains"]
            and not uncertainty["unsupported_root_cause_claim"]
            and not uncertainty["unsupported_safety_claim"]
        )
        result["correct"] = correct
        if not uncertainty["abstains"]:
            failures.append("failed_to_abstain")
    else:
        condition = parse_condition(prediction)
        product = parse_product(prediction)
        location = parse_location(prediction)
        defects = parse_defects(prediction, defect_vocabulary)
        true_defects = split_compound_label(record.metadata.defect_type)
        defect_score = set_metrics(true_defects, defects)
        facts = {
            "condition_correct": condition == record.metadata.condition,
            "product_correct": product in (None, record.metadata.category),
            "defect_f1": float(defect_score["f1"]),
            "location_correct": location in (None, record.metadata.location),
        }
        facts["fact_coverage"] = _mean(
            [
                float(facts["condition_correct"]),
                float(facts["product_correct"]),
                float(facts["defect_f1"]),
                float(facts["location_correct"]),
            ]
        )
        result.update(facts)
        if facts["fact_coverage"] < 1.0:
            failures.append("incomplete_or_incorrect_facts")

    return result, sorted(set(failures))


def _aggregate_task(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if task in {"binary_inspection", "product_identification", "localization"}:
        truths = [str(row["truth"]) for row in rows]
        predictions = [row.get("parsed") for row in rows]
        metrics = classification_metrics(truths, predictions)
        if task == "localization":
            metrics["adjacent_tolerance_accuracy"] = _mean(
                [float(bool(row["adjacent"])) for row in rows]
            )
        return metrics
    if task == "defect_identification":
        semantic = aggregate_set_metrics(
            [
                {
                    "exact_match": row["exact_match"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                }
                for row in rows
            ]
        )
        strict = aggregate_set_metrics(
            [
                {
                    "exact_match": row["strict_exact_match"],
                    "precision": row["strict_precision"],
                    "recall": row["strict_recall"],
                    "f1": row["strict_f1"],
                }
                for row in rows
            ]
        )
        return {**semantic, **{f"strict_{key}": value for key, value in strict.items() if key != "count"}}
    if task == "structured_report":
        return {
            "count": len(rows),
            "json_valid_rate": _mean(
                [float(bool(row.get("json_valid", False))) for row in rows]
            ),
            "schema_valid_rate": _mean(
                [float(bool(row.get("schema_valid", False))) for row in rows]
            ),
            "field_completeness": _mean(
                [float(row.get("field_completeness", 0.0)) for row in rows]
            ),
            "condition_accuracy": _mean(
                [float(bool(row.get("condition_correct", False))) for row in rows]
            ),
            "product_accuracy": _mean(
                [float(bool(row.get("product_correct", False))) for row in rows]
            ),
            "defect_f1": _mean(
                [float(row.get("defect_f1", 0.0)) for row in rows]
            ),
            "strict_defect_f1": _mean(
                [
                    float(
                        row.get(
                            "strict_defect_f1",
                            row.get("defect_f1", 0.0),
                        )
                    )
                    for row in rows
                ]
            ),
            "location_accuracy": _mean(
                [float(bool(row.get("location_correct", False))) for row in rows]
            ),
            "severity_accuracy": _mean(
                [float(bool(row.get("severity_correct", False))) for row in rows]
            ),
            "unsupported_field_rate": _mean(
                [
                    float(int(row.get("unsupported_field_count", 0)) > 0)
                    for row in rows
                ]
            ),
        }
    if task == "uncertainty":
        return {
            "count": len(rows),
            "appropriate_abstention_accuracy": _mean([float(row["correct"]) for row in rows]),
            "unsupported_root_cause_rate": _mean(
                [float(row["unsupported_root_cause_claim"]) for row in rows]
            ),
            "unsupported_safety_claim_rate": _mean(
                [float(row["unsupported_safety_claim"]) for row in rows]
            ),
        }
    return {
        "count": len(rows),
        "mean_fact_coverage": _mean([float(row["fact_coverage"]) for row in rows]),
        "unsupported_root_cause_rate": _mean(
            [float(row["unsupported_root_cause_claim"]) for row in rows]
        ),
        "unsupported_safety_claim_rate": _mean(
            [float(row["unsupported_safety_claim"]) for row in rows]
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_baseline_predictions(
    benchmark_path: Path,
    predictions_path: Path,
    config: EvaluationConfig,
) -> EvaluationResult:
    """Evaluate prediction JSONL against a frozen benchmark."""

    benchmark = _read_benchmark(benchmark_path)
    predictions = _read_predictions(predictions_path)
    missing = sorted(set(benchmark) - set(predictions))
    extras = sorted(set(predictions) - set(benchmark))
    if config.strict_prediction_coverage and (missing or extras):
        raise ValueError(
            f"Prediction coverage mismatch: missing={len(missing)}, extras={len(extras)}"
        )

    defect_vocabulary = {
        record.metadata.defect_type
        for record in benchmark.values()
        if record.metadata.defect_type
    }
    evaluated: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    parsing_errors: list[dict[str, Any]] = []

    for instruction_id, record in benchmark.items():
        payload = predictions.get(instruction_id)
        if payload is None:
            continue
        prediction = str(payload["prediction"])
        row, failure_tags = _evaluate_one(record, prediction, defect_vocabulary, config)
        row["prediction"] = prediction
        row["ground_truth"] = _assistant_target(record)
        evaluated.append(row)
        if failure_tags:
            failures.append(
                {
                    "instruction_id": instruction_id,
                    "image_id": record.image_id,
                    "task_family": record.task_family,
                    "category": record.metadata.category,
                    "condition": record.metadata.condition,
                    "failure_tags": failure_tags,
                    "ground_truth": _assistant_target(record),
                    "prediction": prediction,
                }
            )
        if row.get("parsed") is None and record.task_family in {
            "binary_inspection",
            "product_identification",
            "localization",
        }:
            parsing_errors.append(
                {
                    "instruction_id": instruction_id,
                    "task_family": record.task_family,
                    "prediction": prediction,
                }
            )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_task[str(row["task_family"])].append(row)
        by_category[str(row["category"])].append(row)

    task_metrics = {task: _aggregate_task(task, rows) for task, rows in sorted(by_task.items())}
    category_rows: list[dict[str, Any]] = []
    for category, rows in sorted(by_category.items()):
        category_rows.append(
            {
                "category": category,
                "count": len(rows),
                "failure_rate": safe_divide(
                    sum(1 for item in failures if item["category"] == category), len(rows)
                ),
                "unsupported_root_cause_rate": _mean(
                    [float(row["unsupported_root_cause_claim"]) for row in rows]
                ),
                "unsupported_safety_claim_rate": _mean(
                    [float(row["unsupported_safety_claim"]) for row in rows]
                ),
            }
        )

    overall = {
        "schema_version": "1.1",
        "benchmark_records": len(benchmark),
        "predictions": len(predictions),
        "evaluated": len(evaluated),
        "failure_records": len(failures),
        "failure_rate": safe_divide(len(failures), len(evaluated)),
        "failure_tag_counts": dict(
            sorted(Counter(tag for item in failures for tag in item["failure_tags"]).items())
        ),
        "per_task": task_metrics,
    }

    config.output_root.mkdir(parents=True, exist_ok=True)
    config.metrics_path.write_text(
        json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(
        config.per_task_path,
        [
            {"task_family": task, **metrics}
            for task, metrics in task_metrics.items()
            if all(not isinstance(value, dict) for value in metrics.values())
        ],
    )
    _write_csv(config.per_category_path, category_rows)
    with config.failures_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in failures:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")
    with config.parsing_errors_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in parsing_errors:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")

    return EvaluationResult(
        benchmark_records=len(benchmark),
        predictions=len(predictions),
        failures=len(failures),
        metrics_path=config.metrics_path,
        per_task_path=config.per_task_path,
        per_category_path=config.per_category_path,
        failures_path=config.failures_path,
    )
