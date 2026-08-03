"""Deterministic Phase 11 defect and localization failure analysis."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from visionassist.evaluation.metrics import adjacent_location
from visionassist.evaluation.normalize import split_compound_label
from visionassist.evaluation.parsers import (
    canonicalize_defect_set,
    parse_defects,
    parse_location,
)

DEFECT_TASKS = {"defect_identification", "structured_report"}
LOCALIZATION_TASKS = {"localization", "structured_report"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Prediction line {line_number} is not an object.")
            instruction_id = payload.get("instruction_id")
            if not isinstance(instruction_id, str):
                raise ValueError(
                    f"Prediction line {line_number} has no string instruction_id."
                )
            if instruction_id in seen:
                raise ValueError(f"Duplicate prediction ID: {instruction_id}")
            if not isinstance(payload.get("prediction"), str):
                raise ValueError(
                    f"Prediction line {line_number} has no string prediction."
                )
            seen.add(instruction_id)
            records.append(payload)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(values: set[str]) -> str:
    return "+".join(sorted(values)) if values else "**unparsed**"


def _nested_counts(
    counts: Counter[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for (truth, predicted), count in sorted(counts.items()):
        result.setdefault(truth, {})[predicted] = count
    return result


def _ranked_confusions(
    counts: Counter[tuple[str, str]],
) -> list[dict[str, str | int]]:
    return [
        {"truth": truth, "prediction": predicted, "count": count}
        for (truth, predicted), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        if truth != predicted
    ]


def analyze_predictions(predictions_path: Path) -> dict[str, Any]:
    """Analyze defect and location confusions in an inference prediction file."""

    records = _read_jsonl(predictions_path)
    defect_vocabulary = {
        str(row["defect_type"])
        for row in records
        if isinstance(row.get("defect_type"), str) and row["defect_type"]
    }

    defect_counts: Counter[tuple[str, str]] = Counter()
    defect_errors_by_category: Counter[str] = Counter()
    defect_task_counts: Counter[str] = Counter()
    defect_exact = 0
    defect_unparsed = 0

    location_counts: Counter[tuple[str, str]] = Counter()
    location_errors_by_category: Counter[str] = Counter()
    location_task_counts: Counter[str] = Counter()
    location_exact = 0
    location_adjacent = 0
    location_unparsed = 0

    for row in records:
        task = str(row.get("task_family", ""))
        category = str(row.get("category", "**unknown**"))
        prediction = str(row["prediction"])

        if task in DEFECT_TASKS and row.get("condition") == "anomalous":
            truth = canonicalize_defect_set(
                split_compound_label(str(row.get("defect_type") or ""))
            )
            parsed = parse_defects(prediction, defect_vocabulary, semantic=True)
            truth_label = _label(truth)
            parsed_label = _label(parsed)
            defect_counts[(truth_label, parsed_label)] += 1
            defect_task_counts[task] += 1
            if truth == parsed:
                defect_exact += 1
            else:
                defect_errors_by_category[category] += 1
            if not parsed:
                defect_unparsed += 1

        if task in LOCALIZATION_TASKS and row.get("condition") == "anomalous":
            truth_location = str(row.get("location") or "**missing_truth**")
            parsed_location = parse_location(prediction) or "**unparsed**"
            location_counts[(truth_location, parsed_location)] += 1
            location_task_counts[task] += 1
            exact = parsed_location == truth_location
            adjacent = adjacent_location(truth_location, parsed_location)
            location_exact += int(exact)
            location_adjacent += int(adjacent)
            if not exact:
                location_errors_by_category[category] += 1
            location_unparsed += int(parsed_location == "**unparsed**")

    defect_total = sum(defect_task_counts.values())
    location_total = sum(location_task_counts.values())
    return {
        "schema_version": "1.0",
        "source": predictions_path.as_posix(),
        "source_sha256": _sha256(predictions_path),
        "records": len(records),
        "defect": {
            "records": defect_total,
            "exact_matches": defect_exact,
            "exact_match_rate": defect_exact / defect_total if defect_total else 0.0,
            "unparsed": defect_unparsed,
            "task_counts": dict(sorted(defect_task_counts.items())),
            "errors_by_category": dict(sorted(defect_errors_by_category.items())),
            "confusion_matrix": _nested_counts(defect_counts),
            "ranked_errors": _ranked_confusions(defect_counts),
        },
        "localization": {
            "records": location_total,
            "exact_matches": location_exact,
            "exact_accuracy": (
                location_exact / location_total if location_total else 0.0
            ),
            "adjacent_matches": location_adjacent,
            "adjacent_accuracy": (
                location_adjacent / location_total if location_total else 0.0
            ),
            "unparsed": location_unparsed,
            "task_counts": dict(sorted(location_task_counts.items())),
            "errors_by_category": dict(sorted(location_errors_by_category.items())),
            "confusion_matrix": _nested_counts(location_counts),
            "ranked_errors": _ranked_confusions(location_counts),
        },
    }


def write_failure_analysis(predictions_path: Path, output_path: Path) -> Path:
    """Write one deterministic, atomic Phase 11 analysis report."""

    report = analyze_predictions(predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return output_path
