"""Leakage-safe Phase 11 hard-example selection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from visionassist.evaluation.failure_analysis import DEFECT_TASKS, LOCALIZATION_TASKS
from visionassist.evaluation.normalize import split_compound_label
from visionassist.evaluation.parsers import (
    canonicalize_defect_set,
    parse_defects,
    parse_location,
)
from visionassist.schemas.instruction import InstructionRecord
from visionassist.training.experiment import sha256_file

DEFECT_WEIGHTED_TASKS = {
    "defect_identification",
    "evidence_explanation",
    "structured_report",
    "technician_note",
}
LOCATION_WEIGHTED_TASKS = {
    "localization",
    "evidence_explanation",
    "structured_report",
    "technician_note",
}


class HardExampleConfig(BaseModel):
    """Configuration for deterministic validation-driven train selection."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    seed: int = 42
    train_path: Path
    validation_path: Path
    test_path: Path
    validation_predictions_path: Path
    output_path: Path
    manifest_path: Path
    task_quotas: dict[str, int]
    min_per_category_per_task: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_quotas(self) -> HardExampleConfig:
        if not self.task_quotas:
            raise ValueError("task_quotas must not be empty.")
        if any(quota <= 0 for quota in self.task_quotas.values()):
            raise ValueError("Every task quota must be positive.")
        return self


def load_hard_example_config(path: Path) -> HardExampleConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return HardExampleConfig.model_validate(payload)


def _read_instructions(path: Path) -> list[InstructionRecord]:
    records: list[InstructionRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = InstructionRecord.model_validate_json(line)
            if record.instruction_id in seen:
                raise ValueError(
                    f"Duplicate instruction ID at line {line_number}: "
                    f"{record.instruction_id}"
                )
            seen.add(record.instruction_id)
            records.append(record)
    return records


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            instruction_id = row.get("instruction_id")
            if not isinstance(instruction_id, str):
                raise ValueError(
                    f"Prediction line {line_number} has no string instruction_id."
                )
            if instruction_id in seen:
                raise ValueError(f"Duplicate prediction ID: {instruction_id}")
            if not isinstance(row.get("prediction"), str):
                raise ValueError(
                    f"Prediction line {line_number} has no string prediction."
                )
            seen.add(instruction_id)
            rows.append(row)
    return rows


def _validation_error_profile(
    rows: list[dict[str, Any]],
) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]:
    vocabulary = {
        str(row["defect_type"])
        for row in rows
        if isinstance(row.get("defect_type"), str) and row["defect_type"]
    }
    defect_errors: Counter[tuple[str, str]] = Counter()
    location_errors: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row.get("condition") != "anomalous":
            continue
        task = str(row.get("task_family", ""))
        category = str(row.get("category", ""))
        prediction = str(row["prediction"])
        if task in DEFECT_TASKS:
            truth = canonicalize_defect_set(
                split_compound_label(str(row.get("defect_type") or ""))
            )
            parsed = parse_defects(prediction, vocabulary, semantic=True)
            if truth != parsed:
                for atom in truth:
                    defect_errors[(category, atom)] += 1
        if task in LOCALIZATION_TASKS:
            truth_location = str(row.get("location") or "")
            if truth_location and parse_location(prediction) != truth_location:
                location_errors[(category, truth_location)] += 1
    return defect_errors, location_errors


def _tie_breaker(seed: int, instruction_id: str) -> str:
    return hashlib.sha256(f"{seed}:{instruction_id}".encode()).hexdigest()


def _score(
    record: InstructionRecord,
    defect_errors: Counter[tuple[str, str]],
    location_errors: Counter[tuple[str, str]],
) -> int:
    category = record.metadata.category
    score = 0
    if record.task_family in DEFECT_WEIGHTED_TASKS:
        atoms = canonicalize_defect_set(
            split_compound_label(record.metadata.defect_type)
        )
        score += sum(defect_errors[(category, atom)] for atom in atoms)
    if record.task_family in LOCATION_WEIGHTED_TASKS and record.metadata.location:
        score += location_errors[(category, record.metadata.location)]
    return score


def select_hard_examples(
    config: HardExampleConfig,
) -> tuple[list[InstructionRecord], dict[str, Any]]:
    """Select exact train quotas using only validation-derived error weights."""

    train = _read_instructions(config.train_path)
    validation = _read_instructions(config.validation_path)
    test = _read_instructions(config.test_path)
    predictions = _read_predictions(config.validation_predictions_path)
    if any(record.dataset_split != "train" for record in train):
        raise ValueError("Hard-example source contains non-train records.")

    validation_ids = {record.instruction_id for record in validation}
    prediction_ids = {str(row["instruction_id"]) for row in predictions}
    if prediction_ids - validation_ids:
        raise ValueError("Validation predictions contain IDs outside validation_path.")

    held_out_images = {record.image_id for record in validation + test}
    train_images = {record.image_id for record in train}
    overlap = sorted(train_images & held_out_images)
    if overlap:
        raise ValueError(f"Train/held-out image leakage detected: {overlap[0]}")

    defect_errors, location_errors = _validation_error_profile(predictions)
    buckets: dict[str, list[InstructionRecord]] = defaultdict(list)
    for record in train:
        buckets[record.task_family].append(record)

    selected: list[InstructionRecord] = []
    selected_scores: Counter[int] = Counter()
    categories = sorted({record.metadata.category for record in train})
    for task, quota in sorted(config.task_quotas.items()):
        candidates = buckets.get(task, [])
        if len(candidates) < quota:
            raise ValueError(
                f"Task quota exceeds available records for {task}: "
                f"requested={quota}, available={len(candidates)}"
            )
        ranked = sorted(
            candidates,
            key=lambda record: (
                -_score(record, defect_errors, location_errors),
                _tie_breaker(config.seed, record.instruction_id),
            ),
        )
        chosen: list[InstructionRecord] = []
        chosen_ids: set[str] = set()
        if config.min_per_category_per_task:
            by_category: dict[str, list[InstructionRecord]] = defaultdict(list)
            for record in ranked:
                by_category[record.metadata.category].append(record)
            minimum_total = config.min_per_category_per_task * len(categories)
            if minimum_total > quota:
                raise ValueError(
                    f"Category floor exceeds task quota for {task}: "
                    f"floor_total={minimum_total}, quota={quota}"
                )
            for category in categories:
                category_candidates = by_category.get(category, [])
                if len(category_candidates) < config.min_per_category_per_task:
                    raise ValueError(
                        f"Category floor exceeds available records for {task}/"
                        f"{category}: requested={config.min_per_category_per_task}, "
                        f"available={len(category_candidates)}"
                    )
                for record in category_candidates[
                    : config.min_per_category_per_task
                ]:
                    chosen.append(record)
                    chosen_ids.add(record.instruction_id)
        remaining = quota - len(chosen)
        chosen.extend(
            record
            for record in ranked
            if record.instruction_id not in chosen_ids
        )
        chosen = chosen[: len(chosen_ids) + remaining]
        selected.extend(chosen)
        selected_scores.update(
            _score(record, defect_errors, location_errors) for record in chosen
        )

    selected.sort(key=lambda record: record.instruction_id)
    identifiers = [record.instruction_id for record in selected]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Hard-example selection produced duplicate IDs.")
    selected_images = {record.image_id for record in selected}
    if selected_images & held_out_images:
        raise RuntimeError("Selected hard examples overlap held-out images.")

    id_hash = hashlib.sha256(("\n".join(identifiers) + "\n").encode()).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "seed": config.seed,
        "records": len(selected),
        "unique_instruction_ids": len(set(identifiers)),
        "unique_images": len(selected_images),
        "instruction_ids_sha256": id_hash,
        "task_quotas": dict(sorted(config.task_quotas.items())),
        "task_counts": dict(sorted(Counter(r.task_family for r in selected).items())),
        "min_per_category_per_task": config.min_per_category_per_task,
        "task_category_counts": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(
                {
                    task: Counter(
                        record.metadata.category
                        for record in selected
                        if record.task_family == task
                    )
                    for task in config.task_quotas
                }.items()
            )
        },
        "category_counts": dict(
            sorted(Counter(r.metadata.category for r in selected).items())
        ),
        "condition_counts": dict(
            sorted(Counter(r.metadata.condition for r in selected).items())
        ),
        "score_distribution": {
            str(score): count for score, count in sorted(selected_scores.items())
        },
        "validation_defect_error_strata": len(defect_errors),
        "validation_location_error_strata": len(location_errors),
        "leakage": {
            "validation_image_overlap": 0,
            "test_image_overlap": 0,
        },
        "source_sha256": {
            "train": sha256_file(config.train_path),
            "validation": sha256_file(config.validation_path),
            "test": sha256_file(config.test_path),
            "validation_predictions": sha256_file(
                config.validation_predictions_path
            ),
        },
    }
    return selected, manifest


def write_hard_examples(config: HardExampleConfig) -> tuple[Path, Path]:
    """Atomically write the selected instruction JSONL and its manifest."""

    selected, manifest = select_hard_examples(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = config.output_path.with_suffix(config.output_path.suffix + ".tmp")
    with output_tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(record.model_dump_json())
            handle.write("\n")
    output_tmp.replace(config.output_path)

    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = config.manifest_path.with_suffix(
        config.manifest_path.suffix + ".tmp"
    )
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(config.manifest_path)
    return config.output_path, config.manifest_path
