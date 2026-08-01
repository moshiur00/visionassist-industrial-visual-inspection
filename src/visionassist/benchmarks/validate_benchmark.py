"""Validate the frozen VisionAssist baseline benchmark."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from visionassist.benchmarks.build_visa_baseline import sha256_file
from visionassist.benchmarks.schemas import BenchmarkConfig
from visionassist.schemas.instruction import InstructionRecord


@dataclass(frozen=True)
class BenchmarkValidationResult:
    """Validation result returned to the CLI."""

    records: int
    unique_images: int
    errors: int
    warnings: int
    passed: bool
    report_path: Path
    error_path: Path
    statistics_path: Path


def _image_path(record: InstructionRecord, project_root: Path) -> Path:
    image_items = [
        item
        for item in record.messages[0].content
        if item.type == "image" and item.image is not None
    ]
    if len(image_items) != 1:
        raise ValueError("Expected exactly one user image item.")
    candidate = (project_root / image_items[0].image).resolve()
    root = project_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Image path escapes project root: {candidate}")
    return candidate


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_baseline_benchmark(
    config: BenchmarkConfig,
    project_root: Path,
) -> BenchmarkValidationResult:
    """Validate benchmark counts, hashes, schemas, split safety, and images."""

    project_root = project_root.resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    records: list[InstructionRecord] = []

    if not config.benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {config.benchmark_path}")

    with config.benchmark_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = InstructionRecord.model_validate_json(line)
                records.append(record)
            except Exception as exc:
                errors.append(
                    {
                        "line": line_number,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    ids = [record.instruction_id for record in records]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    for instruction_id in duplicate_ids:
        errors.append(
            {
                "instruction_id": instruction_id,
                "error_type": "DuplicateInstructionId",
                "message": "Instruction ID appears more than once in benchmark.",
            }
        )

    for record in records:
        if record.dataset_split.value != "test":
            errors.append(
                {
                    "instruction_id": record.instruction_id,
                    "error_type": "SplitMismatch",
                    "message": f"Expected test split, got {record.dataset_split.value}.",
                }
            )

    expected_counts = config.task_targets.model_dump()
    actual_counts = Counter(record.task_family for record in records)
    for task, expected in expected_counts.items():
        if actual_counts.get(task, 0) != expected:
            errors.append(
                {
                    "task_family": task,
                    "error_type": "TaskQuotaMismatch",
                    "message": f"Expected {expected}, found {actual_counts.get(task, 0)}.",
                }
            )

    readable_images: set[str] = set()
    image_failures: dict[str, str] = {}
    if config.verify_images:
        for record in records:
            if record.image_id in readable_images or record.image_id in image_failures:
                continue
            try:
                image_path = _image_path(record, project_root)
                if image_path.suffix.lower() not in {
                    extension.lower() for extension in config.allowed_image_extensions
                }:
                    raise ValueError(f"Unsupported image extension: {image_path.suffix}")
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                if image_path.stat().st_size == 0:
                    raise ValueError(f"Image is empty: {image_path}")
                with Image.open(image_path) as image:
                    image.verify()
                readable_images.add(record.image_id)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                image_failures[record.image_id] = str(exc)
                errors.append(
                    {
                        "instruction_id": record.instruction_id,
                        "image_id": record.image_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    manifest: dict[str, Any] = {}
    if config.manifest_path.is_file():
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        current_hash = sha256_file(config.benchmark_path)
        if manifest.get("benchmark_sha256") != current_hash:
            errors.append(
                {
                    "error_type": "BenchmarkHashMismatch",
                    "message": "Benchmark content no longer matches frozen manifest hash.",
                }
            )
    else:
        errors.append(
            {
                "error_type": "MissingManifest",
                "message": f"Benchmark manifest not found: {config.manifest_path}",
            }
        )

    categories_by_task: dict[str, set[str]] = defaultdict(set)
    conditions_by_task: dict[str, set[str]] = defaultdict(set)
    for record in records:
        categories_by_task[record.task_family].add(record.metadata.category)
        conditions_by_task[record.task_family].add(record.metadata.condition)

    statistics = {
        "schema_version": "1.0",
        "benchmark_name": config.benchmark_name,
        "records": len(records),
        "unique_images": len({record.image_id for record in records}),
        "readable_images": len(readable_images) if config.verify_images else None,
        "task_family_counts": dict(sorted(actual_counts.items())),
        "category_counts": dict(
            sorted(Counter(r.metadata.category for r in records).items())
        ),
        "condition_counts": dict(
            sorted(Counter(r.metadata.condition for r in records).items())
        ),
        "severity_counts": dict(
            sorted(Counter(r.metadata.visual_severity for r in records).items())
        ),
        "categories_by_task": {
            task: sorted(values) for task, values in sorted(categories_by_task.items())
        },
        "conditions_by_task": {
            task: sorted(values) for task, values in sorted(conditions_by_task.items())
        },
    }
    _write_json(config.statistics_path, statistics)

    config.validation_error_path.parent.mkdir(parents=True, exist_ok=True)
    with config.validation_error_path.open("w", encoding="utf-8", newline="\n") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False))
            handle.write("\n")

    passed = not errors
    report = {
        "schema_version": "1.0",
        "benchmark_name": config.benchmark_name,
        "records": len(records),
        "expected_records": config.task_targets.total,
        "unique_images": len({record.image_id for record in records}),
        "errors": len(errors),
        "warnings": len(warnings),
        "checks": {
            "record_count_matches": len(records) == config.task_targets.total,
            "test_split_only": all(r.dataset_split.value == "test" for r in records),
            "instruction_ids_unique": not duplicate_ids,
            "task_quotas_match": all(
                actual_counts.get(task, 0) == target
                for task, target in expected_counts.items()
            ),
            "benchmark_hash_matches_manifest": not any(
                error["error_type"] == "BenchmarkHashMismatch" for error in errors
            ),
            "all_images_readable": not image_failures if config.verify_images else None,
        },
        "passed": passed,
        "outputs": {
            "statistics": config.statistics_path.as_posix(),
            "errors": config.validation_error_path.as_posix(),
        },
    }
    _write_json(config.validation_report_path, report)

    if config.strict and not passed:
        raise RuntimeError(
            "Phase 7A benchmark validation failed. See "
            f"{config.validation_report_path} and {config.validation_error_path}."
        )

    return BenchmarkValidationResult(
        records=len(records),
        unique_images=len({record.image_id for record in records}),
        errors=len(errors),
        warnings=len(warnings),
        passed=passed,
        report_path=config.validation_report_path,
        error_path=config.validation_error_path,
        statistics_path=config.statistics_path,
    )
