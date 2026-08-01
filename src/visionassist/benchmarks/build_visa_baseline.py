"""Build and freeze the deterministic VisA Phase 7 baseline benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from visionassist.benchmarks.schemas import BenchmarkConfig
from visionassist.schemas.instruction import InstructionRecord


@dataclass(frozen=True)
class BenchmarkBuildResult:
    """Summary returned by the benchmark builder."""

    benchmark_name: str
    records: int
    benchmark_path: Path
    manifest_path: Path
    distribution_path: Path
    sha256_path: Path
    benchmark_sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: int, *parts: str) -> str:
    payload = "|".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_test_records(path: Path) -> list[InstructionRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Phase 5 test instructions not found: {path}")
    records: list[InstructionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = InstructionRecord.model_validate_json(line)
            except Exception as exc:  # pydantic gives rich context
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
            if record.dataset_split.value != "test":
                raise ValueError(
                    f"Non-test record found in source test file: {record.instruction_id}"
                )
            records.append(record)
    return records


def _stratum(record: InstructionRecord) -> tuple[str, str, str, str, str]:
    metadata = record.metadata
    return (
        metadata.category,
        metadata.condition,
        metadata.defect_type or "none",
        metadata.location or "none",
        metadata.visual_severity,
    )


def _round_robin_select(
    candidates: Iterable[InstructionRecord],
    target: int,
    seed: int,
    task_family: str,
) -> list[InstructionRecord]:
    """Select diverse records deterministically across metadata strata.

    The first pass takes at most one record per image. If the requested quota is
    larger than the number of source images for a task (for example defect
    identification), later template variants are used in a second pass.
    """

    candidate_list = list(candidates)
    if len(candidate_list) < target:
        raise ValueError(
            f"Task {task_family!r} has only {len(candidate_list)} candidates; "
            f"target is {target}."
        )

    candidate_list.sort(
        key=lambda record: _stable_key(seed, task_family, record.instruction_id)
    )
    by_stratum: dict[tuple[str, str, str, str, str], deque[InstructionRecord]] = {}
    grouped: dict[tuple[str, str, str, str, str], list[InstructionRecord]] = defaultdict(list)
    for record in candidate_list:
        grouped[_stratum(record)].append(record)
    for key, values in grouped.items():
        values.sort(
            key=lambda record: _stable_key(seed, task_family, *key, record.instruction_id)
        )
        by_stratum[key] = deque(values)

    keys = sorted(
        by_stratum,
        key=lambda key: _stable_key(seed, task_family, *key),
    )
    selected: list[InstructionRecord] = []
    selected_ids: set[str] = set()
    selected_images: set[str] = set()

    # Pass 1: maximize unique source-image coverage.
    progress = True
    while len(selected) < target and progress:
        progress = False
        for key in keys:
            queue = by_stratum[key]
            deferred: list[InstructionRecord] = []
            chosen: InstructionRecord | None = None
            while queue:
                record = queue.popleft()
                if record.instruction_id in selected_ids:
                    continue
                if record.image_id not in selected_images:
                    chosen = record
                    break
                deferred.append(record)
            queue.extend(deferred)
            if chosen is not None:
                selected.append(chosen)
                selected_ids.add(chosen.instruction_id)
                selected_images.add(chosen.image_id)
                progress = True
                if len(selected) == target:
                    break

    # Pass 2: fill the quota with remaining prompt/template variants.
    while len(selected) < target:
        progress = False
        for key in keys:
            queue = by_stratum[key]
            while queue and queue[0].instruction_id in selected_ids:
                queue.popleft()
            if not queue:
                continue
            record = queue.popleft()
            selected.append(record)
            selected_ids.add(record.instruction_id)
            progress = True
            if len(selected) == target:
                break
        if not progress:
            raise RuntimeError(
                f"Unable to fill target {target} for task {task_family}; "
                f"selected {len(selected)}."
            )

    return selected


def _distribution(records: list[InstructionRecord]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_count": len(records),
        "unique_images": len({record.image_id for record in records}),
        "task_family_counts": dict(sorted(Counter(r.task_family for r in records).items())),
        "category_counts": dict(
            sorted(Counter(r.metadata.category for r in records).items())
        ),
        "condition_counts": dict(
            sorted(Counter(r.metadata.condition for r in records).items())
        ),
        "severity_counts": dict(
            sorted(Counter(r.metadata.visual_severity for r in records).items())
        ),
        "location_counts": dict(
            sorted(Counter(r.metadata.location or "none" for r in records).items())
        ),
        "defect_counts": dict(
            sorted(Counter(r.metadata.defect_type or "none" for r in records).items())
        ),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_visa_baseline(config: BenchmarkConfig) -> BenchmarkBuildResult:
    """Create a deterministic, frozen benchmark from Phase 5 test records."""

    records = _load_test_records(config.source_test_path)
    task_targets = config.task_targets.model_dump()
    selected: list[InstructionRecord] = []
    for task_family, target in task_targets.items():
        if target == 0:
            continue
        candidates = [record for record in records if record.task_family == task_family]
        selected.extend(
            _round_robin_select(candidates, target, config.seed, task_family)
        )

    selected.sort(key=lambda record: (record.task_family, record.instruction_id))
    config.output_root.mkdir(parents=True, exist_ok=True)
    with config.benchmark_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(record.model_dump_json())
            handle.write("\n")

    source_sha256 = sha256_file(config.source_test_path)
    benchmark_sha256 = sha256_file(config.benchmark_path)
    distribution = _distribution(selected)
    _write_json(config.distribution_path, distribution)

    manifest = {
        "benchmark_name": config.benchmark_name,
        "schema_version": config.schema_version,
        "source_split": "test",
        "seed": config.seed,
        "instruction_count": len(selected),
        "unique_images": distribution["unique_images"],
        "source_test_file": config.source_test_path.as_posix(),
        "source_test_sha256": source_sha256,
        "benchmark_file": config.benchmark_path.as_posix(),
        "benchmark_sha256": benchmark_sha256,
        "sampling_policy": (
            "deterministic_stratified_round_robin_with_unique_image_first_pass"
        ),
        "task_targets": task_targets,
        "distribution_file": config.distribution_path.as_posix(),
        "frozen": True,
        "versioning_note": (
            "Do not overwrite this benchmark after model evaluation. Create a new "
            "benchmark version for changed sampling or labels."
        ),
    }
    _write_json(config.manifest_path, manifest)
    config.sha256_path.write_text(f"{benchmark_sha256}\n", encoding="utf-8")

    return BenchmarkBuildResult(
        benchmark_name=config.benchmark_name,
        records=len(selected),
        benchmark_path=config.benchmark_path,
        manifest_path=config.manifest_path,
        distribution_path=config.distribution_path,
        sha256_path=config.sha256_path,
        benchmark_sha256=benchmark_sha256,
    )
