"""Deterministic, leakage-aware supervised splits for VisA Phase 4."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from visionassist.data.config import VisaConfig
from visionassist.schemas.dataset import Condition, DerivedImageRecord, DatasetSplit


class SplitGenerationError(RuntimeError):
    """Raised when Phase 4 cannot produce trustworthy splits."""


@dataclass(frozen=True)
class Phase4Result:
    """Paths and counts emitted by Phase 4."""

    records: int
    train_records: int
    validation_records: int
    test_records: int
    errors: int
    warnings: int
    split_directory: Path
    assignment_path: Path
    report_path: Path
    error_path: Path


def _load_records(path: Path) -> list[DerivedImageRecord]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 3 manifest not found: {path}. Run phase3-visa first."
        )

    records: list[DerivedImageRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(DerivedImageRecord.model_validate_json(line))
            except ValidationError as exc:
                raise SplitGenerationError(
                    f"Invalid Phase 3 record at {path}:{row_number}: {exc}"
                ) from exc
    return records


def _stable_tie_breaker(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _stratum_key(record: DerivedImageRecord) -> tuple[str, str]:
    """Primary strata guarantee category/condition representation."""

    return record.category, record.condition.value


def _defect_key(record: DerivedImageRecord) -> str:
    if record.condition is Condition.NORMAL:
        return "normal"
    return record.defect_type or "unspecified_anomaly"


def _target_counts(
    count: int,
    train_ratio: float,
    validation_ratio: float,
) -> dict[DatasetSplit, int]:
    """Use largest remainders while preserving a non-empty train partition."""

    ratios = {
        DatasetSplit.TRAIN: train_ratio,
        DatasetSplit.VALIDATION: validation_ratio,
        DatasetSplit.TEST: 1.0 - train_ratio - validation_ratio,
    }
    exact = {split: count * ratio for split, ratio in ratios.items()}
    result = {split: int(value) for split, value in exact.items()}
    remaining = count - sum(result.values())

    order = sorted(
        ratios,
        key=lambda split: (exact[split] - result[split], ratios[split]),
        reverse=True,
    )
    for split in order[:remaining]:
        result[split] += 1

    if count > 0 and result[DatasetSplit.TRAIN] == 0:
        donor = max(
            (DatasetSplit.VALIDATION, DatasetSplit.TEST),
            key=lambda split: result[split],
        )
        if result[donor] > 0:
            result[donor] -= 1
            result[DatasetSplit.TRAIN] += 1
    return result


def _interleave_by_defect(
    records: Iterable[DerivedImageRecord], seed: int
) -> list[DerivedImageRecord]:
    """Round-robin defect buckets so rare defects are spread where possible."""

    buckets: dict[str, list[DerivedImageRecord]] = defaultdict(list)
    for record in records:
        buckets[_defect_key(record)].append(record)

    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)

    ordered_keys = sorted(
        buckets,
        key=lambda key: (-len(buckets[key]), _stable_tie_breaker(seed, key)),
    )
    output: list[DerivedImageRecord] = []
    while any(buckets.values()):
        for key in ordered_keys:
            if buckets[key]:
                output.append(buckets[key].pop())
    return output


def _build_duplicate_clusters(
    records: list[DerivedImageRecord],
) -> tuple[dict[str, list[DerivedImageRecord]], list[str]]:
    """Cluster byte-identical images, preventing exact duplicates crossing splits."""

    clusters: dict[str, list[DerivedImageRecord]] = defaultdict(list)
    warnings: list[str] = []
    for record in records:
        key = f"sha256:{record.sha256}" if record.sha256 else f"id:{record.image_id}"
        clusters[key].append(record)

    for key, members in clusters.items():
        if len(members) > 1:
            strata = {_stratum_key(member) for member in members}
            if len(strata) > 1:
                warnings.append(
                    f"Duplicate cluster {key} spans strata {sorted(strata)}; "
                    "all members remain in one split."
                )
    return clusters, warnings


def _assign_records(
    records: list[DerivedImageRecord], config: VisaConfig
) -> tuple[dict[str, DatasetSplit], list[str]]:
    clusters, warnings = _build_duplicate_clusters(records)
    representative_by_key = {key: members[0] for key, members in clusters.items()}

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cluster_key, representative in representative_by_key.items():
        strata[_stratum_key(representative)].append(cluster_key)

    assignments: dict[str, DatasetSplit] = {}
    for stratum, cluster_keys in sorted(strata.items()):
        representatives = [representative_by_key[key] for key in cluster_keys]
        stratum_seed = int(
            hashlib.sha256(f"{config.phase4_seed}:{stratum}".encode()).hexdigest()[:8],
            16,
        )
        ordered_representatives = _interleave_by_defect(representatives, stratum_seed)
        ordered_cluster_keys = [
            next(
                key
                for key in cluster_keys
                if representative_by_key[key].image_id == record.image_id
            )
            for record in ordered_representatives
        ]
        targets = _target_counts(
            len(ordered_cluster_keys),
            config.phase4_train_ratio,
            config.phase4_validation_ratio,
        )

        cursor = 0
        for split in (
            DatasetSplit.TRAIN,
            DatasetSplit.VALIDATION,
            DatasetSplit.TEST,
        ):
            stop = cursor + targets[split]
            for cluster_key in ordered_cluster_keys[cursor:stop]:
                for member in clusters[cluster_key]:
                    assignments[member.image_id] = split
            cursor = stop

    return assignments, warnings


def _validate_uniqueness(records: list[DerivedImageRecord]) -> None:
    for field in ("image_id", "image_path"):
        values = [str(getattr(record, field)) for record in records]
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        if duplicates:
            raise SplitGenerationError(
                f"Duplicate {field} values found, examples: {duplicates[:5]}"
            )


def _validate_leakage(
    records: list[DerivedImageRecord], assignments: dict[str, DatasetSplit]
) -> dict[str, Any]:
    by_sha: dict[str, set[str]] = defaultdict(set)
    by_path: dict[str, set[str]] = defaultdict(set)
    by_id: dict[str, set[str]] = defaultdict(set)

    for record in records:
        split = assignments[record.image_id].value
        by_id[record.image_id].add(split)
        by_path[record.image_path.as_posix()].add(split)
        if record.sha256:
            by_sha[record.sha256].add(split)

    sha_leaks = sorted(key for key, splits in by_sha.items() if len(splits) > 1)
    path_leaks = sorted(key for key, splits in by_path.items() if len(splits) > 1)
    id_leaks = sorted(key for key, splits in by_id.items() if len(splits) > 1)
    return {
        "passed": not (sha_leaks or path_leaks or id_leaks),
        "image_id_cross_split": id_leaks,
        "image_path_cross_split": path_leaks,
        "sha256_cross_split": sha_leaks,
    }


def _stats_for(
    records: list[DerivedImageRecord], assignments: dict[str, DatasetSplit]
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split in DatasetSplit:
        selected = [record for record in records if assignments[record.image_id] is split]
        category_condition = Counter(
            f"{record.category}/{record.condition.value}" for record in selected
        )
        stats[split.value] = {
            "records": len(selected),
            "category_counts": dict(sorted(Counter(r.category for r in selected).items())),
            "condition_counts": dict(
                sorted(Counter(r.condition.value for r in selected).items())
            ),
            "category_condition_counts": dict(sorted(category_condition.items())),
            "defect_type_counts": dict(
                sorted(
                    Counter(
                        _defect_key(record)
                        for record in selected
                        if record.condition is Condition.ANOMALOUS
                    ).items()
                )
            ),
            "severity_counts": dict(
                sorted(Counter(r.visual_severity.value for r in selected).items())
            ),
        }
    return stats


def _write_outputs(
    records: list[DerivedImageRecord],
    assignments: dict[str, DatasetSplit],
    config: VisaConfig,
) -> None:
    config.phase4_split_root.mkdir(parents=True, exist_ok=True)
    config.phase4_assignment_path.parent.mkdir(parents=True, exist_ok=True)

    handles = {
        split: (config.phase4_split_root / f"{split.value}.jsonl").open(
            "w", encoding="utf-8"
        )
        for split in DatasetSplit
    }
    try:
        for record in sorted(records, key=lambda item: item.image_id):
            split = assignments[record.image_id]
            payload = record.model_dump(mode="json")
            payload["dataset_split"] = split.value
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    with config.phase4_assignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "dataset_split",
                "category",
                "condition",
                "defect_type",
                "visual_severity",
                "image_path",
                "sha256",
            ],
        )
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.image_id):
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "dataset_split": assignments[record.image_id].value,
                    "category": record.category,
                    "condition": record.condition.value,
                    "defect_type": record.defect_type or "",
                    "visual_severity": record.visual_severity.value,
                    "image_path": record.image_path.as_posix(),
                    "sha256": record.sha256 or "",
                }
            )


def split_visa(config: VisaConfig) -> Phase4Result:
    """Create deterministic train/validation/test files and audit leakage."""

    records = _load_records(config.phase3_manifest_path)
    _validate_uniqueness(records)
    assignments, warnings = _assign_records(records, config)

    if len(assignments) != len(records):
        raise SplitGenerationError(
            f"Only {len(assignments)} of {len(records)} records received a split."
        )

    leakage = _validate_leakage(records, assignments)
    split_stats = _stats_for(records, assignments)
    counts = Counter(split.value for split in assignments.values())

    errors: list[dict[str, Any]] = []
    if not leakage["passed"]:
        errors.append({"error_type": "LeakageError", "details": leakage})
    if len(records) != config.expected_total_images:
        errors.append(
            {
                "error_type": "RecordCountError",
                "expected": config.expected_total_images,
                "actual": len(records),
            }
        )
    if any(counts.get(split.value, 0) == 0 for split in DatasetSplit):
        errors.append(
            {"error_type": "EmptySplitError", "split_counts": dict(counts)}
        )

    _write_outputs(records, assignments, config)
    config.phase4_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.phase4_error_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": "1.0",
        "dataset": config.dataset_name,
        "source_manifest": config.phase3_manifest_path.as_posix(),
        "seed": config.phase4_seed,
        "ratios": {
            "train": config.phase4_train_ratio,
            "validation": config.phase4_validation_ratio,
            "test": config.phase4_test_ratio,
        },
        "total_records": len(records),
        "split_counts": dict(sorted(counts.items())),
        "split_statistics": split_stats,
        "leakage_checks": leakage,
        "warnings": warnings,
        "checks": {
            "all_records_assigned": len(assignments) == len(records),
            "expected_record_count": len(records) == config.expected_total_images,
            "no_cross_split_leakage": leakage["passed"],
            "all_splits_non_empty": all(
                counts.get(split.value, 0) > 0 for split in DatasetSplit
            ),
        },
    }
    config.phase4_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with config.phase4_error_path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    if config.strict_phase4 and errors:
        raise RuntimeError(
            "Phase 4 validation failed. See "
            f"{config.phase4_report_path} and {config.phase4_error_path}."
        )

    return Phase4Result(
        records=len(records),
        train_records=counts.get(DatasetSplit.TRAIN.value, 0),
        validation_records=counts.get(DatasetSplit.VALIDATION.value, 0),
        test_records=counts.get(DatasetSplit.TEST.value, 0),
        errors=len(errors),
        warnings=len(warnings),
        split_directory=config.phase4_split_root,
        assignment_path=config.phase4_assignment_path,
        report_path=config.phase4_report_path,
        error_path=config.phase4_error_path,
    )
