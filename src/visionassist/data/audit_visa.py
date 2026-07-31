"""Audit a locally downloaded VisA dataset and create a raw manifest."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from visionassist.data.checksum import sha256_file
from visionassist.data.config import VisaAuditConfig
from visionassist.schemas.dataset import Condition, RawImageRecord


@dataclass(frozen=True)
class AuditResult:
    """Paths and counters produced by one audit run."""

    manifest_path: Path
    summary_path: Path
    csv_inventory_path: Path
    record_count: int
    error_count: int


def _normalise_token(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _infer_condition(path: Path) -> Condition | None:
    tokens = {_normalise_token(part) for part in path.parts}
    if tokens & {"normal", "good"}:
        return Condition.NORMAL
    if tokens & {"anomaly", "anomalous", "defect", "defective", "bad"}:
        return Condition.ANOMALOUS
    return None


def _find_category(path: Path, raw_root: Path, expected: set[str]) -> str | None:
    try:
        relative = path.relative_to(raw_root)
    except ValueError:
        return None
    expected_normalised = {_normalise_token(item): item for item in expected}
    for part in relative.parts:
        matched = expected_normalised.get(_normalise_token(part))
        if matched is not None:
            return matched
    return None


def _candidate_mask(image_path: Path, raw_root: Path) -> Path | None:
    relative = image_path.relative_to(raw_root)
    replacements = {
        "Images": "Masks",
        "images": "masks",
        "Anomaly": "Anomaly",
        "anomaly": "anomaly",
    }
    parts = [replacements.get(part, part) for part in relative.parts]
    candidate = raw_root.joinpath(*parts).with_suffix(".png")
    if candidate.is_file() and candidate != image_path:
        return candidate

    category_root = raw_root / relative.parts[0]
    mask_dirs = [path for path in category_root.rglob("*") if path.is_dir() and "mask" in path.name.lower()]
    for directory in mask_dirs:
        same_stem = directory / f"{image_path.stem}.png"
        if same_stem.is_file():
            return same_stem
    return None


def _iter_images(config: VisaAuditConfig) -> Iterable[Path]:
    allowed = {suffix.lower() for suffix in config.allowed_image_extensions}
    for path in sorted(config.raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        tokens = {_normalise_token(part) for part in path.parts}
        if tokens & {"mask", "masks", "ground_truth"}:
            continue
        yield path


def _read_size(path: Path, verify: bool) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            if verify:
                image.verify()
        return width, height
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable image: {path}") from exc


def _csv_inventory(raw_root: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in sorted(raw_root.rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            row_count = sum(1 for _ in reader)
        inventory.append(
            {
                "path": path.as_posix(),
                "columns": header,
                "row_count": row_count,
            }
        )
    return inventory


def audit_visa(config: VisaAuditConfig) -> AuditResult:
    """Validate raw files and write JSONL/JSON audit outputs."""

    if not config.raw_root.is_dir():
        raise FileNotFoundError(
            f"VisA root does not exist: {config.raw_root}. "
            "Download and extract the official dataset first."
        )

    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_root.mkdir(parents=True, exist_ok=True)

    expected = set(config.expected_categories)
    category_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    errors: list[str] = []
    records: list[RawImageRecord] = []

    for image_path in _iter_images(config):
        category = _find_category(image_path, config.raw_root, expected)
        condition = _infer_condition(image_path)
        if category is None or condition is None:
            errors.append(f"Could not classify path: {image_path}")
            continue
        try:
            width, height = _read_size(image_path, config.verify_images)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        mask_path = _candidate_mask(image_path, config.raw_root) if condition is Condition.ANOMALOUS else None
        record = RawImageRecord(
            image_id=f"visa_{category}_{image_path.stem}",
            category=category,
            condition=condition,
            image_path=image_path,
            mask_path=mask_path,
            annotation_path=image_path.parents[3] / "image_anno.csv" if len(image_path.parents) > 3 else None,
            width=width,
            height=height,
            file_size_bytes=image_path.stat().st_size,
            sha256=sha256_file(image_path, config.chunk_size_bytes) if config.compute_sha256 else None,
        )
        records.append(record)
        category_counts[category] += 1
        condition_counts[condition.value] += 1

    with config.manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")

    csv_inventory = _csv_inventory(config.raw_root)
    csv_inventory_path = config.report_root / "visa_csv_inventory.json"
    csv_inventory_path.write_text(json.dumps(csv_inventory, indent=2), encoding="utf-8")

    present_categories = set(category_counts)
    summary = {
        "dataset": config.dataset_name,
        "raw_root": config.raw_root.as_posix(),
        "total_records": len(records),
        "category_counts": dict(sorted(category_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "missing_expected_categories": sorted(expected - present_categories),
        "unexpected_categories": sorted(present_categories - expected),
        "records_without_masks": sum(
            1
            for record in records
            if record.condition is Condition.ANOMALOUS and record.mask_path is None
        ),
        "csv_files": len(csv_inventory),
        "errors": errors,
    }
    summary_path = config.report_root / "visa_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return AuditResult(
        manifest_path=config.manifest_path,
        summary_path=summary_path,
        csv_inventory_path=csv_inventory_path,
        record_count=len(records),
        error_count=len(errors),
    )
