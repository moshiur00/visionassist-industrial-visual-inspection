"""Parse VisA annotation CSVs and build canonical Phase 2 metadata."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, UnidentifiedImageError

from visionassist.data.checksum import sha256_file
from visionassist.data.config import VisaConfig
from visionassist.schemas.dataset import (
    CanonicalImageRecord,
    Condition,
    MaskMetadata,
    SourceSplit,
)

_REQUIRED_COLUMNS = {"label", "image", "mask"}
_COLUMN_ALIASES = {
    "object": {"object", "category", "class", "object_name"},
    "split": {"split", "set", "set_type", "subset"},
    "label": {"label", "condition", "status", "anomaly"},
    "image": {"image", "image_path", "img", "img_path"},
    "mask": {"mask", "mask_path", "ground_truth", "gt_path"},
    "defect_type": {"defect_type", "defect", "anomaly_type", "fault_type"},
}


@dataclass(frozen=True)
class Phase2Result:
    """Output files and counters from one Phase 2 run."""

    manifest_path: Path
    report_path: Path
    error_path: Path
    records: int
    errors: int
    warnings: int


class SampleValidationError(ValueError):
    """Raised when one source annotation row cannot form a valid sample."""


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _canonical_columns(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise SampleValidationError("Annotation CSV has no header.")
    by_normalised = {_normalise_name(name): name for name in fieldnames}
    result: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            original = by_normalised.get(_normalise_name(alias))
            if original is not None:
                result[canonical] = original
                break
    missing = sorted(_REQUIRED_COLUMNS - result.keys())
    if missing:
        raise SampleValidationError(
            f"Unsupported annotation schema. Missing canonical columns: {missing}; "
            f"found: {list(fieldnames)}"
        )
    return result


def _value(row: Mapping[str, str | None], columns: Mapping[str, str], key: str) -> str:
    raw = row.get(columns[key])
    return "" if raw is None else raw.strip()


def _resolve_path(raw_root: Path, csv_path: Path, value: str) -> Path | None:
    if not value or _normalise_name(value) in {"none", "null", "na", "nan", "_"}:
        return None
    normalised = value.replace("\\", "/").lstrip("./")
    candidate = Path(normalised)
    candidates = [csv_path.parent / candidate, raw_root / candidate]
    # Some CSV paths start with "VisA/" while raw_root already points inside it.
    if candidate.parts and _normalise_name(candidate.parts[0]) == "visa":
        candidates.append(raw_root.joinpath(*candidate.parts[1:]))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve()


def _parse_label(value: str) -> tuple[Condition, str | None]:
    """Interpret VisA's label column.

    In the official per-category CSV files, ``label`` is not merely a binary
    status. Normal rows use ``normal`` while anomalous rows contain one or
    more human-readable defect descriptions, often comma-separated.
    """

    raw = value.strip()
    token = _normalise_name(raw)
    if token in {"normal", "good", "0", "false"}:
        return Condition.NORMAL, None

    if not raw or token in {"none", "null", "na", "nan", "_"}:
        raise SampleValidationError("VisA label is empty.")

    # Preserve all source defect concepts while making them machine-friendly.
    defect_parts = [
        _normalise_name(part)
        for part in raw.split(",")
        if _normalise_name(part)
    ]
    if not defect_parts:
        raise SampleValidationError(f"Invalid VisA defect label: {value!r}")
    return Condition.ANOMALOUS, ",".join(defect_parts)


def _parse_split(value: str) -> SourceSplit:
    token = _normalise_name(value)
    if token in {"train", "training"}:
        return SourceSplit.TRAIN
    if token in {"test", "testing", "validation", "val"}:
        return SourceSplit.TEST
    return SourceSplit.UNKNOWN


def _read_image(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise SampleValidationError(f"Image file does not exist: {path}")
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise SampleValidationError(f"Unreadable image: {path}") from exc
    if width <= 0 or height <= 0:
        raise SampleValidationError(f"Invalid image dimensions for {path}: {width}x{height}")
    return width, height


def _read_mask(path: Path, expected_size: tuple[int, int], require_binary: bool) -> MaskMetadata:
    if not path.is_file():
        raise SampleValidationError(f"Mask file does not exist: {path}")
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            width, height = gray.size
            histogram = gray.histogram()
    except (OSError, UnidentifiedImageError) as exc:
        raise SampleValidationError(f"Unreadable mask: {path}") from exc

    if (width, height) != expected_size:
        raise SampleValidationError(
            f"Mask/image size mismatch for {path}: mask={width}x{height}, "
            f"image={expected_size[0]}x{expected_size[1]}"
        )
    unique_values = [value for value, count in enumerate(histogram) if count]
    foreground_values = [value for value in unique_values if value > 0]
    foreground = sum(histogram[value] for value in foreground_values)
    if foreground == 0:
        raise SampleValidationError(f"Anomaly mask contains no foreground pixels: {path}")

    # VisA masks are semantic/instance-style indexed masks. Background is 0,
    # while positive values (1, 2, 3, ...) identify one or more anomaly
    # regions/classes. Such masks are valid for anomaly detection and can be
    # converted losslessly to a binary foreground mask with ``mask > 0``.
    # ``is_binary`` records whether the source mask already has only one
    # foreground value; multi-label masks are preserved and are not rejected.
    is_binary = len(foreground_values) == 1
    if require_binary and 0 not in unique_values:
        raise SampleValidationError(
            f"Mask has no zero-valued background: {path}; values={unique_values[:20]}"
        )
    total = width * height
    return MaskMetadata(
        path=path,
        width=width,
        height=height,
        foreground_pixels=foreground,
        foreground_ratio=foreground / total,
        is_binary=is_binary,
        unique_values=unique_values,
    )


def _category_from_csv(csv_path: Path) -> str:
    return _normalise_name(csv_path.parent.name)


def _record_from_row(
    *,
    row: Mapping[str, str | None],
    columns: Mapping[str, str],
    csv_path: Path,
    row_number: int,
    config: VisaConfig,
) -> CanonicalImageRecord:
    directory_category = _category_from_csv(csv_path)
    if "object" in columns:
        category = _normalise_name(_value(row, columns, "object"))
        if not category:
            category = directory_category
        elif category != directory_category:
            raise SampleValidationError(
                f"CSV object {category!r} does not match directory {directory_category!r}."
            )
    else:
        # Official per-category VisA image_anno.csv files contain only
        # image, label, and mask. The object category is encoded by the
        # parent directory name (for example, pcb1/image_anno.csv).
        category = directory_category
    if category not in {_normalise_name(item) for item in config.expected_categories}:
        raise SampleValidationError(f"Unexpected category: {category}")

    condition, label_defect_type = _parse_label(
        _value(row, columns, "label")
    )
    source_split = (
        _parse_split(_value(row, columns, "split"))
        if "split" in columns
        else SourceSplit.UNKNOWN
    )
    image_path = _resolve_path(config.raw_root, csv_path, _value(row, columns, "image"))
    if image_path is None:
        raise SampleValidationError("Image path is empty.")
    mask_path = _resolve_path(config.raw_root, csv_path, _value(row, columns, "mask"))
    width, height = _read_image(image_path)

    if condition is Condition.NORMAL:
        if mask_path is not None and mask_path.is_file():
            raise SampleValidationError(f"Normal sample unexpectedly has a mask: {mask_path}")
        mask_path = None
        mask_metadata = None
    else:
        if mask_path is None:
            raise SampleValidationError("Anomalous sample has an empty mask path.")
        mask_metadata = _read_mask(
            mask_path,
            expected_size=(width, height),
            require_binary=config.require_binary_masks,
        )

    defect_type = label_defect_type
    if "defect_type" in columns:
        candidate = _normalise_name(_value(row, columns, "defect_type"))
        if candidate:
            defect_type = candidate

    # Category + stem is stable and unique in the official release.
    image_id = f"visa_{category}_{condition.value}_{image_path.stem}"
    return CanonicalImageRecord(
        image_id=image_id,
        source_version=config.version,
        category=category,
        condition=condition,
        source_split=source_split,
        defect_type=defect_type,
        image_path=image_path,
        mask_path=mask_path,
        annotation_path=csv_path.resolve(),
        annotation_row=row_number,
        width=width,
        height=height,
        file_size_bytes=image_path.stat().st_size,
        sha256=(
            sha256_file(image_path, config.chunk_size_bytes)
            if config.compute_sha256
            else None
        ),
        mask=mask_metadata,
    )


def _iter_annotation_csvs(config: VisaConfig) -> list[Path]:
    return sorted(config.raw_root.rglob("image_anno.csv"))


def _error_payload(csv_path: Path, row_number: int, error: Exception) -> dict[str, Any]:
    return {
        "annotation_path": csv_path.as_posix(),
        "annotation_row": row_number,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def parse_visa(config: VisaConfig) -> Phase2Result:
    """Parse all official annotation CSVs and validate every referenced sample."""

    csv_paths = _iter_annotation_csvs(config)
    if not csv_paths:
        raise FileNotFoundError(f"No image_anno.csv files found below {config.raw_root}")

    records: list[CanonicalImageRecord] = []
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    ids: set[str] = set()
    image_paths: set[Path] = set()
    category_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            try:
                columns = _canonical_columns(reader.fieldnames)
            except SampleValidationError as exc:
                errors.append(_error_payload(csv_path, 1, exc))
                continue
            for row_number, row in enumerate(reader, start=2):
                try:
                    record = _record_from_row(
                        row=row,
                        columns=columns,
                        csv_path=csv_path,
                        row_number=row_number,
                        config=config,
                    )
                    resolved_image = record.image_path.resolve()
                    if record.image_id in ids:
                        raise SampleValidationError(f"Duplicate image_id: {record.image_id}")
                    if resolved_image in image_paths:
                        raise SampleValidationError(
                            f"Image occurs in more than one annotation row: {resolved_image}"
                        )
                    ids.add(record.image_id)
                    image_paths.add(resolved_image)
                    records.append(record)
                    category_counts[record.category] += 1
                    condition_counts[record.condition.value] += 1
                    split_counts[record.source_split.value] += 1
                except (SampleValidationError, ValueError) as exc:
                    errors.append(_error_payload(csv_path, row_number, exc))

    expected_categories = {_normalise_name(item) for item in config.expected_categories}
    missing_categories = sorted(expected_categories - set(category_counts))
    if missing_categories:
        warnings.append(f"Missing categories: {missing_categories}")

    if len(records) != config.expected_total_images:
        warnings.append(
            f"Record count differs from expected: {len(records)} != "
            f"{config.expected_total_images}"
        )
    if condition_counts[Condition.NORMAL.value] != config.expected_normal_images:
        warnings.append(
            "Normal count differs from expected: "
            f"{condition_counts[Condition.NORMAL.value]} != {config.expected_normal_images}"
        )
    if condition_counts[Condition.ANOMALOUS.value] != config.expected_anomalous_images:
        warnings.append(
            "Anomalous count differs from expected: "
            f"{condition_counts[Condition.ANOMALOUS.value]} != "
            f"{config.expected_anomalous_images}"
        )

    config.canonical_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.phase2_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.phase2_error_path.parent.mkdir(parents=True, exist_ok=True)

    with config.canonical_manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    with config.phase2_error_path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    mask_ratios = [
        record.mask.foreground_ratio
        for record in records
        if record.mask is not None
    ]
    report = {
        "schema_version": "1.0",
        "dataset": config.dataset_name,
        "source_version": config.version,
        "annotation_csv_count": len(csv_paths),
        "valid_records": len(records),
        "invalid_records": len(errors),
        "warnings": warnings,
        "category_counts": dict(sorted(category_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "source_split_counts": dict(sorted(split_counts.items())),
        "records_with_masks": sum(record.mask is not None for record in records),
        "source_binary_masks": sum(
            record.mask is not None and record.mask.is_binary for record in records
        ),
        "multi_label_masks": sum(
            record.mask is not None and not record.mask.is_binary for record in records
        ),
        "binary_compatible_masks": sum(record.mask is not None for record in records),
        "mask_foreground_ratio": {
            "minimum": min(mask_ratios) if mask_ratios else None,
            "maximum": max(mask_ratios) if mask_ratios else None,
            "mean": sum(mask_ratios) / len(mask_ratios) if mask_ratios else None,
        },
        "checks": {
            "all_rows_valid": not errors,
            "record_count_matches": len(records) == config.expected_total_images,
            "normal_count_matches": (
                condition_counts[Condition.NORMAL.value] == config.expected_normal_images
            ),
            "anomalous_count_matches": (
                condition_counts[Condition.ANOMALOUS.value]
                == config.expected_anomalous_images
            ),
            "all_expected_categories_present": not missing_categories,
            "all_anomalies_have_masks": all(
                record.mask is not None
                for record in records
                if record.condition is Condition.ANOMALOUS
            ),
        },
    }
    config.phase2_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if config.strict_phase2 and (errors or warnings):
        raise RuntimeError(
            "Phase 2 validation failed. "
            f"See {config.phase2_report_path} and {config.phase2_error_path}."
        )

    return Phase2Result(
        manifest_path=config.canonical_manifest_path,
        report_path=config.phase2_report_path,
        error_path=config.phase2_error_path,
        records=len(records),
        errors=len(errors),
        warnings=len(warnings),
    )
