"""Derive grounded spatial features from VisA anomaly masks."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import ValidationError

from visionassist.data.config import VisaConfig
from visionassist.schemas.dataset import (
    BoundingBox,
    CanonicalImageRecord,
    Centroid,
    Condition,
    DerivedImageRecord,
    NineGridLocation,
    VisualSeverity,
)


@dataclass(frozen=True)
class Phase3Result:
    """Files and counts produced by Phase 3."""

    manifest_path: Path
    report_path: Path
    error_path: Path
    records: int
    anomalous_records: int
    errors: int
    warnings: int


class FeatureDerivationError(ValueError):
    """Raised when a canonical sample cannot be enriched safely."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical Phase 2 manifest not found: {path}. Run phase2-visa first."
        )

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise FeatureDerivationError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise FeatureDerivationError(
                    f"Expected a JSON object at {path}:{line_number}."
                )
            rows.append(payload)
    return rows


def _read_binary_foreground(mask_path: Path) -> np.ndarray:
    if not mask_path.is_file():
        raise FeatureDerivationError(f"Mask file not found: {mask_path}")

    try:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"))
    except (OSError, ValueError) as exc:
        raise FeatureDerivationError(f"Cannot read mask {mask_path}: {exc}") from exc

    if mask.ndim != 2:
        raise FeatureDerivationError(
            f"Expected a 2D mask at {mask_path}; shape={mask.shape}."
        )
    return mask > 0


def _derive_bbox(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> BoundingBox:
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    # Pixel coordinates are inclusive; normalized x_max/y_max use the outer edge.
    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        width=x_max - x_min + 1,
        height=y_max - y_min + 1,
        x_min_normalized=x_min / width,
        y_min_normalized=y_min / height,
        x_max_normalized=(x_max + 1) / width,
        y_max_normalized=(y_max + 1) / height,
    )


def _derive_centroid(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> Centroid:
    x = float(xs.mean())
    y = float(ys.mean())
    return Centroid(
        x=x,
        y=y,
        x_normalized=x / width,
        y_normalized=y / height,
    )


def _nine_grid_location(centroid: Centroid) -> NineGridLocation:
    column = min(int(centroid.x_normalized * 3), 2)
    row = min(int(centroid.y_normalized * 3), 2)
    locations = (
        (NineGridLocation.TOP_LEFT, NineGridLocation.TOP_CENTER, NineGridLocation.TOP_RIGHT),
        (
            NineGridLocation.CENTER_LEFT,
            NineGridLocation.CENTER,
            NineGridLocation.CENTER_RIGHT,
        ),
        (
            NineGridLocation.BOTTOM_LEFT,
            NineGridLocation.BOTTOM_CENTER,
            NineGridLocation.BOTTOM_RIGHT,
        ),
    )
    return locations[row][column]


def _visual_severity(
    *,
    area_ratio: float,
    defect_type: str | None,
    config: VisaConfig,
) -> tuple[VisualSeverity, str]:
    normalized_defect = (defect_type or "").lower().replace("_", " ")
    if any(keyword.lower() in normalized_defect for keyword in config.phase3_major_keywords):
        return VisualSeverity.MAJOR, "defect_keyword_override"
    if area_ratio < config.phase3_minor_max_area_ratio:
        return VisualSeverity.MINOR, "area_ratio"
    if area_ratio < config.phase3_moderate_max_area_ratio:
        return VisualSeverity.MODERATE, "area_ratio"
    return VisualSeverity.MAJOR, "area_ratio"


def derive_record(record: CanonicalImageRecord, config: VisaConfig) -> DerivedImageRecord:
    """Enrich one validated Phase 2 record with deterministic spatial features."""

    base = record.model_dump(mode="python")
    base["schema_version"] = "1.1"
    if record.condition is Condition.NORMAL:
        return DerivedImageRecord(
            **base,
            anomaly_area_pixels=0,
            anomaly_area_ratio=0.0,
            bounding_box=None,
            centroid=None,
            nine_grid_location=None,
            visual_severity=VisualSeverity.NONE,
            severity_basis="normal_sample",
        )

    if record.mask_path is None or record.mask is None:
        raise FeatureDerivationError(
            f"Anomalous record {record.image_id} has no parsed mask metadata."
        )

    foreground = _read_binary_foreground(record.mask_path)
    mask_height, mask_width = foreground.shape
    if (mask_width, mask_height) != (record.width, record.height):
        raise FeatureDerivationError(
            f"Mask/image size mismatch for {record.image_id}: "
            f"mask={mask_width}x{mask_height}, image={record.width}x{record.height}."
        )

    ys, xs = np.nonzero(foreground)
    if xs.size == 0:
        raise FeatureDerivationError(f"Anomaly mask is empty for {record.image_id}.")

    area_pixels = int(xs.size)
    area_ratio = area_pixels / (record.width * record.height)
    if area_pixels != record.mask.foreground_pixels:
        raise FeatureDerivationError(
            f"Foreground count changed for {record.image_id}: "
            f"phase2={record.mask.foreground_pixels}, phase3={area_pixels}."
        )

    bbox = _derive_bbox(xs, ys, record.width, record.height)
    centroid = _derive_centroid(xs, ys, record.width, record.height)
    severity, severity_basis = _visual_severity(
        area_ratio=area_ratio,
        defect_type=record.defect_type,
        config=config,
    )

    return DerivedImageRecord(
        **base,
        anomaly_area_pixels=area_pixels,
        anomaly_area_ratio=area_ratio,
        bounding_box=bbox,
        centroid=centroid,
        nine_grid_location=_nine_grid_location(centroid),
        visual_severity=severity,
        severity_basis=severity_basis,
    )


def derive_visa_features(config: VisaConfig) -> Phase3Result:
    """Run Phase 3 over the canonical VisA manifest and write validated outputs."""

    config.phase3_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.phase3_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.phase3_error_path.parent.mkdir(parents=True, exist_ok=True)

    payloads = _load_jsonl(config.canonical_manifest_path)
    records: list[DerivedImageRecord] = []
    errors: list[dict[str, Any]] = []

    for index, payload in enumerate(payloads, start=1):
        image_id = str(payload.get("image_id", "unknown"))
        try:
            canonical = CanonicalImageRecord.model_validate(payload)
            records.append(derive_record(canonical, config))
        except (ValidationError, FeatureDerivationError, OSError, ValueError) as exc:
            errors.append(
                {
                    "manifest_path": config.canonical_manifest_path.as_posix(),
                    "manifest_row": index,
                    "image_id": image_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    with config.phase3_manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")

    with config.phase3_error_path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    condition_counts = Counter(record.condition.value for record in records)
    location_counts = Counter(
        record.nine_grid_location.value
        for record in records
        if record.nine_grid_location is not None
    )
    severity_counts = Counter(record.visual_severity.value for record in records)
    anomalous_records = [
        record for record in records if record.condition is Condition.ANOMALOUS
    ]
    ratios = [record.anomaly_area_ratio for record in anomalous_records]

    warnings: list[str] = []
    if len(records) != config.expected_total_images:
        warnings.append(
            f"Record count differs from expected: {len(records)} != "
            f"{config.expected_total_images}"
        )
    if len(anomalous_records) != config.expected_anomalous_images:
        warnings.append(
            f"Anomalous count differs from expected: {len(anomalous_records)} != "
            f"{config.expected_anomalous_images}"
        )

    report = {
        "schema_version": "1.0",
        "dataset": config.dataset_name,
        "source_version": config.version,
        "input_manifest": config.canonical_manifest_path.as_posix(),
        "output_manifest": config.phase3_manifest_path.as_posix(),
        "valid_records": len(records),
        "invalid_records": len(errors),
        "warnings": warnings,
        "condition_counts": dict(sorted(condition_counts.items())),
        "nine_grid_location_counts": dict(sorted(location_counts.items())),
        "visual_severity_counts": dict(sorted(severity_counts.items())),
        "anomaly_area_ratio": {
            "minimum": min(ratios) if ratios else None,
            "maximum": max(ratios) if ratios else None,
            "mean": sum(ratios) / len(ratios) if ratios else None,
        },
        "severity_policy": {
            "name": "project_defined_visual_severity",
            "minor": f"area ratio < {config.phase3_minor_max_area_ratio}",
            "moderate": (
                f"{config.phase3_minor_max_area_ratio} <= area ratio < "
                f"{config.phase3_moderate_max_area_ratio}"
            ),
            "major": f"area ratio >= {config.phase3_moderate_max_area_ratio}",
            "major_keyword_overrides": config.phase3_major_keywords,
            "disclaimer": (
                "Visual severity is a project-defined annotation and does not "
                "represent mechanical or safety risk."
            ),
        },
        "checks": {
            "all_rows_valid": not errors,
            "record_count_matches": len(records) == config.expected_total_images,
            "anomalous_count_matches": (
                len(anomalous_records) == config.expected_anomalous_images
            ),
            "all_anomalies_have_features": all(
                record.bounding_box is not None
                and record.centroid is not None
                and record.nine_grid_location is not None
                and record.anomaly_area_pixels > 0
                for record in anomalous_records
            ),
            "all_normals_have_empty_spatial_features": all(
                record.bounding_box is None
                and record.centroid is None
                and record.nine_grid_location is None
                and record.anomaly_area_pixels == 0
                for record in records
                if record.condition is Condition.NORMAL
            ),
        },
    }
    config.phase3_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if config.strict_phase3 and (errors or warnings):
        raise RuntimeError(
            "Phase 3 validation failed. "
            f"See {config.phase3_report_path} and {config.phase3_error_path}."
        )

    return Phase3Result(
        manifest_path=config.phase3_manifest_path,
        report_path=config.phase3_report_path,
        error_path=config.phase3_error_path,
        records=len(records),
        anomalous_records=len(anomalous_records),
        errors=len(errors),
        warnings=len(warnings),
    )
