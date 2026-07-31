"""Validated schemas used by the VisionAssist data pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Condition(StrEnum):
    """Image-level inspection condition."""

    NORMAL = "normal"
    ANOMALOUS = "anomalous"


class SourceSplit(StrEnum):
    """Split value carried by the original VisA annotation CSV."""

    TRAIN = "train"
    TEST = "test"
    UNKNOWN = "unknown"


class DatasetSplit(StrEnum):
    """Supervised split assigned in Phase 4."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RawImageRecord(BaseModel):
    """One immutable record in the raw-dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    source: str = "visa"
    category: str = Field(min_length=1)
    condition: Condition
    image_path: Path
    mask_path: Path | None = None
    annotation_path: Path | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    file_size_bytes: int = Field(ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def normal_images_must_not_have_masks(self) -> RawImageRecord:
        if self.condition is Condition.NORMAL and self.mask_path is not None:
            raise ValueError("Normal records must not point to anomaly masks.")
        return self


class MaskMetadata(BaseModel):
    """Validated information read from an anomaly segmentation mask."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    foreground_pixels: int = Field(ge=0)
    foreground_ratio: float = Field(ge=0.0, le=1.0)
    is_binary: bool
    unique_values: list[int]


class CanonicalImageRecord(BaseModel):
    """Canonical Phase 2 record produced from a VisA CSV row and its files."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    image_id: str = Field(min_length=1)
    source: str = "visa"
    source_version: str = Field(min_length=1)
    category: str = Field(min_length=1)
    condition: Condition
    source_split: SourceSplit = SourceSplit.UNKNOWN
    defect_type: str | None = None
    image_path: Path
    mask_path: Path | None = None
    annotation_path: Path
    annotation_row: int = Field(ge=2)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    file_size_bytes: int = Field(gt=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    mask: MaskMetadata | None = None

    @model_validator(mode="after")
    def validate_condition_and_mask(self) -> CanonicalImageRecord:
        if self.condition is Condition.NORMAL:
            if self.mask_path is not None or self.mask is not None:
                raise ValueError("Normal samples must not contain a mask.")
        elif self.mask_path is None or self.mask is None:
            raise ValueError("Anomalous samples must contain a parsed mask.")
        return self


class NineGridLocation(StrEnum):
    """Coarse anomaly location derived from the foreground centroid."""

    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class VisualSeverity(StrEnum):
    """Project-defined visual severity, not mechanical safety severity."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class BoundingBox(BaseModel):
    """Inclusive pixel bounds and normalized outer-edge coordinates."""

    model_config = ConfigDict(extra="forbid")

    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    x_min_normalized: float = Field(ge=0.0, le=1.0)
    y_min_normalized: float = Field(ge=0.0, le=1.0)
    x_max_normalized: float = Field(ge=0.0, le=1.0)
    y_max_normalized: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> BoundingBox:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("Bounding-box maximums must not be below minimums.")
        if self.width != self.x_max - self.x_min + 1:
            raise ValueError("Bounding-box width is inconsistent with pixel bounds.")
        if self.height != self.y_max - self.y_min + 1:
            raise ValueError("Bounding-box height is inconsistent with pixel bounds.")
        return self


class Centroid(BaseModel):
    """Foreground-pixel centroid in pixel and normalized coordinates."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0.0)
    y: float = Field(ge=0.0)
    x_normalized: float = Field(ge=0.0, lt=1.0)
    y_normalized: float = Field(ge=0.0, lt=1.0)


class DerivedImageRecord(CanonicalImageRecord):
    """Phase 3 record with deterministic mask-derived spatial features."""

    schema_version: str = "1.1"
    anomaly_area_pixels: int = Field(ge=0)
    anomaly_area_ratio: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = None
    centroid: Centroid | None = None
    nine_grid_location: NineGridLocation | None = None
    visual_severity: VisualSeverity
    severity_basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_derived_features(self) -> DerivedImageRecord:
        if self.condition is Condition.NORMAL:
            if self.anomaly_area_pixels != 0 or self.anomaly_area_ratio != 0.0:
                raise ValueError("Normal samples must have zero anomaly area.")
            if any(
                value is not None
                for value in (
                    self.bounding_box,
                    self.centroid,
                    self.nine_grid_location,
                )
            ):
                raise ValueError("Normal samples must not have spatial anomaly features.")
            if self.visual_severity is not VisualSeverity.NONE:
                raise ValueError("Normal samples must have visual severity 'none'.")
        else:
            if self.anomaly_area_pixels <= 0 or self.anomaly_area_ratio <= 0.0:
                raise ValueError("Anomalous samples must have positive anomaly area.")
            if (
                self.bounding_box is None
                or self.centroid is None
                or self.nine_grid_location is None
            ):
                raise ValueError("Anomalous samples require all spatial features.")
            if self.visual_severity is VisualSeverity.NONE:
                raise ValueError("Anomalous samples cannot have severity 'none'.")
        return self
