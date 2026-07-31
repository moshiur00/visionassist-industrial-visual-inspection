from pathlib import Path

from PIL import Image

from visionassist.data.config import VisaConfig
from visionassist.data.derive_features import derive_record
from visionassist.schemas.dataset import (
    CanonicalImageRecord,
    Condition,
    MaskMetadata,
    NineGridLocation,
    SourceSplit,
    VisualSeverity,
)


def _config(tmp_path: Path) -> VisaConfig:
    return VisaConfig.model_validate(
        {
            "version": "2022-09-22",
            "download_url": "https://example.com/visa.tar",
            "archive_path": tmp_path / "visa.tar",
            "raw_root": tmp_path / "raw",
            "report_root": tmp_path / "reports",
            "manifest_path": tmp_path / "raw.jsonl",
            "archive_receipt_path": tmp_path / "receipt.json",
            "license_report_path": tmp_path / "license.md",
            "dataset_card_path": tmp_path / "card.md",
            "expected_total_images": 2,
            "expected_normal_images": 1,
            "expected_anomalous_images": 1,
            "expected_categories": ["pcb1"],
            "allowed_image_extensions": [".png"],
        }
    )


def _canonical(
    tmp_path: Path,
    *,
    condition: Condition,
    mask_path: Path | None,
    defect_type: str | None = None,
) -> CanonicalImageRecord:
    mask = None
    if mask_path is not None:
        mask = MaskMetadata(
            path=mask_path,
            width=6,
            height=6,
            foreground_pixels=4,
            foreground_ratio=4 / 36,
            is_binary=True,
            unique_values=[0, 255],
        )
    return CanonicalImageRecord(
        image_id=f"visa_pcb1_{condition.value}_001",
        source_version="2022-09-22",
        category="pcb1",
        condition=condition,
        source_split=SourceSplit.UNKNOWN,
        defect_type=defect_type,
        image_path=tmp_path / "image.png",
        mask_path=mask_path,
        annotation_path=tmp_path / "image_anno.csv",
        annotation_row=2,
        width=6,
        height=6,
        file_size_bytes=100,
        mask=mask,
    )


def test_derives_bbox_centroid_grid_area_and_severity(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    pixels = [0] * 36
    for y in (4, 5):
        for x in (4, 5):
            pixels[y * 6 + x] = 2
    image = Image.new("L", (6, 6))
    image.putdata(pixels)
    image.save(mask_path)

    result = derive_record(
        _canonical(
            tmp_path,
            condition=Condition.ANOMALOUS,
            mask_path=mask_path,
        ),
        _config(tmp_path),
    )

    assert result.bounding_box is not None
    assert result.bounding_box.model_dump() == {
        "x_min": 4,
        "y_min": 4,
        "x_max": 5,
        "y_max": 5,
        "width": 2,
        "height": 2,
        "x_min_normalized": 4 / 6,
        "y_min_normalized": 4 / 6,
        "x_max_normalized": 1.0,
        "y_max_normalized": 1.0,
    }
    assert result.centroid is not None
    assert result.centroid.x == 4.5
    assert result.centroid.y == 4.5
    assert result.nine_grid_location is NineGridLocation.BOTTOM_RIGHT
    assert result.anomaly_area_pixels == 4
    assert result.anomaly_area_ratio == 4 / 36
    assert result.visual_severity is VisualSeverity.MAJOR
    assert result.severity_basis == "area_ratio"


def test_major_defect_keyword_overrides_small_area(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    image = Image.new("L", (20, 20), 0)
    image.putpixel((1, 1), 1)
    image.save(mask_path)

    record = _canonical(
        tmp_path,
        condition=Condition.ANOMALOUS,
        mask_path=mask_path,
        defect_type="missing_component",
    )
    record.width = 20
    record.height = 20
    assert record.mask is not None
    record.mask.width = 20
    record.mask.height = 20
    record.mask.foreground_pixels = 1
    record.mask.foreground_ratio = 1 / 400

    result = derive_record(record, _config(tmp_path))
    assert result.visual_severity is VisualSeverity.MAJOR
    assert result.severity_basis == "defect_keyword_override"


def test_normal_record_has_empty_features(tmp_path: Path) -> None:
    result = derive_record(
        _canonical(
            tmp_path,
            condition=Condition.NORMAL,
            mask_path=None,
        ),
        _config(tmp_path),
    )

    assert result.anomaly_area_pixels == 0
    assert result.anomaly_area_ratio == 0.0
    assert result.bounding_box is None
    assert result.centroid is None
    assert result.nine_grid_location is None
    assert result.visual_severity is VisualSeverity.NONE
