from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from visionassist.data.config import VisaConfig
from visionassist.data.parse_visa import parse_visa


def _config(tmp_path: Path) -> VisaConfig:
    return VisaConfig(
        version="2022-09-22",
        download_url="https://example.com/visa.tar",
        archive_path=tmp_path / "visa.tar",
        raw_root=tmp_path / "visa",
        report_root=tmp_path / "reports",
        manifest_path=tmp_path / "raw.jsonl",
        archive_receipt_path=tmp_path / "receipt.json",
        license_report_path=tmp_path / "license.md",
        dataset_card_path=tmp_path / "card.md",
        expected_total_images=2,
        expected_normal_images=1,
        expected_anomalous_images=1,
        expected_categories=["candle"],
        allowed_image_extensions=[".jpg", ".png"],
        canonical_manifest_path=tmp_path / "interim" / "canonical.jsonl",
        phase2_report_path=tmp_path / "reports" / "phase2.json",
        phase2_error_path=tmp_path / "reports" / "errors.jsonl",
        strict_phase2=True,
    )


def _build_dataset(root: Path) -> None:
    category = root / "candle"
    normal = category / "Data" / "Images" / "Normal" / "000.JPG"
    anomaly = category / "Data" / "Images" / "Anomaly" / "001.JPG"
    mask = category / "Data" / "Masks" / "Anomaly" / "001.png"
    for path in (normal, anomaly, mask):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), "white").save(normal)
    Image.new("RGB", (8, 6), "white").save(anomaly)
    mask_image = Image.new("L", (8, 6), 0)
    mask_image.putpixel((3, 2), 255)
    mask_image.save(mask)

    with (category / "image_anno.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "split", "label", "image", "mask"])
        writer.writerow(["candle", "train", "normal", "Data/Images/Normal/000.JPG", ""])
        writer.writerow(
            ["candle", "test", "chunk of wax missing", "Data/Images/Anomaly/001.JPG", "Data/Masks/Anomaly/001.png"]
        )


def test_parse_visa_builds_canonical_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _build_dataset(config.raw_root)

    result = parse_visa(config)

    assert result.records == 2
    assert result.errors == 0
    lines = config.canonical_manifest_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    anomalous = next(item for item in records if item["condition"] == "anomalous")
    assert anomalous["defect_type"] == "chunk_of_wax_missing"
    assert anomalous["mask"]["foreground_pixels"] == 1
    assert anomalous["mask"]["is_binary"] is True


def test_parse_visa_reports_missing_mask(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(update={"strict_phase2": False})
    _build_dataset(config.raw_root)
    (config.raw_root / "candle" / "Data" / "Masks" / "Anomaly" / "001.png").unlink()

    result = parse_visa(config)

    assert result.records == 1
    assert result.errors == 1
    error = json.loads(config.phase2_error_path.read_text(encoding="utf-8").splitlines()[0])
    assert "Mask file does not exist" in error["message"]


def test_parse_official_per_category_schema_without_object_or_split(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _build_dataset(config.raw_root)
    csv_path = config.raw_root / "candle" / "image_anno.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "mask"])
        writer.writerow(["Data/Images/Normal/000.JPG", "normal", ""])
        writer.writerow(
            ["Data/Images/Anomaly/001.JPG", "chunk of wax missing,foreign particals on candle", "Data/Masks/Anomaly/001.png"]
        )

    result = parse_visa(config)

    assert result.records == 2
    records = [
        json.loads(line)
        for line in config.canonical_manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {record["category"] for record in records} == {"candle"}
    assert {record["source_split"] for record in records} == {"unknown"}
    anomalous = next(item for item in records if item["condition"] == "anomalous")
    assert anomalous["defect_type"] == (
        "chunk_of_wax_missing,foreign_particals_on_candle"
    )


def test_parse_visa_accepts_multilabel_indexed_mask(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _build_dataset(config.raw_root)
    mask_path = (
        config.raw_root
        / "candle"
        / "Data"
        / "Masks"
        / "Anomaly"
        / "001.png"
    )
    mask = Image.new("L", (8, 6), 0)
    mask.putpixel((1, 1), 1)
    mask.putpixel((2, 2), 2)
    mask.putpixel((3, 3), 3)
    mask.save(mask_path)

    result = parse_visa(config)

    assert result.records == 2
    assert result.errors == 0
    records = [
        json.loads(line)
        for line in config.canonical_manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    anomalous = next(item for item in records if item["condition"] == "anomalous")
    assert anomalous["mask"]["foreground_pixels"] == 3
    assert anomalous["mask"]["foreground_ratio"] == 3 / 48
    assert anomalous["mask"]["unique_values"] == [0, 1, 2, 3]
    assert anomalous["mask"]["is_binary"] is False

    report = json.loads(config.phase2_report_path.read_text(encoding="utf-8"))
    assert report["source_binary_masks"] == 0
    assert report["multi_label_masks"] == 1
    assert report["binary_compatible_masks"] == 1
