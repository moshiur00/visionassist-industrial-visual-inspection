from pathlib import Path

import pytest
from pydantic import ValidationError

from visionassist.schemas.dataset import Condition, RawImageRecord


def test_normal_record_rejects_mask() -> None:
    with pytest.raises(ValidationError):
        RawImageRecord(
            image_id="visa_pcb1_001",
            category="pcb1",
            condition=Condition.NORMAL,
            image_path=Path("normal.png"),
            mask_path=Path("mask.png"),
            width=10,
            height=10,
            file_size_bytes=100,
        )


def test_anomalous_record_accepts_mask() -> None:
    record = RawImageRecord(
        image_id="visa_pcb1_002",
        category="pcb1",
        condition=Condition.ANOMALOUS,
        image_path=Path("anomaly.png"),
        mask_path=Path("mask.png"),
        width=10,
        height=10,
        file_size_bytes=100,
    )
    assert record.condition is Condition.ANOMALOUS
