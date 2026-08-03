import json
from pathlib import Path

import pytest

from visionassist.evaluation.failure_analysis import (
    analyze_predictions,
    write_failure_analysis,
)


def _row(
    instruction_id: str,
    task: str,
    defect: str,
    location: str,
    prediction: str,
    *,
    category: str = "candle",
) -> dict[str, object]:
    return {
        "instruction_id": instruction_id,
        "task_family": task,
        "category": category,
        "condition": "anomalous",
        "defect_type": defect,
        "location": location,
        "prediction": prediction,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_analyzes_defect_and_location_confusions_deterministically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "predictions.jsonl"
    _write(
        source,
        [
            _row(
                "one",
                "defect_identification",
                "small scratches",
                "top_left",
                "The annotated defect is small cracks.",
            ),
            _row(
                "two",
                "localization",
                "small scratches",
                "top_left",
                "The defect is in the top center region.",
            ),
            _row(
                "three",
                "structured_report",
                "small scratches",
                "bottom_right",
                '{"condition":"anomalous","defect_type":"small scratches",'
                '"location":"bottom_right"}',
                category="pcb1",
            ),
        ],
    )

    first = analyze_predictions(source)
    output = write_failure_analysis(source, tmp_path / "analysis.json")
    second = json.loads(output.read_text(encoding="utf-8"))

    assert first == second
    assert first["defect"]["records"] == 2
    assert first["defect"]["exact_matches"] == 1
    assert first["defect"]["errors_by_category"] == {"candle": 1}
    assert first["localization"]["records"] == 2
    assert first["localization"]["exact_matches"] == 1
    assert first["localization"]["adjacent_matches"] == 2
    assert first["localization"]["errors_by_category"] == {"candle": 1}


def test_rejects_duplicate_prediction_ids(tmp_path: Path) -> None:
    source = tmp_path / "predictions.jsonl"
    row = _row(
        "duplicate",
        "defect_identification",
        "scratch",
        "center",
        "scratch",
    )
    _write(source, [row, row])

    with pytest.raises(ValueError, match="Duplicate prediction ID"):
        analyze_predictions(source)
