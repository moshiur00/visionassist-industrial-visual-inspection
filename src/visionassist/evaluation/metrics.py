"""Small dependency-free metrics used by Phase 7B."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(
    truths: Sequence[str],
    predictions: Sequence[str | None],
    labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro precision/recall/F1, and confusion counts."""

    if len(truths) != len(predictions):
        raise ValueError("Truth and prediction lengths differ.")
    label_list = sorted(set(labels or truths) | {p for p in predictions if p is not None})
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    unparsed = 0
    parsed_predictions: list[str] = []
    for truth, prediction in zip(truths, predictions, strict=True):
        predicted = prediction if prediction is not None else "__unparsed__"
        if prediction is None:
            unparsed += 1
        parsed_predictions.append(predicted)
        confusion[(truth, predicted)] += 1
        correct += int(truth == predicted)

    per_label: dict[str, dict[str, float]] = {}
    for label in label_list:
        tp = confusion[(label, label)]
        fp = sum(count for (truth, pred), count in confusion.items() if pred == label and truth != label)
        fn = sum(count for (truth, pred), count in confusion.items() if truth == label and pred != label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for truth in truths if truth == label),
        }

    return {
        "count": len(truths),
        "accuracy": safe_divide(correct, len(truths)),
        "macro_precision": safe_divide(
            sum(item["precision"] for item in per_label.values()), len(per_label)
        ),
        "macro_recall": safe_divide(
            sum(item["recall"] for item in per_label.values()), len(per_label)
        ),
        "macro_f1": safe_divide(
            sum(item["f1"] for item in per_label.values()), len(per_label)
        ),
        "unparseable_count": unparsed,
        "unparseable_rate": safe_divide(unparsed, len(truths)),
        "per_label": per_label,
        "confusion_matrix": {
            truth: {
                pred: confusion[(truth, pred)]
                for pred in sorted(set(parsed_predictions))
            }
            for truth in sorted(set(truths))
        },
    }


def set_metrics(truth: set[str], prediction: set[str]) -> dict[str, float | bool]:
    """Compute exact, precision, recall, and F1 for one set-valued label."""

    intersection = truth & prediction
    precision = safe_divide(len(intersection), len(prediction))
    recall = safe_divide(len(intersection), len(truth))
    if not truth and not prediction:
        precision = recall = 1.0
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "exact_match": truth == prediction,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_set_metrics(items: Sequence[dict[str, float | bool]]) -> dict[str, float]:
    """Average set-valued metrics across records."""

    if not items:
        return {"count": 0.0, "exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "count": float(len(items)),
        "exact_match": sum(bool(item["exact_match"]) for item in items) / len(items),
        "precision": sum(float(item["precision"]) for item in items) / len(items),
        "recall": sum(float(item["recall"]) for item in items) / len(items),
        "f1": sum(float(item["f1"]) for item in items) / len(items),
    }


GRID_COORDS = {
    "top_left": (0, 0),
    "top_center": (0, 1),
    "top_right": (0, 2),
    "center_left": (1, 0),
    "center": (1, 1),
    "center_right": (1, 2),
    "bottom_left": (2, 0),
    "bottom_center": (2, 1),
    "bottom_right": (2, 2),
}


def adjacent_location(truth: str, prediction: str | None) -> bool:
    """Return true for exact or immediately neighboring nine-grid cells."""

    if prediction is None or truth not in GRID_COORDS or prediction not in GRID_COORDS:
        return False
    truth_row, truth_col = GRID_COORDS[truth]
    pred_row, pred_col = GRID_COORDS[prediction]
    return max(abs(truth_row - pred_row), abs(truth_col - pred_col)) <= 1
