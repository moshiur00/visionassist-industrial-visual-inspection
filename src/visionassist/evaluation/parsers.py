"""Task-specific parsers for VisionAssist generated predictions."""

from __future__ import annotations

from collections.abc import Iterable

from visionassist.evaluation.normalize import (
    normalize_label,
    normalize_text,
    parse_json_object,
    split_compound_label,
)

CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)

CATEGORY_ALIASES = {
    "chewing_gum": "chewinggum",
    "chewinggum": "chewinggum",
    "pipe_fryum": "pipe_fryum",
    "pipefryum": "pipe_fryum",
    "macaroni_1": "macaroni1",
    "macaroni_2": "macaroni2",
    "pcb_1": "pcb1",
    "pcb_2": "pcb2",
    "pcb_3": "pcb3",
    "pcb_4": "pcb4",
}

LOCATIONS = (
    "top_left",
    "top_center",
    "top_right",
    "center_left",
    "center",
    "center_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)

LOCATION_ALIASES = {
    "upper_left": "top_left",
    "upper_center": "top_center",
    "upper_centre": "top_center",
    "upper_right": "top_right",
    "middle_left": "center_left",
    "centre_left": "center_left",
    "middle": "center",
    "centre": "center",
    "middle_center": "center",
    "middle_centre": "center",
    "middle_right": "center_right",
    "centre_right": "center_right",
    "lower_left": "bottom_left",
    "lower_center": "bottom_center",
    "lower_centre": "bottom_center",
    "lower_right": "bottom_right",
}


def parse_condition(text: str) -> str | None:
    """Extract normal or anomalous/defective condition from model text."""

    payload = parse_json_object(text)
    if payload is not None and isinstance(payload.get("condition"), str):
        return parse_condition(str(payload["condition"]))
    normalized = normalize_text(text)
    defective_terms = (
        "defective",
        "anomalous",
        "anomaly is present",
        "should not pass",
        "reject",
        "fails inspection",
        "failed inspection",
    )
    normal_terms = (
        "normal",
        "no anomaly",
        "no defect",
        "acceptable",
        "pass visual quality control",
        "passes inspection",
        "accept",
    )
    if any(term in normalized for term in defective_terms):
        return "anomalous"
    if any(term in normalized for term in normal_terms):
        return "normal"
    return None


def parse_product(text: str) -> str | None:
    """Extract one canonical VisA product category."""

    payload = parse_json_object(text)
    if payload is not None:
        for key in ("product", "category", "object"):
            if isinstance(payload.get(key), str):
                parsed = parse_product(str(payload[key]))
                if parsed is not None:
                    return parsed
    normalized = normalize_label(text)
    padded = f"_{normalized}_"
    candidates = {**{category: category for category in CATEGORIES}, **CATEGORY_ALIASES}
    for alias in sorted(candidates, key=len, reverse=True):
        if f"_{alias}_" in padded:
            return candidates[alias]
    return None


def parse_location(text: str) -> str | None:
    """Extract one canonical nine-grid location."""

    payload = parse_json_object(text)
    if payload is not None and isinstance(payload.get("location"), str):
        return parse_location(str(payload["location"]))
    normalized = normalize_label(text)
    padded = f"_{normalized}_"
    aliases = {**{location: location for location in LOCATIONS}, **LOCATION_ALIASES}
    for alias in sorted(aliases, key=len, reverse=True):
        if f"_{alias}_" in padded:
            return aliases[alias]
    return None


def parse_severity(text: str) -> str | None:
    """Extract none/minor/moderate/major visual severity."""

    payload = parse_json_object(text)
    if payload is not None:
        value = payload.get("visual_severity", payload.get("severity"))
        if isinstance(value, str):
            return parse_severity(value)
    normalized = normalize_label(text)
    for severity in ("major", "moderate", "minor", "none"):
        if severity in normalized.split("_"):
            return severity
    return None


def parse_defects(text: str, vocabulary: Iterable[str]) -> set[str]:
    """Extract normalized defect concepts using JSON fields and known vocabulary."""

    payload = parse_json_object(text)
    if payload is not None:
        value = payload.get("defect_type", payload.get("defect"))
        if value is None:
            return set()
        if isinstance(value, str):
            return split_compound_label(value)
        if isinstance(value, list):
            return {
                normalized
                for item in value
                if isinstance(item, str) and (normalized := normalize_label(item))
            }

    normalized_text = normalize_label(text)
    padded = f"_{normalized_text}_"
    atoms: set[str] = set()
    for compound in vocabulary:
        atoms.update(split_compound_label(compound))
    matches = {
        atom
        for atom in atoms
        if atom and f"_{atom}_" in padded
    }
    return matches


def parse_uncertainty(text: str) -> dict[str, bool]:
    """Detect appropriate abstention and unsupported causal/safety claims."""

    normalized = normalize_text(text)
    abstention_phrases = (
        "cannot be determined",
        "cannot determine",
        "cannot be established",
        "not possible to determine",
        "insufficient information",
        "from this image alone",
    )
    root_cause_claims = (
        "caused by",
        "the cause is",
        "root cause is",
        "due to overheating",
        "due to manufacturing",
        "because the machine",
    )
    safety_claims = (
        "safe to use",
        "unsafe to use",
        "poses a safety risk",
        "no safety risk",
        "hazardous",
        "dangerous",
    )
    return {
        "abstains": any(phrase in normalized for phrase in abstention_phrases),
        "unsupported_root_cause_claim": any(
            phrase in normalized for phrase in root_cause_claims
        ),
        "unsupported_safety_claim": any(phrase in normalized for phrase in safety_claims),
    }
