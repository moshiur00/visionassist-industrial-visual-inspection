"""Task-specific parsers for VisionAssist generated predictions.

The parsers are intentionally conservative: they normalize formatting and a
small set of semantically equivalent phrases, but they do not turn generic
object descriptions into VisA labels. This keeps the evaluator from giving the
base model credit for guesses that do not identify the requested category.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

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
    "chewing_gums": "chewinggum",
    "pipe_fryum": "pipe_fryum",
    "pipe_fryums": "pipe_fryum",
    "pipefryum": "pipe_fryum",
    "macaroni_1": "macaroni1",
    "macaroni_type_1": "macaroni1",
    "macaroni_2": "macaroni2",
    "macaroni_type_2": "macaroni2",
    "pcb_1": "pcb1",
    "pcb_board_1": "pcb1",
    "printed_circuit_board_1": "pcb1",
    "pcb_2": "pcb2",
    "pcb_board_2": "pcb2",
    "printed_circuit_board_2": "pcb2",
    "pcb_3": "pcb3",
    "pcb_board_3": "pcb3",
    "printed_circuit_board_3": "pcb3",
    "pcb_4": "pcb4",
    "pcb_board_4": "pcb4",
    "printed_circuit_board_4": "pcb4",
    "capsule": "capsules",
    "cashew_nut": "cashew",
    "cashew_nuts": "cashew",
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
    "upper_middle": "top_center",
    "upper_right": "top_right",
    "middle_left": "center_left",
    "centre_left": "center_left",
    "middle": "center",
    "centre": "center",
    "middle_center": "center",
    "middle_centre": "center",
    "central": "center",
    "middle_right": "center_right",
    "centre_right": "center_right",
    "lower_left": "bottom_left",
    "lower_center": "bottom_center",
    "lower_centre": "bottom_center",
    "lower_middle": "bottom_center",
    "lower_right": "bottom_right",
}

# Semantically equivalent defect words used only for the semantic metric. The
# strict metric still evaluates the original source-label atoms unchanged.
DEFECT_ATOM_ALIASES = {
    "scratch": "scratch",
    "scratches": "scratch",
    "small_scratch": "small_scratch",
    "small_scratches": "small_scratch",
    "crack": "crack",
    "cracks": "crack",
    "small_crack": "small_crack",
    "small_cracks": "small_crack",
    "hole": "hole",
    "holes": "hole",
    "small_hole": "small_hole",
    "small_holes": "small_hole",
    "colour_spot": "color_spot",
    "color_spot": "color_spot",
    "different_colour_spot": "different_color_spot",
    "different_color_spot": "different_color_spot",
    "same_colour_spot": "same_color_spot",
    "same_color_spot": "same_color_spot",
    "similar_colour_spot": "similar_color_spot",
    "similar_color_spot": "similar_color_spot",
    "discolouration": "discolor",
    "discoloration": "discolor",
    "discoloured": "discolor",
    "discolored": "discolor",
    "misshapen": "misshape",
    "deformed": "misshape",
    "burned": "burnt",
    "burned_area": "burnt",
    "broken_corner": "corner_missing",
    "missing_corner": "corner_missing",
    "edge_breakage": "corner_or_edge_breakage",
    "corner_breakage": "corner_or_edge_breakage",
    "middle_broken": "middle_breakage",
    "center_breakage": "middle_breakage",
    "centre_breakage": "middle_breakage",
    "stuck": "stuck_together",
    "joined_together": "stuck_together",
    "wrong_position": "wrong_place",
    "misplaced": "wrong_place",
    "foreign_particle": "foreign_particals_on_candle",
    "foreign_particles": "foreign_particals_on_candle",
}

NEGATED_ANOMALY_PATTERNS = (
    r"\bno\s+(?:annotated\s+)?anomal(?:y|ies)\b",
    r"\bthere\s+(?:is|are)\s+no\s+(?:annotated\s+)?anomal(?:y|ies)\b",
    r"\bno\s+(?:visible\s+)?defects?\b",
    r"\bwithout\s+(?:any\s+)?(?:visible\s+)?defects?\b",
    r"\bnot\s+defective\b",
)

POSITIVE_ANOMALY_PATTERNS = (
    r"\byes\s*,?\s+there\s+(?:is|are)\s+(?:an\s+)?(?:annotated\s+)?anomal(?:y|ies)\b",
    r"\ban\s+anomaly\s+is\s+present\b",
    r"\bannotated\s+anomaly\s+is\s+present\b",
    r"\bdefect(?:ive)?\b",
    r"\bshould\s+not\s+pass\b",
    r"\breject(?:ed)?\b",
    r"\bfails?\s+(?:visual\s+)?inspection\b",
)


def _json_string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
    return None


def parse_condition(text: str) -> str | None:
    """Extract normal or anomalous condition from model text.

    Negated anomaly phrases are checked before the word ``anomaly`` itself, so
    statements such as "no annotated anomalies" are not misclassified.
    """

    payload = parse_json_object(text)
    if payload is not None:
        value = _json_string(payload, "condition", "status", "result", "pass")
        if value is not None:
            normalized_value = normalize_label(value)
            if normalized_value in {"pass", "passed", "normal", "acceptable", "ok", "true"}:
                return "normal"
            if normalized_value in {"fail", "failed", "defective", "anomalous", "reject", "rejected", "false"}:
                return "anomalous"
            parsed = parse_condition(value)
            if parsed is not None:
                return parsed

        defect = payload.get("defect_type", payload.get("defect"))
        if defect is None and any(key in payload for key in ("defect_type", "defect")):
            return "normal"
        if isinstance(defect, str):
            normalized_defect = normalize_label(defect)
            if normalized_defect in {"none", "no_defect", "no_anomaly", "normal"}:
                return "normal"

    normalized = normalize_text(text)
    if any(re.search(pattern, normalized) for pattern in NEGATED_ANOMALY_PATTERNS):
        return "normal"
    if any(re.search(pattern, normalized) for pattern in POSITIVE_ANOMALY_PATTERNS):
        return "anomalous"

    normal_terms = (
        "appears to be normal",
        "item is normal",
        "looks normal",
        "acceptable",
        "should pass visual quality control",
        "passes inspection",
        "pass inspection",
        "no visible irregularities",
        "no visible imperfections",
        "uniform size and shape",
    )
    if any(term in normalized for term in normal_terms):
        return "normal"
    return None


def parse_product(text: str) -> str | None:
    """Extract one canonical VisA category without guessing generic objects."""

    payload = parse_json_object(text)
    if payload is not None:
        for key in ("product", "category", "object", "product_category"):
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


def _grid_from_row_column(text: str) -> str | None:
    normalized = normalize_text(text)
    word_number = {
        "first": 1,
        "1st": 1,
        "one": 1,
        "second": 2,
        "2nd": 2,
        "two": 2,
        "third": 3,
        "3rd": 3,
        "three": 3,
    }
    pattern = re.compile(
        r"\b(first|1st|one|second|2nd|two|third|3rd|three)\s+row\s*,?\s*"
        r"(?:and\s+)?(?:the\s+)?(first|1st|one|second|2nd|two|third|3rd|three)\s+column\b"
    )
    match = pattern.search(normalized)
    if not match:
        pattern = re.compile(
            r"\b(first|1st|one|second|2nd|two|third|3rd|three)\s+row\s*,?\s*"
            r"(first|1st|one|second|2nd|two|third|3rd|three)\s+column\b"
        )
        match = pattern.search(normalized)
    if not match:
        return None
    row = word_number[match.group(1)]
    column = word_number[match.group(2)]
    rows = {1: "top", 2: "center", 3: "bottom"}
    columns = {1: "left", 2: "center", 3: "right"}
    if row == 2 and column == 2:
        return "center"
    return f"{rows[row]}_{columns[column]}"


def parse_location(text: str) -> str | None:
    """Extract one canonical nine-grid location."""

    payload = parse_json_object(text)
    if payload is not None:
        value = payload.get("location", payload.get("region"))
        if isinstance(value, str):
            parsed = parse_location(value)
            if parsed is not None:
                return parsed

    row_column = _grid_from_row_column(text)
    if row_column is not None:
        return row_column

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
            normalized = normalize_label(value)
            severity_aliases = {
                "none": "none",
                "no_severity": "none",
                "minor": "minor",
                "low": "minor",
                "mild": "minor",
                "moderate": "moderate",
                "medium": "moderate",
                "major": "major",
                "high": "major",
                "severe": "major",
            }
            if normalized in severity_aliases:
                return severity_aliases[normalized]
    normalized = normalize_label(text)
    for severity in ("major", "moderate", "minor", "none"):
        if severity in normalized.split("_"):
            return severity
    return None


def canonicalize_defect_atom(atom: str) -> str:
    """Canonicalize a defect atom for semantic (not strict) comparison."""

    normalized = normalize_label(atom)
    return DEFECT_ATOM_ALIASES.get(normalized, normalized)


def canonicalize_defect_set(values: Iterable[str]) -> set[str]:
    return {canonicalize_defect_atom(value) for value in values if normalize_label(value)}


def _defect_atoms(vocabulary: Iterable[str]) -> set[str]:
    atoms: set[str] = set()
    for compound in vocabulary:
        atoms.update(split_compound_label(compound))
    return atoms


def parse_defects(text: str, vocabulary: Iterable[str], *, semantic: bool = False) -> set[str]:
    """Extract defect concepts using JSON fields and known source vocabulary.

    ``semantic=False`` preserves strict source atoms. ``semantic=True`` applies
    conservative singular/plural and spelling equivalences.
    """

    payload = parse_json_object(text)
    if payload is not None:
        value = payload.get("defect_type", payload.get("defect", payload.get("issue")))
        if value is None:
            return set()
        if isinstance(value, str):
            result = split_compound_label(value)
            return canonicalize_defect_set(result) if semantic else result
        if isinstance(value, list):
            result = {
                normalized
                for item in value
                if isinstance(item, str) and (normalized := normalize_label(item))
            }
            return canonicalize_defect_set(result) if semantic else result

    normalized_text = normalize_label(text)
    padded = f"_{normalized_text}_"
    atoms = _defect_atoms(vocabulary)
    matches = {atom for atom in atoms if atom and f"_{atom}_" in padded}

    if semantic:
        semantic_atoms = {canonicalize_defect_atom(atom) for atom in atoms}
        words = set(normalized_text.split("_"))
        # Phrase matching for conservative aliases such as "small hole" and
        # British/American spelling variants.
        alias_candidates = set(DEFECT_ATOM_ALIASES)
        for alias in alias_candidates:
            if f"_{alias}_" in padded:
                semantic_atoms.add(canonicalize_defect_atom(alias))
        semantic_matches = canonicalize_defect_set(matches)
        matched_aliases = [
            alias
            for alias in sorted(DEFECT_ATOM_ALIASES, key=len, reverse=True)
            if f"_{alias}_" in padded or ("_" not in alias and alias in words)
        ]
        accepted_aliases: list[str] = []
        for alias in matched_aliases:
            if any(
                alias != longer
                and alias in longer.split("_")
                for longer in accepted_aliases
            ):
                continue
            accepted_aliases.append(alias)
            semantic_matches.add(DEFECT_ATOM_ALIASES[alias])
        return semantic_matches & semantic_atoms
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
