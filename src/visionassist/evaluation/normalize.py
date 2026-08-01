"""Normalization helpers for deterministic baseline evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_label(value: str | None) -> str:
    """Normalize a free-text label to a stable underscore form."""

    if not value:
        return ""
    normalized = value.strip().lower().replace("&", " and ")
    normalized = NON_ALNUM_RE.sub("_", normalized)
    return normalized.strip("_")


def normalize_text(value: str) -> str:
    """Normalize text for phrase matching without destroying punctuation first."""

    return SPACE_RE.sub(" ", value.strip().lower())


def split_compound_label(value: str | None) -> set[str]:
    """Split comma-delimited source labels into normalized atomic concepts."""

    if not value:
        return set()
    return {
        normalized
        for part in value.split(",")
        if (normalized := normalize_label(part))
    }


def parse_json_object(text: str) -> dict[str, object] | None:
    """Parse a JSON object, tolerating fenced or surrounding explanatory text."""

    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        unfenced = re.sub(r"```(?:json)?", "", stripped, flags=re.IGNORECASE)
        candidates.append(unfenced.replace("```", "").strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    """Return whether normalized text contains any normalized phrase."""

    normalized = normalize_label(text)
    padded = f"_{normalized}_"
    return any(f"_{normalize_label(phrase)}_" in padded for phrase in phrases)
