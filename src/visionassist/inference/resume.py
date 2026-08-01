"""Crash-safe JSONL helpers for resumable inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete JSON objects, ignoring a truncated final line."""

    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted write can leave only the final line incomplete.
                remaining = handle.read()
                if remaining.strip():
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}")
                break
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(payload)
    return records


def completed_instruction_ids(path: Path) -> set[str]:
    """Return completed instruction IDs from a partial/final prediction file."""

    identifiers: set[str] = set()
    for payload in read_jsonl(path):
        instruction_id = payload.get("instruction_id")
        if not isinstance(instruction_id, str) or not instruction_id:
            raise ValueError(f"Prediction without instruction_id in {path}")
        if instruction_id in identifiers:
            raise ValueError(f"Duplicate prediction ID in {path}: {instruction_id}")
        identifiers.add(instruction_id)
    return identifiers


def append_jsonl(path: Path, payload: dict[str, Any], *, fsync: bool = True) -> None:
    """Append one durable JSONL record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON through a temporary file and atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write a complete JSONL file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in records:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)
