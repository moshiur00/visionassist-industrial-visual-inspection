"""Lazy JSONL dataset adapter for VisionAssist multimodal instructions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visionassist.schemas.instruction import InstructionRecord


class VisionAssistJsonlDataset:
    """Load validated instruction records from one JSONL split.

    This lightweight adapter intentionally returns project ``InstructionRecord``
    objects. Image loading and Qwen processing remain the collator's responsibility.
    """

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Instruction dataset not found: {path}")
        self.path = path
        self._offsets: list[int] = []
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> InstructionRecord:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            line = handle.readline()
        return InstructionRecord.model_validate_json(line)

    def to_hf_generator(self) -> Any:
        """Yield JSON-compatible records for ``datasets.Dataset.from_generator``."""

        for index in range(len(self)):
            yield self[index].model_dump(mode="json")
