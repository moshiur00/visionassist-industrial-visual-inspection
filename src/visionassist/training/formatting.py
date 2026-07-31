"""Formatting helpers for VisionAssist multimodal instruction records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visionassist.schemas.instruction import InstructionRecord


def user_prompt(record: InstructionRecord) -> str:
    """Return the single user text prompt from an instruction record."""

    for item in record.messages[0].content:
        if item.type == "text" and item.text:
            return item.text
    raise ValueError(f"Instruction {record.instruction_id} has no user prompt.")


def image_reference(record: InstructionRecord) -> Path:
    """Return the image path stored in an instruction record."""

    for item in record.messages[0].content:
        if item.type == "image" and item.image:
            return Path(item.image)
    raise ValueError(f"Instruction {record.instruction_id} has no image reference.")


def assistant_target(record: InstructionRecord) -> str:
    """Return the assistant target text."""

    text = record.messages[1].content[0].text
    if not text:
        raise ValueError(f"Instruction {record.instruction_id} has no assistant target.")
    return text


def resolve_image_path(record: InstructionRecord, project_root: Path) -> Path:
    """Resolve a project-relative image reference without allowing path escape."""

    project_root = project_root.resolve()
    reference = image_reference(record)
    candidate = reference if reference.is_absolute() else project_root / reference
    candidate = candidate.resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"Instruction {record.instruction_id} points outside the project root: "
            f"{reference}"
        ) from exc
    return candidate


def qwen_messages(
    record: InstructionRecord,
    project_root: Path,
    *,
    include_assistant: bool = True,
) -> list[dict[str, Any]]:
    """Convert a project instruction into Qwen-compatible chat messages."""

    image_path = resolve_image_path(record, project_root)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.as_posix()},
                {"type": "text", "text": user_prompt(record)},
            ],
        }
    ]
    if include_assistant:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_target(record)}],
            }
        )
    return messages
