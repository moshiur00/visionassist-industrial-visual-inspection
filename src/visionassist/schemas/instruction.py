"""Schemas for grounded multimodal instruction records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from visionassist.schemas.dataset import DatasetSplit


class MessageContent(BaseModel):
    """One text or image item in a chat message."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image", "text"]
    image: str | None = None
    text: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> MessageContent:
        if self.type == "image":
            if not self.image or self.text is not None:
                raise ValueError("Image content requires only the image field.")
        elif not self.text or self.image is not None:
            raise ValueError("Text content requires only the text field.")
        return self


class ChatMessage(BaseModel):
    """One multimodal conversational message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: list[MessageContent] = Field(min_length=1)


class InstructionMetadata(BaseModel):
    """Machine-readable supervision carried beside each conversation."""

    model_config = ConfigDict(extra="forbid")

    source: str = "visa"
    category: str = Field(min_length=1)
    condition: Literal["normal", "anomalous"]
    defect_type: str | None = None
    location: str | None = None
    visual_severity: Literal["none", "minor", "moderate", "major"]
    answer_format: Literal["text", "json"]
    grounded_by: list[str] = Field(min_length=1)


class InstructionRecord(BaseModel):
    """One Phase 5 VLM instruction example."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    instruction_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    dataset_split: DatasetSplit
    task_family: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=2, max_length=2)
    metadata: InstructionMetadata

    @model_validator(mode="after")
    def validate_conversation(self) -> InstructionRecord:
        if self.messages[0].role != "user" or self.messages[1].role != "assistant":
            raise ValueError("Instruction records require user then assistant messages.")
        user_types = [item.type for item in self.messages[0].content]
        if user_types.count("image") != 1 or user_types.count("text") != 1:
            raise ValueError("User message must contain exactly one image and one text item.")
        if len(self.messages[1].content) != 1 or self.messages[1].content[0].type != "text":
            raise ValueError("Assistant message must contain exactly one text item.")
        return self
