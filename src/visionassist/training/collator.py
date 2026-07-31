"""Qwen2.5-VL supervised fine-tuning collator with assistant-only labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from visionassist.schemas.instruction import InstructionRecord
from visionassist.training.formatting import qwen_messages, resolve_image_path


class QwenAssistantOnlyCollator:
    """Build VLM batches while masking image, user, and padding tokens.

    PyTorch and Transformers are imported lazily so the core data audit can run
    with only the project's standard dependencies.
    """

    def __init__(
        self,
        processor: Any,
        project_root: Path,
        *,
        max_length: int | None = None,
    ) -> None:
        self.processor = processor
        self.project_root = project_root.resolve()
        self.max_length = max_length

    def _encode(self, record: InstructionRecord) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on training extra
            raise RuntimeError("Install the training extra to use the VLM collator.") from exc

        image_path = resolve_image_path(record, self.project_root)
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        full_messages = qwen_messages(record, self.project_root, include_assistant=True)
        prompt_messages = qwen_messages(record, self.project_root, include_assistant=False)
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        common = {
            "images": [image],
            "return_tensors": "pt",
            "padding": False,
        }
        if self.max_length is not None:
            common.update({"truncation": True, "max_length": self.max_length})

        full = self.processor(text=[full_text], **common)
        prompt = self.processor(text=[prompt_text], **common)
        full_ids = full["input_ids"][0]
        prompt_ids = prompt["input_ids"][0]

        prefix_length = 0
        maximum = min(full_ids.numel(), prompt_ids.numel())
        while prefix_length < maximum and full_ids[prefix_length] == prompt_ids[prefix_length]:
            prefix_length += 1
        if prefix_length == full_ids.numel():
            raise ValueError(
                f"No assistant target tokens remained for {record.instruction_id}."
            )

        labels = full_ids.clone()
        labels[:prefix_length] = -100
        output = {key: value for key, value in full.items()}
        output["labels"] = labels.unsqueeze(0)
        output["assistant_prefix_length"] = prefix_length
        output["instruction_id"] = record.instruction_id
        return output

    def __call__(self, records: Sequence[InstructionRecord]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on training extra
            raise RuntimeError("Install the training extra to use the VLM collator.") from exc

        encoded = [self._encode(record) for record in records]
        tokenizer = self.processor.tokenizer
        token_batch = tokenizer.pad(
            [
                {
                    "input_ids": item["input_ids"][0],
                    "attention_mask": item["attention_mask"][0],
                }
                for item in encoded
            ],
            return_tensors="pt",
        )
        sequence_length = token_batch["input_ids"].shape[1]
        labels = torch.full(
            (len(encoded), sequence_length),
            -100,
            dtype=token_batch["input_ids"].dtype,
        )
        for index, item in enumerate(encoded):
            row = item["labels"][0]
            if tokenizer.padding_side == "left":
                labels[index, -row.numel() :] = row
            else:
                labels[index, : row.numel()] = row

        batch: dict[str, Any] = dict(token_batch)
        batch["labels"] = labels
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [item[key] for item in encoded if key in item]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch
