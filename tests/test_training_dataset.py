"""Tests for the lazy Phase 6 JSONL dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path

from visionassist.training.dataset import VisionAssistJsonlDataset


def test_lazy_jsonl_dataset_reads_records(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    payload = {
        "schema_version": "1.0",
        "instruction_id": "sample-1",
        "image_id": "image-1",
        "dataset_split": "train",
        "task_family": "product_identification",
        "template_id": "product_01",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data/raw/visa/a.png"},
                    {"type": "text", "text": "What product is shown?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The product is pcb1."}],
            },
        ],
        "metadata": {
            "source": "visa",
            "category": "pcb1",
            "condition": "normal",
            "defect_type": None,
            "location": None,
            "visual_severity": "none",
            "answer_format": "text",
            "grounded_by": ["product_category"],
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    dataset = VisionAssistJsonlDataset(path)
    assert len(dataset) == 1
    assert dataset[0].instruction_id == "sample-1"
