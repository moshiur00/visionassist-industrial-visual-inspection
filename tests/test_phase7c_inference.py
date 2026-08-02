from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from visionassist.benchmarks.build_visa_baseline import sha256_file
from visionassist.evaluation.adapter import run_adapter_evaluation
from visionassist.inference.generate import run_baseline_inference
from visionassist.inference.model_loader import LoadedInferenceModel
from visionassist.inference.resume import completed_instruction_ids, read_jsonl
from visionassist.inference.schemas import InferenceConfig


def _instruction(instruction_id: str, image_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "instruction_id": instruction_id,
        "image_id": instruction_id.split("__")[0],
        "dataset_split": "test",
        "task_family": "product_identification",
        "template_id": "product_01",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.as_posix()},
                    {"type": "text", "text": "What product category is shown?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The product category is pcb1."}],
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


class FakeBatch(dict[str, torch.Tensor]):
    def to(self, _device: object) -> FakeBatch:
        return self


class FakeProcessor:
    def apply_chat_template(self, *_args: object, **_kwargs: object) -> FakeBatch:
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    def batch_decode(self, _ids: torch.Tensor, **_kwargs: object) -> list[str]:
        return ["The product category is pcb1."]


class FakeModel:
    device = torch.device("cpu")

    def generate(self, **_kwargs: object) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4, 5]])


def _loader(_config: InferenceConfig) -> LoadedInferenceModel:
    return LoadedInferenceModel(
        model=FakeModel(),
        processor=FakeProcessor(),
        resolved_precision="float32",
        quantized_4bit=False,
    )


def _config(tmp_path: Path, benchmark: Path, manifest: Path, stop_after: int | None) -> InferenceConfig:
    output = tmp_path / "outputs"
    return InferenceConfig(
        run_id="test_run",
        benchmark_path=benchmark,
        benchmark_manifest_path=manifest,
        output_dir=output,
        partial_predictions_path=output / "predictions.partial.jsonl",
        predictions_path=output / "predictions.jsonl",
        errors_path=output / "errors.jsonl",
        run_manifest_path=output / "run_manifest.json",
        stop_after=stop_after,
    )


def test_resumable_inference_finalizes_in_benchmark_order(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image)
    benchmark = tmp_path / "benchmark.jsonl"
    rows = [_instruction(f"visa_pcb1_normal_00{i}__product_01", image) for i in range(3)]
    benchmark.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"benchmark_sha256": sha256_file(benchmark)}), encoding="utf-8"
    )

    first = run_baseline_inference(
        _config(tmp_path, benchmark, manifest, stop_after=1),
        project_root=tmp_path,
        loader=_loader,
    )
    assert not first.complete
    assert first.completed_predictions == 1

    second = run_baseline_inference(
        _config(tmp_path, benchmark, manifest, stop_after=None),
        project_root=tmp_path,
        loader=_loader,
    )
    assert second.complete
    assert second.new_predictions == 2
    predictions = read_jsonl(second.predictions_path or Path("missing"))
    assert [row["instruction_id"] for row in predictions] == [
        row["instruction_id"] for row in rows
    ]
    assert completed_instruction_ids(second.partial_predictions_path) == {
        row["instruction_id"] for row in rows
    }


def test_frozen_benchmark_hash_is_enforced(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image)
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(_instruction("visa_pcb1_normal_001__product_01", image)) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"benchmark_sha256": "wrong"}), encoding="utf-8")

    try:
        run_baseline_inference(
            _config(tmp_path, benchmark, manifest, stop_after=None),
            project_root=tmp_path,
            loader=_loader,
        )
    except RuntimeError as exc:
        assert "hash differs" in str(exc)
    else:
        raise AssertionError("Expected a frozen benchmark hash failure")


def test_adapter_evaluation_uses_exact_deterministic_subset(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image)
    dataset = tmp_path / "train.jsonl"
    rows = [
        _instruction(f"visa_pcb1_normal_00{i}__product_01", image)
        for i in range(5)
    ]
    for row in rows:
        row["dataset_split"] = "train"
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "adapter-output"
    adapter = tmp_path / "checkpoint-50"
    adapter.mkdir()
    config = InferenceConfig(
        run_id="adapter_test",
        adapter_path=adapter,
        benchmark_path=dataset,
        benchmark_manifest_path=tmp_path / "missing-manifest.json",
        output_dir=output,
        partial_predictions_path=output / "predictions.partial.jsonl",
        predictions_path=output / "predictions.jsonl",
        errors_path=output / "errors.jsonl",
        run_manifest_path=output / "run_manifest.json",
        evaluation_records_path=output / "evaluation_records.jsonl",
        expected_dataset_split="train",
        subset_limit=3,
        subset_seed=42,
    )

    result = run_adapter_evaluation(config, project_root=tmp_path, loader=_loader)

    assert result.inference.complete
    assert result.inference.completed_predictions == 3
    selected = read_jsonl(config.evaluation_records_path or Path("missing"))
    predictions = read_jsonl(config.predictions_path)
    assert [row["instruction_id"] for row in selected] == [
        row["instruction_id"] for row in predictions
    ]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["adapter_path"] == adapter.as_posix()
    assert summary["records"] == 3
    assert summary["inference_errors"] == 0


def test_inference_restores_partial_predictions_from_persistent_storage(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image)
    benchmark = tmp_path / "benchmark.jsonl"
    rows = [
        _instruction(f"visa_pcb1_normal_00{i}__product_01", image)
        for i in range(3)
    ]
    benchmark.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"benchmark_sha256": sha256_file(benchmark)}),
        encoding="utf-8",
    )
    persistent = tmp_path / "persistent"
    first_config = _config(tmp_path, benchmark, manifest, stop_after=1).model_copy(
        update={
            "persistent_output_dir": persistent,
            "persistent_sync_every": 1,
        }
    )
    first = run_baseline_inference(
        first_config,
        project_root=tmp_path,
        loader=_loader,
    )
    assert first.completed_predictions == 1
    assert (persistent / "predictions.partial.jsonl").is_file()

    shutil.rmtree(first_config.output_dir)
    second_config = first_config.model_copy(update={"stop_after": None})
    second = run_baseline_inference(
        second_config,
        project_root=tmp_path,
        loader=_loader,
    )

    assert second.complete
    assert second.new_predictions == 2
    assert (persistent / "predictions.jsonl").is_file()
