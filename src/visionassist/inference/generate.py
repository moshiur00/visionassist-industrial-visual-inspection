"""Resumable untouched-model inference for the frozen VisionAssist benchmark."""

from __future__ import annotations

import gc
import json
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from visionassist.benchmarks.build_visa_baseline import sha256_file
from visionassist.inference.model_loader import LoadedInferenceModel, load_qwen25vl
from visionassist.inference.resume import (
    append_jsonl,
    atomic_write_json,
    completed_instruction_ids,
    read_jsonl,
    write_jsonl_atomic,
)
from visionassist.inference.runtime import runtime_metadata
from visionassist.inference.schemas import InferenceConfig
from visionassist.schemas.instruction import InstructionRecord
from visionassist.training.formatting import (
    assistant_target,
    resolve_image_path,
    user_prompt,
)
from visionassist.training.dataset import VisionAssistJsonlDataset
from visionassist.training.experiment import subset_dataset


@dataclass(frozen=True)
class BaselineInferenceResult:
    """Summary returned after a complete or intentionally limited run."""

    run_id: str
    benchmark_records: int
    completed_predictions: int
    new_predictions: int
    errors: int
    complete: bool
    predictions_path: Path | None
    partial_predictions_path: Path
    manifest_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_benchmark(config: InferenceConfig) -> list[InstructionRecord]:
    path = config.benchmark_path
    if not path.is_file():
        raise FileNotFoundError(f"Frozen benchmark not found: {path}")
    records: list[InstructionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = InstructionRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid benchmark row {path}:{line_number}: {exc}") from exc
            records.append(record)
    wrong_split = [
        record.instruction_id
        for record in records
        if record.dataset_split.value != config.expected_dataset_split
    ]
    if wrong_split:
        raise ValueError(
            f"Expected {config.expected_dataset_split} records; first mismatch: "
            f"{wrong_split[0]}"
        )
    if config.subset_limit is not None:
        dataset = VisionAssistJsonlDataset(path)
        subset = subset_dataset(dataset, config.subset_limit, config.subset_seed)
        records = [subset[index] for index in range(len(subset))]
    identifiers = [record.instruction_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Frozen benchmark contains duplicate instruction IDs.")
    return records


def _manifest_benchmark_hash(config: InferenceConfig) -> str | None:
    if not config.benchmark_manifest_path.is_file():
        return None
    payload = json.loads(config.benchmark_manifest_path.read_text(encoding="utf-8"))
    key = {
        "train": "train_sha256",
        "validation": "validation_sha256",
        "test": "benchmark_sha256",
    }[config.expected_dataset_split]
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _verify_frozen_benchmark(config: InferenceConfig) -> str:
    actual = sha256_file(config.benchmark_path)
    expected = _manifest_benchmark_hash(config)
    if (
        expected is not None
        and actual != expected
        and not config.allow_path_normalized_hash_mismatch
    ):
        raise RuntimeError(
            "Frozen benchmark hash differs from its manifest. Revalidate Phase 7A."
        )
    return actual


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _messages(record: InstructionRecord, image: Image.Image, system_prompt: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt(record)},
            ],
        }
    )
    return messages


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    """Move tensors for non-dispatched models; accelerate-dispatched models self-route."""

    device = getattr(model, "device", None)
    if device is None or str(device) == "meta":
        return inputs
    try:
        return inputs.to(device)
    except (AttributeError, RuntimeError):
        return inputs


def _generate_one(
    record: InstructionRecord,
    loaded: LoadedInferenceModel,
    config: InferenceConfig,
    project_root: Path,
) -> dict[str, Any]:
    import torch

    image_path = resolve_image_path(record, project_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    started = time.perf_counter()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        messages = _messages(record, image, config.system_prompt)
        inputs = loaded.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    inputs = _move_inputs_to_model(inputs, loaded.model)
    input_length = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        generated = loaded.model.generate(
            **inputs,
            **config.generation.model_kwargs(),
        )
    new_token_ids = generated[:, input_length:]
    prediction = loaded.processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    elapsed = time.perf_counter() - started

    generated_tokens = int(new_token_ids.shape[-1])
    return {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "instruction_id": record.instruction_id,
        "image_id": record.image_id,
        "task_family": record.task_family,
        "category": record.metadata.category,
        "condition": record.metadata.condition,
        "defect_type": record.metadata.defect_type,
        "location": record.metadata.location,
        "visual_severity": record.metadata.visual_severity,
        "prompt": user_prompt(record),
        "ground_truth": assistant_target(record),
        "prediction": prediction,
        "input_token_count": input_length,
        "generated_token_count": generated_tokens,
        "generation_time_seconds": round(elapsed, 6),
        "image_path": image_path.relative_to(project_root.resolve()).as_posix(),
        "generated_at_utc": _utc_now(),
        "error": None,
    }


def _error_payload(record: InstructionRecord, config: InferenceConfig, exc: Exception) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "instruction_id": record.instruction_id,
        "image_id": record.image_id,
        "task_family": record.task_family,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "recorded_at_utc": _utc_now(),
    }


def _load_failed_ids(path: Path) -> set[str]:
    return {
        str(payload["instruction_id"])
        for payload in read_jsonl(path)
        if isinstance(payload.get("instruction_id"), str)
    }


def _restore_persistent_inference(config: InferenceConfig) -> None:
    if config.persistent_output_dir is None or config.overwrite:
        return
    for local in (
        config.partial_predictions_path,
        config.predictions_path,
        config.errors_path,
        config.run_manifest_path,
    ):
        persistent = config.persistent_output_dir / local.name
        if persistent.is_file() and not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(persistent, local)


def _sync_persistent_inference(config: InferenceConfig) -> None:
    if config.persistent_output_dir is None:
        return
    config.persistent_output_dir.mkdir(parents=True, exist_ok=True)
    for source in (
        config.partial_predictions_path,
        config.predictions_path,
        config.errors_path,
        config.run_manifest_path,
        config.evaluation_records_path,
    ):
        if source is None or not source.is_file():
            continue
        destination = config.persistent_output_dir / source.name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def _initial_manifest(
    config: InferenceConfig,
    benchmark_hash: str,
    records: int,
    loaded: LoadedInferenceModel,
) -> dict[str, Any]:
    adapter_files: dict[str, str] = {}
    if config.adapter_path is not None and config.adapter_path.is_dir():
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = config.adapter_path / name
            if path.is_file():
                adapter_files[name] = sha256_file(path)
    return {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "status": "running",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "processor_revision": config.processor_revision,
        "benchmark_path": config.benchmark_path.as_posix(),
        "benchmark_sha256": benchmark_hash,
        "benchmark_records": records,
        "seed": config.seed,
        "system_prompt": config.system_prompt,
        "precision_requested": config.precision,
        "precision_resolved": loaded.resolved_precision,
        "load_in_4bit": loaded.quantized_4bit,
        "adapter_path": loaded.adapter_path,
        "adapter_checkpoint": (
            config.adapter_path.as_posix() if config.adapter_path else None
        ),
        "adapter_file_sha256": adapter_files,
        "expected_dataset_split": config.expected_dataset_split,
        "subset_limit": config.subset_limit,
        "subset_seed": config.subset_seed,
        "source_hash_path_normalization_override": (
            config.allow_path_normalized_hash_mismatch
        ),
        "image_min_pixels": config.image_min_pixels,
        "image_max_pixels": config.image_max_pixels,
        "device_map": config.device_map,
        "attention_implementation": config.attention_implementation,
        "generation_config": config.generation.model_dump(),
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "completed_predictions": 0,
        "errors": 0,
        "runtime": runtime_metadata(),
    }


def _finalize_predictions(config: InferenceConfig, benchmark_ids: list[str]) -> Path:
    rows = read_jsonl(config.partial_predictions_path)
    by_id = {str(row["instruction_id"]): row for row in rows}
    if set(by_id) != set(benchmark_ids):
        missing = sorted(set(benchmark_ids) - set(by_id))
        extra = sorted(set(by_id) - set(benchmark_ids))
        raise RuntimeError(
            f"Cannot finalize predictions: missing={len(missing)}, extra={len(extra)}"
        )
    ordered = [by_id[instruction_id] for instruction_id in benchmark_ids]
    write_jsonl_atomic(config.predictions_path, ordered)
    return config.predictions_path


def run_baseline_inference(
    config: InferenceConfig,
    *,
    project_root: Path = Path.cwd(),
    loader: Callable[[InferenceConfig], LoadedInferenceModel] = load_qwen25vl,
) -> BaselineInferenceResult:
    """Generate raw untouched-model predictions with durable resume support."""

    project_root = project_root.resolve()
    benchmark_hash = _verify_frozen_benchmark(config)
    records = _load_benchmark(config)
    benchmark_ids = [record.instruction_id for record in records]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _restore_persistent_inference(config)
    if config.evaluation_records_path is not None:
        write_jsonl_atomic(
            config.evaluation_records_path,
            [record.model_dump(mode="json") for record in records],
        )

    if config.overwrite:
        for path in (
            config.partial_predictions_path,
            config.predictions_path,
            config.errors_path,
            config.run_manifest_path,
        ):
            path.unlink(missing_ok=True)
    elif config.predictions_path.is_file():
        completed = completed_instruction_ids(config.predictions_path)
        if completed == set(benchmark_ids):
            return BaselineInferenceResult(
                run_id=config.run_id,
                benchmark_records=len(records),
                completed_predictions=len(completed),
                new_predictions=0,
                errors=len(read_jsonl(config.errors_path)),
                complete=True,
                predictions_path=config.predictions_path,
                partial_predictions_path=config.partial_predictions_path,
                manifest_path=config.run_manifest_path,
            )
        raise RuntimeError(
            f"Incomplete final predictions file exists: {config.predictions_path}"
        )

    completed_ids = completed_instruction_ids(config.partial_predictions_path)
    failed_ids = _load_failed_ids(config.errors_path)
    _seed_everything(config.seed)
    loaded = loader(config)
    manifest = _initial_manifest(config, benchmark_hash, len(records), loaded)
    manifest["resumed_predictions"] = len(completed_ids)
    atomic_write_json(config.run_manifest_path, manifest)

    new_predictions = 0
    error_count = 0
    for record in records:
        if record.instruction_id in completed_ids:
            continue
        if not config.retry_failed and record.instruction_id in failed_ids:
            continue
        if config.stop_after is not None and new_predictions >= config.stop_after:
            break

        try:
            payload = _generate_one(record, loaded, config, project_root)
            append_jsonl(
                config.partial_predictions_path,
                payload,
                fsync=(new_predictions + 1) % config.save_every == 0,
            )
            completed_ids.add(record.instruction_id)
            new_predictions += 1
        except Exception as exc:
            append_jsonl(config.errors_path, _error_payload(record, config, exc))
            error_count += 1
            if error_count > config.max_errors:
                manifest.update(
                    {
                        "status": "failed",
                        "completed_predictions": len(completed_ids),
                        "errors": error_count,
                        "completed_at_utc": _utc_now(),
                    }
                )
                atomic_write_json(config.run_manifest_path, manifest)
                raise RuntimeError(
                    f"Inference stopped after {error_count} errors. See {config.errors_path}"
                ) from exc
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        if (new_predictions + error_count) % 10 == 0:
            manifest.update(
                {
                    "completed_predictions": len(completed_ids),
                    "errors": error_count,
                    "last_updated_at_utc": _utc_now(),
                }
            )
            atomic_write_json(config.run_manifest_path, manifest)
        if (
            (new_predictions + error_count) % config.persistent_sync_every == 0
        ):
            _sync_persistent_inference(config)

    complete = set(benchmark_ids) == completed_ids
    final_path: Path | None = None
    if complete:
        final_path = _finalize_predictions(config, benchmark_ids)

    manifest.update(
        {
            "status": "complete" if complete else "paused",
            "completed_predictions": len(completed_ids),
            "new_predictions_this_run": new_predictions,
            "errors": error_count,
            "completed_at_utc": _utc_now(),
            "predictions_path": final_path.as_posix() if final_path else None,
            "partial_predictions_path": config.partial_predictions_path.as_posix(),
            "errors_path": config.errors_path.as_posix(),
        }
    )
    atomic_write_json(config.run_manifest_path, manifest)
    _sync_persistent_inference(config)

    return BaselineInferenceResult(
        run_id=config.run_id,
        benchmark_records=len(records),
        completed_predictions=len(completed_ids),
        new_predictions=new_predictions,
        errors=error_count,
        complete=complete,
        predictions_path=final_path,
        partial_predictions_path=config.partial_predictions_path,
        manifest_path=config.run_manifest_path,
    )
