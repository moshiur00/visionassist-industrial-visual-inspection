"""Phase 8 one-batch validation and resumable QLoRA training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visionassist.training.checkpointing import (
    make_persistent_checkpoint_callback,
    resolve_resume_checkpoint,
)
from visionassist.training.collator import QwenAssistantOnlyCollator
from visionassist.training.config import Phase8TrainingConfig
from visionassist.training.experiment import training_datasets, write_experiment_files
from visionassist.training.hardware import inspect_hardware, select_profile
from visionassist.training.modeling import build_qlora_model


@dataclass(frozen=True)
class TrainingRunResult:
    run_id: str
    output_dir: Path
    resumed_from: Path | None
    global_step: int
    best_checkpoint: str | None
    final_adapter: Path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _datasets(config: Phase8TrainingConfig) -> tuple[Any, Any]:
    return training_datasets(config)


def _training_arguments(config: Phase8TrainingConfig, hardware: Any) -> Any:
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(config.output_dir),
        max_steps=config.training.max_steps,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=config.training.dataloader_num_workers,
        optim=config.training.optim,
        lr_scheduler_type=config.training.lr_scheduler_type,
        report_to=config.training.report_to,
        remove_unused_columns=False,
        bf16=hardware.bf16_supported,
        fp16=not hardware.bf16_supported,
        tf32=config.training.tf32,
        seed=config.seed,
        data_seed=config.data.subset_seed,
    )


def validate_one_batch(
    config: Phase8TrainingConfig,
    *,
    project_root: Path,
) -> dict[str, object]:
    """Load the model and prove one collated batch has a finite loss."""

    hardware = inspect_hardware(config.output_dir)

    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    build = build_qlora_model(config, hardware)
    train, validation = _datasets(config)
    write_experiment_files(
        config,
        project_root,
        hardware,
        train_count=len(train),
        validation_count=len(validation),
        train_selection=train,
        validation_selection=validation,
        trainable_parameters=build.trainable_parameters,
        total_parameters=build.total_parameters,
        target_modules=build.target_modules,
    )
    collator = QwenAssistantOnlyCollator(
        build.processor,
        project_root,
        max_length=config.data.max_sequence_length,
    )
    count = min(config.training.per_device_train_batch_size, len(train))
    batch = collator([train[index] for index in range(count)])
    device = next(build.model.parameters()).device
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    build.model.train()
    output = build.model(**batch)
    loss_tensor = output.loss
    loss = float(loss_tensor.detach().cpu())
    if not (loss >= 0.0 and loss < float("inf")):
        raise RuntimeError(f"Non-finite smoke-test loss: {loss}")

    # Validate the real training path, not only inference-like forward memory.
    loss_tensor.backward()
    gradient_tensors = [
        parameter.grad
        for parameter in build.model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradient_tensors:
        raise RuntimeError("No LoRA gradients were produced by the smoke test.")

    finite_gradients = all(bool(gradient.isfinite().all()) for gradient in gradient_tensors)
    nonzero_gradients = any(bool(gradient.detach().abs().sum() > 0) for gradient in gradient_tensors)
    if not finite_gradients:
        raise RuntimeError("Non-finite LoRA gradients were produced.")
    if not nonzero_gradients:
        raise RuntimeError("All LoRA gradients are zero.")

    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0
    if torch is not None and torch.cuda.is_available():
        peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024**3)

    build.model.zero_grad(set_to_none=True)
    result = {
        "run_id": config.run_id,
        "hardware_profile": select_profile(hardware),
        "loss": loss,
        "batch_shape": list(batch["input_ids"].shape),
        "sequence_length": int(batch["input_ids"].shape[-1]),
        "image_min_pixels": config.data.image_min_pixels,
        "image_max_pixels": config.data.image_max_pixels,
        "trainable_parameters": build.trainable_parameters,
        "total_parameters": build.total_parameters,
        "gradient_tensor_count": len(gradient_tensors),
        "finite_gradients": finite_gradients,
        "nonzero_gradients": nonzero_gradients,
        "peak_allocated_vram_gib": round(peak_allocated_gib, 3),
        "peak_reserved_vram_gib": round(peak_reserved_gib, 3),
        "forward_passed": True,
        "backward_passed": True,
        "passed": True,
    }
    path = config.output_dir / "one_batch_smoke_test.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_qlora_training(
    config: Phase8TrainingConfig,
    *,
    project_root: Path,
    resume_override: str | Path | None = None,
) -> TrainingRunResult:
    """Run resumable QLoRA training with bounded local and persistent checkpoints."""

    _seed_everything(config.seed)
    hardware = inspect_hardware(config.output_dir)
    if not hardware.cuda_available:
        raise RuntimeError(
            "No CUDA GPU is available. Run QLoRA in Google Colab Pro. "
            "Local CPU execution is supported only for tests/configuration."
        )
    train, validation = _datasets(config)
    build = build_qlora_model(config, hardware)
    write_experiment_files(
        config,
        project_root,
        hardware,
        train_count=len(train),
        validation_count=len(validation),
        train_selection=train,
        validation_selection=validation,
        trainable_parameters=build.trainable_parameters,
        total_parameters=build.total_parameters,
        target_modules=build.target_modules,
    )

    collator = QwenAssistantOnlyCollator(
        build.processor,
        project_root,
        max_length=config.data.max_sequence_length,
    )
    args = _training_arguments(config, hardware)
    callbacks: list[Any] = []
    persistent_root = config.checkpoints.persistent_output_dir
    if persistent_root is not None and config.checkpoints.sync_every_save:
        callbacks.append(
            make_persistent_checkpoint_callback(
                persistent_root,
                keep_latest=config.checkpoints.keep_latest,
                keep_best=config.checkpoints.keep_best,
            )
        )

    from transformers import Trainer

    trainer = Trainer(
        model=build.model,
        args=args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=collator,
        callbacks=callbacks,
    )
    policy = resume_override if resume_override is not None else config.checkpoints.resume
    if isinstance(policy, str) and policy not in {"none", "latest", "best"}:
        policy = Path(policy)
    resume = resolve_resume_checkpoint(
        config.output_dir,
        persistent_root,
        policy,
    )
    train_result = trainer.train(
        resume_from_checkpoint=str(resume) if resume is not None else None
    )
    trainer.save_state()
    final_adapter = config.output_dir / "final_adapter"
    trainer.model.save_pretrained(final_adapter, safe_serialization=True)
    build.processor.save_pretrained(final_adapter)

    manifest_path = config.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "resumed_from": str(resume) if resume is not None else None,
            "global_step": trainer.state.global_step,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "training_loss": train_result.training_loss,
            "final_adapter": str(final_adapter),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return TrainingRunResult(
        run_id=config.run_id,
        output_dir=config.output_dir,
        resumed_from=resume,
        global_step=trainer.state.global_step,
        best_checkpoint=trainer.state.best_model_checkpoint,
        final_adapter=final_adapter,
    )
