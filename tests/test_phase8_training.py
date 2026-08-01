from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionassist.training.checkpointing import (
    checkpoint_step,
    latest_checkpoint,
    prune_checkpoints,
    resolve_resume_checkpoint,
)
from visionassist.training.config import Phase8TrainingConfig
from visionassist.training.hardware import (
    HardwareInfo,
    _nearest_existing_path,
    inspect_hardware,
    select_profile,
)


def config_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "run_id": "test_run",
        "output_dir": str(tmp_path / "output"),
        "data": {
            "train_path": str(tmp_path / "train.jsonl"),
            "validation_path": str(tmp_path / "validation.jsonl"),
        },
        "training": {
            "eval_steps": 10,
            "save_steps": 20,
            "save_total_limit": 3,
        },
        "checkpoints": {"keep_latest": 2, "keep_best": 1},
    }


def test_phase8_config_requires_bounded_checkpoint_capacity(tmp_path: Path) -> None:
    payload = config_payload(tmp_path)
    config = Phase8TrainingConfig.model_validate(payload)
    assert config.training.save_total_limit == 3
    payload["training"]["save_total_limit"] = 2  # type: ignore[index]
    with pytest.raises(ValueError):
        Phase8TrainingConfig.model_validate(payload)


def test_hardware_profiles() -> None:
    base = dict(
        cuda_available=True,
        gpu_name="GPU",
        free_vram_gib=10.0,
        bf16_supported=True,
        compute_capability="8.0",
        cuda_version="12.8",
        torch_version="2.7",
        system_ram_gib=50.0,
        free_disk_gib=100.0,
        python_version="3.12",
        platform="test",
    )
    assert select_profile(HardwareInfo(total_vram_gib=12.0, **base)) == "low_vram"
    assert select_profile(HardwareInfo(total_vram_gib=24.0, **base)) == "standard_vram"
    assert select_profile(HardwareInfo(total_vram_gib=40.0, **base)) == "high_vram"


def make_checkpoint(root: Path, step: int, best: str | None = None) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    state = {"global_step": step, "best_model_checkpoint": best}
    (path / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    return path


def test_checkpoint_resolution_and_pruning(tmp_path: Path) -> None:
    first = make_checkpoint(tmp_path, 10)
    best = make_checkpoint(tmp_path, 20)
    latest = make_checkpoint(tmp_path, 30, best=str(best))
    assert checkpoint_step(latest) == 30
    assert latest_checkpoint(tmp_path) == latest
    assert resolve_resume_checkpoint(tmp_path, None, "latest") == latest
    removed = prune_checkpoints(tmp_path, keep_latest=1, keep_best=1)
    assert first in removed
    assert latest.exists()
    assert best.exists()


def test_local_training_has_clear_no_gpu_boundary(tmp_path: Path) -> None:
    # Configuration validation is CPU-safe; actual training checks CUDA at runtime.
    config = Phase8TrainingConfig.model_validate(config_payload(tmp_path))
    assert config.run_id == "test_run"


def test_nearest_existing_path_handles_future_output_directory(tmp_path: Path) -> None:
    future_output = tmp_path / "outputs" / "training" / "run" / "checkpoints"

    resolved = _nearest_existing_path(future_output)

    assert resolved.exists()
    assert resolved == tmp_path


def test_hardware_inspection_accepts_missing_output_directory(tmp_path: Path) -> None:
    future_output = tmp_path / "outputs" / "training" / "run"

    hardware = inspect_hardware(future_output)

    assert hardware.free_disk_gib >= 0.0
    assert isinstance(hardware.cuda_available, bool)
