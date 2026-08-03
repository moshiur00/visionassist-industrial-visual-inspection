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
from visionassist.training.hardware import HardwareInfo, select_profile


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


def test_a100_safe_overfit_profile_is_memory_bounded() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "training"
        / "qwen25vl3b_qlora_overfit.yaml"
    )
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["attention_implementation"] == "sdpa"
    assert payload["data"]["max_sequence_length"] == 2048
    assert payload["data"]["image_min_pixels"] == 100352
    assert payload["data"]["image_max_pixels"] == 200704
    assert payload["lora"]["rank"] == 8
    assert payload["lora"]["alpha"] == 16
    assert payload["lora"]["target_suffixes"] == ["q_proj", "v_proj", "o_proj"]


def test_training_arguments_avoid_removed_save_safetensors_option() -> None:
    """Transformers 5.x always uses safe checkpoint serialization."""

    training_source = (
        Path(__file__).parents[1]
        / "src"
        / "visionassist"
        / "training"
        / "train.py"
    ).read_text(encoding="utf-8")

    assert "save_safetensors=" not in training_source


def test_hard_example_training_starts_from_promoted_adapter() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "training"
        / "qwen25vl3b_qlora_hard_examples.yaml"
    )
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = Phase8TrainingConfig.model_validate(payload)

    assert config.initial_adapter_path == Path(
        "outputs/training/qwen25vl3b_qlora_pilot_v1/final_adapter"
    )
    assert config.output_dir != config.initial_adapter_path.parent
    assert config.training.learning_rate == 5e-5
    assert config.data.train_path == Path(
        "outputs/training_data/phase11/hard_examples.jsonl"
    )
