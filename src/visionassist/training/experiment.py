"""Experiment manifests and deterministic dataset subsets."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from visionassist.training.config import Phase8TrainingConfig
from visionassist.training.dataset import VisionAssistJsonlDataset
from visionassist.training.hardware import HardwareInfo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


class DeterministicSubset:
    """Stable subset wrapper that preserves InstructionRecord objects."""

    def __init__(self, dataset: VisionAssistJsonlDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]


def subset_dataset(
    dataset: VisionAssistJsonlDataset,
    limit: int | None,
    seed: int,
) -> VisionAssistJsonlDataset | DeterministicSubset:
    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return DeterministicSubset(dataset, sorted(indices[:limit]))


def write_experiment_files(
    config: Phase8TrainingConfig,
    project_root: Path,
    hardware: HardwareInfo,
    *,
    train_count: int,
    validation_count: int,
    trainable_parameters: int | None = None,
    total_parameters: int | None = None,
    target_modules: list[str] | None = None,
) -> None:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    (output / "environment.json").write_text(
        json.dumps(hardware.to_dict(), indent=2), encoding="utf-8"
    )
    dataset_manifest = {
        "train_path": str(config.data.train_path),
        "train_sha256": sha256_file(config.data.train_path),
        "train_records_used": train_count,
        "validation_path": str(config.data.validation_path),
        "validation_sha256": sha256_file(config.data.validation_path),
        "validation_records_used": validation_count,
        "subset_seed": config.data.subset_seed,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(project_root, "rev-parse", "HEAD"),
        "git_branch": git_value(project_root, "branch", "--show-current"),
        "git_status": git_value(project_root, "status", "--short"),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "processor_revision": config.processor_revision,
        "seed": config.seed,
        "status": "initialized",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if trainable_parameters is not None and total_parameters is not None:
        parameter_report = {
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_percent": 100.0 * trainable_parameters / total_parameters,
            "target_modules": target_modules or [],
        }
        (output / "trainable_parameters.json").write_text(
            json.dumps(parameter_report, indent=2), encoding="utf-8"
        )
