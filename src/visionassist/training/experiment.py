"""Experiment manifests and deterministic dataset subsets."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

    def __init__(
        self,
        dataset: VisionAssistJsonlDataset,
        indices: Sequence[int],
    ) -> None:
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
    task_quotas: dict[str, int] | None = None,
) -> VisionAssistJsonlDataset | DeterministicSubset:
    if task_quotas is not None:
        buckets: dict[str, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            buckets[dataset[index].task_family].append(index)
        selected: list[int] = []
        randomizer = random.Random(seed)
        for task, quota in sorted(task_quotas.items()):
            candidates = list(buckets.get(task, []))
            if len(candidates) < quota:
                raise ValueError(
                    f"Task quota exceeds available records for {task}: "
                    f"requested={quota}, available={len(candidates)}"
                )
            randomizer.shuffle(candidates)
            selected.extend(candidates[:quota])
        if len(selected) != len(set(selected)):
            raise RuntimeError("Task-quota selection produced duplicate indices.")
        return DeterministicSubset(dataset, sorted(selected))
    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return DeterministicSubset(dataset, sorted(indices[:limit]))


def training_datasets(
    config: Phase8TrainingConfig,
) -> tuple[
    VisionAssistJsonlDataset | DeterministicSubset,
    VisionAssistJsonlDataset | DeterministicSubset,
]:
    """Build deterministic train and validation selections from configuration."""

    train = VisionAssistJsonlDataset(config.data.train_path)
    validation = VisionAssistJsonlDataset(config.data.validation_path)
    return (
        subset_dataset(
            train,
            config.data.train_limit,
            config.data.subset_seed,
            config.data.train_task_quotas,
        ),
        subset_dataset(
            validation,
            config.data.validation_limit,
            config.data.subset_seed + 1,
        ),
    )


def selection_summary(dataset: Any) -> dict[str, Any]:
    """Describe and fingerprint the exact ordered instruction selection."""

    task_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    task_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_condition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    identifiers: list[str] = []
    image_ids: set[str] = set()
    for index in range(len(dataset)):
        record = dataset[index]
        identifiers.append(record.instruction_id)
        image_ids.add(record.image_id)
        task_counts[record.task_family] += 1
        category_counts[record.metadata.category] += 1
        condition_counts[record.metadata.condition] += 1
        task_category_counts[record.task_family][record.metadata.category] += 1
        task_condition_counts[record.task_family][record.metadata.condition] += 1
    digest = hashlib.sha256(("\n".join(identifiers) + "\n").encode()).hexdigest()
    return {
        "records": len(identifiers),
        "unique_instruction_ids": len(set(identifiers)),
        "unique_images": len(image_ids),
        "instruction_ids_sha256": digest,
        "task_families": dict(sorted(task_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "conditions": dict(sorted(condition_counts.items())),
        "task_categories": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(task_category_counts.items())
        },
        "task_conditions": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(task_condition_counts.items())
        },
    }


def write_dataset_selection_audit(
    config: Phase8TrainingConfig,
    output_path: Path | None = None,
) -> Path:
    """Write a CPU-safe audit of the exact configured train/validation subsets."""

    train, validation = training_datasets(config)
    path = output_path or config.output_dir / "dataset_selection_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "subset_seed": config.data.subset_seed,
        "train_task_quotas": config.data.train_task_quotas,
        "train_source_sha256": sha256_file(config.data.train_path),
        "validation_source_sha256": sha256_file(config.data.validation_path),
        "train": selection_summary(train),
        "validation": selection_summary(validation),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_experiment_files(
    config: Phase8TrainingConfig,
    project_root: Path,
    hardware: HardwareInfo,
    *,
    train_count: int,
    validation_count: int,
    train_selection: Any | None = None,
    validation_selection: Any | None = None,
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
    dataset_manifest: dict[str, Any] = {
        "train_path": str(config.data.train_path),
        "train_sha256": sha256_file(config.data.train_path),
        "train_records_used": train_count,
        "validation_path": str(config.data.validation_path),
        "validation_sha256": sha256_file(config.data.validation_path),
        "validation_records_used": validation_count,
        "subset_seed": config.data.subset_seed,
        "train_task_quotas": config.data.train_task_quotas,
    }
    if train_selection is not None:
        dataset_manifest["train_selection"] = selection_summary(train_selection)
    if validation_selection is not None:
        dataset_manifest["validation_selection"] = selection_summary(
            validation_selection
        )
    (output / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_value(project_root, "rev-parse", "HEAD"),
        "git_branch": git_value(project_root, "branch", "--show-current"),
        "git_status": git_value(project_root, "status", "--short"),
        "model_id": config.model_id,
        "initial_adapter_path": (
            str(config.initial_adapter_path)
            if config.initial_adapter_path is not None
            else None
        ),
        "initial_adapter_sha256": (
            sha256_file(config.initial_adapter_path / "adapter_model.safetensors")
            if config.initial_adapter_path is not None
            else None
        ),
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
