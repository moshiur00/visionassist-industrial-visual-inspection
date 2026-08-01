"""Checkpoint discovery, persistence, and bounded retention."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def checkpoint_step(path: Path) -> int:
    """Return the global step encoded in a Trainer checkpoint directory."""

    match = CHECKPOINT_PATTERN.search(path.name)
    return int(match.group(1)) if match else -1


def list_checkpoints(root: Path) -> list[Path]:
    """List valid checkpoint directories in ascending step order."""

    if not root.exists():
        return []
    paths = [path for path in root.glob("checkpoint-*") if path.is_dir()]
    return sorted(paths, key=checkpoint_step)


def latest_checkpoint(root: Path) -> Path | None:
    """Return the newest checkpoint."""

    checkpoints = list_checkpoints(root)
    return checkpoints[-1] if checkpoints else None


def best_checkpoint(root: Path) -> Path | None:
    """Read the best checkpoint from the newest Trainer state when available."""

    newest = latest_checkpoint(root)
    state_paths = []
    if newest is not None:
        state_paths.append(newest / "trainer_state.json")
    state_paths.append(root / "trainer_state.json")
    for path in state_paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("best_model_checkpoint")
        if isinstance(value, str):
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = root / candidate.name
            if candidate.exists():
                return candidate
    return None


def restore_checkpoint(source: Path, local_root: Path) -> Path:
    """Copy a persistent checkpoint back to fast local storage."""

    destination = local_root / source.name
    if destination.exists():
        return destination
    local_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def resolve_resume_checkpoint(
    local_root: Path,
    persistent_root: Path | None,
    policy: str | Path,
) -> Path | None:
    """Resolve none/latest/best/explicit resume policies."""

    if isinstance(policy, Path):
        if not policy.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {policy}")
        return policy
    if policy == "none":
        return None

    resolver = best_checkpoint if policy == "best" else latest_checkpoint
    local = resolver(local_root)
    persistent = resolver(persistent_root) if persistent_root is not None else None
    candidates = [path for path in (local, persistent) if path is not None]
    if not candidates:
        return None
    chosen = max(candidates, key=checkpoint_step)
    if persistent_root is not None and persistent_root in chosen.parents:
        return restore_checkpoint(chosen, local_root)
    return chosen


def prune_checkpoints(
    root: Path,
    *,
    keep_latest: int,
    keep_best: int,
) -> list[Path]:
    """Keep a bounded union of newest and best checkpoints."""

    checkpoints = list_checkpoints(root)
    if not checkpoints:
        return []
    keep = set(checkpoints[-keep_latest:])
    best = best_checkpoint(root)
    if best is not None and keep_best > 0:
        keep.add(best)
    removed: list[Path] = []
    for checkpoint in checkpoints:
        if checkpoint not in keep:
            shutil.rmtree(checkpoint)
            removed.append(checkpoint)
    return removed


def sync_checkpoint(
    checkpoint: Path,
    persistent_root: Path,
    *,
    keep_latest: int,
    keep_best: int,
) -> Path:
    """Atomically copy one checkpoint to persistent storage and prune old ones."""

    persistent_root.mkdir(parents=True, exist_ok=True)
    destination = persistent_root / checkpoint.name
    temporary = persistent_root / f".{checkpoint.name}.copying"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(checkpoint, temporary)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    prune_checkpoints(
        persistent_root,
        keep_latest=keep_latest,
        keep_best=keep_best,
    )
    return destination


def make_persistent_checkpoint_callback(
    persistent_root: Path,
    *,
    keep_latest: int,
    keep_best: int,
) -> Any:
    """Create a real Transformers callback without importing Transformers locally."""

    try:
        from transformers import TrainerCallback
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("Install the training extra to use checkpoint syncing.") from exc

    class _PersistentCheckpointCallback(TrainerCallback):
        def on_save(
            self,
            args: Any,
            state: Any,
            control: Any,
            **_: Any,
        ) -> Any:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint.is_dir():
                sync_checkpoint(
                    checkpoint,
                    persistent_root,
                    keep_latest=keep_latest,
                    keep_best=keep_best,
                )
            return control

    return _PersistentCheckpointCallback()
