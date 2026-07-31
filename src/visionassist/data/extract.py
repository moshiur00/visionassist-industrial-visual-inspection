"""Safe archive extraction helpers."""

from __future__ import annotations

import tarfile
from pathlib import Path


def _is_within_directory(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def safe_extract_tar(archive_path: Path, destination: Path, *, force: bool = False) -> Path:
    """Extract a tar archive while rejecting path traversal and link entries."""

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            target = destination / member.name
            if not _is_within_directory(destination, target):
                raise ValueError(f"Unsafe archive member path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name}")
            if target.exists() and not force:
                continue
            archive.extract(member, destination, filter="data")

    # Official archive normally contains a top-level `VisA` directory.
    candidate = destination / "VisA"
    return candidate if candidate.is_dir() else destination
