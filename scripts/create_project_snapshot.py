#!/usr/bin/env python3
"""Create a clean, upload-ready ZIP snapshot of the VisionAssist project.

Run from the project root:

    uv run python scripts/create_project_snapshot.py

The archive is written to:

    project_snapshots/visionassist_snapshot_<timestamp>.zip

The script includes source code, configuration, tests, documentation, and
lightweight reports while excluding datasets, model artifacts, environments,
caches, secrets, and other large/generated files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_EXCLUDED_DIR_NAMES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "mlruns",
    "model_outputs",
    "models",
    "node_modules",
    "outputs",
    "project_snapshots",
    "wandb",
}

DEFAULT_EXCLUDED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".coverage",
    "coverage.xml",
    "Thumbs.db",
    "desktop.ini",
}

DEFAULT_EXCLUDED_SUFFIXES = {
    ".7z",
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".iso",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".part",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tmp",
    ".zip",
}

# Raw/intermediate/processed datasets are deliberately excluded.
DEFAULT_EXCLUDED_PATH_PREFIXES = {
    PurePosixPath("data/downloads"),
    PurePosixPath("data/raw"),
    PurePosixPath("data/interim"),
    PurePosixPath("data/processed"),
    PurePosixPath("data/splits"),
}

# These lightweight generated artifacts are useful for understanding the
# current state and are therefore included when present.
INCLUDED_GENERATED_PREFIXES = {
    PurePosixPath("data/manifests"),
    PurePosixPath("reports/dataset_audit"),
}

TEXT_SECRET_MARKERS = (
    "api_key=",
    "apikey=",
    "secret_key=",
    "access_token=",
    "private_key=",
    "password=",
    "token=",
)


@dataclass(frozen=True)
class SnapshotFile:
    absolute_path: Path
    archive_path: PurePosixPath
    size_bytes: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a clean ZIP snapshot of the current project."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("project_snapshots"),
        help="Directory in which to write the ZIP.",
    )
    parser.add_argument(
        "--name",
        default="visionassist_snapshot",
        help="Base archive name.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=10.0,
        help="Skip individual files larger than this size. Default: 10 MB.",
    )
    parser.add_argument(
        "--include-git-diff",
        action="store_true",
        help="Include git status and patch files in snapshot metadata.",
    )
    return parser.parse_args()


def run_git(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to_any(
    relative_path: PurePosixPath,
    prefixes: Iterable[PurePosixPath],
) -> bool:
    return any(
        relative_path == prefix or prefix in relative_path.parents
        for prefix in prefixes
    )


def should_exclude_path(
    relative_path: PurePosixPath,
    absolute_path: Path,
    max_file_size_bytes: int,
) -> tuple[bool, str | None]:
    parts = set(relative_path.parts)

    if parts.intersection(DEFAULT_EXCLUDED_DIR_NAMES):
        return True, "excluded directory"

    if relative_path.name in DEFAULT_EXCLUDED_FILE_NAMES:
        return True, "excluded file"

    lower_name = relative_path.name.lower()
    if lower_name.startswith(".env"):
        return True, "environment/secrets file"

    if is_relative_to_any(relative_path, DEFAULT_EXCLUDED_PATH_PREFIXES):
        if not is_relative_to_any(relative_path, INCLUDED_GENERATED_PREFIXES):
            return True, "dataset/generated-data directory"

    # Path.suffix does not recognize compound extensions such as .tar.gz.
    if any(lower_name.endswith(suffix) for suffix in DEFAULT_EXCLUDED_SUFFIXES):
        return True, "excluded binary/archive suffix"

    try:
        size = absolute_path.stat().st_size
    except OSError:
        return True, "unreadable file"

    if size > max_file_size_bytes:
        return True, f"larger than {max_file_size_bytes} bytes"

    return False, None


def looks_like_secret_file(path: Path) -> bool:
    """Conservatively inspect small text files for obvious secret assignments."""
    try:
        if path.stat().st_size > 1_000_000:
            return False
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False

    return any(marker in text for marker in TEXT_SECRET_MARKERS)


def collect_files(
    project_root: Path,
    max_file_size_bytes: int,
) -> tuple[list[SnapshotFile], list[dict[str, str]]]:
    included: list[SnapshotFile] = []
    skipped: list[dict[str, str]] = []

    for directory, dir_names, file_names in os.walk(project_root):
        current_dir = Path(directory)

        # Prevent traversal into excluded directories.
        dir_names[:] = sorted(
            name
            for name in dir_names
            if name not in DEFAULT_EXCLUDED_DIR_NAMES
        )

        for file_name in sorted(file_names):
            absolute_path = current_dir / file_name
            relative_path = PurePosixPath(
                absolute_path.relative_to(project_root).as_posix()
            )

            excluded, reason = should_exclude_path(
                relative_path,
                absolute_path,
                max_file_size_bytes,
            )
            if excluded:
                skipped.append({"path": str(relative_path), "reason": reason or ""})
                continue

            # Never include the script's own output ZIP if output is inside root.
            if relative_path.parts and relative_path.parts[0] == "project_snapshots":
                skipped.append(
                    {"path": str(relative_path), "reason": "snapshot output directory"}
                )
                continue

            if looks_like_secret_file(absolute_path):
                skipped.append(
                    {
                        "path": str(relative_path),
                        "reason": "possible secret assignment detected",
                    }
                )
                continue

            included.append(
                SnapshotFile(
                    absolute_path=absolute_path,
                    archive_path=relative_path,
                    size_bytes=absolute_path.stat().st_size,
                    sha256=sha256_file(absolute_path),
                )
            )

    included.sort(key=lambda item: str(item.archive_path))
    skipped.sort(key=lambda item: item["path"])
    return included, skipped


def build_metadata(
    project_root: Path,
    files: list[SnapshotFile],
    skipped: list[dict[str, str]],
) -> dict[str, object]:
    git_commit = run_git(project_root, "rev-parse", "HEAD")
    git_branch = run_git(project_root, "branch", "--show-current")
    git_status = run_git(project_root, "status", "--short")

    return {
        "snapshot_schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_name": project_root.name,
        "python_version": sys.version,
        "platform": sys.platform,
        "git": {
            "branch": git_branch,
            "commit": git_commit,
            "status": git_status,
        },
        "summary": {
            "included_file_count": len(files),
            "included_size_bytes": sum(item.size_bytes for item in files),
            "skipped_file_count": len(skipped),
        },
        "files": [
            {
                "path": str(item.archive_path),
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in files
        ],
        "skipped": skipped,
    }


def create_snapshot(
    project_root: Path,
    output_dir: Path,
    base_name: str,
    max_file_size_mb: float,
    include_git_diff: bool,
) -> Path:
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    # A lightweight sanity check to avoid zipping the wrong directory.
    expected_markers = ("pyproject.toml", "src")
    missing = [name for name in expected_markers if not (project_root / name).exists()]
    if missing:
        raise RuntimeError(
            "This does not look like the project root. "
            f"Missing: {', '.join(missing)}"
        )

    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = output_dir / f"{base_name}_{timestamp}.zip"
    max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    files, skipped = collect_files(project_root, max_file_size_bytes)
    metadata = build_metadata(project_root, files, skipped)

    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=compression,
        compresslevel=9,
    ) as archive:
        for item in files:
            archive.write(item.absolute_path, arcname=str(item.archive_path))

        archive.writestr(
            "_snapshot/manifest.json",
            json.dumps(metadata, indent=2, ensure_ascii=False),
        )

        archive.writestr(
            "_snapshot/README.txt",
            (
                "VisionAssist project snapshot\n"
                "================================\n\n"
                "This archive contains project source code, configuration, tests,\n"
                "documentation, manifests, and lightweight reports.\n\n"
                "Excluded intentionally:\n"
                "- .env and probable secret-bearing files\n"
                "- virtual environments and caches\n"
                "- raw/intermediate/processed datasets\n"
                "- model weights and training outputs\n"
                "- Git internals\n"
                "- other archives and large binary files\n\n"
                "See _snapshot/manifest.json for the complete included/skipped list.\n"
            ),
        )

        if include_git_diff:
            status = run_git(project_root, "status", "--short")
            diff = run_git(project_root, "diff", "--binary")
            staged_diff = run_git(project_root, "diff", "--cached", "--binary")

            if status is not None:
                archive.writestr("_snapshot/git_status.txt", status + "\n")
            if diff is not None:
                archive.writestr("_snapshot/git_diff.patch", diff + "\n")
            if staged_diff is not None:
                archive.writestr(
                    "_snapshot/git_staged_diff.patch",
                    staged_diff + "\n",
                )

    return archive_path


def main() -> int:
    args = parse_args()

    try:
        archive_path = create_snapshot(
            project_root=args.project_root,
            output_dir=args.output_dir,
            base_name=args.name,
            max_file_size_mb=args.max_file_size_mb,
            include_git_diff=args.include_git_diff,
        )
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"Snapshot failed: {exc}", file=sys.stderr)
        return 1

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print("Project snapshot created successfully.")
    print(f"Archive: {archive_path}")
    print(f"Size: {size_mb:.2f} MB")
    print("Upload this ZIP in the chat to synchronize the project code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
