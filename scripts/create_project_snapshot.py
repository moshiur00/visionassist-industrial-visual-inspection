#!/usr/bin/env python3
"""Create an upload-ready VisionAssist project snapshot.

Run from the project root:

    uv run python scripts/create_project_snapshot.py --include-git-diff

The snapshot includes source code, configuration, tests, documentation,
lightweight reports, benchmark artifacts, and lightweight baseline results.
Large datasets, model weights, environments, caches, and raw prediction files
remain excluded.

Useful options:

    --include-git-diff
    --include-output-samples
    --sample-lines 50
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


EXCLUDED_DIR_NAMES = {
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
    "wandb",
}

EXCLUDED_FILE_NAMES = {
    ".coverage",
    "coverage.xml",
    "Thumbs.db",
    "desktop.ini",
}

EXCLUDED_SUFFIXES = {
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

EXCLUDED_PATH_PREFIXES = {
    PurePosixPath("data/downloads"),
    PurePosixPath("data/raw"),
    PurePosixPath("data/interim"),
    PurePosixPath("data/processed"),
    PurePosixPath("data/splits"),
    PurePosixPath("checkpoints"),
    PurePosixPath("training_outputs"),
    PurePosixPath("trainer_output"),
    PurePosixPath("runs"),
    PurePosixPath("project_snapshots"),
}

# Generated artifacts that are small and important for project understanding.
INCLUDED_GENERATED_PREFIXES = {
    PurePosixPath("data/benchmarks"),
    PurePosixPath("data/manifests"),
    PurePosixPath("reports/baseline"),
    PurePosixPath("reports/dataset_audit"),
    PurePosixPath("reports/training_readiness"),
}

# Include only reproducibility-critical output files by default.
INCLUDED_OUTPUT_NAMES = {
    "metrics.json",
    "run_manifest.json",
    "per_task_metrics.csv",
    "per_category_metrics.csv",
}

RAW_OUTPUT_NAMES = {
    "predictions.jsonl",
    "predictions.partial.jsonl",
    "failures.jsonl",
    "parsing_errors.jsonl",
    "inference_errors.jsonl",
}

SECRET_MARKERS = (
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
        description="Create a clean ZIP snapshot of VisionAssist."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("project_snapshots"),
    )
    parser.add_argument(
        "--name",
        default="visionassist_snapshot",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--include-git-diff",
        action="store_true",
    )
    parser.add_argument(
        "--include-output-samples",
        action="store_true",
        help=(
            "Include bounded samples from raw baseline JSONL outputs under "
            "_snapshot/output_samples/."
        ),
    )
    parser.add_argument(
        "--sample-lines",
        type=int,
        default=50,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to_any(
    path: PurePosixPath,
    prefixes: Iterable[PurePosixPath],
) -> bool:
    return any(path == prefix or prefix in path.parents for prefix in prefixes)


def is_allowed_output(relative_path: PurePosixPath) -> bool:
    if not relative_path.parts or relative_path.parts[0] != "outputs":
        return True
    return relative_path.name in INCLUDED_OUTPUT_NAMES


def looks_like_secret_file(path: Path) -> bool:
    lower_name = path.name.lower()
    if lower_name == ".env" or lower_name.startswith(".env."):
        return True

    try:
        if path.stat().st_size > 1_000_000:
            return False
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False

    return any(marker in text for marker in SECRET_MARKERS)


def should_exclude(
    relative_path: PurePosixPath,
    absolute_path: Path,
    max_file_size_bytes: int,
) -> tuple[bool, str]:
    if set(relative_path.parts).intersection(EXCLUDED_DIR_NAMES):
        return True, "excluded directory"

    if relative_path.name in EXCLUDED_FILE_NAMES:
        return True, "excluded file"

    if relative_path.parts and relative_path.parts[0] == "outputs":
        if not is_allowed_output(relative_path):
            return True, "raw or nonessential output"

    if is_relative_to_any(relative_path, EXCLUDED_PATH_PREFIXES):
        if not is_relative_to_any(relative_path, INCLUDED_GENERATED_PREFIXES):
            return True, "large generated-data path"

    lower_name = relative_path.name.lower()
    if any(lower_name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return True, "excluded archive/binary suffix"

    try:
        size = absolute_path.stat().st_size
    except OSError:
        return True, "unreadable"

    if size > max_file_size_bytes:
        return True, "file exceeds size limit"

    if looks_like_secret_file(absolute_path):
        return True, "possible secret-bearing file"

    return False, ""


def collect_files(
    project_root: Path,
    max_file_size_bytes: int,
) -> tuple[list[SnapshotFile], list[dict[str, str]]]:
    included: list[SnapshotFile] = []
    skipped: list[dict[str, str]] = []

    for directory, dir_names, file_names in os.walk(project_root):
        current = Path(directory)
        dir_names[:] = sorted(
            name for name in dir_names if name not in EXCLUDED_DIR_NAMES
        )

        for file_name in sorted(file_names):
            absolute = current / file_name
            relative = PurePosixPath(absolute.relative_to(project_root).as_posix())
            excluded, reason = should_exclude(
                relative,
                absolute,
                max_file_size_bytes,
            )
            if excluded:
                skipped.append({"path": str(relative), "reason": reason})
                continue

            included.append(
                SnapshotFile(
                    absolute_path=absolute,
                    archive_path=relative,
                    size_bytes=absolute.stat().st_size,
                    sha256=sha256_file(absolute),
                )
            )

    included.sort(key=lambda item: str(item.archive_path))
    skipped.sort(key=lambda item: item["path"])
    return included, skipped


def read_jsonl_sample(path: Path, line_limit: int) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index >= line_limit:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines) + ("\n" if lines else "")


def add_output_samples(
    archive: zipfile.ZipFile,
    project_root: Path,
    line_limit: int,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []

    output_root = project_root / "outputs"
    if not output_root.exists():
        return samples

    for path in sorted(output_root.rglob("*.jsonl")):
        if path.name not in RAW_OUTPUT_NAMES:
            continue

        relative = path.relative_to(project_root)
        sample_text = read_jsonl_sample(path, line_limit)
        archive_path = (
            PurePosixPath("_snapshot/output_samples")
            / PurePosixPath(relative.as_posix())
        )
        archive.writestr(str(archive_path), sample_text)
        samples.append(
            {
                "source": relative.as_posix(),
                "sample_path": str(archive_path),
                "sample_lines": len(sample_text.splitlines()),
                "source_size_bytes": path.stat().st_size,
                "source_sha256": sha256_file(path),
            }
        )

    return samples


def build_metadata(
    project_root: Path,
    files: list[SnapshotFile],
    skipped: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "snapshot_schema_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_name": project_root.name,
        "python_version": sys.version,
        "platform": sys.platform,
        "git": {
            "branch": run_git(project_root, "branch", "--show-current"),
            "commit": run_git(project_root, "rev-parse", "HEAD"),
            "status": run_git(project_root, "status", "--short"),
            "latest_commit": run_git(
                project_root,
                "log",
                "-1",
                "--pretty=format:%H%n%an%n%ad%n%s",
                "--date=iso-strict",
            ),
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


def create_snapshot(args: argparse.Namespace) -> Path:
    root = args.project_root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src").is_dir():
        raise RuntimeError("Run this script from the VisionAssist project root.")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = output_dir / f"{args.name}_{timestamp}.zip"

    files, skipped = collect_files(
        root,
        int(args.max_file_size_mb * 1024 * 1024),
    )
    metadata = build_metadata(root, files, skipped)

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for item in files:
            archive.write(item.absolute_path, str(item.archive_path))

        samples: list[dict[str, object]] = []
        if args.include_output_samples:
            samples = add_output_samples(
                archive,
                root,
                max(1, args.sample_lines),
            )
        metadata["output_samples"] = samples

        if args.include_git_diff:
            for name, git_args in (
                ("git_status.txt", ("status", "--short")),
                ("git_diff.patch", ("diff", "--binary")),
                ("git_staged_diff.patch", ("diff", "--cached", "--binary")),
            ):
                content = run_git(root, *git_args)
                if content is not None:
                    archive.writestr(f"_snapshot/{name}", content + "\n")

        archive.writestr(
            "_snapshot/manifest.json",
            json.dumps(metadata, indent=2, ensure_ascii=False),
        )
        archive.writestr(
            "_snapshot/README.txt",
            (
                "VisionAssist project snapshot\n"
                "=============================\n\n"
                "Includes code, configs, tests, docs, benchmark artifacts, "
                "reports, and lightweight baseline metrics/manifests.\n"
                "Raw data, model weights, checkpoints, and complete prediction "
                "JSONL files are excluded.\n"
            ),
        )

    return archive_path


def main() -> int:
    args = parse_args()
    try:
        archive_path = create_snapshot(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Snapshot failed: {exc}", file=sys.stderr)
        return 1

    print("Project snapshot created successfully.")
    print(f"Archive: {archive_path}")
    print(f"Size: {archive_path.stat().st_size / (1024 * 1024):.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
