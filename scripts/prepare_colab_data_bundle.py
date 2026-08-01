#!/usr/bin/env python3
"""Create a reusable VisionAssist prepared-data archive for Google Drive.

Run from the project root:

    uv run python scripts/prepare_colab_data_bundle.py

The archive includes the prepared dataset and benchmark artifacts needed for
Phase 7C and later training phases, while excluding the original download
archive, environments, caches, model outputs, and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


DEFAULT_PATHS = (
    Path("data/raw/visa"),
    Path("data/interim"),
    Path("data/processed"),
    Path("data/splits"),
    Path("data/benchmarks"),
    Path("data/manifests"),
    Path("reports/dataset_audit"),
    Path("reports/training_readiness"),
)


def count_files(path: Path) -> int:
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "project_snapshots/visionassist_prepared_data.tar.gz"
        ),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = (root / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    existing: list[Path] = []
    missing: list[str] = []

    for relative in DEFAULT_PATHS:
        source = root / relative
        if source.exists():
            existing.append(relative)
        else:
            missing.append(str(relative))

    required = (
        Path("data/raw/visa"),
        Path("data/processed/visa_instructions"),
        Path("data/benchmarks/visa_baseline_v1"),
    )
    missing_required = [str(path) for path in required if not (root / path).exists()]
    if missing_required:
        raise FileNotFoundError(
            "Required prepared-data paths are missing: "
            + ", ".join(missing_required)
        )

    manifest = {
        "schema_version": "1.0",
        "included_paths": [str(path).replace("\\", "/") for path in existing],
        "missing_optional_paths": missing,
        "file_counts": {
            str(path).replace("\\", "/"): count_files(root / path)
            for path in existing
        },
    }

    with tarfile.open(output, "w:gz") as archive:
        for relative in existing:
            source = root / relative
            archive.add(source, arcname=relative.as_posix())

        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo("_prepared_data_manifest.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, fileobj=__import__("io").BytesIO(manifest_bytes))

    size_gib = output.stat().st_size / (1024**3)
    print("Prepared-data archive created.")
    print(f"Archive: {output}")
    print(f"Size: {size_gib:.2f} GiB")
    print(
        "Upload it to Google Drive at "
        "MyDrive/visionassist/data/visionassist_prepared_data.tar.gz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
