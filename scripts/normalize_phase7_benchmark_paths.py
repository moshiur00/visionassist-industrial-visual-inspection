#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(value: str) -> str:
    parts = PurePosixPath(value.replace("\\", "/")).parts
    marker = ("data", "raw", "visa")
    lowered = tuple(part.lower() for part in parts)

    for index in range(len(parts) - len(marker) + 1):
        if lowered[index:index + len(marker)] == marker:
            return PurePosixPath(*parts[index:]).as_posix()

    raise ValueError(f"Cannot find data/raw/visa in path: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/benchmarks/visa_baseline_v1"),
    )
    args = parser.parse_args()

    directory = args.benchmark_dir.resolve()
    benchmark = directory / "benchmark.jsonl"
    manifest_path = directory / "benchmark_manifest.json"
    sha_path = directory / "benchmark_sha256.txt"

    records = []
    changed_records = 0

    with benchmark.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            changed = False
            for message in record.get("messages", []):
                if message.get("role") != "user":
                    continue
                for item in message.get("content", []):
                    if item.get("type") != "image":
                        continue
                    old = str(item["image"])
                    new = normalize_path(old)
                    if new != old:
                        item["image"] = new
                        changed = True
            changed_records += int(changed)
            records.append(record)

    temporary = benchmark.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(benchmark)

    digest = sha256_file(benchmark)
    sha_path.write_text(digest + "\n", encoding="utf-8")

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["benchmark_sha256"] = digest
        for key in ("sha256", "benchmark_hash"):
            if key in manifest:
                manifest[key] = digest
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    print("Benchmark paths normalized.")
    print(f"Records: {len(records)}")
    print(f"Changed records: {changed_records}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
