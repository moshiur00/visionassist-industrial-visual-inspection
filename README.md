# VisionAssist

VisionAssist is a reproducible multimodal fine-tuning project for industrial visual inspection. Phase 1 acquires and validates the official VisA dataset before any split or instruction generation occurs.

## Phase 1 deliverables

- resumable official-source download;
- safe TAR extraction with path-traversal protection;
- archive SHA-256 fingerprint and acquisition receipt;
- structural verification against the official 10,821-image release counts;
- per-image SHA-256 manifest and image-readability checks;
- CSV annotation inventory and dataset statistics;
- CC BY 4.0 license report;
- generated dataset card.

## Requirements

- Python 3.12
- `uv`
- sufficient disk space for the archive, extracted dataset, and temporary extraction copy

## Setup on Windows PowerShell

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv sync --extra dev
```

## Run all of Phase 1

```powershell
uv run visionassist phase1-visa --config configs/data/visa.yaml
```

Interrupted downloads are retained as `data/downloads/VisA_20220922.tar.part`. Run the same command again to resume. To start from scratch:

```powershell
uv run visionassist phase1-visa --force-download --force-extract
```

## Generated outputs

```text
data/downloads/VisA_20220922.tar
data/raw/visa/
data/manifests/visa_archive_receipt.json
data/manifests/visa_raw_manifest.jsonl
reports/dataset_audit/visa_audit_summary.json
reports/dataset_audit/visa_csv_inventory.json
reports/dataset_audit/VISA_LICENSE_REPORT.md
DATASET_CARD.md
```

The official registry does not publish a SHA-256 digest in its dataset entry. The pipeline therefore records the SHA-256 of the acquired official archive for repeat-run verification and additionally validates image counts, category coverage, image readability, masks, and source annotations.

## Re-audit an existing extraction

```powershell
uv run visionassist audit-visa --config configs/data/visa.yaml
```

## Quality checks

```powershell
uv run pytest
uv run ruff check .
uv run mypy src
```

Raw data, archives, manifests, and generated reports are excluded from Git.

## Phase 2: annotation and mask parsing

Phase 2 treats the official `image_anno.csv` files as the source of truth. It:

- discovers all 12 category annotation files;
- supports the official columns `object, split, label, image, mask`;
- resolves and validates every referenced image and anomaly mask;
- verifies image/mask dimension equality;
- rejects missing, unreadable, empty, or non-binary anomaly masks;
- prevents duplicate image IDs and repeated image rows;
- writes one versioned canonical JSONL record per source image;
- produces machine-readable validation and error reports.

Run Phase 1 first, then:

```powershell
uv run visionassist phase2-visa --config configs/data/visa.yaml
```

Generated outputs:

```text
data/interim/visa_canonical.jsonl
reports/dataset_audit/visa_phase2_validation.json
reports/dataset_audit/visa_phase2_errors.jsonl
```

`strict_phase2: true` makes the command fail if any row is invalid or the official
counts do not match. This prevents later phases from silently training on an
incomplete dataset.

The official CSV provides image-level condition and mask paths, but it does not
provide a universal fine-grained defect taxonomy. Therefore `defect_type` remains
`null` unless a compatible source column is present. Phase 3 will derive spatial
and geometric labels from masks without inventing unsupported defect names.
