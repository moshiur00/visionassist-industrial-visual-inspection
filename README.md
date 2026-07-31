# VisionAssist

VisionAssist is a reproducible multimodal fine-tuning project for industrial visual inspection and defect explanation.

The project converts the official VisA industrial anomaly dataset into a validated multimodal instruction dataset suitable for vision-language model fine-tuning. The current pipeline covers dataset acquisition, annotation parsing, mask-derived feature generation, leakage-safe splitting, and grounded instruction generation.

## Project objective

Given an industrial inspection image and a natural-language request, the final system should be able to:

- classify the item as normal or defective;
- identify the product category;
- describe the visible defect;
- localize the anomaly coarsely;
- explain the visible evidence;
- return a structured quality-control report;
- produce a concise technician note;
- abstain from unsupported root-cause or safety claims.

The project does **not** claim to determine mechanical root cause, hidden internal damage, safety impact, or authoritative repair instructions from an image alone.

---

## Current status

The data-engineering pipeline is complete through **Phase 5**.

| Phase                            | Status   | Main output                               |
| -------------------------------- | -------- | ----------------------------------------- |
| Phase 1 — Dataset acquisition    | Complete | Verified VisA raw dataset                 |
| Phase 2 — Dataset parsing        | Complete | Canonical metadata JSONL                  |
| Phase 3 — Feature derivation     | Complete | Spatially enriched records                |
| Phase 4 — Data splitting         | Complete | Leakage-safe train/validation/test splits |
| Phase 5 — Instruction generation | Complete | 52,863 multimodal instruction records     |
| Phase 6 — Baseline evaluation    | Next     | Untouched-model benchmark                 |

Final dataset counts:

| Split      | Source images | Instructions |
| ---------- | ------------: | -----------: |
| Train      |         7,575 |       37,005 |
| Validation |         1,622 |        7,926 |
| Test       |         1,624 |        7,932 |
| **Total**  |    **10,821** |   **52,863** |

Final validation status:

```text
Errors: 0
Warnings: 0
```

---

## Dataset

Primary dataset:

```text
VisA
```

Expected release statistics:

| Item               |  Count |
| ------------------ | -----: |
| Total images       | 10,821 |
| Normal images      |  9,621 |
| Anomalous images   |  1,200 |
| Product categories |     12 |

Categories:

```text
candle
capsules
cashew
chewinggum
fryum
macaroni1
macaroni2
pcb1
pcb2
pcb3
pcb4
pipe_fryum
```

The configured dataset license is CC BY 4.0. The generated license report and dataset card are local audit artifacts.

---

## Requirements

- Python 3.12
- `uv`
- sufficient disk space for the VisA archive, extracted dataset, and generated artifacts

Optional training dependencies are defined separately and include PyTorch, Transformers, Datasets, Accelerate, PEFT, TRL, and bitsandbytes where supported.

---

## Setup on Windows PowerShell

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv sync --extra dev
```

Keep both of these files under version control:

```text
pyproject.toml
uv.lock
```

---

## Run the full data pipeline

### Phase 1 — Dataset acquisition and audit

```powershell
uv run visionassist phase1-visa --config configs/data/visa.yaml
```

This phase performs:

- resumable official-source download;
- safe TAR extraction with path-traversal protection;
- archive SHA-256 fingerprinting;
- acquisition receipt generation;
- per-image SHA-256 calculation;
- image readability validation;
- category and condition statistics;
- annotation CSV inventory;
- license report generation;
- dataset card generation.

Interrupted downloads remain as:

```text
data/downloads/VisA_20220922.tar.part
```

Run the same command again to resume.

To force a clean download and extraction:

```powershell
uv run visionassist phase1-visa `
    --config configs/data/visa.yaml `
    --force-download `
    --force-extract
```

Re-audit an existing extraction:

```powershell
uv run visionassist audit-visa --config configs/data/visa.yaml
```

### Phase 2 — Annotation and mask parsing

```powershell
uv run visionassist phase2-visa --config configs/data/visa.yaml
```

Phase 2:

- discovers all 12 `image_anno.csv` files;
- supports the actual per-category VisA schema;
- infers product category from the CSV directory;
- parses normal and fine-grained defect labels;
- resolves and validates image and mask paths;
- supports indexed masks with values greater than 1;
- treats every positive mask value as anomaly foreground;
- verifies image/mask dimension equality;
- prevents duplicate image IDs and duplicate annotated image paths;
- writes one versioned canonical record per image.

Important parsing behavior:

```text
label == normal
    → condition = normal
    → defect_type = null

non-empty non-normal label
    → condition = anomalous
    → defect_type = normalized source label
```

Indexed masks are converted to foreground using:

```python
binary_mask = mask > 0
```

Generated outputs:

```text
data/interim/visa_canonical.jsonl
reports/dataset_audit/visa_phase2_validation.json
reports/dataset_audit/visa_phase2_errors.jsonl
```

Final Phase 2 result:

```text
Canonical records: 10,821
Errors: 0
Warnings: 0
```

### Phase 3 — Feature derivation

```powershell
uv run visionassist phase3-visa --config configs/data/visa.yaml
```

Phase 3 derives:

- anomaly area in pixels;
- anomaly-to-image area ratio;
- inclusive bounding box;
- normalized bounding-box coordinates;
- foreground-pixel centroid;
- normalized centroid;
- centroid-based nine-grid location;
- project-defined visual severity.

Nine-grid labels:

```text
top_left       top_center       top_right
center_left    center           center_right
bottom_left    bottom_center    bottom_right
```

Default visual-severity policy:

```text
minor: anomaly area ratio < 0.005
moderate: 0.005 <= ratio < 0.02
major: ratio >= 0.02
```

Configured defect keywords can override severity to `major`.

This severity is a synthetic visual-inspection label, not a mechanical-risk or safety assessment.

Generated outputs:

```text
data/interim/visa_features.jsonl
reports/dataset_audit/visa_phase3_validation.json
reports/dataset_audit/visa_phase3_errors.jsonl
```

Final Phase 3 result:

```text
Feature records: 10,821
Anomalous records: 1,200
Errors: 0
Warnings: 0
```

### Phase 4 — Leakage-safe data splitting

```powershell
uv run visionassist phase4-visa --config configs/data/visa.yaml
```

Configured split policy:

```text
Train:      70%
Validation: 15%
Test:       15%
Seed:       42
```

Phase 4:

- splits at the source-image level;
- stratifies by category and condition;
- distributes defect concepts where frequency permits;
- keeps byte-identical SHA-256 duplicate clusters together;
- verifies no cross-split image ID leakage;
- verifies no cross-split image path leakage;
- verifies no cross-split SHA-256 leakage;
- writes per-category, per-condition, per-defect, and per-severity statistics.

Generated outputs:

```text
data/splits/vlm_supervised/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── split_assignments.csv
```

Reports:

```text
reports/dataset_audit/visa_phase4_validation.json
reports/dataset_audit/visa_phase4_errors.jsonl
```

Final image split:

```text
Train:       7,575
Validation:  1,622
Test:        1,624
Total:      10,821
```

### Phase 5 — Multimodal instruction generation

```powershell
uv run visionassist phase5-visa --config configs/data/visa.yaml
```

Default instruction policy:

```text
Normal image:     3 instructions
Anomalous image: 20 instructions
```

Task families:

- binary inspection;
- product identification;
- defect identification;
- localization;
- evidence explanation;
- structured report;
- technician note;
- uncertainty and abstention.

The generator uses deterministic templates and grounds answers only in:

- image-level condition;
- product category;
- source defect label;
- segmentation mask;
- derived spatial features.

It does not use another language model to invent training answers.

Generated outputs:

```text
data/processed/visa_instructions/
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

Reports:

```text
reports/dataset_audit/visa_phase5_validation.json
reports/dataset_audit/visa_phase5_errors.jsonl
```

Final instruction counts:

```text
Train:       37,005
Validation:   7,926
Test:         7,932
Total:       52,863
```

---

## Instruction record format

Each record follows a multimodal conversation structure:

```json
{
  "schema_version": "1.0",
  "instruction_id": "visa_pcb1_anomalous_001__json_01",
  "image_id": "visa_pcb1_anomalous_001",
  "dataset_split": "train",
  "task_family": "structured_report",
  "template_id": "json_01",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "data/raw/visa/pcb1/Data/Images/Anomaly/001.JPG"
        },
        {
          "type": "text",
          "text": "Return a concise quality-control report as valid JSON."
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "{\"condition\":\"defective\",\"product\":\"pcb1\"}"
        }
      ]
    }
  ],
  "metadata": {
    "source": "visa",
    "category": "pcb1",
    "condition": "anomalous",
    "defect_type": "missing_component",
    "location": "top_left",
    "visual_severity": "major",
    "answer_format": "json"
  }
}
```

All instruction variants from one image inherit the same Phase 4 split.

---

## Repository structure

```text
visionassist/
├── configs/
│   └── data/
│       └── visa.yaml
├── data/
│   ├── downloads/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── splits/
│   └── manifests/
├── reports/
│   └── dataset_audit/
├── scripts/
├── src/
│   └── visionassist/
│       ├── cli.py
│       ├── data/
│       │   ├── audit_visa.py
│       │   ├── checksum.py
│       │   ├── config.py
│       │   ├── dataset_card.py
│       │   ├── derive_features.py
│       │   ├── download.py
│       │   ├── extract.py
│       │   ├── generate_instructions.py
│       │   ├── license_report.py
│       │   ├── parse_visa.py
│       │   ├── phase1.py
│       │   ├── phase2.py
│       │   ├── phase3.py
│       │   ├── phase4.py
│       │   ├── phase5.py
│       │   └── split_visa.py
│       └── schemas/
│           ├── dataset.py
│           └── instruction.py
├── tests/
├── docs/
│   └── VISIONASSIST_DEVELOPMENT_RECORD.md
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Quality checks

Run the complete test suite:

```powershell
uv run pytest
```

Static checks:

```powershell
uv run ruff check .
uv run mypy src
```

Phase-specific tests:

```powershell
uv run pytest tests/test_parse_visa.py
uv run pytest tests/test_derive_features.py
uv run pytest tests/test_split_visa.py
uv run pytest tests/test_generate_instructions.py
```

---

## Generated files and Git

The following are intentionally excluded from Git:

- downloaded archives;
- raw VisA images and masks;
- canonical and processed datasets;
- split files;
- generated manifests and audit reports;
- model weights;
- checkpoints;
- training outputs;
- local environment files;
- project snapshot ZIPs.

The following should remain tracked:

```text
pyproject.toml
uv.lock
configs/
src/
tests/
scripts/
docs/
README.md
.gitignore
```

---

## Project snapshot

To create an upload-ready code snapshot without datasets or secrets:

```powershell
uv run python scripts/create_project_snapshot.py
```

Optional Git diff:

```powershell
uv run python scripts/create_project_snapshot.py --include-git-diff
```

The snapshot includes source code, configuration, tests, documentation, and lightweight metadata while excluding datasets, model artifacts, environments, and secrets.

---

## Next phase

The next phase is **Phase 6 — Training-readiness validation and untouched-model baseline evaluation**.

Recommended work:

1. verify all image references;
2. visually inspect samples from every task family;
3. implement the Qwen2.5-VL processor adapter;
4. implement the VLM data collator;
5. measure prompt and answer token-length distributions;
6. measure image-resolution and memory requirements;
7. run a small forward-pass smoke test;
8. evaluate the untouched model on a fixed benchmark subset;
9. save baseline metrics before QLoRA fine-tuning.

Training should begin only after the dataset adapter, batching logic, and baseline evaluation are validated.

---

## Detailed development record

For a full record of decisions, implementation details, fixes, counts, and phase outputs, see:

```text
docs/VISIONASSIST_DEVELOPMENT_RECORD.md
```
