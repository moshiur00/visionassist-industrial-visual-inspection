# VisionAssist

VisionAssist is a reproducible multimodal fine-tuning project for industrial visual inspection and defect explanation.

The project converts the official VisA industrial anomaly dataset into a validated multimodal instruction dataset, benchmarks an untouched vision-language model, and provides GPU-aware QLoRA training infrastructure for fine-tuning Qwen2.5-VL-3B-Instruct.

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

The data pipeline, training infrastructure, frozen baseline, adapter evaluation,
task-balanced 10,000-example pilot, and balanced-replay correction are complete.
The Phase 11b adapter is the promoted model, with Phase 10 retained as rollback.
The planned training cycle is closed. Phase 12 release-readiness development is
in progress: immutable identities, acceptance gates, model documentation, and
rollback controls are implemented, with the clean-runtime GPU check outstanding.

| Phase                            | Status   | Main output                               |
| -------------------------------- | -------- | ----------------------------------------- |
| Phase 1 — Dataset acquisition    | Complete | Verified VisA raw dataset                 |
| Phase 2 — Dataset parsing        | Complete | Canonical metadata JSONL                  |
| Phase 3 — Feature derivation     | Complete | Spatially enriched records                |
| Phase 4 — Data splitting         | Complete | Leakage-safe train/validation/test splits |
| Phase 5 — Instruction generation | Complete | 52,863 multimodal instruction records     |
| Phase 6 — Training readiness     | Complete | Processor, masking, and data validation   |
| Phase 7 — Baseline evaluation    | Complete | Frozen benchmark and baseline results     |
| Phase 8 — QLoRA infrastructure   | Complete | Memory-safe training and checkpointing    |
| Phase 9 — Adapter evaluation     | Complete | Overfit and 1,000-example adapter results |
| Phase 10 — Task-balanced pilot   | Complete | 10,000-example pilot and full evaluation  |
| Phase 11 — Hard-example iteration | Complete | Balanced replay and promoted final adapter |
| Phase 12 — Release readiness      | In progress | Packaging, runtime QA, and model card    |

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

Untouched-model baseline highlights:

| Metric | Result |
| --- | ---: |
| Benchmark records | 2,100 |
| Inference errors | 0 |
| Overall failure rate | 82.9% |
| Binary inspection accuracy | 35.5% |
| Product identification accuracy | 16.0% |
| Defect exact match | 0.0% |
| Localization accuracy | 19.7% |
| Structured-report schema validity | 0.0% |
| Appropriate abstention accuracy | 74.0% |

The baseline was run in BF16 without 4-bit quantization on an NVIDIA A100
40 GB. These results are the frozen reference for measuring improvement after
fine-tuning.

Phase 8 validation highlights:

- the revised one-batch forward-and-backward smoke test passed;
- peak allocated VRAM was 3.72 GiB on an A100 40 GB;
- 3,022,848 LoRA parameters were trainable;
- gradients were finite and nonzero;
- the 32-example, 100-step overfit run completed;
- the best validation loss was 1.246 at checkpoint 50;
- the final reported training loss was 0.750.

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

### Phase 6 — Training-readiness validation

```powershell
uv run visionassist phase6-visa --config configs/data/visa.yaml
```

Phase 6 validates all image references, instruction schemas, task coverage,
assistant-only label masking, Qwen processor compatibility, sequence lengths,
and visual sample quality. The final report covers all 52,863 instructions and
10,821 source images with zero errors or warnings.

Generated reports are stored in:

```text
reports/training_readiness/
```

### Phase 7 — Frozen baseline benchmark

Phase 7 builds a deterministic 2,100-record benchmark, runs untouched-model
inference, and evaluates task-specific predictions.

```powershell
uv run visionassist baseline-inference `
    --config configs/inference/qwen25vl3b_direct.yaml
```

Inference is append-only and resumable through `predictions.partial.jsonl`.
The completed direct baseline and evaluation reports are stored under:

```text
outputs/baseline/qwen2_5_vl_3b_direct/
outputs/baseline/direct_evaluation/
```

### Phase 8 — QLoRA training infrastructure

Phase 8 adds hardware inspection, 4-bit model loading, language-only LoRA
target discovery, deterministic training subsets, forward-and-backward smoke
validation, resumable training, bounded checkpoint retention, and optional
Google Drive mirroring.

Local CPU-safe checks:

```powershell
uv run pytest tests/test_phase8_training.py
uv run visionassist training-environment `
    --config configs/training/qwen25vl3b_qlora_overfit.yaml
```

GPU workflow:

```bash
uv run visionassist training-smoke-test \
  --config configs/training/qwen25vl3b_qlora_overfit.yaml

uv run visionassist train-qlora \
  --config configs/training/qwen25vl3b_qlora_overfit.yaml \
  --resume latest
```

The default overfit profile is designed for an A100 40 GB and uses:

- a maximum sequence length of 2,048;
- 100,352–200,704 image pixels;
- LoRA rank 8 and alpha 16;
- `q_proj`, `v_proj`, and `o_proj` targets;
- batch size 1 with gradient accumulation;
- SDPA attention and gradient checkpointing.

Use [the Phase 8 Colab notebook](scripts/VisionAssist_Phase8_Colab.ipynb) in a
fresh GPU runtime. Do not start a larger training run until the one-batch report
confirms finite loss, finite nonzero gradients, and adequate VRAM headroom.

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
│   ├── benchmark/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   └── training/
├── data/
│   ├── downloads/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── splits/
│   └── manifests/
├── outputs/
│   ├── baseline/
│   └── training/
├── reports/
│   ├── baseline/
│   ├── dataset_audit/
│   └── training_readiness/
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
│       ├── benchmarks/
│       ├── evaluation/
│       ├── inference/
│       ├── training/
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
uv run pytest tests/test_phase6_readiness.py
uv run pytest tests/test_phase7_benchmark.py tests/test_phase7_metrics.py
uv run pytest tests/test_phase7c_inference.py
uv run pytest tests/test_phase8_training.py
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

The active phase is **Phase 12 — Release readiness**.

Phase 11b is the promoted final adapter for this training cycle. On the complete
2,100-record frozen test it reduced failures from 966 to 937 (46.00% to 44.62%),
raised direct defect F1 from 0.3210 to 0.4239, and raised evidence fact coverage
from 47.38% to 49.70%. It completed with zero inference errors and zero
unsupported root-cause or safety claims. Phase 10 remains the rollback model.

Phase 12 now pins the model and processor revision, verifies promoted and
rollback adapter hashes, publishes the model card and known limitations, and
defines a task-balanced 96-record acceptance suite. The remaining release gate
is to run that suite in a clean Colab GPU runtime, require a `ready` report, and
build the immutable release bundle. Further training is out of scope unless
release testing reveals a specific, measurable blocker.

See [README_PHASE10.md](README_PHASE10.md) for the pilot baseline,
[README_PHASE11.md](README_PHASE11.md) for the completed correction sequence,
and the [Phase 11b promotion record](docs/results/phase11b/promoted_balanced_replay_results.json)
for compact frozen-test evidence and artifact hashes. Follow
[README_PHASE12.md](README_PHASE12.md) for the active release sequence.

---

## Detailed development record

For a full record of decisions, implementation details, fixes, counts, and phase outputs, see:

```text
docs/VISIONASSIST_DEVELOPMENT_RECORD.md
```
