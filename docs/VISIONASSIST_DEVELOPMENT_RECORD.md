# VisionAssist Development Record

**Project:** VisionAssist - Industrial Visual Inspection and Defect-Explanation Assistant  
**Repository version:** `0.1.0`  
**Document purpose:** A detailed record of the project decisions, implemented workflow, generated artifacts, validation results, and current development status through Phase 5.  
**Last updated:** 31 July 2026

---

## 1. Project overview

VisionAssist is a multimodal vision-language model project for industrial visual inspection.

The intended system accepts an industrial inspection image and a natural-language request, then produces a grounded answer such as:

- whether the item is normal or defective;
- the product category;
- the visible defect type;
- the approximate defect location;
- a concise evidence-based explanation;
- a structured quality-control report;
- a technician-oriented inspection note;
- an uncertainty statement when root cause or safety impact cannot be inferred from the image.

The initial system is deliberately limited to visual inspection. It does not claim to determine mechanical root causes, provide authoritative repair instructions, replace qualified inspectors, or infer safety impact from an image alone.

The project is designed as a reproducible pipeline:

```text
Official VisA dataset
        ↓
Acquisition and integrity verification
        ↓
Canonical image-level metadata
        ↓
Mask-derived spatial features
        ↓
Leakage-safe train/validation/test splits
        ↓
Grounded multimodal instruction records
        ↓
Baseline evaluation
        ↓
QLoRA fine-tuning
        ↓
Post-training evaluation and deployment
```

---

## 2. Current project scope

### Primary dataset

The project currently uses the official **VisA industrial anomaly dataset**.

Configured source version:

```text
2022-09-22
```

Expected official release statistics:

| Item               |  Count |
| ------------------ | -----: |
| Total images       | 10,821 |
| Normal images      |  9,621 |
| Anomalous images   |  1,200 |
| Product categories |     12 |

Configured categories:

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

### Planned model

The currently planned initial model is:

```text
Qwen2.5-VL-3B-Instruct
```

The intended fine-tuning method is QLoRA or LoRA, depending on the available GPU environment.

Training has not started yet. The project has completed the dataset engineering stages through multimodal instruction generation.

---

## 3. Technology stack

### Runtime

```text
Python: >=3.12,<3.13
Package manager: uv
Build backend: Hatchling
CLI: Typer
Configuration: YAML + Pydantic
```

### Core dependencies

```text
pydantic
pydantic-settings
PyYAML
Pillow
pandas
numpy
typer
rich
```

### Development dependencies

```text
pytest
pytest-cov
ruff
mypy
```

### Optional training dependencies

```text
torch
transformers
datasets
accelerate
peft
trl
bitsandbytes
```

`bitsandbytes` is excluded on Windows by the project dependency marker and will normally be used in a Linux training environment such as RunPod, Colab, or another CUDA host.

---

## 4. Repository structure

The synchronized repository currently contains:

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
│   └── bootstrap.ps1
├── src/
│   └── visionassist/
│       ├── __init__.py
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
│   ├── test_checksum.py
│   ├── test_dataset_schema.py
│   ├── test_derive_features.py
│   ├── test_download.py
│   ├── test_extract.py
│   ├── test_generate_instructions.py
│   ├── test_parse_visa.py
│   └── test_split_visa.py
├── DATASET_CARD.md
├── README.md
├── README_PHASE2_LABEL_FIX.md
├── README_PHASE2_MULTILABEL_MASK_FIX.md
├── README_PHASE3.md
├── README_PHASE4.md
├── README_PHASE5.md
├── pyproject.toml
└── uv.lock
```

Large datasets, generated processed data, model weights, virtual environments, caches, secrets, and temporary archives are intentionally not included in project snapshots.

---

## 5. Configuration

The central dataset configuration is:

```text
configs/data/visa.yaml
```

It contains:

- dataset source and version;
- official download URL;
- archive and extraction paths;
- expected counts and category names;
- image extensions;
- checksum and download settings;
- Phase 2 canonical-manifest paths;
- Phase 3 feature thresholds;
- Phase 4 split ratios and seed;
- Phase 5 instruction-generation policy;
- strict validation switches.

Important current values:

```yaml
phase4_seed: 42
phase4_train_ratio: 0.70
phase4_validation_ratio: 0.15

phase5_normal_instructions_per_image: 3
phase5_anomalous_instructions_per_image: 20
```

The remaining Phase 4 ratio is assigned to the test split:

```text
1.00 - 0.70 - 0.15 = 0.15
```

---

# 6. Phase 1 - Dataset acquisition and audit

## 6.1 Objective

Phase 1 acquires the official VisA dataset and verifies that the raw source is structurally complete before downstream parsing begins.

## 6.2 Implemented capabilities

Phase 1 implements:

- resumable official-source downloading;
- `.part` files for interrupted downloads;
- archive SHA-256 calculation;
- acquisition receipt generation;
- safe TAR extraction;
- path-traversal prevention during extraction;
- raw image discovery;
- image readability verification;
- image dimensions and file-size collection;
- per-image SHA-256 generation;
- category and condition inference from paths;
- anomaly-mask candidate lookup;
- annotation CSV inventory;
- structural verification against official counts;
- license-report generation;
- dataset-card generation.

## 6.3 Main modules

### `download.py`

Responsibilities:

- inspect remote metadata where possible;
- download the archive;
- resume interrupted downloads;
- return a structured `DownloadResult`.

Important functions:

```text
_remote_metadata()
download_file()
```

### `extract.py`

Responsibilities:

- safely extract TAR files;
- prevent archive members from escaping the configured destination;
- optionally replace an existing extraction.

Important functions:

```text
_is_within_directory()
safe_extract_tar()
```

### `checksum.py`

Responsibilities:

- compute SHA-256 hashes in chunks.

Important function:

```text
sha256_file()
```

### `audit_visa.py`

Responsibilities:

- scan the extracted dataset;
- infer product category and condition;
- verify image files;
- locate masks;
- inventory annotation CSVs;
- write the raw manifest and audit reports.

Important functions:

```text
_infer_condition()
_find_category()
_candidate_mask()
_iter_images()
_read_size()
_csv_inventory()
audit_visa()
```

### `dataset_card.py`

Generates:

```text
DATASET_CARD.md
```

### `license_report.py`

Generates:

```text
reports/dataset_audit/VISA_LICENSE_REPORT.md
```

### `phase1.py`

Coordinates the complete acquisition workflow:

```text
download
→ checksum
→ extraction
→ dataset placement
→ audit
→ documentation
```

## 6.4 Phase 1 command

```powershell
uv run visionassist phase1-visa --config configs/data/visa.yaml
```

Force a clean acquisition:

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

## 6.5 Phase 1 generated artifacts

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

## 6.6 Checksum limitation

The official dataset registry did not provide an authoritative SHA-256 digest in the configured source metadata. Therefore, the pipeline records the hash of the downloaded official archive for repeat-run comparison and combines it with structural verification:

- expected image counts;
- category coverage;
- image readability;
- annotation CSV discovery;
- mask availability.

---

# 7. Phase 2 - Annotation and mask parsing

## 7.1 Objective

Phase 2 converts the raw VisA annotations into one validated canonical metadata record per source image.

Output:

```text
data/interim/visa_canonical.jsonl
```

## 7.2 Actual VisA CSV schema discovered

During development, the real per-category `image_anno.csv` files were found to contain:

```text
image
label
mask
```

They do not consistently contain explicit `object` or `split` columns.

The parser was corrected to:

- infer the category from the parent directory of the CSV;
- assign `source_split: unknown` when no split column is present;
- remain compatible with equivalent optional column names where available.

## 7.3 Defect-label interpretation

The `label` column is not merely a binary label. It includes actual defect descriptions, for example:

```text
normal
chunk of wax missing
damaged corner of packaging
chunk of wax missing,foreign particals on candle
```

The final parsing policy is:

```text
label == normal
    → condition = normal
    → defect_type = null

non-empty non-normal label
    → condition = anomalous
    → defect_type = normalized source label
```

Multi-defect labels are preserved as normalized comma-separated concepts.

The project intentionally preserves source terminology rather than inventing a new defect taxonomy at this stage.

## 7.4 Multi-label mask correction

An important dataset-specific issue was discovered in PCB masks.

Some masks contain values such as:

```text
[0, 2]
[0, 3]
[0, 1, 2]
[0, 1, 2, 3]
```

These are valid indexed anomaly masks, not invalid grayscale masks.

The final mask policy is:

```python
foreground = mask > 0
```

Therefore:

- `0` means background;
- every positive value means anomaly foreground;
- original unique values are preserved in metadata;
- masks can still be converted to binary for geometric calculations;
- source-binary and multi-label masks are reported separately.

## 7.5 Validation performed per record

For every annotation row, the parser verifies:

- supported annotation fields;
- resolvable image path;
- readable image;
- valid dimensions;
- known category;
- interpretable normal/anomalous condition;
- resolvable anomaly mask;
- readable mask;
- equal image and mask dimensions;
- non-empty anomaly foreground;
- unique image ID;
- no duplicate annotated image path.

## 7.6 Main modules

### `parse_visa.py`

Important responsibilities:

- normalize column names;
- infer missing category information;
- parse source labels;
- parse optional source split;
- resolve image and mask paths;
- inspect images;
- inspect indexed masks;
- produce canonical records;
- generate strict validation reports.

Important functions:

```text
_canonical_columns()
_resolve_path()
_parse_label()
_parse_split()
_read_image()
_read_mask()
_category_from_csv()
_record_from_row()
_iter_annotation_csvs()
parse_visa()
```

### `schemas/dataset.py`

Defines:

```text
Condition
SourceSplit
DatasetSplit
RawImageRecord
MaskMetadata
CanonicalImageRecord
```

Pydantic validation makes canonical records explicit and versioned.

## 7.7 Phase 2 command

```powershell
uv run visionassist phase2-visa --config configs/data/visa.yaml
```

## 7.8 Phase 2 outputs

```text
data/interim/visa_canonical.jsonl
reports/dataset_audit/visa_phase2_validation.json
reports/dataset_audit/visa_phase2_errors.jsonl
```

## 7.9 Final Phase 2 results

| Metric                  | Result |
| ----------------------- | -----: |
| Annotation CSV files    |     12 |
| Valid records           | 10,821 |
| Invalid records         |      0 |
| Normal records          |  9,621 |
| Anomalous records       |  1,200 |
| Records with masks      |  1,200 |
| Source-binary masks     |    916 |
| Multi-label masks       |    284 |
| Binary-compatible masks |  1,200 |
| Warnings                |      0 |

Foreground-ratio statistics:

| Statistic |        Value |
| --------- | -----------: |
| Minimum   | 0.0000200038 |
| Maximum   | 0.3195656991 |
| Mean      | 0.0099883123 |

All Phase 2 strict checks passed.

---

# 8. Phase 3 - Feature derivation

## 8.1 Objective

Phase 3 converts anomaly masks into deterministic spatial and severity metadata suitable for instruction generation.

Input:

```text
data/interim/visa_canonical.jsonl
```

Output:

```text
data/interim/visa_features.jsonl
```

## 8.2 Features derived

For anomalous images:

- foreground area in pixels;
- anomaly-to-image area ratio;
- inclusive pixel bounding box;
- normalized bounding-box coordinates;
- foreground-pixel centroid;
- normalized centroid;
- centroid-based nine-grid location;
- project-defined visual severity;
- severity basis.

For normal images:

```text
anomaly area = 0
anomaly ratio = 0.0
bounding box = null
centroid = null
nine-grid location = null
visual severity = none
```

## 8.3 Bounding-box convention

The bounding box is based on all positive mask pixels.

Conceptually:

```text
x_min = minimum foreground x
y_min = minimum foreground y
x_max = maximum foreground x
y_max = maximum foreground y
```

The box is inclusive in pixel space.

Normalized coordinates are also stored to make records independent of image resolution.

## 8.4 Centroid convention

The centroid is computed from all foreground pixels, not merely from the bounding-box center.

This is important for irregular or disconnected anomalies.

## 8.5 Nine-grid location

The normalized centroid is mapped to:

```text
top_left       top_center       top_right
center_left    center           center_right
bottom_left    bottom_center    bottom_right
```

Each image dimension is divided into thirds.

## 8.6 Visual-severity policy

The configured area thresholds are:

```yaml
minor: anomaly area ratio < 0.005
moderate: 0.005 <= ratio < 0.02
major: ratio >= 0.02
```

Configured major keyword overrides:

```text
missing
misplaced
damaged
crack
broken
```

This is explicitly a project-defined visual severity label. It does not represent real mechanical risk, operational danger, product safety, or regulatory severity.

## 8.7 Main module

### `derive_features.py`

Important classes:

```text
Phase3Result
FeatureDerivationError
```

Important functions:

```text
_read_binary_foreground()
_derive_bbox()
_derive_centroid()
_nine_grid_location()
_visual_severity()
derive_record()
derive_visa_features()
```

Additional schemas:

```text
NineGridLocation
VisualSeverity
BoundingBox
Centroid
DerivedImageRecord
```

## 8.8 Phase 3 command

```powershell
uv run visionassist phase3-visa --config configs/data/visa.yaml
```

## 8.9 Phase 3 outputs

```text
data/interim/visa_features.jsonl
reports/dataset_audit/visa_phase3_validation.json
reports/dataset_audit/visa_phase3_errors.jsonl
```

## 8.10 Final Phase 3 results

| Metric            | Result |
| ----------------- | -----: |
| Valid records     | 10,821 |
| Invalid records   |      0 |
| Anomalous records |  1,200 |
| Normal records    |  9,621 |
| Warnings          |      0 |

Nine-grid distribution:

| Location      | Count |
| ------------- | ----: |
| top_left      |    48 |
| top_center    |    91 |
| top_right     |    43 |
| center_left   |   142 |
| center        |   569 |
| center_right  |   111 |
| bottom_left   |    47 |
| bottom_center |   108 |
| bottom_right  |    41 |

Visual-severity distribution:

| Severity | Count |
| -------- | ----: |
| none     | 9,621 |
| minor    |   791 |
| moderate |   117 |
| major    |   292 |

All anomalies have derived features, and all normal images have empty spatial features.

---

# 9. Phase 4 - Data splitting

## 9.1 Objective

Phase 4 creates deterministic, leakage-safe image-level splits before instruction generation.

This ordering is critical.

All instruction variants generated from a source image must inherit the same image-level split. Otherwise, the model could train on one question about an image and be evaluated on another question about the identical image.

## 9.2 Split policy

Configured ratios:

```text
Train:      70%
Validation: 15%
Test:       15%
```

Configured seed:

```text
42
```

Primary stratification:

```text
product category × normal/anomalous condition
```

Within each stratum, defect labels are interleaved where frequency permits so that defect concepts are distributed across splits rather than concentrated in one split.

## 9.3 Duplicate and leakage handling

The Phase 4 implementation:

- validates unique image IDs;
- validates unique image paths;
- groups byte-identical images by SHA-256;
- keeps duplicate clusters in one split;
- checks cross-split image ID leakage;
- checks cross-split path leakage;
- checks cross-split SHA-256 leakage;
- fails in strict mode when leakage is found.

## 9.4 Main module

### `split_visa.py`

Important classes:

```text
SplitGenerationError
Phase4Result
```

Important functions:

```text
_load_records()
_stable_tie_breaker()
_stratum_key()
_defect_key()
_target_counts()
_interleave_by_defect()
_build_duplicate_clusters()
_assign_records()
_validate_uniqueness()
_validate_leakage()
_stats_for()
_write_outputs()
split_visa()
```

The deterministic tie-breaker ensures reproducible assignment with the same seed and source records.

## 9.5 Phase 4 command

```powershell
uv run visionassist phase4-visa --config configs/data/visa.yaml
```

## 9.6 Phase 4 outputs

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

## 9.7 Final split results

| Split      | Images |          Percentage |
| ---------- | -----: | ------------------: |
| Train      |  7,575 | approximately 70.0% |
| Validation |  1,622 | approximately 15.0% |
| Test       |  1,624 | approximately 15.0% |
| Total      | 10,821 |                100% |

Condition counts:

| Split      | Normal | Anomalous |
| ---------- | -----: | --------: |
| Train      |  6,735 |       840 |
| Validation |  1,442 |       180 |
| Test       |  1,444 |       180 |

Each product category contributes anomalous examples to all three splits.

Phase 4 completed with:

```text
Errors: 0
Warnings: 0
```

---

# 10. Phase 5 - Multimodal instruction generation

## 10.1 Objective

Phase 5 converts the Phase 4 source-image splits into deterministic multimodal conversation records for VLM training and evaluation.

## 10.2 Instruction policy

Configured records per image:

```text
Normal image:    3 instructions
Anomalous image: 20 instructions
```

This policy increases anomalous-task representation while preserving all source images.

Calculated total:

```text
9,621 normal images × 3 instructions = 28,863
1,200 anomalous images × 20          = 24,000
Total                                = 52,863
```

This is instruction-level rebalancing, not image duplication across splits.

## 10.3 Task families

The generator creates:

- binary inspection;
- product identification;
- defect identification;
- localization;
- evidence explanation;
- structured report;
- technician note;
- uncertainty and abstention.

Final task-family counts:

| Task family            | Instructions |
| ---------------------- | -----------: |
| binary_inspection      |       13,221 |
| product_identification |       12,021 |
| defect_identification  |        3,600 |
| localization           |        3,600 |
| evidence_explanation   |        3,600 |
| structured_report      |       13,221 |
| technician_note        |        2,400 |
| uncertainty            |        1,200 |
| **Total**              |   **52,863** |

## 10.4 Grounding policy

Answers are generated only from:

- source image-level condition;
- source product category;
- source defect label;
- source segmentation mask;
- deterministic Phase 3 features.

The generator does not use another language model to invent answers.

This gives the dataset:

- deterministic regeneration;
- inspectable supervision;
- reproducibility;
- low annotation cost;
- reduced hallucination risk.

## 10.5 Safety and uncertainty policy

The generated language distinguishes visible evidence from unsupported inference.

The model is taught that it can:

- describe a visible anomaly;
- identify its coarse location;
- report project-defined visual severity;
- recommend manual verification.

It is also taught that it cannot determine from the image alone:

- mechanical root cause;
- internal damage;
- actual safety impact;
- authoritative repair action;
- manufacturing-process cause.

## 10.6 Instruction schema

### `schemas/instruction.py`

Defines:

```text
MessageContent
ChatMessage
InstructionMetadata
InstructionRecord
```

A record contains:

- schema version;
- unique instruction ID;
- source image ID;
- dataset split;
- task family;
- template ID;
- user multimodal message;
- assistant answer;
- structured metadata.

Representative shape:

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
          "text": "{\"condition\":\"defective\", ...}"
        }
      ]
    }
  ],
  "metadata": {
    "source": "visa",
    "category": "pcb1",
    "condition": "anomalous",
    "defect_type": "missing",
    "location": "top_left",
    "visual_severity": "major",
    "answer_format": "json"
  }
}
```

## 10.7 Main module

### `generate_instructions.py`

Important classes:

```text
InstructionGenerationError
Phase5Result
Template
```

Important functions:

```text
_read_split()
_humanize()
_status()
_location()
_binary_answer()
_product_answer()
_defect_answer()
_location_answer()
_explanation_answer()
_json_answer()
_technician_answer()
_uncertainty_answer()
_templates()
_selected_templates()
_record_to_instruction()
_validate_json_answers()
generate_instructions()
```

The module validates generated structured JSON answers before marking Phase 5 complete.

## 10.8 Phase 5 command

```powershell
uv run visionassist phase5-visa --config configs/data/visa.yaml
```

## 10.9 Phase 5 outputs

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

## 10.10 Final instruction results

| Split      | Source images | Instructions |
| ---------- | ------------: | -----------: |
| Train      |         7,575 |       37,005 |
| Validation |         1,622 |        7,926 |
| Test       |         1,624 |        7,932 |
| **Total**  |    **10,821** |   **52,863** |

Condition-level instruction counts:

| Condition | Instructions |
| --------- | -----------: |
| Normal    |       28,863 |
| Anomalous |       24,000 |

Validation checks:

```text
all_source_images_preserved: true
instruction_ids_unique: true
image_split_leakage_absent: true
json_answers_valid: true
all_task_families_present: true
```

Final Phase 5 status:

```text
Errors: 0
Warnings: 0
Passed: true
```

---

# 11. Command-line interface

The project exposes the following CLI commands through:

```text
visionassist = visionassist.cli:app
```

Commands:

```powershell
uv run visionassist phase1-visa --config configs/data/visa.yaml
uv run visionassist audit-visa --config configs/data/visa.yaml
uv run visionassist phase2-visa --config configs/data/visa.yaml
uv run visionassist phase3-visa --config configs/data/visa.yaml
uv run visionassist phase4-visa --config configs/data/visa.yaml
uv run visionassist phase5-visa --config configs/data/visa.yaml
```

Each phase has a small orchestration module:

```text
phase1.py
phase2.py
phase3.py
phase4.py
phase5.py
```

The orchestration layer keeps the CLI separate from implementation details.

---

# 12. Testing strategy

Current test modules:

| Test file                       | Main coverage                              |
| ------------------------------- | ------------------------------------------ |
| `test_checksum.py`              | SHA-256 behavior                           |
| `test_download.py`              | download logic                             |
| `test_extract.py`               | safe extraction and traversal prevention   |
| `test_dataset_schema.py`        | Pydantic dataset models                    |
| `test_parse_visa.py`            | CSV parsing, labels, multi-label masks     |
| `test_derive_features.py`       | boxes, centroids, location, severity       |
| `test_split_visa.py`            | deterministic splitting and leakage checks |
| `test_generate_instructions.py` | instruction selection and validation       |

Phase-specific tests run successfully during development:

```text
Phase 2 parser tests:           4 passed
Phase 3 feature tests:          3 passed
Phase 4 split tests:            3 passed
Phase 5 instruction tests:      4 passed
```

The full suite should be run before each new phase:

```powershell
uv run pytest
```

Static-quality commands:

```powershell
uv run ruff check .
uv run mypy src
```

---

# 13. Strict validation philosophy

Each data phase has a strict-mode configuration:

```yaml
strict_phase2: true
strict_phase3: true
strict_phase4: true
strict_phase5: true
```

The intended behavior is to fail loudly rather than silently produce an incomplete or unsafe training dataset.

Examples of strict failures detected during development:

- unsupported assumptions about CSV columns;
- incorrect binary-label assumptions;
- indexed masks mistakenly interpreted as invalid;
- missing count expectations;
- cross-split leakage;
- malformed structured JSON answers.

This strict approach is a core quality feature of the project.

---

# 14. Important development corrections

The following issues were found and fixed while building the pipeline.

## 14.1 Incorrect CSV-schema assumption

Initial assumption:

```text
object, split, label, image, mask
```

Actual per-category schema:

```text
image, label, mask
```

Fix:

- infer object category from the annotation file path;
- represent unavailable source split as `unknown`.

## 14.2 Incorrect binary-label assumption

Initial assumption:

```text
label contains normal/anomaly
```

Actual behavior:

```text
label contains normal or fine-grained defect descriptions
```

Fix:

- treat `normal` as normal;
- treat other non-empty values as anomalous;
- preserve normalized defect descriptions.

## 14.3 Incorrect binary-mask assumption

Initial assumption:

```text
valid mask values must be 0/1 or 0/255
```

Actual behavior:

- some PCB masks are indexed and use positive class values such as 2 and 3.

Fix:

```python
binary_foreground = mask > 0
```

Original unique values remain available in mask metadata.

These corrections are important because they show that the pipeline was adapted to the actual source data rather than forcing the dataset into guessed conventions.

---

# 15. Reproducibility

The pipeline is reproducible because it uses:

- versioned source configuration;
- an official download URL;
- archive and per-file SHA-256 values;
- explicit expected counts;
- versioned Pydantic schemas;
- deterministic feature derivation;
- deterministic split seed;
- deterministic instruction templates;
- machine-readable reports;
- strict validation;
- test coverage.

The same source version and configuration should produce the same canonical records, features, image assignments, and instruction dataset.

---

# 16. Current dataset readiness

The multimodal instruction dataset is currently ready for training-readiness validation.

Primary files:

```text
data/processed/visa_instructions/train.jsonl
data/processed/visa_instructions/validation.jsonl
data/processed/visa_instructions/test.jsonl
```

Current total:

```text
52,863 instruction records
10,821 unique images
```

No source image crosses train, validation, and test boundaries.

However, “dataset generated successfully” is not yet identical to “training pipeline fully verified.”

Before fine-tuning, the next phase should verify:

- every referenced image path exists in the intended training environment;
- sampled conversations render correctly using the target processor;
- the target chat template accepts the record format;
- JSON responses remain valid after tokenizer formatting;
- image-size and token-length distributions are understood;
- the data collator masks labels correctly;
- a small batch can complete a forward pass;
- the untouched model has a recorded baseline.

---

# 17. Recommended next phase

## Phase 6 - Training-readiness validation and baseline evaluation

Recommended subphases:

### 6A. Dataset inspection

- randomly sample records from every task family;
- render image, prompt, and target answer;
- inspect normal/anomalous balance;
- inspect category and defect distributions;
- inspect duplicate prompts and repetitive answers;
- verify all image paths.

### 6B. Qwen2.5-VL format adapter

- load the processor;
- apply the official chat template;
- convert project JSONL into model-ready examples;
- verify image preprocessing;
- verify label masking;
- implement a VLM data collator.

### 6C. Token and image statistics

- prompt token distribution;
- answer token distribution;
- total sequence-length distribution;
- image-resolution distribution;
- estimated memory requirements;
- truncation analysis.

### 6D. Smoke test

- load a very small model or the intended model;
- process a few examples;
- run one forward pass;
- verify finite loss;
- verify gradient flow for LoRA parameters.

### 6E. Untouched-model baseline

Evaluate the base model on a fixed subset before fine-tuning.

Suggested baseline metrics:

```text
condition accuracy
condition macro-F1
product accuracy
defect macro-F1
location accuracy
JSON validity
required-field accuracy
unsupported-claim rate
uncertainty-response accuracy
```

The baseline must be saved so that the later fine-tuned model can be compared fairly against exactly the same test records.

---

# 18. Known limitations

## Dataset imbalance

At image level:

```text
Normal:    9,621
Anomalous: 1,200
```

Instruction generation reduces the imbalance but does not create new visual anomaly diversity.

## Synthetic language

The answer text is template-generated. This provides reliable grounding but may produce repetitive language.

A later controlled paraphrasing stage may improve linguistic diversity, but the structured source label must remain the ground truth.

## Visual severity

Severity is project-defined from area thresholds and keyword overrides.

It is not a real industrial standard.

## Coarse localization

Nine-grid labels provide coarse localization. They do not replace pixel segmentation or precise object detection.

## Defect taxonomy

Defect labels originate from VisA source strings and can contain spelling variation, compound labels, and inconsistent terminology.

A later taxonomy-normalization stage may be useful, but it must preserve the original source label.

## Generalization

Current data comes only from VisA. Performance on new factories, lighting conditions, cameras, products, and defect types remains unknown.

External evaluation using selected MVTec AD or MVTec LOCO AD categories is planned but has not been implemented.

## Root cause and safety

The model is not trained to infer hidden mechanical causes or safety risk from an image.

---

# 19. Project status checklist

## Completed

- [x] Define the initial visual-inspection scope
- [x] Select VisA as the primary dataset
- [x] Configure Python 3.12 project
- [x] Implement official dataset download
- [x] Implement resumable acquisition
- [x] Implement safe extraction
- [x] Implement archive and image checksums
- [x] Verify dataset structure
- [x] Generate dataset card
- [x] Generate license report
- [x] Parse all annotation CSVs
- [x] Infer missing category metadata
- [x] Parse fine-grained defect labels
- [x] Parse binary and indexed masks
- [x] Generate canonical records
- [x] Derive bounding boxes
- [x] Derive centroids
- [x] Derive nine-grid locations
- [x] Derive anomaly-area ratios
- [x] Derive synthetic visual severity
- [x] Create deterministic image-level splits
- [x] Prevent image/path/hash leakage
- [x] Generate split statistics
- [x] Generate multimodal instruction records
- [x] Generate uncertainty and abstention samples
- [x] Validate JSON answers
- [x] Produce 52,863 instructions
- [x] Complete all phases with zero final errors and warnings

## Not yet completed

- [ ] Full visual audit of sampled instruction records
- [ ] Qwen2.5-VL processor adapter
- [ ] Training data collator
- [ ] Token-length analysis
- [ ] Image-resolution and memory analysis
- [ ] Base-model inference pipeline
- [ ] Baseline evaluation
- [ ] QLoRA training configuration
- [ ] Fine-tuning
- [ ] Post-training evaluation
- [ ] External generalization evaluation
- [ ] FastAPI inference service
- [ ] Streamlit or React application
- [ ] Dockerized deployment
- [ ] Model card for the trained model

---

# 20. Environment setup and full reproduction

## Create the environment

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv sync --extra dev
```

## Run the full data pipeline

```powershell
uv run visionassist phase1-visa --config configs/data/visa.yaml
uv run visionassist phase2-visa --config configs/data/visa.yaml
uv run visionassist phase3-visa --config configs/data/visa.yaml
uv run visionassist phase4-visa --config configs/data/visa.yaml
uv run visionassist phase5-visa --config configs/data/visa.yaml
```

## Run quality checks

```powershell
uv run pytest
uv run ruff check .
uv run mypy src
```

---

# 21. Snapshot workflow

A project snapshot script was created so the repository code can be synchronized without uploading the raw dataset.

Recommended location:

```text
scripts/create_project_snapshot.py
```

Run:

```powershell
uv run python scripts/create_project_snapshot.py
```

Optional Git diff:

```powershell
uv run python scripts/create_project_snapshot.py --include-git-diff
```

The generated ZIP excludes:

- raw datasets;
- processed datasets;
- model weights;
- `.env`;
- virtual environments;
- caches;
- Git internals;
- large binary artifacts.

It includes:

- source code;
- tests;
- configuration;
- documentation;
- manifests;
- lightweight validation reports;
- a SHA-256 snapshot manifest.

---

# 22. Final status summary

At the end of Phase 5, VisionAssist has a complete and validated data-engineering pipeline.

The project has transformed the official VisA anomaly dataset into:

```text
10,821 validated canonical image records
1,200 validated anomalous masks
10,821 spatially enriched records
7,575 / 1,622 / 1,624 leakage-safe image splits
52,863 grounded multimodal instruction records
```

Final instruction split:

```text
Train:       37,005
Validation:   7,926
Test:         7,932
```

Final validation status:

```text
Errors:   0
Warnings: 0
Passed:   true
```

The data-engineering foundation is complete.

The correct next step is not immediate full training. The next step is to build the Qwen2.5-VL dataset adapter, inspect the generated samples, validate batching and label masking, measure token and image distributions, and record an untouched-model baseline before QLoRA fine-tuning.

---

# 23. Phase 10 task-balanced pilot completion

Phase 10 trained a fresh QLoRA adapter from
`Qwen/Qwen2.5-VL-3B-Instruct` using a deterministic 10,000-instruction sample.
The audit contained 10,000 unique instruction IDs, 3,435 unique images, all
eight task families, and all 12 VisA categories. Its ordered instruction-ID
fingerprint is
`98bcf9ab5831746b3c3399723a3f83b2a118fafba837d9613fa8245a860dae1e`.

The one-batch forward/backward gate passed with finite, nonzero gradients and
3.721 GiB peak allocated A100 memory. Training completed at step 1,250 with
checkpoint 1,250 selected as best, evaluation loss 0.0693785, and training loss
0.2725075.

The 1,000-record validation assessment completed with zero inference errors and
a 28.8% failure rate. The complete frozen 2,100-record test also completed with
zero inference errors and a 46.0% failure rate. Held-out metrics included 85.0%
binary accuracy, 0.3210 defect F1, 47.38% evidence coverage, 46.67% exact
localization, 98.67% product accuracy, and 99.33% appropriate abstention.
Unsupported root-cause and safety-claim rates remained zero.

The pilot is promoted over the 1,000-example smoke adapter. The primary
remaining errors are wrong defect (326), incomplete or incorrect evidence
(330), and wrong or adjacent location (265 combined). Phase 11 therefore starts
with deterministic error analysis and leakage-safe hard-example selection, not
an immediate larger training run.

Compact metrics are tracked in
`docs/results/phase10/pilot_results.json`. Adapter weights, checkpoints, raw
predictions, and evaluation records remain external Google Drive artifacts.

---

# 24. Phase 11 hard-example rejection and Phase 11b promotion

Phase 11 introduced deterministic failure analysis and selected 6,000
leakage-safe hard examples. The selection contained 5,464 anomalous records,
and all 1,000 structured-report records were anomalous. Although direct defect
and evidence metrics improved, this imbalance caused structured-report behavior
to collapse on validation. The run was rejected before frozen testing. Its best
checkpoint and final adapter hashes matched, excluding an export mismatch as
the cause.

Phase 11b corrected the replay distribution with exact per-task condition
quotas: 4,220 anomalous and 1,780 normal records overall, including 300
anomalous and 800 normal structured reports. The selection contained 6,000
unique instruction IDs, 2,392 images, and no validation or test image overlap.
Training restarted from the promoted Phase 10 adapter with a fresh optimizer,
a `1e-5` learning rate, and a 150-step limit. The GPU smoke gate passed and
training completed at step 150; checkpoint 150 was selected as best with
evaluation loss 0.0646632 and training loss 0.0887208.

Validation completed over 1,000 records with zero inference errors and a 28.7%
failure rate. The complete 2,100-record frozen test also had zero inference
errors. Compared with Phase 10, failures fell from 966 to 937 and the failure
rate from 46.00% to 44.62%. Direct defect F1 rose from 0.3210 to 0.4239,
evidence fact coverage from 0.4738 to 0.4970, and structured-report defect F1
from 0.5236 to 0.5442. Unsupported root-cause and safety-claim rates remained
zero.

Phase 11b is promoted as the current best adapter. Phase 10 remains the rollback
because Phase 11b has small regressions in exact localization (46.67% to
46.33%), structured schema validity (99.67% to 99.33%), and appropriate
abstention (99.33% to 98.00%). The training cycle is closed. Phase 12 focuses on
packaging, deterministic inference acceptance tests, documented limitations,
and release readiness rather than additional training.

The promoted final adapter and checkpoint-150 adapter share SHA-256
`d335e37fdd8c96c0e9a823992f4aa458b0500b542986dedccde21561a142590f`.
Compact metrics and archive hashes are tracked in
`docs/results/phase11b/promoted_balanced_replay_results.json`; large weights,
predictions, and evaluation records remain ignored local or Google Drive
artifacts.

---

# 25. Phase 12 release-readiness implementation

Phase 12 begins after the planned training cycle closes. It adds no optimizer
steps and does not change the promoted weights. Its purpose is to make artifact
identity, clean-runtime behavior, limitations, and rollback enforceable.

The release contract pins `Qwen/Qwen2.5-VL-3B-Instruct` and its processor to
revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. Earlier training and
evaluation manifests recorded null revisions, so a clean-runtime acceptance run
must prove that the promoted adapter is compatible with this exact upstream
revision. The contract hashes every required promoted-adapter file and the two
critical rollback-adapter files.

The `release-readiness` command verifies adapter contents, base-model alignment,
model-card and rollback documentation, the Phase 11b promotion evidence, and
all configured metric thresholds. Before GPU acceptance it must return
`pending` with no failed checks. With valid acceptance artifacts it must return
`ready`; any identity, hash, metric, count, or unsupported-claim violation makes
the release `blocked`.

Inference configuration now supports exact per-task subset quotas. The Phase 12
acceptance suite selects 96 frozen test records with 12 records from each of the
eight task families. It uses deterministic decoding, pinned revisions, BF16
computation, 4-bit NF4 loading, SDPA attention, and the promoted adapter. The
acceptance manifest must contain the exact task quotas, adapter hash, revision,
96 completed predictions, and zero errors. Its ordered instruction-ID
fingerprint is
`afb6725901b21a7f013bb082f644b202ec6856fb8f3895a465331737c537b2ef`.

The repository now contains:

- `configs/release/phase12.yaml`, the validated release contract;
- `configs/inference/qwen25vl3b_phase12_acceptance.yaml`, the clean-runtime run;
- `MODEL_CARD.md`, including frozen-test metrics and known limitations;
- `docs/PHASE12_ROLLBACK.md`, the immutable rollback procedure;
- `tests/test_phase12_release.py`, covering pending, ready, and tamper states;
- `scripts/VisionAssist_Phase12_Colab.ipynb`, with separate acceptance and
  packaging gates.

The remaining external step is the clean Colab GPU run. A release bundle may be
built only after `visionassist release-readiness --require-ready` succeeds.
