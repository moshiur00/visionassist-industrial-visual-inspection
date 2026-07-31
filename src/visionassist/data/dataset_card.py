"""Generate the project dataset card for the acquired VisA release."""

from __future__ import annotations

import json
from pathlib import Path


def write_dataset_card(path: Path, summary_path: Path, *, version: str) -> Path:
    """Create a dataset card using measured audit results when available."""

    summary: dict[str, object] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    total = summary.get("total_records", "pending audit")
    conditions = summary.get("condition_counts", {})
    categories = summary.get("category_counts", {})
    errors = summary.get("errors", [])

    text = f"""---
pretty_name: VisionAssist VisA Acquisition
license: cc-by-4.0
task_categories:
  - image-classification
  - image-segmentation
  - visual-question-answering
---

# VisionAssist — VisA Dataset Card

## Dataset summary

This project uses the **Visual Anomaly (VisA)** dataset as the primary source for
industrial visual inspection and grounded defect-report generation. The official
release contains image-level normal/anomalous labels and pixel-level masks for
anomalous images.

- **Source release:** {version}
- **Measured image records:** {total}
- **Measured condition counts:** `{json.dumps(conditions, sort_keys=True)}`
- **Measured category counts:** `{json.dumps(categories, sort_keys=True)}`
- **Audit errors:** {len(errors) if isinstance(errors, list) else "unknown"}

## Intended use

The acquired source data will be transformed into reproducible multimodal
instruction records for product identification, defect classification, coarse
localisation, evidence-grounded explanation, uncertainty handling, and
structured quality-control reporting.

## Out-of-scope use

The dataset and resulting models must not be presented as sufficient to:

- determine mechanical root causes from an image alone;
- certify safety or regulatory compliance;
- replace qualified industrial inspectors;
- provide authoritative repair instructions;
- generalise to arbitrary machinery without evaluation.

## Data fields in the raw manifest

Each image record contains a stable image ID, category, condition, relative image
and optional mask paths, dimensions, file size, and optional SHA-256 digest.

## Known limitations

VisA contains only twelve object subsets and a limited set of capture conditions.
Its class distribution is imbalanced toward normal images. Project-derived
severity and natural-language descriptions are not original industrial safety
labels and must be documented as synthetic supervision.

## License and citation

VisA is released under **CC BY 4.0**. See
`reports/dataset_audit/VISA_LICENSE_REPORT.md` for attribution and citation details.

## Reproducibility

Run:

```powershell
uv run visionassist phase1-visa --config configs/data/visa.yaml
```

The command downloads, fingerprints, safely extracts, audits, and documents the
dataset. Raw data and generated checksums are excluded from Git by default.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
