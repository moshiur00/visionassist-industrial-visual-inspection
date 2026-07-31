# Phase 3 — Feature derivation

This project-root patch adds deterministic mask-derived features to every Phase 2 canonical record.

## Run

```powershell
uv sync --extra dev
uv run pytest tests/test_derive_features.py
uv run visionassist phase3-visa --config configs/data/visa.yaml
```

## Outputs

- `data/interim/visa_features.jsonl`
- `reports/dataset_audit/visa_phase3_validation.json`
- `reports/dataset_audit/visa_phase3_errors.jsonl`

For anomalous images, Phase 3 computes an inclusive pixel bounding box, foreground-pixel centroid, normalized coordinates, centroid-based nine-grid location, anomaly area, area ratio, and project-defined visual severity. Normal images receive zero anomaly area, no spatial features, and severity `none`.

Visual severity is a synthetic project annotation, not a mechanical or safety-risk assessment. Thresholds and keyword overrides are explicit in `configs/data/visa.yaml` and recorded in the validation report.
