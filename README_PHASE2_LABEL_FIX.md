# Phase 2 VisA Label Fix

Extract this archive directly into the project root and allow overwriting.

The official per-category VisA CSV uses `label` as follows:

- `normal` for normal images
- one or more defect descriptions for anomalous images

This patch therefore derives `condition=anomalous` from any non-empty,
non-normal label and stores the normalized source defect description in
`defect_type`. Comma-separated source labels are preserved as comma-separated
normalized defect types.

Run:

```powershell
uv run pytest tests/test_parse_visa.py
uv run visionassist phase2-visa --config configs/data/visa.yaml
```
