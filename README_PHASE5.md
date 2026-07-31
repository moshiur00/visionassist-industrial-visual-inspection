# Phase 5 — Multimodal instruction generation

Phase 5 converts the leakage-safe Phase 4 image splits into deterministic,
grounded conversations for VLM fine-tuning.

## Run

```powershell
uv run pytest tests/test_generate_instructions.py
uv run visionassist phase5-visa --config configs/data/visa.yaml
```

## Outputs

```text
data/processed/visa_instructions/
├── train.jsonl
├── validation.jsonl
└── test.jsonl

reports/dataset_audit/
├── visa_phase5_validation.json
└── visa_phase5_errors.jsonl
```

The default policy generates three instruction records per normal image and
twenty per anomalous image. This yields approximately 52,863 records for the
official VisA distribution while increasing the representation of anomalous
examples.

The answers are generated only from source labels, masks, and deterministic
features. They do not claim mechanical root causes or safety conclusions.
