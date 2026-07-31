# Phase 4 — Data splitting

Run after Phase 3:

```powershell
uv run pytest tests/test_split_visa.py
uv run visionassist phase4-visa --config configs/data/visa.yaml
```

Outputs:

```text
data/splits/vlm_supervised/train.jsonl
data/splits/vlm_supervised/validation.jsonl
data/splits/vlm_supervised/test.jsonl
data/splits/vlm_supervised/split_assignments.csv
reports/dataset_audit/visa_phase4_validation.json
reports/dataset_audit/visa_phase4_errors.jsonl
```

The split is deterministic from `phase4_seed`. Primary stratification is by
object category and normal/anomalous condition. Within each stratum, defect
labels are interleaved so they are distributed across splits where their
frequency permits. Byte-identical images sharing a SHA-256 digest are kept in
the same split. The validation report fails strict mode if image IDs, paths, or
SHA-256 digests cross split boundaries.
