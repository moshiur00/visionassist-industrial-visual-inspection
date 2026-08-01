# Phase 7B evaluator hardening

This patch improves deterministic parsing without rerunning model inference.
It adds:

- negation-aware binary condition parsing;
- explicit yes/no annotated-anomaly handling;
- conservative product aliases (formatting and canonical naming only);
- row/column to nine-grid localization parsing;
- severity aliases such as severe → major;
- strict and semantic defect metrics side by side;
- schema version 1.1 for evaluation reports.

The evaluator remains conservative. It does not map generic descriptions such
as `buttons`, `pills`, or `tokens` to VisA categories.

Rerun only evaluation against the existing frozen predictions:

```powershell
uv run visionassist evaluate-baseline `
  --benchmark data/benchmarks/visa_baseline_v1/benchmark.jsonl `
  --predictions outputs/baseline/qwen2_5_vl_3b_direct/predictions.jsonl `
  --config configs/evaluation/visa_baseline.yaml
```
