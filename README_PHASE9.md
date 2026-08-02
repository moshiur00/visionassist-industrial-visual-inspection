# Phase 9 — Post-training adapter evaluation and training continuation

Phase 9 verifies that the best QLoRA checkpoint improves generated answers
before training is scaled. A low training loss alone is not a promotion signal.

## Required checkpoint layout

Restore the persisted Phase 8 run beneath the project root:

```text
outputs/training/qwen25vl3b_qlora_overfit_v1/
├── final_adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── dataset_manifest.json
├── resolved_config.yaml
└── run_manifest.json
```

The completed Phase 8 run used `load_best_model_at_end`, selected checkpoint 50,
and then exported that selected model as `final_adapter`. Therefore the final
adapter is the checkpoint-50 evaluation artifact. The optimizer, scheduler, and
RNG files are required to resume training from a checkpoint but are not required
to evaluate its adapter.

## Evaluate checkpoint 50

Run all three assessments in a GPU environment:

```bash
uv run visionassist evaluate-adapter \
  --config configs/inference/qwen25vl3b_overfit_checkpoint50_train.yaml

uv run visionassist evaluate-adapter \
  --config configs/inference/qwen25vl3b_overfit_checkpoint50_validation.yaml

uv run visionassist evaluate-adapter \
  --config configs/inference/qwen25vl3b_overfit_checkpoint50_test.yaml
```

The train and validation configurations reproduce the exact deterministic
32-record subsets used by Phase 8. The test configuration evaluates a fixed
64-record subset of the frozen Phase 7 benchmark. Each run writes:

```text
evaluation_records.jsonl
predictions.partial.jsonl
predictions.jsonl
run_manifest.json
assessment_summary.json
evaluation/metrics.json
evaluation/per_task_metrics.csv
evaluation/per_category_metrics.csv
evaluation/failures.jsonl
evaluation/parsing_errors.jsonl
```

The run manifest records hashes for `adapter_config.json` and
`adapter_model.safetensors` when present.

## Promotion gate

Do not scale training until checkpoint 50 satisfies all of these checks:

1. all three inference runs complete without errors;
2. training-subset predictions show that the adapter learned the supervised
   formats and facts;
3. validation and held-out outputs are grounded and non-repetitive;
4. structured reports are valid JSON with the required fields;
5. uncertainty prompts abstain from unsupported root-cause and safety claims;
6. the held-out subset improves over the corresponding untouched-model
   predictions or provides a documented explanation for any regression.

Inspect both aggregate metrics and representative predictions. The 32-example
run is an infrastructure and learnability proof, not a candidate production
model.

## Continue training

### Stage A — 1,000-example smoke run

Use `configs/training/qwen25vl3b_qlora_smoke.yaml`. This is a new experiment;
do not resume it from the 32-example overfit optimizer state.

```bash
uv run visionassist training-smoke-test \
  --config configs/training/qwen25vl3b_qlora_smoke.yaml

uv run visionassist train-qlora \
  --config configs/training/qwen25vl3b_qlora_smoke.yaml \
  --resume latest
```

Persist checkpoints to Google Drive and verify one interruption/resume cycle.
Evaluate the best checkpoint using the Phase 9 pipeline before promotion.

Promotion criteria:

- finite loss and gradients with safe VRAM headroom;
- evaluation loss does not diverge persistently;
- improved held-out task metrics over the untouched baseline;
- structured output and abstention behavior remain acceptable;
- no repeated-output or answer-format collapse.

### Stage B — 10,000-example pilot

Use `configs/training/qwen25vl3b_qlora_pilot.yaml` only after Stage A passes.

```bash
uv run visionassist training-smoke-test \
  --config configs/training/qwen25vl3b_qlora_pilot.yaml

uv run visionassist train-qlora \
  --config configs/training/qwen25vl3b_qlora_pilot.yaml \
  --resume latest
```

Evaluate the best pilot checkpoint on the complete frozen 2,100-record Phase 7
benchmark. Compare per-task, per-category, safety, abstention, and structured
report metrics against the untouched-model baseline.

### Stage C — Full training decision

Choose the full-run data volume, epochs, learning rate, and checkpoint cadence
from the pilot learning curves and benchmark results. Do not automatically
extend the pilot configuration without reviewing category balance, normal versus
anomalous performance, and task-family regressions.
