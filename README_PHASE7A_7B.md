# VisionAssist Phase 7A and 7B

Phase 7A freezes a deterministic benchmark from the Phase 5 test split. Phase 7B provides dependency-free parsers and task-specific metrics so baseline predictions can be evaluated consistently before and after QLoRA fine-tuning.

## Phase 7A — Build the benchmark

```powershell
uv run visionassist build-baseline-benchmark `
    --config configs/benchmark/visa_baseline_v1.yaml
```

This creates:

```text
data/benchmarks/visa_baseline_v1/
├── benchmark.jsonl
├── benchmark_manifest.json
├── benchmark_distribution.json
└── benchmark_sha256.txt
```

The default benchmark contains 2,100 test instructions:

| Task | Records |
|---|---:|
| Binary inspection | 400 |
| Product identification | 300 |
| Defect identification | 300 |
| Localization | 300 |
| Evidence explanation | 200 |
| Structured report | 300 |
| Technician note | 150 |
| Uncertainty | 150 |

Sampling is deterministic with seed 42. It uses a metadata-stratified round-robin policy and prioritizes unique source images before using additional template variants when a task quota is larger than the number of available test images.

The frozen benchmark and manifest should be committed. Do not overwrite `visa_baseline_v1` after running model evaluation; create `visa_baseline_v2` for a changed sampling policy or corrected labels.

## Validate the benchmark

```powershell
uv run visionassist validate-baseline-benchmark `
    --config configs/benchmark/visa_baseline_v1.yaml
```

Validation checks:

- only test-split records;
- exact task quotas;
- unique instruction IDs;
- benchmark SHA-256 matches the frozen manifest;
- image paths remain under the project root;
- supported image extensions;
- image existence, non-zero size, readability;
- task/category/condition distribution reports.

Outputs:

```text
reports/baseline/
├── visa_baseline_v1_validation.json
├── visa_baseline_v1_statistics.json
└── visa_baseline_v1_errors.jsonl
```

## Phase 7B — Prediction format

The evaluator expects one JSON object per benchmark record:

```json
{
  "instruction_id": "visa_pcb2_anomalous_054__json_01",
  "prediction": "{\"condition\":\"defective\", ...}"
}
```

Additional inference metadata fields are allowed and preserved by the inference stage, but only `instruction_id` and `prediction` are required by the evaluator.

## Evaluate predictions

```powershell
uv run visionassist evaluate-baseline `
    --benchmark data/benchmarks/visa_baseline_v1/benchmark.jsonl `
    --predictions outputs/baseline/qwen2_5_vl_3b_direct/predictions.jsonl `
    --config configs/evaluation/visa_baseline.yaml
```

Outputs:

```text
outputs/baseline/evaluation/
├── metrics.json
├── per_task_metrics.csv
├── per_category_metrics.csv
├── failures.jsonl
└── parsing_errors.jsonl
```

Implemented deterministic scoring includes:

- normal/defective classification accuracy and macro-F1;
- product-category accuracy and macro-F1;
- compound-defect exact match and set precision/recall/F1;
- exact and adjacent-tolerance nine-grid localization;
- structured JSON validity, schema completeness, and field accuracy;
- appropriate uncertainty/abstention accuracy;
- unsupported root-cause and safety-claim detection;
- fact-coverage scoring for explanations and technician notes;
- machine-readable failure tags.

## Tests

```powershell
uv run pytest tests/test_phase7_benchmark.py tests/test_phase7_metrics.py
```

The complete repository test suite passed with this patch:

```text
31 passed
```
