# Phase 10 — Task-balanced 10,000-example pilot

Phase 10 scales VisionAssist from the successful 1,000-example smoke run to a
task-balanced 10,000-example pilot. The pilot starts from the untouched
`Qwen/Qwen2.5-VL-3B-Instruct` base model. It does not continue the optimizer or
adapter state from the smoke experiment.

Use [the Phase 10 Colab notebook](scripts/VisionAssist_Phase10_Colab.ipynb) for
the GPU audit, smoke test, training, resume, and evaluation workflow.

## Evidence from the smoke run

The completed smoke experiment reached step 250 and selected checkpoint 250:

| Item | Result |
| --- | ---: |
| Training records | 1,000 |
| Validation records | 200 |
| Best evaluation loss | 0.2052 |
| Training loss | 0.5746 |
| Frozen benchmark records | 2,100 |
| Inference errors | 0 |
| Benchmark failure rate | 53.86% |

Compared with the untouched baseline, binary inspection, product
identification, localization, structured reporting, and abstention improved
substantially. Defect recognition and evidence coverage remain the primary
weaknesses.

## Pilot sampling policy

The existing proportional random sampler overrepresents already-strong binary
and structured-report tasks. Implement deterministic task quotas before pilot
training:

| Task family | Pilot records |
| --- | ---: |
| Binary inspection | 1,400 |
| Product identification | 1,200 |
| Defect identification | 1,700 |
| Localization | 1,400 |
| Evidence explanation | 1,500 |
| Structured report | 1,400 |
| Technician note | 700 |
| Uncertainty | 700 |
| **Total** | **10,000** |

Selection must be deterministic, without duplicate instruction IDs, and must
retain all 12 product categories. Category and condition coverage must be
reported for every task family. The selected instruction-ID sequence and source
dataset must be fingerprinted in the experiment manifest.

## Development work

1. Extend the training-data configuration with an optional task-quota mapping.
2. Implement deterministic quota selection without changing existing random
   subset behavior used by the overfit and smoke experiments.
3. Reject quota sums that differ from `train_limit` and quotas that exceed
   available records.
4. Add tests for reproducibility, uniqueness, exact quotas, invalid
   configurations, and category/condition reporting.
5. Update the pilot YAML with the approved quota policy.
6. Add a preflight command/report that writes the exact pilot distribution and
   instruction-ID hash before model loading.
7. Remove the deprecated inference `torch_dtype` argument in favor of `dtype`.
8. Add periodic Drive mirroring for long adapter-inference runs.
9. Extend the Colab notebook with a separate pilot section and explicit launch
   gate.

## Execution gates

Before pilot training:

- all tests and static checks pass;
- the quota audit totals exactly 10,000 unique instructions;
- all task quotas match configuration;
- every category is represented among anomalous records;
- one-batch forward and backward validation passes;
- gradients are finite and nonzero;
- A100 VRAM headroom remains safely above the observed peak;
- the persistent checkpoint directory is empty or belongs to the same pilot
  run ID.

## Training policy

- start from the untouched base model;
- use the existing A100-safe image, sequence, quantization, and LoRA profile;
- store checkpoints on local `/content` and mirror them to Drive;
- keep the newest two checkpoints plus the best checkpoint;
- verify at least one resume cycle;
- do not change hyperparameters after the run begins.

## Post-training evaluation

Evaluate the best pilot adapter on:

1. the deterministic pilot training sample;
2. the configured validation subset;
3. the complete frozen 2,100-record Phase 7 benchmark.

Compare it directly with both the untouched baseline and the 1,000-example
adapter. The review must include per-task and per-category metrics, anomalous
recall, failure tags, structured schema validity, unsupported claims, and
representative defect/evidence failures.

## Promotion criteria

The pilot should improve the smoke adapter without regressing its strongest
behaviors. Target gates are:

- overall benchmark failure rate below 45%;
- binary accuracy above 85% with improved anomalous recall;
- defect F1 above 0.30;
- evidence fact coverage above 0.50;
- exact localization accuracy above 0.50;
- product accuracy above 0.90;
- structured-report schema validity of 1.0;
- appropriate abstention accuracy above 0.90;
- zero unsupported root-cause and safety claims.

Missing a target does not automatically invalidate the experiment, but every
miss requires failure analysis before a full-data training decision.
