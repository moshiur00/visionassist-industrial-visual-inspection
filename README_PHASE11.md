# Phase 11 — Defect and localization hard-example iteration

Phase 11 targets the remaining errors of the promoted 10,000-example pilot.
The Phase 10 adapter is the frozen comparison point; its weights and raw
predictions remain external artifacts, while its compact results are tracked in
[`docs/results/phase10/pilot_results.json`](docs/results/phase10/pilot_results.json).

Use [the Phase 11 Colab notebook](scripts/VisionAssist_Phase11_Colab.ipynb) to
reproduce the selection, restore the promoted adapter, run the GPU smoke gate,
train resumably, and evaluate behind separate validation and frozen-test gates.

## Evidence

The pilot reduced the complete held-out benchmark failure rate from 53.86% to
46.00%. Product identification reached 98.67%, abstention reached 99.33%, and
defect-identification F1 improved from 0.1333 to 0.3210. The largest remaining
failure groups are wrong defect (326), incomplete or incorrect facts (330), and
wrong or adjacent location (265 combined).

## Development sequence

1. Build a deterministic failure-analysis command over evaluation records.
2. Produce defect confusion tables by category and identify label aliases,
   compound-label partial matches, and visually close defect pairs.
3. Produce exact and adjacent localization confusion tables by category and
   anomaly size.
4. Audit the single invalid structured report and add a bounded JSON recovery
   path if the failure is syntactic rather than semantic.
5. Generate a reproducible hard-example training manifest without drawing from
   validation or held-out benchmark records.
6. Run CPU tests and a Colab one-batch gate before any additional training.
7. Train a separately named adapter and compare it with the frozen Phase 10
   result on exactly the same validation and test records.

## Original promotion targets

These strict targets were defined before the experiments. The final decision
records every exception instead of silently treating all targets as passed.

- held-out failure rate below 46%;
- defect-identification F1 above 0.321;
- evidence fact coverage above 0.474;
- exact localization accuracy above 0.467 without reducing adjacent-tolerance
  accuracy below 0.863;
- product accuracy at least 0.98;
- structured JSON and schema validity of 1.0;
- abstention accuracy at least 0.99;
- zero unsupported root-cause and safety claims.

Phase 11b met the overall failure-rate, defect, evidence, product, and unsupported
claim targets. It narrowly missed the exact-localization, structured-validity,
and abstention targets. It was promoted on aggregate improvement with Phase 10
retained as an explicit rollback.

Phase 11 training remained gated until the failure analysis and hard-example
selection were reproducible and leakage checks passed.

## Progress

- [x] Implement a deterministic `analyze-adapter-failures` CLI command.
- [x] Add duplicate-ID, deterministic-output, defect-confusion, and
  localization-confusion tests.
- [x] Analyze all 2,100 held-out pilot predictions.
- [ ] Audit defect aliases and compound-label parsing.
- [ ] Add anomaly-size-aware localization analysis.
- [x] Build the leakage-safe hard-example selector.
- [x] Select 6,000 unique train instructions with exact task quotas, a
  20-record floor for every task/category pair, and zero held-out image overlap.
- [x] Add explicit promoted-adapter initialization with a fresh optimizer and
  lower `5e-5` learning rate.
- [x] Create and syntax-check the gated Phase 11 Colab notebook.
- [x] Run the promoted-adapter one-batch GPU gate.
- [x] Train and validate the Phase 11 adapter.
- [x] Reject Phase 11 before frozen testing because structured-report behavior
  regressed catastrophically.
- [x] Add per-task condition quotas and build the Phase 11b balanced-replay
  selection.
- [x] Create and syntax-check the Phase 11b gated Colab notebook.
- [x] Run the Phase 11b smoke gate, short training, and validation-only review.
- [x] Run the complete frozen test after the validation review.
- [x] Promote Phase 11b and retain Phase 10 as the rollback adapter.

The first report found 88 exact defect matches among 479 anomalous direct and
structured-report records. The dominant defect failure is PCB `melt` being
unparsed or predicted as `missing`. Macaroni1, macaroni2, and chewinggum have the
largest total defect-error counts.

Localization produced 218 exact and 383 adjacent-tolerant matches among the
same 479 records. The dominant pattern is collapsing `center_left`,
`center_right`, and `bottom_center` into `center`. Macaroni1, candle, and
macaroni2 have the largest localization-error counts. See the
[compact analysis summary](docs/results/phase11/pilot_test_failure_analysis_summary.json).

The accepted hard-example selection contains 6,000 unique instructions over
1,284 images. Validation errors influenced 4,989 selected records; the remaining
1,011 preserve coverage for binary inspection, product identification, and
other strata without a matching validation error. Every task/category pair has
at least 20 examples, and overlap with validation and test images is zero. Its
instruction-ID fingerprint is
`440390b2a4ab6b5491eeaba806f5c5b45a9f460781bd67d9008fe13d33e1e3e6`.
See the [selection summary](docs/results/phase11/hard_example_selection_summary.json).

## Phase 11 outcome and Phase 11b correction

Phase 11 improved direct defect F1 from 0.3755 to 0.4551 and evidence coverage
from 0.4381 to 0.4869, but validation failure rate increased from 28.8% to
44.9%. Structured-report condition accuracy fell to 25.1%, defect F1 to 9.6%,
and location accuracy to 12.8%. The best checkpoint and exported adapter hashes
were identical, ruling out an export bug. The frozen test was intentionally not
run, and Phase 10 remains promoted.

The cause was selection imbalance: all 1,000 structured-report records and
5,464 of 6,000 total records were anomalous. Phase 11b adds exact per-task
condition quotas. Its 6,000-record replay selection contains 4,220 anomalous and
1,780 normal records, including 300 anomalous and 800 normal structured reports.
It starts again from Phase 10 at `1e-5` for at most 150 steps, evaluating every
25 steps. See the [rejected-run record](docs/results/phase11/rejected_hard_example_run.json)
and [Phase 11b selection](docs/results/phase11b/balanced_replay_selection_summary.json).

Phase 11b completed at step 150, with checkpoint 150 selected as best. The
checkpoint and exported final adapter are byte-identical (SHA-256
`d335e37fdd8c96c0e9a823992f4aa458b0500b542986dedccde21561a142590f`).
The complete 2,100-record frozen test produced zero inference errors and 937
failure records (44.62%), improving on Phase 10 by 29 records and 1.38
percentage points. Direct defect F1 rose from 0.3210 to 0.4239, evidence fact
coverage from 0.4738 to 0.4970, and structured-report defect F1 from 0.5236 to
0.5442.

The trade-offs are retained in the promotion record: exact localization fell
slightly from 46.67% to 46.33%, while adjacent-tolerance localization improved
to 86.67%; structured schema validity fell from 99.67% to 99.33%; and
appropriate abstention fell from 99.33% to 98.00%. Product accuracy improved to
99.33%, and unsupported root-cause and safety-claim rates remained zero.

Phase 11b is promoted as the best overall adapter, with Phase 10 retained as a
rollback. No additional training run is scheduled by default. See the
[complete compact result](docs/results/phase11b/promoted_balanced_replay_results.json)
and use [the cleaned Phase 11b Colab notebook](scripts/VisionAssist_Phase11b_Colab.ipynb)
to reproduce the gated workflow, including the separately gated frozen test.
