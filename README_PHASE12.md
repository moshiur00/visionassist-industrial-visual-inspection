# Phase 12 — Release readiness

Phase 12 freezes the promoted Phase 11b adapter and proves that it can be loaded
and evaluated in a clean Colab runtime without identity drift. It does not
perform further training.

## Release contract

- pin base model and processor to commit
  `66285546d2b821cf421d4f5eb2576359d3770cd3`;
- verify every required promoted-adapter file by SHA-256;
- verify the Phase 10 rollback adapter by SHA-256;
- enforce the compact frozen-test promotion thresholds;
- run a deterministic 96-record acceptance suite with 12 records from each task;
- require acceptance instruction fingerprint
  `afb6725901b21a7f013bb082f644b202ec6856fb8f3895a465331737c537b2ef`;
- require zero inference errors and zero unsupported root-cause or safety claims;
- publish the model card, limitations, and rollback runbook;
- produce a final readiness report with status `ready`.

The base revision is the latest verified upstream Qwen model revision identified
when Phase 12 began. Because earlier training runs recorded a null revision, the
clean-runtime acceptance run is the compatibility proof for this pin.

## Sequence

1. Restore the prepared VisA data, promoted adapter, rollback adapter, and
   compact promotion evidence.
2. Run Phase 12 CPU regression tests.
3. Run `visionassist release-readiness`; status should be `pending`, with no
   failed checks and only clean-runtime acceptance outstanding.
4. Audit the exact task-balanced 96-record selection.
5. Run the gated adapter acceptance evaluation on a GPU.
6. Re-run readiness with `--require-ready`.
7. Build and hash the release bundle only after status is `ready`.
8. Persist the acceptance output, report, and bundle to Drive.

Use [the Phase 12 Colab notebook](scripts/VisionAssist_Phase12_Colab.ipynb) for
the clean-runtime sequence.

## Status

- [x] Freeze promoted and rollback adapter hashes.
- [x] Pin model and processor revisions.
- [x] Add validated release configuration and CLI checks.
- [x] Add task-balanced inference subset support.
- [x] Add Phase 12 CPU regression tests.
- [x] Publish the model card and rollback runbook.
- [ ] Run the clean-runtime 96-record GPU acceptance suite.
- [ ] Produce a `ready` report and immutable release bundle.
