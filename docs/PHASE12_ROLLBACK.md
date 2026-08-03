# Phase 12 rollback runbook

## Immutable rollback target

- Run: `qwen25vl3b_qlora_pilot_v1`
- Adapter: `outputs/training/qwen25vl3b_qlora_pilot_v1/final_adapter`
- Adapter SHA-256:
  `eec9217a60d3897a48c925b0ad8658e0245e70d269d87ca73d627ebf0b7db8a7`
- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Pinned release revision:
  `66285546d2b821cf421d4f5eb2576359d3770cd3`

## Rollback triggers

Rollback is required when any of these conditions is confirmed:

- adapter or processor hash verification fails;
- clean-runtime startup or inference cannot complete without errors;
- structured JSON/schema validity falls below the Phase 12 acceptance floor;
- unsupported root-cause or safety claims appear in the acceptance suite;
- operator review identifies a material regression affecting safe use;
- required release identity or audit metadata is absent.

## Procedure

1. Stop new inference requests and preserve the failing request, response,
   runtime manifest, logs, and artifact hashes.
2. Mark the Phase 11b release unavailable; do not delete its artifacts.
3. Verify the rollback adapter SHA-256 against the value above.
4. Keep the same pinned base-model revision, processor revision, image bounds,
   quantization, prompt, and deterministic generation settings.
5. Change only `adapter_path` to the rollback directory.
6. Run the 96-record Phase 12 acceptance suite under a new run ID and output
   directory.
7. Require a `ready` release-readiness report before restoring service.
8. Record the operator, time, reason, failing release identity, rollback
   identity, and acceptance report path.

## Recovery

Phase 11b may be restored only after the original trigger has a documented root
cause, a reproducible fix, a clean acceptance run, and explicit human approval.
Do not retrain automatically as part of rollback.
