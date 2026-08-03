# VisionAssist Qwen2.5-VL-3B Phase 11b adapter

## Summary

VisionAssist is a QLoRA adapter for `Qwen/Qwen2.5-VL-3B-Instruct` that assists
with image-grounded industrial visual inspection on the VisA product domains.
The promoted adapter is `qwen25vl3b_qlora_balanced_replay_v1/final_adapter`.

This is a decision-support research model, not an autonomous quality-control,
safety, repair, or root-cause system. A trained technician must review outputs
before operational use.

## Immutable identities

| Component | Identity |
| --- | --- |
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Base model revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Processor revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Adapter SHA-256 | `d335e37fdd8c96c0e9a823992f4aa458b0500b542986dedccde21561a142590f` |
| Rollback adapter SHA-256 | `eec9217a60d3897a48c925b0ad8658e0245e70d269d87ca73d627ebf0b7db8a7` |
| Frozen benchmark SHA-256 | `785928ea4775e0540e3851a1ce9900d0a34c49672404186635b35d64ecabb706` |

The base revision is the verified upstream `main` revision used for Phase 12
clean-runtime verification. The original training manifests did not pin a
revision, so release acceptance must demonstrate compatibility with this exact
revision before the release status can become `ready`.

## Intended uses

- normal-versus-anomalous visual inspection;
- VisA product identification;
- visible defect description and coarse localization;
- image-grounded evidence explanations;
- structured inspection reports and concise technician notes;
- abstention when root cause or safety impact is not visually supported.

## Out-of-scope uses

- autonomous accept/reject decisions without human review;
- determining mechanical root cause, hidden damage, or future failure;
- safety certification, regulatory compliance, or repair authorization;
- medical, personnel, surveillance, or non-industrial classification;
- products, imaging conditions, or defect vocabularies not validated here.

## Training

The base Qwen checkpoint was first adapted on a deterministic, task-balanced
10,000-instruction pilot. Phase 11b then continued from that adapter for 150
optimizer steps using 6,000 leakage-safe balanced-replay records: 4,220
anomalous and 1,780 normal. The replay selection contained all eight task
families, all 12 VisA categories, 2,392 unique images, and no validation or test
image overlap.

QLoRA used rank 8, alpha 16, dropout 0.05, NF4 4-bit base loading, and LoRA on
`q_proj`, `v_proj`, and `o_proj`. The exported adapter and best checkpoint are
byte-identical.

## Frozen-test results

The held-out benchmark contains 2,100 instructions over 591 VisA test images.
The promoted adapter completed all records with zero inference errors.

| Metric | Result |
| --- | ---: |
| Overall failure rate | 44.62% |
| Binary inspection accuracy | 85.00% |
| Product identification accuracy | 99.33% |
| Defect-identification F1 | 0.4239 |
| Evidence fact coverage | 49.70% |
| Exact localization accuracy | 46.33% |
| Adjacent-tolerance localization | 86.67% |
| Structured JSON validity | 99.67% |
| Structured schema validity | 99.33% |
| Structured defect F1 | 0.5442 |
| Technician-note fact coverage | 73.41% |
| Appropriate abstention accuracy | 98.00% |
| Unsupported root-cause rate | 0.00% |
| Unsupported safety-claim rate | 0.00% |

Compact evidence is stored in
[`docs/results/phase11b/promoted_balanced_replay_results.json`](docs/results/phase11b/promoted_balanced_replay_results.json).

## Limitations and known failure modes

- 937 of 2,100 frozen-test records triggered at least one failure tag.
- Defect vocabulary remains the weakest direct task; visually similar labels
  and compound defects are frequently confused.
- Exact localization is only 46.33%, although adjacent-tolerance accuracy is
  substantially higher.
- Evidence explanations cover only about half of annotated facts on average.
- One frozen structured report was invalid JSON and two records missed schema
  validity.
- Abstention is strong but imperfect; three held-out uncertainty cases failed
  to abstain appropriately.
- Performance is specific to VisA categories and its image distribution.

## Required safeguards

1. Display the source image and model output together.
2. Require technician confirmation for condition, defect, and location.
3. Treat root-cause, safety, and repair statements as unsupported unless a
   separate authoritative process supplies them.
4. Log the model revision, processor revision, adapter hash, generation
   settings, and operator decision.
5. Reject startup when release-readiness verification fails.
6. Use the documented Phase 10 rollback when a release trigger is met.

## Runtime contract

- deterministic decoding (`do_sample=false`, one beam);
- BF16 computation with 4-bit NF4 base loading on a supported CUDA GPU;
- SDPA attention;
- image pixel bounds 100,352 to 200,704;
- maximum 256 generated tokens;
- exact pinned model and processor revisions listed above.

Run `visionassist release-readiness --config configs/release/phase12.yaml
--require-ready` before promoting a packaged runtime.

## Licensing and attribution

Repository code is licensed under MIT. VisA source data is distributed under
CC BY 4.0 and requires attribution. Use of the Qwen base model and redistribution
of combined artifacts remain subject to the upstream model license. Confirm all
applicable terms before external distribution.
