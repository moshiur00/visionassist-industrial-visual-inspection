# Phase 6 completion patch

This patch completes the remaining training-readiness evidence requested after the initial Phase 6 run.

## Added evidence

- defect-label, location, and severity distributions;
- repeated-prompt and repeated-answer frequency summaries;
- instructions-per-image distribution;
- unique prompt/template combination count;
- deterministic stratified processor sampling across task, category, and condition;
- dedicated sequence and visual-token statistics;
- examples exceeding the configured sequence limit;
- explicit assistant-only label masking verification by decoding trainable labels;
- padding-token masking verification in a collated batch;
- richer sample gallery cards and coverage summary;
- machine-readable gallery approval record;
- lazy JSONL training dataset adapter.

## Run the complete automated check

```powershell
uv sync --extra dev --extra training
uv run visionassist phase6-visa `
    --config configs/data/visa.yaml `
    --processor-smoke-test
```

New output:

```text
reports/training_readiness/visa_phase6_sequence_statistics.json
reports/training_readiness/visa_phase6_gallery_review.json
```

The default processor sample increases from 64 to 256 and is selected using a deterministic round-robin strategy over task family, category, and condition.

## Review and approve the gallery

Open the gallery:

```powershell
Start-Process reports/training_readiness/visa_phase6_sample_gallery.html
```

After visually checking image/target alignment, record approval:

```powershell
uv run visionassist phase6-visa `
    --config configs/data/visa.yaml `
    --processor-smoke-test `
    --approve-gallery `
    --reviewer "Md Moshiur Rahman"
```

The validation report distinguishes:

- `automated_passed`: all machine checks passed;
- `phase_complete`: processor evidence exists and the gallery has been manually approved.

## Important

`--approve-gallery` is a human attestation. Use it only after inspecting the images and targets in the generated HTML gallery.
