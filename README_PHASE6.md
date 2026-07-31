# Phase 6 — Training-readiness validation

Phase 6 validates the Phase 5 multimodal instruction dataset before baseline inference or QLoRA training.

## Core validation

Run locally after Phase 5:

```powershell
uv run visionassist phase6-visa --config configs/data/visa.yaml
```

This checks all 52,863 instruction records, resolves every image path, verifies image readability, validates structured JSON answers, checks metadata consistency, detects duplicate instruction IDs and split leakage, computes dataset statistics, and creates an HTML sample gallery.

Outputs:

```text
reports/training_readiness/
├── visa_phase6_validation.json
├── visa_phase6_errors.jsonl
├── visa_phase6_statistics.json
└── visa_phase6_sample_gallery.html
```

## Qwen processor and collator smoke test

This mode requires the training dependency group and downloads the configured processor:

```powershell
uv sync --extra dev --extra training
uv run visionassist phase6-visa `
    --config configs/data/visa.yaml `
    --processor-smoke-test
```

On Google Colab, use the same command after mounting or copying the project and data into the runtime.

Additional output:

```text
reports/training_readiness/visa_phase6_processor.json
```

The processor smoke test verifies:

- Qwen chat-template formatting;
- image preprocessing;
- non-empty assistant target labels;
- assistant-only label masking;
- two-example batch collation;
- visual tensor presence.

The project-owned collator is implemented in:

```text
src/visionassist/training/collator.py
```

It masks user, image-prefix, and padding tokens with `-100`, leaving only assistant target tokens trainable.
