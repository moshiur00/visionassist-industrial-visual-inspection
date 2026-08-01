# Phase 7C — Resumable Untouched-Model Inference

Phase 7C runs `Qwen/Qwen2.5-VL-3B-Instruct` over the frozen Phase 7A benchmark and writes raw predictions durably so a Colab interruption can be resumed.

## Direct baseline

```bash
uv run visionassist baseline-inference \
  --config configs/inference/qwen25vl3b_direct.yaml
```

## Prompted baseline

```bash
uv run visionassist baseline-inference \
  --config configs/inference/qwen25vl3b_prompted.yaml
```

The same command resumes automatically from `predictions.partial.jsonl`.

## Small smoke run

Set `stop_after: 5` in the selected inference YAML, run the command, inspect outputs, then restore `stop_after: null` and rerun. The completed five records are skipped.

## Outputs

```text
outputs/baseline/<run>/
├── predictions.partial.jsonl
├── predictions.jsonl
├── inference_errors.jsonl
└── run_manifest.json
```

`predictions.jsonl` is created only after every frozen benchmark instruction has a successful prediction. Raw generations are preserved without evaluation-time cleanup.

## Colab guidance

Copy the repository and VisA images to local `/content` storage before inference. Persist the output directory on Google Drive or periodically copy it there. The partial JSONL is append-only and each completed instruction ID is skipped on restart.

Use full precision/BF16 or FP16 for the official direct baseline when the assigned GPU has enough VRAM. Set `load_in_4bit: true` only when memory requires it, and record that quantized run as a distinct baseline because quantization can change predictions.
