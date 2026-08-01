# Phase 8 hardware inspection fix

This patch fixes `training-environment` on Windows when the configured training
output directory has not been created yet.

## Cause

`shutil.disk_usage()` requires an existing path on Windows. Phase 8 passed the
future experiment output directory directly, causing `WinError 3` before any
hardware report could be generated.

## Fix

- resolve relative output paths against the current project directory;
- walk upward to the nearest existing parent;
- inspect disk capacity using that existing path;
- remain CPU-safe and avoid loading the training model;
- add Windows-compatible regression tests.

## Verification

```powershell
uv run pytest tests/test_phase8_training.py

uv run visionassist training-environment `
    --config configs/training/qwen25vl3b_qlora_overfit.yaml
```

On a CPU-only PC, the second command should complete and report that CUDA is
unavailable. Actual QLoRA training remains restricted to a CUDA environment
such as Google Colab Pro.
