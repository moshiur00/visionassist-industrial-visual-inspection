# Phase 8 — Colab-resilient QLoRA training infrastructure

Phase 8 adds validated QLoRA configuration, GPU inspection, language-only LoRA
target discovery, deterministic subsets, one-batch validation, resumable
Transformers training, bounded checkpoint retention, and optional mirroring to
Google Drive.

## Local PC

The local PC does not need a GPU. Run configuration/checkpoint tests locally:

```powershell
uv sync --extra dev
uv run pytest tests/test_phase8_training.py
```

Do not run QLoRA training locally. The command fails clearly when CUDA is absent.

## Colab checkpoint policy

Set `checkpoints.persistent_output_dir` to a Drive directory, for example:

```yaml
checkpoints:
  resume: latest
  keep_latest: 2
  keep_best: 1
  persistent_output_dir: /content/drive/MyDrive/visionassist/checkpoints/qwen25vl3b_qlora_overfit_v1
  sync_every_save: true
```

`save_total_limit: 3` keeps a bounded local set. The persistent callback keeps
the newest two checkpoints plus the checkpoint referenced as best by Trainer.
On restart, `resume: latest` restores the newest checkpoint from Drive to local
`/content` storage and continues training.

## Commands

```bash
uv run visionassist training-environment \
  --config configs/training/qwen25vl3b_qlora_overfit.yaml

uv run visionassist training-smoke-test \
  --config configs/training/qwen25vl3b_qlora_overfit.yaml

uv run visionassist train-qlora \
  --config configs/training/qwen25vl3b_qlora_overfit.yaml \
  --resume latest
```

Order:

1. one-batch smoke test;
2. 32-example overfit run;
3. checkpoint resume test;
4. 1,000-example smoke run;
5. pilot run.
