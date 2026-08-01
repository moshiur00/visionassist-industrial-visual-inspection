# Phase 8 A100 memory-safety patch

This patch addresses the one-batch QLoRA CUDA out-of-memory failure observed on
an NVIDIA A100 40 GB. The previous profile allowed unrestricted image
resolution, a 4,096-token sequence, LoRA rank 16, and attention plus MLP adapter
targets. The forward pass consumed almost all available VRAM.

## New default profile

- `max_sequence_length: 2048`
- `image_min_pixels: 100352`
- `image_max_pixels: 200704`
- LoRA rank `8`, alpha `16`
- LoRA targets: `q_proj`, `v_proj`, `o_proj`
- vision tower and multimodal projector remain frozen
- SDPA attention
- batch size 1 with gradient accumulation
- newest two checkpoints plus the best checkpoint retained

The Colab notebook sets `PYTORCH_CUDA_ALLOC_CONF` before importing Torch,
normalizes legacy Windows image paths, patches the runtime configuration,
clears stale CUDA allocations, and runs a forward-and-backward smoke test.

Start the notebook in a fresh GPU runtime. Do not begin the 100-step overfit run
until `one_batch_smoke_test.json` reports finite loss, finite nonzero gradients,
and several GiB of VRAM headroom.
