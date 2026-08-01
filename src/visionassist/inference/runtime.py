"""Runtime and reproducibility metadata for model inference."""

from __future__ import annotations

import platform
import sys
from typing import Any


def runtime_metadata() -> dict[str, Any]:
    """Collect software and accelerator details without requiring CUDA."""

    metadata: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        metadata.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            metadata.update(
                {
                    "gpu_name": torch.cuda.get_device_name(device),
                    "gpu_total_memory_bytes": properties.total_memory,
                    "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    except ImportError:
        metadata["torch_available"] = False

    for module_name in ("transformers", "accelerate", "bitsandbytes", "PIL"):
        try:
            module = __import__(module_name)
            metadata[f"{module_name}_version"] = getattr(module, "__version__", None)
        except ImportError:
            metadata[f"{module_name}_version"] = None
    return metadata
