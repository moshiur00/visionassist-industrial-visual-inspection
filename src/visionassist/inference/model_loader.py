"""Load Qwen2.5-VL for reproducible baseline inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visionassist.inference.schemas import InferenceConfig


@dataclass(frozen=True)
class LoadedInferenceModel:
    """Model, processor, and resolved loading decisions."""

    model: Any
    processor: Any
    resolved_precision: str
    quantized_4bit: bool
    adapter_path: str | None = None


def _resolve_dtype(config: InferenceConfig, torch: Any) -> tuple[Any, str]:
    if config.precision == "bfloat16":
        return torch.bfloat16, "bfloat16"
    if config.precision == "float16":
        return torch.float16, "float16"
    if config.precision == "float32":
        return torch.float32, "float32"
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    return torch.float32, "float32"


def load_qwen25vl(config: InferenceConfig) -> LoadedInferenceModel:
    """Load the untouched Qwen2.5-VL checkpoint and processor."""

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Install inference dependencies with: uv sync --extra training"
        ) from exc

    dtype, resolved_precision = _resolve_dtype(config, torch)
    processor_kwargs: dict[str, Any] = {
        "revision": config.processor_revision or config.model_revision,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.image_min_pixels is not None:
        processor_kwargs["min_pixels"] = config.image_min_pixels
    if config.image_max_pixels is not None:
        processor_kwargs["max_pixels"] = config.image_max_pixels
    processor = AutoProcessor.from_pretrained(config.model_id, **processor_kwargs)

    model_kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "torch_dtype": dtype,
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attention_implementation != "auto":
        model_kwargs["attn_implementation"] = config.attention_implementation

    if config.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("4-bit loading requires bitsandbytes.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        **model_kwargs,
    )
    if config.adapter_path is not None:
        if not config.adapter_path.is_dir():
            raise FileNotFoundError(
                f"Adapter checkpoint not found: {config.adapter_path}"
            )
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Adapter inference requires PEFT.") from exc
        model = PeftModel.from_pretrained(
            model, config.adapter_path, is_trainable=False
        )
    model.eval()
    return LoadedInferenceModel(
        model=model,
        processor=processor,
        resolved_precision=resolved_precision,
        quantized_4bit=config.load_in_4bit,
        adapter_path=str(config.adapter_path) if config.adapter_path else None,
    )
