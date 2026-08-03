"""Qwen2.5-VL QLoRA model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visionassist.training.config import Phase8TrainingConfig
from visionassist.training.hardware import HardwareInfo


@dataclass(frozen=True)
class ModelBuildResult:
    """Loaded processor/model plus parameter accounting."""

    model: Any
    processor: Any
    trainable_parameters: int
    total_parameters: int
    target_modules: list[str]
    compute_dtype: str


def _dtype_name(configured: str, hardware: HardwareInfo) -> str:
    if configured != "auto":
        return configured
    return "bfloat16" if hardware.bf16_supported else "float16"


def _torch_dtype(name: str, torch: Any) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _language_target_modules(model: Any, suffixes: list[str]) -> list[str]:
    """Discover exact language-model linear names, excluding the vision tower."""

    found: list[str] = []
    suffix_set = set(suffixes)
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Linear" and not name.endswith(tuple(suffixes)):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in suffix_set:
            continue
        lower = name.lower()
        if "visual" in lower or "vision" in lower:
            continue
        if "language_model" in lower or ".model.layers." in lower:
            found.append(name)
    if not found:
        raise RuntimeError(
            "No language-model LoRA targets were discovered. Inspect "
            "model.named_modules() before changing target policy."
        )
    return sorted(set(found))


def build_qlora_model(
    config: Phase8TrainingConfig,
    hardware: HardwareInfo,
) -> ModelBuildResult:
    """Load Qwen2.5-VL in 4-bit and attach LoRA to language layers only."""

    if not hardware.cuda_available:
        raise RuntimeError(
            "QLoRA training requires a CUDA GPU. Run this command in Google Colab Pro; "
            "the local PC can run tests and configuration validation only."
        )
    try:
        import torch
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the training extra: uv sync --extra training"
        ) from exc

    dtype_name = _dtype_name(config.quantization.compute_dtype, hardware)
    dtype = _torch_dtype(dtype_name, torch)
    quantization_config = None
    if config.quantization.enabled and config.quantization.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=dtype,
        )

    processor_kwargs: dict[str, object] = {
        "revision": config.processor_revision,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.data.image_min_pixels is not None:
        processor_kwargs["min_pixels"] = config.data.image_min_pixels
    if config.data.image_max_pixels is not None:
        processor_kwargs["max_pixels"] = config.data.image_max_pixels
    processor = AutoProcessor.from_pretrained(config.model_id, **processor_kwargs)

    model_kwargs: dict[str, object] = {
        "revision": config.model_revision,
        "trust_remote_code": config.trust_remote_code,
        "quantization_config": quantization_config,
        "device_map": "auto",
        "dtype": dtype,
    }
    if config.attention_implementation != "auto":
        model_kwargs["attn_implementation"] = config.attention_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        **model_kwargs,
    )
    model.config.use_cache = False
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )

    target_modules = _language_target_modules(model, config.lora.target_suffixes)
    if config.initial_adapter_path is not None:
        adapter_config = config.initial_adapter_path / "adapter_config.json"
        adapter_weights = config.initial_adapter_path / "adapter_model.safetensors"
        if not adapter_config.is_file() or not adapter_weights.is_file():
            raise FileNotFoundError(
                "initial_adapter_path requires adapter_config.json and "
                f"adapter_model.safetensors: {config.initial_adapter_path}"
            )
        model = PeftModel.from_pretrained(
            model,
            config.initial_adapter_path,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Explicitly prevent accidental vision-tower training.
    for name, parameter in model.named_parameters():
        if "visual" in name.lower() or "vision" in name.lower():
            parameter.requires_grad = False

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters were created.")
    return ModelBuildResult(
        model=model,
        processor=processor,
        trainable_parameters=trainable,
        total_parameters=total,
        target_modules=target_modules,
        compute_dtype=dtype_name,
    )
