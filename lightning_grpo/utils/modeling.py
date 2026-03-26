"""Model and tokenizer utilities for the Lightning GRPO pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from lightning.pytorch.utilities import rank_zero_info
from lightning_grpo.module.minimind.model_minimind import MiniMindConfig, MiniMindForCausalLM, register_minimind_for_auto_class

from lightning_grpo.utils.config import load_json_config
from lightning_grpo.utils.configs.base import ModelConfig, PrecisionConfig

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

CustomModelBuilder = Callable[[ModelConfig, PrecisionConfig], PreTrainedModel]


def resolve_torch_dtype(precision_config: PrecisionConfig) -> torch.dtype:
    """Resolve the parameter dtype from configuration."""

    return DTYPE_MAP[precision_config.parameter_dtype]


def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizerBase:
    """Load and normalize the tokenizer."""

    if model_config.model_family == "minimind" and not model_config.tokenizer_name_or_path:
        raise ValueError("MiniMind requires `tokenizer_name_or_path` for tokenizer loading and HF export.")

    tokenizer_name = model_config.tokenizer_name_or_path or model_config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )

    if model_config.chat_template:
        tokenizer.chat_template = model_config.chat_template
    if model_config.eos_token:
        tokenizer.eos_token = model_config.eos_token
    if model_config.pad_token:
        tokenizer.pad_token = model_config.pad_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def _freeze_embeddings_if_needed(model: PreTrainedModel, freeze_embeddings: bool) -> None:
    """Freeze token embeddings for parameter-efficient adaptation."""

    if not freeze_embeddings:
        return

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        for parameter in input_embeddings.parameters():
            parameter.requires_grad = False

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        for parameter in output_embeddings.parameters():
            parameter.requires_grad = False


def _apply_lora_if_needed(model: PreTrainedModel, model_config: ModelConfig) -> PreTrainedModel:
    """Wrap the model with LoRA adapters when enabled."""

    if not model_config.lora.enabled:
        return model

    if model_config.lora.init_path:
        lora_path = Path(model_config.lora.init_path).expanduser()
        rank_zero_info(f"Loading LoRA adapters from {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=True)

        if model_config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        return model

    lora_config = LoraConfig(
        r=model_config.lora.r,
        lora_alpha=model_config.lora.alpha,
        lora_dropout=model_config.lora.dropout,
        bias=model_config.lora.bias,
        target_modules=model_config.lora.target_modules,
        modules_to_save=model_config.lora.modules_to_save or None,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if model_config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model


def _resolve_model_init_kwargs(model_config: ModelConfig) -> dict:
    """Resolve custom model init kwargs from JSON config and inline overrides."""

    init_kwargs: dict = {}
    if model_config.model_config_path:
        config_path = Path(model_config.model_config_path)
        raw_config = load_json_config(config_path)
        init_kwargs.update(raw_config)

    init_kwargs.update(model_config.model_init_kwargs)
    return init_kwargs


def _build_minimind_model(model_config: ModelConfig, precision_config: PrecisionConfig) -> PreTrainedModel:
    """Build a MiniMind model from local modules with optional checkpoint loading."""

    init_kwargs = _resolve_model_init_kwargs(model_config)
    init_kwargs.setdefault("hidden_size", 768)
    init_kwargs.setdefault("num_hidden_layers", 8)
    init_kwargs.setdefault("use_moe", False)

    config_cls = getattr(MiniMindForCausalLM, "config_class", None)
    if config_cls is None:
        raise ValueError("MiniMindForCausalLM.config_class is required for MiniMind initialization.")

    minimind_config = config_cls(**init_kwargs)
    model = MiniMindForCausalLM(minimind_config)
    model = model.to(dtype=resolve_torch_dtype(precision_config))

    if model_config.pretrained_weight and model_config.pretrained_weight.lower() != "none":
        moe_suffix = "_moe" if getattr(minimind_config, "use_moe", False) else ""
        hidden_size = getattr(minimind_config, "hidden_size", init_kwargs["hidden_size"])
        weight_path = Path(model_config.custom_weight_dir) / f"{model_config.pretrained_weight}_{hidden_size}{moe_suffix}.pth"
        weights = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(weights, strict=False)

    return model


CUSTOM_MODEL_BUILDERS: dict[str, CustomModelBuilder] = {
    "minimind": _build_minimind_model,
}


def load_causal_lm(model_config: ModelConfig, precision_config: PrecisionConfig) -> PreTrainedModel:
    """Load a decoder-only language model with optional LoRA support."""

    if model_config.model_family == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
            dtype=resolve_torch_dtype(precision_config),
        )
    else:
        builder = CUSTOM_MODEL_BUILDERS.get(model_config.model_family)
        if builder is None:
            raise ValueError(
                f"Unsupported model_family '{model_config.model_family}'. Expected one of {sorted(CUSTOM_MODEL_BUILDERS)} or 'auto'."
            )
        model = builder(model_config, precision_config)

    if hasattr(model, "config"):
        model.config.use_cache = model_config.use_cache

    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    _freeze_embeddings_if_needed(model, model_config.freeze_embeddings)
    model = _apply_lora_if_needed(model, model_config)

    if model_config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model


def count_trainable_parameters(model: PreTrainedModel) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def register_custom_model_builder(name: str, builder: CustomModelBuilder) -> None:
    """Register a custom model loader for future extensions."""

    CUSTOM_MODEL_BUILDERS[name] = builder


def describe_model_source(model_config: ModelConfig) -> str:
    """Describe the selected model source for logging."""

    if model_config.model_family == "auto":
        return model_config.model_name_or_path
    return f"{model_config.model_family}:{model_config.pretrained_weight or 'none'}"


def unwrap_model(model: PreTrainedModel) -> PreTrainedModel:
    return model.get_base_model() if isinstance(model, PeftModel) else model


def export_hf_model(
    model: PreTrainedModel,
    model_config: ModelConfig,
    export_dir: str | Path,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
    safe_serialization: bool = False,
) -> Path:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    save_model = model
    if isinstance(model, PeftModel):
        save_model = model
    else:
        save_model = unwrap_model(model)

    save_kwargs = {"safe_serialization": safe_serialization}
    if state_dict is not None:
        save_kwargs["state_dict"] = state_dict
    save_model.save_pretrained(str(export_path), **save_kwargs)

    resolved_tokenizer = tokenizer
    if resolved_tokenizer is None and model_config.tokenizer_name_or_path:
        resolved_tokenizer = load_tokenizer(model_config)
    if resolved_tokenizer is not None:
        resolved_tokenizer.save_pretrained(str(export_path))

    return export_path
