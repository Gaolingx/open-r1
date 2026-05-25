"""Model and tokenizer utilities for the Lightning GRPO pipeline."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
from typing import Any

import lightning as L
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, GenerationConfig, PreTrainedModel
from lightning.pytorch.utilities import rank_zero_info
from transformers.configuration_utils import PreTrainedConfig

from lightning_grpo.models.common import resolve_torch_dtype
from lightning_grpo.utils.config import load_json_config
from lightning_grpo.utils.configs.base import ModelConfig, PrecisionConfig
from lightning_grpo.utils.reflection import import_causal_lm_class


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
        modules_to_save=model_config.lora.modules_to_save,
        task_type=TaskType.CAUSAL_LM,
        **model_config.lora.lora_kwargs,
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

    init_kwargs["attn_implementation"] = model_config.attn_implementation
    init_kwargs.update(model_config.model_init_kwargs)
    return init_kwargs


def _resolve_checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Normalize checkpoint containers to a plain state dict."""

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected checkpoint to resolve to a mapping-based state dict.")

    return state_dict


def _maybe_load_custom_weights(model: PreTrainedModel, model_config: ModelConfig) -> PreTrainedModel:
    """Load an optional local PyTorch checkpoint into a freshly built model."""

    if model_config.model_name_or_path is None:
        return model

    weight_path = Path(model_config.model_name_or_path).expanduser()
    if weight_path.suffix == "":
        pth_candidate = weight_path.with_suffix(".pth")
        if pth_candidate.exists():
            weight_path = pth_candidate

    if not weight_path.exists():
        raise FileNotFoundError(f"Custom checkpoint not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = _resolve_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model


def _build_configured_model_class(model_config: ModelConfig, precision_config: PrecisionConfig) -> PreTrainedModel:
    """Build a local model from an explicit YAML-configured class path."""

    init_kwargs = _resolve_model_init_kwargs(model_config)
    model_class = import_causal_lm_class(model_config.model_class_path or "")

    config_class = getattr(model_class, "config_class", None)
    if not inspect.isclass(config_class) or not issubclass(config_class, PreTrainedConfig):
        raise TypeError(
            f"Configured model class must expose a PreTrainedConfig `config_class`: {model_config.model_class_path}"
        )

    model_hf_config = config_class(**init_kwargs)
    model = model_class(model_hf_config)
    model = model.to(dtype=resolve_torch_dtype(precision_config))

    return _maybe_load_custom_weights(model, model_config)


def load_causal_lm(model_config: ModelConfig, precision_config: PrecisionConfig) -> PreTrainedModel:
    """Load a decoder-only language model with optional LoRA support."""

    if model_config.custom_model:
        model = _build_configured_model_class(model_config, precision_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
            dtype=resolve_torch_dtype(precision_config),
        )

    if model_config.model_generation_config_path:
        generation_config = GenerationConfig.from_pretrained(model_config.model_generation_config_path)
        model.generation_config = generation_config

    if hasattr(model, "config"):
        model.config.use_cache = model_config.use_cache

    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=model_config.gradient_checkpointing_kwargs)

    _freeze_embeddings_if_needed(model, model_config.freeze_embeddings)
    model = _apply_lora_if_needed(model, model_config)

    return model


def resolve_export_model(pl_module: L.LightningModule) -> torch.nn.Module | None:
    """Return the underlying trainable model that should be exported."""

    return getattr(pl_module, "policy", None) or getattr(pl_module, "model", None)
