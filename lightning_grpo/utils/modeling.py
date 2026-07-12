"""Model and tokenizer utilities for the Lightning GRPO pipeline."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, GenerationConfig, PreTrainedModel, AutoConfig
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.models.common import resolve_torch_dtype
from lightning_grpo.utils.models.loader import build_configured_model_class
from lightning_grpo.utils.configs.base import ModelConfig, PrecisionConfig


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


def load_causal_lm(model_config: ModelConfig, precision_config: PrecisionConfig) -> PreTrainedModel:
    """Load a decoder-only language model with optional LoRA support."""

    if model_config.custom_model:
        model = build_configured_model_class(model_config, precision_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
            dtype=resolve_torch_dtype(precision_config.model_param_dtype),
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
