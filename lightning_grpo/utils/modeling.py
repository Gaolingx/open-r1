"""Model and tokenizer utilities for the Lightning GRPO pipeline."""

from __future__ import annotations

from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from lightning_grpo.configs.base import ModelConfig, PrecisionConfig


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def resolve_torch_dtype(precision_config: PrecisionConfig) -> torch.dtype:
    """Resolve the parameter dtype from configuration."""

    return DTYPE_MAP[precision_config.parameter_dtype]


def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizerBase:
    """Load and normalize the tokenizer."""

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

    if not model_config.use_lora and not model_config.lora.enabled:
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
    return get_peft_model(model, lora_config)


def load_causal_lm(model_config: ModelConfig, precision_config: PrecisionConfig) -> PreTrainedModel:
    """Load a decoder-only language model with optional LoRA support."""

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=resolve_torch_dtype(precision_config),
        use_cache=not model_config.gradient_checkpointing,
    )

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
