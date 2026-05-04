"""Shared optimization helpers for Lightning modules."""

from __future__ import annotations

from typing import Any

import torch
from transformers import GenerationConfig, PreTrainedConfig
from transformers.optimization import get_scheduler

from lightning_grpo.utils.configs.base import OptimizationConfig


def build_optimizer(parameters: Any, optimization: OptimizationConfig) -> torch.optim.Optimizer:
    """Create the optimizer from configuration."""

    optimizer_config = optimization.optimizer
    optimizer_type = optimizer_config.type.lower()

    if optimizer_type in {"adamw8bit", "adam8bit"}:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ImportError("bitsandbytes is required for 8-bit optimizers.") from error

        optimizer_class = bnb.optim.AdamW8bit if optimizer_type == "adamw8bit" else bnb.optim.Adam8bit
        return optimizer_class(
            parameters,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            amsgrad=optimizer_config.amsgrad,
            foreach=optimizer_config.foreach,
            maximize=optimizer_config.maximize,
        )

    if optimizer_type == "adam":
        return torch.optim.Adam(
            parameters,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            amsgrad=optimizer_config.amsgrad,
            foreach=optimizer_config.foreach,
            maximize=optimizer_config.maximize,
        )

    if optimizer_type == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=optimizer_config.learning_rate,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
            dampening=optimizer_config.dampening,
            nesterov=optimizer_config.nesterov,
            foreach=optimizer_config.foreach,
            maximize=optimizer_config.maximize,
        )

    if optimizer_type == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=optimizer_config.learning_rate,
            alpha=optimizer_config.alpha,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            momentum=optimizer_config.momentum,
            centered=optimizer_config.centered,
            foreach=optimizer_config.foreach,
            maximize=optimizer_config.maximize,
        )

    if optimizer_type == "adagrad":
        return torch.optim.Adagrad(
            parameters,
            lr=optimizer_config.learning_rate,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            foreach=optimizer_config.foreach,
            maximize=optimizer_config.maximize,
        )

    raise ValueError(
        f"Unknown optimizer type: {optimizer_config.type}. Supported: adamw, adam, adamw8bit, adam8bit, sgd, rmsprop, adagrad."
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    optimization: OptimizationConfig,
    estimated_stepping_batches: int,
) -> dict[str, Any]:
    """Create a per-step learning rate scheduler."""
    scheduler_config = optimization.scheduler

    if optimization.max_steps and optimization.max_steps > 0:
        total_steps = optimization.max_steps
    else:
        total_steps = max(1, estimated_stepping_batches)

    num_warmup_steps = min(max(0, scheduler_config.warmup_steps), total_steps)
    scheduler_specific_kwargs = dict(scheduler_config.scheduler_specific_kwargs)

    scheduler = get_scheduler(
        name=scheduler_config.type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs or None,
    )

    return {
        "scheduler": scheduler,
        "interval": "step",
        "frequency": 1,
        "name": "lr",
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute a mask-aware mean."""

    mask = mask.to(values.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (values * mask).sum() / denom


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute token-level entropy from logits."""

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def approx_kl_divergence(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """Approximate KL divergence between policy and reference log-probabilities."""

    return torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1.0


def selective_log_softmax(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Gather token log-probabilities for target ids."""

    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


def masked_token_stats(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> dict[str, torch.Tensor]:
    """Compute reusable masked token-level metrics for LM training."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels != ignore_index
    if not mask.any():
        zero = shift_logits.new_tensor(0.0)
        return {
            "token_accuracy": zero,
            "entropy": zero,
            "mean_logprob": zero,
            "perplexity": torch.exp(zero),
        }

    predictions = shift_logits.argmax(dim=-1)
    token_accuracy = (((predictions == shift_labels) & mask).sum().to(dtype=torch.float32) / mask.sum().to(dtype=torch.float32))
    entropy = masked_mean(entropy_from_logits(shift_logits), mask)
    per_token_logps = selective_log_softmax(shift_logits, shift_labels.masked_fill(~mask, 0))
    mean_logprob = masked_mean(per_token_logps, mask)
    perplexity = torch.exp(-mean_logprob)
    return {
        "token_accuracy": token_accuracy,
        "entropy": entropy,
        "mean_logprob": mean_logprob,
        "perplexity": perplexity,
    }


def resolve_generation_config(generation_config_path: str, model_config: PreTrainedConfig) -> GenerationConfig:
    """Resolve the generation config that should be saved with final exports."""

    if generation_config_path:
        return GenerationConfig.from_pretrained(generation_config_path)
    elif model_config:
        return GenerationConfig.from_model_config(model_config)
    else:
        raise ValueError("generation_config must be provided when generation_config_path is empty")
