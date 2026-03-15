"""Shared optimization helpers for Lightning modules."""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from lightning_grpo.configs.base import OptimizationConfig


def build_optimizer(parameters: Any, optimization: OptimizationConfig) -> torch.optim.Optimizer:
    """Create the optimizer from configuration."""

    if optimization.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ImportError("bitsandbytes is required for adamw8bit optimizer.") from error

        return bnb.optim.AdamW8bit(
            parameters,
            lr=optimization.learning_rate,
            betas=optimization.betas,
            eps=optimization.eps,
            weight_decay=optimization.weight_decay,
        )

    return torch.optim.AdamW(
        parameters,
        lr=optimization.learning_rate,
        betas=optimization.betas,
        eps=optimization.eps,
        weight_decay=optimization.weight_decay,
    )


def build_scheduler(
        optimizer: torch.optim.Optimizer,
        optimization: OptimizationConfig,
        estimated_stepping_batches: int,
) -> dict[str, Any]:
    """Create a per-step learning rate scheduler."""

    if optimization.max_steps and optimization.max_steps > 0:
        total_steps = optimization.max_steps
    else:
        total_steps = max(1, estimated_stepping_batches)

    if optimization.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=optimization.warmup_steps,
            num_training_steps=total_steps,
        )
    elif optimization.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=optimization.warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

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
