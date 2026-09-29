"""Masked tensor helpers vendored from ``verl.utils.torch_functional``.

Only the handful of helpers that :mod:`lightning_grpo.utils.loss.core_algos`
needs are implemented here so that the ported PPO/GRPO algorithm bodies stay
free of any ``verl`` runtime dependency.

Semantics intentionally match verl's implementations:

* ``masked_mean`` / ``masked_sum`` clamp the denominator at ``1.0`` (not at a
  tiny epsilon), so a fully masked row yields exactly ``0`` instead of ``nan``.
* ``masked_whiten`` uses the masked variance and ``rsqrt(var + 1e-8)``.
"""

from __future__ import annotations

import torch


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Sum ``values`` where ``mask`` is non-zero."""

    mask = mask.to(values.dtype)
    return (values * mask).sum(dim=dim)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Mask-aware mean of ``values``."""

    mask = mask.to(values.dtype)
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1.0)


def masked_var(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Mask-aware (biased) variance of ``values``."""

    mask = mask.to(values.dtype)
    mean = masked_mean(values, mask, dim=dim)
    centered = values - mean
    return masked_mean(centered * centered, mask, dim=dim)


def masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    shift_mean: bool = True,
    dim: int | None = None,
) -> torch.Tensor:
    """Standardize ``values`` using statistics gathered under ``mask``."""

    mean = masked_mean(values, mask, dim=dim)
    var = masked_var(values, mask, dim=dim)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy of ``logits`` along the last dimension."""

    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def clip_by_value(x: torch.Tensor, tensor_min: torch.Tensor, tensor_max: torch.Tensor) -> torch.Tensor:
    """Element-wise clip that broadcasts tensor bounds.

    This exists because ``torch.clamp`` requires the bounds to be scalars or
    tensors with a compatible shape; several vendored algorithms pass per-token
    bounds of shape ``(bs, 1)`` against a ``(bs, response_length)`` input.
    """

    return torch.max(torch.min(x, tensor_max), tensor_min)


__all__ = [
    "clip_by_value",
    "entropy_from_logits",
    "masked_mean",
    "masked_sum",
    "masked_var",
    "masked_whiten",
]
