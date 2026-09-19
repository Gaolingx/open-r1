"""Metrics helper for the Lightning GRPO pipeline."""

from __future__ import annotations

from typing import Any, Optional

import lightning as L
import torch


def _get_output_value(outputs: Any, key: str) -> Any:
    if isinstance(outputs, dict):
        return outputs.get(key)
    return getattr(outputs, key, None)


def format_metric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu().item())
    return float(value)


def collect_moe_metrics(outputs: Any, top_k: int = 1) -> dict[str, torch.Tensor]:
    """Extract aggregate MoE routing diagnostics from model outputs.

    These are router-*logit* level diagnostics. Realized (post-bias, post-grouped-top-k)
    expert load is tracked separately via ``tokens_per_expert`` by
    ``RouterBiasUpdateCallback``. Load balancing itself is handled by updating
    ``e_score_correction_bias`` instead of adding an auxiliary loss.

    Args:
        outputs: Model forward outputs containing ``router_logits``.
        top_k: Number of top experts selected per token (``num_experts_per_tok``).
    """

    metrics: dict[str, torch.Tensor] = {}

    if outputs is None:
        return metrics

    router_logits = _get_output_value(outputs, "router_logits")

    if router_logits is None:
        return metrics

    if torch.is_tensor(router_logits):
        router_logits = (router_logits,)
    elif not isinstance(router_logits, (tuple, list)):
        return metrics

    valid_router_logits = [layer_logits for layer_logits in router_logits if torch.is_tensor(layer_logits)]
    if not valid_router_logits:
        return metrics

    layer_entropies: list[torch.Tensor] = []
    layer_dead_expert_fractions: list[torch.Tensor] = []
    layer_load_imbalance_means: list[torch.Tensor] = []
    layer_load_imbalance_maxs: list[torch.Tensor] = []

    for layer_logits in valid_router_logits:
        probs = layer_logits.detach().to(dtype=torch.float32)
        if probs.ndim == 0 or probs.shape[-1] == 0:
            continue

        probs = probs.reshape(-1, probs.shape[-1])
        if probs.numel() == 0:
            continue

        row_sums = probs.sum(dim=-1)
        is_probability_distribution = torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1.0e-4,
            rtol=1.0e-4,
        ) and torch.all(probs >= 0)
        if not is_probability_distribution:
            probs = torch.softmax(probs, dim=-1)

        entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=-1).mean()
        num_experts = probs.shape[-1]
        num_tokens = probs.shape[0]

        # Compute top-k: count each token's k selected experts via one-hot → sum
        _, selected_experts = torch.topk(probs, top_k, dim=-1)  # [tokens, top_k]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)  # [tokens, top_k, num_experts]
        expert_counts = expert_mask.sum(dim=(0, 1)).to(dtype=torch.float32)  # [num_experts]
        # Each token contributes top_k assignments; ideal = tokens * top_k / num_experts
        ideal_load = (num_tokens * top_k) / max(num_experts, 1)

        # Load imbalance: ratio of actual tokens per expert vs ideal uniform load
        load_ratios = expert_counts / max(ideal_load, 1.0)
        load_imbalance_mean = (load_ratios - 1.0).abs().mean()
        load_imbalance_max = load_ratios.max()
        dead_expert_fraction = (expert_counts == 0).to(dtype=torch.float32).mean()

        layer_entropies.append(entropy)
        layer_dead_expert_fractions.append(dead_expert_fraction)
        layer_load_imbalance_means.append(load_imbalance_mean)
        layer_load_imbalance_maxs.append(load_imbalance_max)

    if layer_entropies:
        metrics["router_entropy"] = torch.stack(layer_entropies).mean()
        metrics["dead_expert_fraction"] = torch.stack(layer_dead_expert_fractions).mean()
        metrics["load_imbalance_mean"] = torch.stack(layer_load_imbalance_means).mean()
        metrics["load_imbalance_max"] = torch.stack(layer_load_imbalance_maxs).mean()

    return metrics


def log_moe_metrics(
    module: L.LightningModule,
    metrics: dict[str, Any],
    stage: str,
    *,
    on_step: bool,
    on_epoch: bool = True,
    sync_dist: bool = True,
) -> None:
    """Log shared MoE diagnostics from a precomputed metric dict."""

    if not isinstance(metrics, dict):
        raise ValueError(
            f"log_moe_metrics expects a precomputed metrics dictionary, but got {type(metrics).__name__}. "
            "Please call `collect_moe_metrics(outputs)` beforehand and pass the resulting dictionary."
        )

    moe_keys = [
        "router_entropy",
        "dead_expert_fraction",
        "load_imbalance_mean",
        "load_imbalance_max",
    ]

    # Extract only MoE-related metrics to prevent logging generic metrics (e.g. ce_loss) with a 'moe_' prefix
    moe_metrics = {key: metrics[key] for key in moe_keys if metrics.get(key) is not None}

    if not moe_metrics:
        return

    log_kwargs = {"prog_bar": False, "on_step": on_step, "on_epoch": on_epoch, "sync_dist": sync_dist}

    for key, value in moe_metrics.items():
        formatted_value = format_metric_value(value)
        if formatted_value is not None:
            module.log(f"{stage}/moe_{key}", formatted_value, **log_kwargs)
