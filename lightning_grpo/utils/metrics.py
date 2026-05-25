"""Metrics helper for the Lightning GRPO pipeline."""

from __future__ import annotations

from typing import Any, Optional

import lightning as L
import torch


@staticmethod
def format_metric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu().item())
    return float(value)


def collect_moe_metrics(outputs: Any) -> dict[str, torch.Tensor]:
    """Extract aggregate MoE routing diagnostics from model outputs."""

    metrics: dict[str, torch.Tensor] = {}

    def _get_output_value(outputs: Any, key: str) -> Any:
        if isinstance(outputs, dict):
            return outputs.get(key)
        return getattr(outputs, key, None)

    if outputs is None:
        return metrics

    router_logits = _get_output_value(outputs, "router_logits")
    aux_loss = _get_output_value(outputs, "aux_loss")

    if aux_loss is not None:
        metrics["aux_loss"] = aux_loss.detach().to(dtype=torch.float32)

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
    layer_load_std: list[torch.Tensor] = []
    layer_top1_occupancies: list[torch.Tensor] = []
    layer_dead_expert_fractions: list[torch.Tensor] = []

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

        mean_probs = probs.mean(dim=0)
        entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=-1).mean()
        load_std = mean_probs.std(unbiased=False)
        top1_experts = probs.argmax(dim=-1)
        num_experts = probs.shape[-1]
        top1_counts = torch.bincount(top1_experts, minlength=num_experts).to(dtype=torch.float32)
        top1_occupancy = (top1_counts / max(top1_experts.numel(), 1)).max()
        dead_expert_fraction = (top1_counts == 0).to(dtype=torch.float32).mean()

        layer_entropies.append(entropy)
        layer_load_std.append(load_std)
        layer_top1_occupancies.append(top1_occupancy)
        layer_dead_expert_fractions.append(dead_expert_fraction)

    if layer_entropies:
        metrics["router_entropy"] = torch.stack(layer_entropies).mean()
        metrics["expert_load_std"] = torch.stack(layer_load_std).mean()
        metrics["top1_expert_occupancy"] = torch.stack(layer_top1_occupancies).mean()
        metrics["dead_expert_fraction"] = torch.stack(layer_dead_expert_fractions).mean()

    return metrics


def log_moe_metrics(
    module: L.LightningModule,
    outputs_or_metrics: Any,
    stage: str,
    *,
    on_step: bool,
    on_epoch: bool = True,
    sync_dist: bool = True,
) -> None:
    """Log shared MoE diagnostics from raw outputs or a precomputed metric dict."""

    metrics = collect_moe_metrics(outputs_or_metrics)

    if isinstance(outputs_or_metrics, dict):
        get_metric = outputs_or_metrics.get
        metrics.update({
            "aux_loss": get_metric("aux_loss", metrics.get("aux_loss")),
            "router_entropy": get_metric("router_entropy", metrics.get("router_entropy")),
            "expert_load_std": get_metric("expert_load_std", metrics.get("expert_load_std")),
            "top1_expert_occupancy": get_metric("top1_expert_occupancy", metrics.get("top1_expert_occupancy")),
            "dead_expert_fraction": get_metric("dead_expert_fraction", metrics.get("dead_expert_fraction")),
        })
        metrics = {key: value for key, value in metrics.items() if value is not None}

    if not metrics:
        return

    log_kwargs = {"prog_bar": False, "on_step": on_step, "on_epoch": on_epoch, "sync_dist": sync_dist}

    for key, value in metrics.items():
        formatted_value = format_metric_value(value)
        if formatted_value is not None:
            module.log(f"{stage}/moe_{key}", formatted_value, **log_kwargs)
