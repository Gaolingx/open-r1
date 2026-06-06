"""Metrics helper for the Lightning GRPO pipeline."""

from __future__ import annotations

from typing import Any, Optional

import lightning as L
import torch
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
from lightning.pytorch.utilities import rank_zero_warn


def _get_output_value(outputs: Any, key: str) -> Any:
    if isinstance(outputs, dict):
        return outputs.get(key)
    return getattr(outputs, key, None)


def _unwrap_model(model: Any) -> Any:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


class MoEAuxLossComputer:
    """Compute MoE router auxiliary loss from captured router logits."""

    def __init__(self, model: Any) -> None:
        self.model = _unwrap_model(model)
        self.config = getattr(self.model, "config", None)
        self.aux_loss_enabled = getattr(self.config, "output_router_logits", False)
        self.aux_loss_coef = getattr(self.config, "router_aux_loss_coef", 0.0)
        self.num_experts = getattr(self.config, "num_experts", 0)
        self.num_experts_per_tok = getattr(self.config, "num_experts_per_tok", 0)

        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            rank_zero_warn(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
            )

    def compute(self, outputs: Any, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Return weighted aux loss for optimization and detached aux metric for logging."""

        metrics: dict[str, torch.Tensor] = {}
        if not self.aux_loss_enabled or self.aux_loss_coef == 0.0:
            return None, metrics

        router_logits = _get_output_value(outputs, "router_logits")
        aux_loss = _get_output_value(outputs, "aux_loss")

        if aux_loss is None and router_logits is not None:
            aux_loss = load_balancing_loss_func(
                router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )

        if aux_loss is None or not torch.is_tensor(aux_loss):
            return None, metrics

        metrics["aux_loss"] = aux_loss.detach().to(dtype=torch.float32)
        return self.aux_loss_coef * aux_loss, metrics


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
        "aux_loss",
        "router_entropy",
        "expert_load_std",
        "top1_expert_occupancy",
        "dead_expert_fraction",
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
