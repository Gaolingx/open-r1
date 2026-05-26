from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import loss_parallel


# CE Loss
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


def tensor_parallel_loss_context(enabled: bool) -> Any:
    """Return the PyTorch loss-parallel context when vocab logits are sharded."""

    return loss_parallel() if enabled else nullcontext()


def materialize_vocab_parallel_logits(logits: torch.Tensor) -> torch.Tensor:
    """Gather DTensor vocabulary-sharded logits for metrics that require full vocab tensors."""

    if isinstance(logits, DTensor):
        return logits.redistribute(placements=[Replicate()]).to_local()
    return logits


def masked_token_stats(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> dict[str, torch.Tensor]:
    """Compute reusable masked token-level metrics for LM training."""

    logits = materialize_vocab_parallel_logits(logits)
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


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_parallel_enabled: bool = False,
) -> torch.Tensor:
    """Compute token-level next-token loss with optional label smoothing."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)

    with tensor_parallel_loss_context(loss_parallel_enabled):
        return F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )


def compute_liger_cross_entropy_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_parallel_enabled: bool = False,
) -> tuple[torch.Tensor, Any]:
    """Compute token-level next-token loss using Liger Kernel's fused kernel."""
    from lightning_grpo.models.grpo.liger_loss import LigerCELossComputer

    loss_computer = LigerCELossComputer(model, loss_parallel_enabled=loss_parallel_enabled)
    return loss_computer.compute_loss(
        batch=batch,
        labels=labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


# DPO Loss
def _dpo_loss(
    loss_type: str,
    beta: float,
    chosen_logratios: torch.Tensor,
    rejected_logratios: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the DPO loss given log-ratios."""

    delta_score = chosen_logratios - rejected_logratios

    if loss_type == "sigmoid":
        loss = -torch.nn.functional.logsigmoid(beta * delta_score)
    elif loss_type == "hinge":
        loss = torch.relu(1 - beta * delta_score)
    elif loss_type == "ipo":
        # IPO normalizes by completion length
        chosen_mask, rejected_mask = completion_mask.chunk(2, dim=0)
        chosen_avg = chosen_logratios / chosen_mask.sum(dim=1).clamp(min=1.0)
        rejected_avg = rejected_logratios / rejected_mask.sum(dim=1).clamp(min=1.0)
        ipo_delta = chosen_avg - rejected_avg
        loss = (ipo_delta - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"Unknown DPO loss_type: {loss_type}")

    return loss.mean()


def compute_liger_dpo_loss(
    liger_loss_computer: Any,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute DPO loss using the shared Liger loss computer."""

    if liger_loss_computer is None:
        raise RuntimeError("Liger DPO loss computer is not initialized. Call configure_model() first.")
    return liger_loss_computer.compute_loss(batch)


def compute_standard_dpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    beta: float,
    loss_type: str,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute DPO loss using standard logits computation (fallback when Liger is disabled)."""

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    completion_mask = batch["completion_mask"]

    # Forward through policy model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_completion_mask = completion_mask[..., 1:].contiguous()

    # Compute per-token log-probabilities
    per_token_logps = selective_log_softmax(shift_logits, shift_labels)
    per_token_logps[shift_completion_mask == 0] = 0.0

    # Sum log-probs over sequence
    logps = per_token_logps.sum(dim=1)
    chosen_logps, rejected_logps = logps.chunk(2, dim=0)

    # Reference model forward (no gradients)
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
        ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
        ref_per_token_logps[shift_completion_mask == 0] = 0.0
        ref_logps = ref_per_token_logps.sum(dim=1)
        ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

    # Compute log-ratios
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps

    # Compute DPO loss based on loss_type
    loss = _dpo_loss(loss_type, beta, chosen_logratios, rejected_logratios, completion_mask)

    # Compute rewards for logging
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    metrics_dict = {
        "chosen_logps": chosen_logps.detach(),
        "rejected_logps": rejected_logps.detach(),
        "chosen_logits_mean": shift_logits[:shift_logits.size(0) // 2].mean().detach(),
        "rejected_logits_mean": shift_logits[shift_logits.size(0) // 2:].mean().detach(),
        "nll_loss": torch.tensor(0.0, device=loss.device),
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
    }

    return loss, metrics_dict
