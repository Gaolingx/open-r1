from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import loss_parallel


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


# CE Loss
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
    liger_loss_computer: Any,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, Any]:
    """Compute token-level next-token loss using the shared Liger CE loss computer."""

    if liger_loss_computer is None:
        raise RuntimeError("Liger CE loss computer is not initialized. Call configure_model() first.")
    return liger_loss_computer.compute_loss(
        batch=batch,
        labels=labels,
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
    nll_coeff: float = 0.0,
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

    # NLL regularization on chosen completions to prevent logps collapse
    nll_loss = torch.tensor(0.0, device=loss.device)
    if nll_coeff > 0.0:
        batch_size = shift_logits.size(0) // 2
        chosen_logits = shift_logits[:batch_size]
        chosen_labels = shift_labels[:batch_size]
        chosen_completion_mask = shift_completion_mask[:batch_size]
        # Mask non-completion tokens with ignore_index
        chosen_nll_labels = chosen_labels.clone()
        chosen_nll_labels[chosen_completion_mask == 0] = -100
        vocab_size = chosen_logits.size(-1)
        nll_loss = F.cross_entropy(
            chosen_logits.reshape(-1, vocab_size),
            chosen_nll_labels.reshape(-1),
            ignore_index=-100,
        )
        loss = loss + nll_coeff * nll_loss

    # Compute rewards for logging
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    metrics_dict = {
        "chosen_logps": chosen_logps.detach(),
        "rejected_logps": rejected_logps.detach(),
        "chosen_logits_mean": shift_logits[:shift_logits.size(0) // 2].mean().detach(),
        "rejected_logits_mean": shift_logits[shift_logits.size(0) // 2:].mean().detach(),
        "nll_loss": nll_loss.detach(),
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "_policy_outputs": outputs,
    }

    return loss, metrics_dict


## GRPO
def compute_grpo_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    advantage_epsilon: float = 1.0e-4,
) -> torch.Tensor:
    """Normalize rewards within each prompt group to produce GRPO advantages."""

    if num_generations <= 1:
        return torch.zeros_like(rewards)
    grouped_rewards = rewards.view(-1, num_generations)
    grouped_mean = grouped_rewards.mean(dim=1, keepdim=True)
    grouped_std = grouped_rewards.std(dim=1, keepdim=True)
    grouped_advantages = (grouped_rewards - grouped_mean) / (grouped_std + advantage_epsilon)
    return grouped_advantages.reshape(-1)


def compute_grpo_per_token_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor | None,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    ref_per_token_logps: torch.Tensor | None = None,
    beta: float = 0.04,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    loss_type: str = "bnpo",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute token-level GRPO losses and diagnostics with regular PyTorch ops.

    This mirrors the Liger fused GRPO kernel for the loss types used by this
    project: ``grpo``/``bnpo`` use PPO-style clipped ratios, while ``cispo``
    clips detached importance weights and multiplies them by current log-probs.
    """

    if loss_type == "grpo":
        loss_type = "bnpo"
    if loss_type not in {"bnpo", "cispo"}:
        raise ValueError(f"Unknown GRPO loss_type: {loss_type}")

    if ref_per_token_logps is None:
        ref_per_token_logps = per_token_logps.detach()
    if old_per_token_logps is None:
        old_per_token_logps = per_token_logps.detach()

    mask = completion_mask.to(per_token_logps.dtype)
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)
    expanded_advantages = advantages.unsqueeze(1)

    if loss_type == "cispo":
        clipped_ratio = torch.clamp(ratio, max=epsilon_high)
        per_token_policy_loss = -clipped_ratio.detach() * expanded_advantages * per_token_logps
    else:
        clipped_ratio = torch.clamp(ratio, min=1.0 - epsilon_low, max=1.0 + epsilon_high)
        unclipped_loss = ratio * expanded_advantages
        clipped_loss = clipped_ratio * expanded_advantages
        per_token_policy_loss = -torch.min(unclipped_loss, clipped_loss)

    per_token_kl = approx_kl_divergence(per_token_logps, ref_per_token_logps)
    per_token_loss = per_token_policy_loss + beta * per_token_kl

    is_low_clipped = (ratio < 1.0 - epsilon_low) & (expanded_advantages < 0)
    is_high_clipped = (ratio > 1.0 + epsilon_high) & (expanded_advantages > 0)
    is_region_clipped = is_low_clipped | is_high_clipped
    if loss_type == "cispo":
        is_low_clipped = torch.zeros_like(is_low_clipped)
        is_high_clipped = torch.zeros_like(is_high_clipped)
        is_region_clipped = torch.zeros_like(is_region_clipped)
        is_cispo_clipped = (ratio > epsilon_high) & (expanded_advantages > 0)
    else:
        is_cispo_clipped = torch.zeros_like(is_region_clipped)

    metrics = {
        "per_token_kl": per_token_kl.detach(),
        "is_low_clipped": is_low_clipped.to(mask.dtype),
        "is_high_clipped": is_high_clipped.to(mask.dtype),
        "is_region_clipped": is_region_clipped.to(mask.dtype),
        "is_cispo_clipped": is_cispo_clipped.to(mask.dtype),
    }
    return per_token_loss * mask, metrics


def reduce_grpo_loss(
    per_token_loss: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    loss_type: str = "bnpo",
    max_completion_length: int | None = None,
) -> torch.Tensor:
    """Reduce token-level GRPO loss using the same normalizers as Liger."""

    if loss_type == "grpo":
        loss_type = "bnpo"
    mask = completion_mask.to(per_token_loss.dtype)
    if loss_type == "bnpo":
        return per_token_loss.sum() / torch.clamp(mask.sum(), min=1.0)
    if loss_type == "cispo":
        return per_token_loss.sum() / torch.clamp(mask.sum(), min=1.0)
    if loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("max_completion_length must be provided for loss_type='dr_grpo'")
        return per_token_loss.sum() / (mask.shape[0] * max_completion_length)
    raise ValueError(f"Unknown GRPO loss_type: {loss_type}")


def compute_standard_grpo_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    *,
    ref_model: torch.nn.Module | None = None,
    beta: float = 0.04,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    loss_type: str = "bnpo",
    temperature: float = 1.0,
    max_completion_length: int | None = None,
    loss_parallel_enabled: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a plain PyTorch GRPO loss without the Liger fused kernel."""

    prompt_ids = batch["prompt_ids"]
    prompt_mask = batch["prompt_mask"]
    completion_ids = batch["completion_ids"]
    completion_mask = batch["completion_mask"]
    old_per_token_logps = batch.get("old_per_token_logps")

    model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)

    outputs = model(input_ids=model_input_ids, attention_mask=model_attention_mask, use_cache=False, output_router_logits=True)
    logits = materialize_vocab_parallel_logits(outputs.logits)
    shift_logits = logits[:, :-1, :]
    completion_logits = shift_logits[:, -logits_to_keep:, :]
    if temperature != 1.0:
        completion_logits = completion_logits / temperature
    per_token_logps = selective_log_softmax(completion_logits, completion_ids)

    ref_per_token_logps = None
    if ref_model is not None:
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=model_input_ids, attention_mask=model_attention_mask, use_cache=False)
            ref_logits = materialize_vocab_parallel_logits(ref_outputs.logits)
            ref_shift_logits = ref_logits[:, :-1, :]
            ref_completion_logits = ref_shift_logits[:, -logits_to_keep:, :]
            if temperature != 1.0:
                ref_completion_logits = ref_completion_logits / temperature
            ref_per_token_logps = selective_log_softmax(ref_completion_logits, completion_ids)

    loss_mask = completion_mask
    if "tool_mask" in batch:
        loss_mask = completion_mask * batch["tool_mask"]

    epsilon_high = epsilon_high if loss_type == "cispo" else epsilon_low
    per_token_loss, loss_metrics = compute_grpo_per_token_loss(
        per_token_logps,
        old_per_token_logps,
        advantages,
        loss_mask,
        ref_per_token_logps=ref_per_token_logps,
        beta=beta,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        loss_type=loss_type,
    )
    loss = reduce_grpo_loss(
        per_token_loss,
        loss_mask,
        loss_type=loss_type,
        max_completion_length=max_completion_length,
    )

    with torch.no_grad():
        entropy = entropy_from_logits(completion_logits)
        completion_lengths = completion_mask.sum(dim=1).float()
        completion_truncated = batch.get("completion_truncated", completion_lengths >= completion_mask.shape[1])
        local_metrics = {
            "loss_mask": loss_mask.detach(),
            "per_token_kl": loss_metrics["per_token_kl"].detach(),
            "entropy": entropy.detach(),
            "completion_lengths": completion_lengths.detach(),
            "completion_truncated": completion_truncated.to(torch.float32).detach(),
            "is_low_clipped": loss_metrics["is_low_clipped"].detach(),
            "is_high_clipped": loss_metrics["is_high_clipped"].detach(),
            "is_region_clipped": loss_metrics["is_region_clipped"].detach(),
            "is_cispo_clipped": loss_metrics["is_cispo_clipped"].detach(),
        }
    local_metrics["_policy_outputs"] = outputs
    return loss, local_metrics


class StandardGRPOLossComputer:
    """Compute standard PyTorch GRPO loss and metrics with the same orchestration as Liger."""

    def __init__(
        self,
        module: Any,
        reward_manager: Any,
        metrics_aggregator: Any,
        *,
        rollout_temperature: float,
        loss_parallel_enabled: bool = False,
    ) -> None:

        self.module = module
        self.reward_manager = reward_manager
        self.metrics_aggregator = metrics_aggregator
        self.rollout_temperature = rollout_temperature
        self.loss_parallel_enabled = loss_parallel_enabled

    def compute_advantages(self, rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
        return compute_grpo_advantages(
            rewards,
            num_generations,
            advantage_epsilon=self.module.config.rollout.advantage_epsilon,
        )

    def normalize_grouped_advantages(
        self,
        *,
        local_rewards: torch.Tensor,
        global_rewards: torch.Tensor,
        local_sample_ids: torch.Tensor,
        global_sample_ids: torch.Tensor,
        num_generations: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize rewards by globally gathered prompt groups, matching the Liger GRPO loss."""

        if global_rewards.numel() % num_generations != 0:
            raise ValueError(
                f"Global rollout batch ({global_rewards.numel()}) must be divisible by num_generations "
                f"({num_generations}) so prompt groups can be normalized correctly."
            )

        grouped_sample_ids = global_sample_ids.view(-1, num_generations)
        if not torch.all(grouped_sample_ids == grouped_sample_ids[:, :1]):
            raise RuntimeError(
                "Distributed GRPO prompt groups are not contiguous after gather. "
                "Each prompt must contribute exactly num_generations completions before advantage normalization."
            )

        global_advantages = self.compute_advantages(global_rewards, num_generations)
        local_batch_size = local_rewards.numel()
        process_index = getattr(getattr(self.module, "trainer", None), "global_rank", 0)
        start = process_index * local_batch_size
        end = start + local_batch_size
        local_advantages = global_advantages[start:end]
        expected_sample_ids = global_sample_ids[start:end].to(local_sample_ids.device)
        if local_advantages.numel() != local_batch_size or not torch.equal(expected_sample_ids, local_sample_ids):
            raise RuntimeError(
                "Failed to recover this rank's advantages from the gathered reward tensor. "
                "Ensure every rank receives the same local rollout batch size."
            )
        return local_advantages, global_advantages

    def compute_loss(
        self,
        rollout_batch: dict[str, torch.Tensor | list[object]],
        *,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute standard GRPO loss and logged metrics for a rollout batch."""

        sample_ids = rollout_batch["sample_ids"]
        completion_truncated = rollout_batch["completion_truncated"]
        rewards, rewards_per_func = self.reward_manager.compute_rewards(
            prompts=rollout_batch["prompts"],
            completions=rollout_batch["completions"],
            completion_id_lists=rollout_batch["completion_id_lists"],
            metadata=rollout_batch["metadata"],
        )
        global_rewards_per_func = self.metrics_aggregator.gather_tensor(rewards_per_func.detach())
        global_sample_ids = self.metrics_aggregator.gather_tensor(sample_ids.detach())
        num_generations = self.module.rollout_coordinator.resolve_num_generations(training)
        reward_weights = self.reward_manager.reward_weight_tensor.to(global_rewards_per_func.device)
        global_rewards = (global_rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=-1)
        local_advantages, global_advantages = self.normalize_grouped_advantages(
            local_rewards=rewards.detach(),
            global_rewards=global_rewards,
            local_sample_ids=sample_ids.detach(),
            global_sample_ids=global_sample_ids,
            num_generations=num_generations,
        )

        config = self.module.config
        loss_type = "bnpo" if config.rollout.loss_type == "grpo" else config.rollout.loss_type
        epsilon_high = config.rollout.epsilon_high if config.rollout.loss_type == "cispo" else config.rollout.epsilon
        loss, local_metrics = compute_standard_grpo_loss(
            self.module.policy,
            rollout_batch,
            local_advantages.to(self.module.device),
            ref_model=self.module.reference_model,
            beta=config.rollout.kl_beta,
            epsilon_low=config.rollout.epsilon,
            epsilon_high=epsilon_high,
            loss_type=loss_type,
            temperature=self.rollout_temperature,
            max_completion_length=config.rollout.max_completion_length,
            loss_parallel_enabled=self.loss_parallel_enabled,
        )

        metrics = build_standard_grpo_training_metrics(
            self.metrics_aggregator,
            rewards_per_func=rewards_per_func,
            reward_weights=reward_weights,
            num_generations=num_generations,
            local_metrics={**local_metrics, "completion_truncated": completion_truncated.to(torch.float32)},
            global_advantages=global_advantages,
            reward_names=config.reward.reward_funcs,
            moe_outputs=local_metrics.get("_policy_outputs"),
        )

        return loss, metrics


def build_standard_grpo_training_metrics(
    metrics_aggregator: Any,
    *,
    rewards_per_func: torch.Tensor,
    reward_weights: torch.Tensor,
    num_generations: int,
    local_metrics: dict[str, torch.Tensor],
    global_advantages: torch.Tensor,
    reward_names: list[str],
    moe_outputs: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Gather local standard-GRPO diagnostics and build the logged metrics dict."""

    global_rewards_per_func = metrics_aggregator.gather_tensor(rewards_per_func.detach())
    global_loss_mask = metrics_aggregator.gather_tensor(local_metrics["loss_mask"].detach())
    global_per_token_kl = metrics_aggregator.gather_tensor(local_metrics["per_token_kl"].detach())
    global_entropy = metrics_aggregator.gather_tensor(local_metrics["entropy"].detach())
    global_completion_lengths = metrics_aggregator.gather_tensor(local_metrics["completion_lengths"].detach())
    global_completion_truncated = metrics_aggregator.gather_tensor(local_metrics["completion_truncated"].detach())
    global_is_low_clipped = metrics_aggregator.gather_tensor(local_metrics["is_low_clipped"].detach())
    global_is_high_clipped = metrics_aggregator.gather_tensor(local_metrics["is_high_clipped"].detach())
    global_is_region_clipped = metrics_aggregator.gather_tensor(local_metrics["is_region_clipped"].detach())
    global_is_cispo_clipped = metrics_aggregator.gather_tensor(local_metrics["is_cispo_clipped"].detach())

    return metrics_aggregator.build_training_metrics(
        global_rewards_per_func=global_rewards_per_func,
        reward_weights=reward_weights,
        num_generations=num_generations,
        global_per_token_kl=global_per_token_kl,
        global_loss_mask=global_loss_mask,
        global_entropy=global_entropy,
        global_completion_lengths=global_completion_lengths,
        global_completion_truncated=global_completion_truncated,
        global_is_low_clipped=global_is_low_clipped,
        global_is_high_clipped=global_is_high_clipped,
        global_is_region_clipped=global_is_region_clipped,
        global_is_cispo_clipped=global_is_cispo_clipped,
        global_advantages=global_advantages,
        reward_names=reward_names,
        moe_outputs=moe_outputs,
    )
