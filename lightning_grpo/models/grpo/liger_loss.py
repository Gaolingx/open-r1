"""Liger Kernel fused GRPO loss for memory-efficient training.

This module wraps the LigerFusedLinearGRPOLoss kernel which fuses the LM head
linear projection with the GRPO policy gradient loss computation. By avoiding
materialization of the full vocabulary logits tensor, it can reduce peak VRAM
usage by 30-50% for large vocabulary models.

Reference: https://github.com/linkedin/Liger-Kernel
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch.distributed.tensor import DTensor, Replicate

from lightning_grpo.utils.modeling import collect_moe_metrics

logger = logging.getLogger(__name__)


def _materialize_liger_lm_head_parameter(
    tensor: torch.Tensor | None,
    *,
    parameter_name: str,
    loss_parallel_enabled: bool,
) -> torch.Tensor | None:
    """Return a regular Tensor for Liger fused kernels, gathering TP shards when needed."""

    if tensor is None:
        return None
    if not isinstance(tensor, DTensor):
        return tensor
    if loss_parallel_enabled:
        raise ValueError(
            "Liger fused loss is incompatible with tensor_parallel.loss_parallel=True. "
            "Disable either Liger Kernel fused loss or tensor-parallel loss parallelism."
        )
    return tensor.redistribute(placements=[Replicate()]).to_local()


def _materialize_liger_lm_head(
    lm_head: torch.nn.Module,
    *,
    loss_parallel_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gather DTensor lm_head parameters for Liger, or reject loss-parallel TP."""

    weight = _materialize_liger_lm_head_parameter(
        lm_head.weight,
        parameter_name="lm_head.weight",
        loss_parallel_enabled=loss_parallel_enabled,
    )
    bias = _materialize_liger_lm_head_parameter(
        getattr(lm_head, "bias", None),
        parameter_name="lm_head.bias",
        loss_parallel_enabled=loss_parallel_enabled,
    )
    return weight, bias


def is_liger_kernel_available() -> bool:
    """Check if liger-kernel is installed."""
    try:
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss  # noqa: F401
        return True
    except ImportError:
        return False


class LigerGRPOLossComputer:
    """Compute GRPO loss using Liger Kernel's fused linear + GRPO kernel.

    Instead of materializing the full [batch, seq, vocab] logits tensor, this
    kernel computes the loss in a chunked, fused manner that dramatically
    reduces peak memory usage. The trade-off is slightly higher compute due to
    recomputation, but the memory savings enable larger batch sizes or longer
    sequences.
    """

    def __init__(
        self,
        module: Any,
        reward_manager: Any,
        metrics_aggregator: Any,
        *,
        rollout_temperature: float,
        loss_parallel_enabled: bool = False,
    ) -> None:
        if not is_liger_kernel_available():
            raise ImportError(
                "Liger Kernel is required for fused GRPO loss. "
                "Install it with: pip install liger-kernel"
            )

        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

        self.module = module
        self.reward_manager = reward_manager
        self.metrics_aggregator = metrics_aggregator
        self.rollout_temperature = rollout_temperature
        self.loss_parallel_enabled = loss_parallel_enabled

        config = module.config
        self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
            beta=config.rollout.kl_beta,
            epsilon_low=config.rollout.epsilon,
            epsilon_high=config.rollout.epsilon_high,
            temperature=rollout_temperature,
            use_ref_model=config.rollout.use_reference_model,
            loss_type=config.rollout.loss_type,
            max_completion_length=config.rollout.max_completion_length,
        )

    def _get_last_hidden_state(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass to get last hidden state and MoE metrics without computing logits."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        moe_metrics = collect_moe_metrics(outputs)
        # Get the last hidden state and slice to keep only completion tokens
        last_hidden_state = outputs.hidden_states[-1]
        # We need logits_to_keep + 1 positions because we predict next token
        # Slice: keep positions that correspond to completion token predictions
        last_hidden_state = last_hidden_state[:, -(logits_to_keep + 1):-1, :]
        return last_hidden_state, moe_metrics

    def compute_advantages(self, rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
        """Compute group-relative advantages."""
        grouped_rewards = rewards.view(-1, num_generations)
        grouped_mean = grouped_rewards.mean(dim=1, keepdim=True)
        grouped_std = grouped_rewards.std(dim=1, keepdim=True)
        eps = self.module.config.rollout.advantage_epsilon
        grouped_advantages = (grouped_rewards - grouped_mean) / (grouped_std + eps)
        return grouped_advantages.reshape(-1)

    def normalize_grouped_advantages(
        self,
        *,
        local_rewards: torch.Tensor,
        global_rewards: torch.Tensor,
        local_sample_ids: torch.Tensor,
        global_sample_ids: torch.Tensor,
        num_generations: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize rewards by prompt groups gathered across all ranks."""
        if global_rewards.numel() % num_generations != 0:
            raise ValueError(
                f"Global rollout batch ({global_rewards.numel()}) must be divisible by "
                f"num_generations ({num_generations})."
            )

        grouped_sample_ids = global_sample_ids.view(-1, num_generations)
        if not torch.all(grouped_sample_ids == grouped_sample_ids[:, :1]):
            raise RuntimeError(
                "Distributed GRPO prompt groups are not contiguous after gather."
            )

        global_advantages = self.compute_advantages(global_rewards, num_generations)
        local_batch_size = local_rewards.numel()
        process_index = getattr(getattr(self.module, "trainer", None), "global_rank", 0)
        start = process_index * local_batch_size
        end = start + local_batch_size
        local_advantages = global_advantages[start:end]
        return local_advantages, global_advantages

    def compute_loss(
        self,
        rollout_batch: dict[str, torch.Tensor | list[Any]],
        *,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute GRPO loss using Liger Kernel fused operation.

        The fused kernel combines the LM head projection and loss computation
        in a single pass, avoiding full logits materialization.
        """
        from lightning_grpo.models.rollout_engine import compute_per_token_logps

        prompt_ids = rollout_batch["prompt_ids"]
        prompt_mask = rollout_batch["prompt_mask"]
        completion_ids = rollout_batch["completion_ids"]
        completion_mask = rollout_batch["completion_mask"]
        old_per_token_logps = rollout_batch["old_per_token_logps"]
        sample_ids = rollout_batch["sample_ids"]

        model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.shape[1]

        # Get the unwrapped model for accessing lm_head
        policy = self.module.policy
        unwrapped = policy.module if hasattr(policy, "module") else policy

        # Get last hidden state (without computing logits)
        last_hidden_state, moe_metrics = self._get_last_hidden_state(
            policy, model_input_ids, model_attention_mask, logits_to_keep
        )

        # Compute reference log-probs if needed
        ref_per_token_logps: Optional[torch.Tensor] = None
        if self.module.reference_model is not None:
            with torch.no_grad():
                ref_per_token_logps = compute_per_token_logps(
                    self.module.reference_model,
                    model_input_ids,
                    logits_to_keep,
                    attention_mask=model_attention_mask,
                    temperature=self.rollout_temperature,
                )

        # Compute rewards and advantages
        rewards, rewards_per_func = self.reward_manager.compute_rewards(
            prompts=rollout_batch["prompts"],
            completions=rollout_batch["completions"],
            completion_id_lists=rollout_batch["completion_id_lists"],
            metadata=rollout_batch["metadata"],
        )
        global_rewards_per_func = self.metrics_aggregator.gather_tensor(rewards_per_func.detach())
        global_sample_ids = self.metrics_aggregator.gather_tensor(sample_ids.detach())
        num_generations = self.module.rollout_coordinator.resolve_num_generations(training)
        global_rewards = (
            global_rewards_per_func * self.module.reward_weights.to(global_rewards_per_func.device).unsqueeze(0)
        ).nansum(dim=-1)

        local_advantages, global_advantages = self.normalize_grouped_advantages(
            local_rewards=rewards.detach(),
            global_rewards=global_rewards,
            local_sample_ids=sample_ids.detach(),
            global_sample_ids=global_sample_ids,
            num_generations=num_generations,
        )
        advantages = local_advantages.to(last_hidden_state.device)

        # Apply tool_mask if present (multi-turn training)
        loss_mask = completion_mask
        if "tool_mask" in rollout_batch:
            loss_mask = completion_mask * rollout_batch["tool_mask"]

        # Compute fused loss via Liger Kernel
        lm_head_weight, lm_head_bias = _materialize_liger_lm_head(
            unwrapped.lm_head,
            loss_parallel_enabled=self.loss_parallel_enabled,
        )
        loss, liger_metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=lm_head_weight,
            selected_token_ids=completion_ids,
            attention_mask=loss_mask,
            advantages=advantages,
            bias=lm_head_bias,
            old_per_token_logps=old_per_token_logps,
            ref_per_token_logps=ref_per_token_logps,
        )

        # Build metrics dict compatible with the standard metrics aggregator
        with torch.no_grad():
            mean_kl = liger_metrics[0] if self.module.config.rollout.use_reference_model else torch.tensor(0.0)
            clip_ratio = liger_metrics[-1]

            completion_lengths = completion_mask.sum(dim=1).float()
            global_completion_lengths = self.metrics_aggregator.gather_tensor(completion_lengths.detach())
            completion_truncated = rollout_batch.get(
                "completion_truncated",
                torch.zeros(completion_ids.size(0), dtype=torch.bool, device=completion_ids.device),
            )
            global_completion_truncated = self.metrics_aggregator.gather_tensor(completion_truncated.to(torch.float32))

        metrics = {
            "kl": self.metrics_aggregator.gather_tensor(mean_kl.detach()).mean() if mean_kl is not None else torch.tensor(0.0),
            "clip_ratio": self.metrics_aggregator.gather_tensor(clip_ratio.detach()).mean(),
            "reward": global_rewards.mean(),
            "reward_std": global_rewards.std(),
            "advantage_mean": global_advantages.mean(),
            "advantage_std": global_advantages.std(),
            "completion_length": global_completion_lengths.mean(),
            "completion_length_min": global_completion_lengths.min(),
            "completion_length_max": global_completion_lengths.max(),
            "completion_clipped_ratio": global_completion_truncated.mean(),
        }

        # Add per-function reward metrics
        reward_names = self.module.config.reward.active.reward_funcs
        for i, name in enumerate(reward_names):
            func_rewards = global_rewards_per_func[:, i]
            metrics[f"reward/{name}"] = func_rewards.nanmean()

        metrics.update(moe_metrics)

        return loss, metrics


class LigerCELossComputer:
    """Compute cross entropy loss using Liger Kernel's fused linear + CELoss kernel."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        loss_parallel_enabled: bool = False,
    ) -> None:
        if not is_liger_kernel_available():
            raise ImportError(
                "Liger Kernel is required for fused cross entropy loss. "
                "Install it with: pip install liger-kernel"
            )

        self.model = model
        self.loss_parallel_enabled = loss_parallel_enabled

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Compute CE loss using Liger Kernel's fused linear + cross-entropy.

        Instead of materializing the full [batch*seq, vocab] logits tensor, this
        kernel fuses the LM head projection with the cross-entropy computation in
        a chunked manner, reducing peak VRAM by ~30-50% for large vocabularies.

        Args:
            model: The language model (must have a `lm_head` attribute).
            hidden_states: Last hidden states from the model, shape [B, S, H].
            labels: Target token IDs, shape [B, S]. Uses ignore_index for masked positions.
            ignore_index: Label value to ignore in loss computation.
            label_smoothing: Label smoothing factor.

        Returns:
            Scalar loss tensor.
        """
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        # Shift for next-token prediction: hidden_states[:-1] predicts labels[1:]
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Reshape to 2D for the fused kernel
        batch_seq = shift_hidden.shape[0] * shift_hidden.shape[1]
        hidden_dim = shift_hidden.shape[-1]
        shift_hidden_2d = shift_hidden.reshape(batch_seq, hidden_dim)
        shift_labels_1d = shift_labels.reshape(batch_seq)

        # Get LM head weight (and optional bias)
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        lm_head = unwrapped.lm_head
        weight, bias = _materialize_liger_lm_head(
            lm_head,
            loss_parallel_enabled=self.loss_parallel_enabled,
        )

        # Use Liger fused kernel
        loss_fn = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        loss = loss_fn(shift_hidden_2d, weight, shift_labels_1d, bias)
        return loss
