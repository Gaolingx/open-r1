"""Liger Kernel fused GRPO loss for memory-efficient training.

This module wraps the LigerFusedLinearGRPOLoss kernel which fuses the LM head
linear projection with the GRPO policy gradient loss computation. By avoiding
materialization of the full vocabulary logits tensor, it can reduce peak VRAM
usage by 30-50% for large vocabulary models.

Reference: https://github.com/linkedin/Liger-Kernel
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate

from lightning_grpo.models.common import get_lm_head_model, get_transformer_backbone_model

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
        loss_type = "bnpo" if config.rollout.loss_type == "grpo" else config.rollout.loss_type
        epsilon_high = config.rollout.epsilon_high if config.rollout.loss_type == "cispo" else config.rollout.epsilon
        self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
            beta=config.rollout.kl_beta,
            epsilon_low=config.rollout.epsilon,
            epsilon_high=epsilon_high,
            temperature=rollout_temperature,
            use_ref_model=config.rollout.use_reference_model,
            loss_type=loss_type,
            max_completion_length=config.rollout.max_completion_length,
        )

    def compute_advantages(self, rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
        grouped_rewards = rewards.view(-1, num_generations)
        grouped_mean = grouped_rewards.mean(dim=1, keepdim=True)
        grouped_std = grouped_rewards.std(dim=1, keepdim=True)
        grouped_advantages = (grouped_rewards - grouped_mean) / (grouped_std + self.module.config.rollout.advantage_epsilon)
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
        """Normalize rewards by globally gathered prompt groups, matching the standard GRPO loss."""

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

    def _get_last_hidden_state(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        output_router_logits: bool = True,
    ) -> tuple[torch.Tensor, Any]:
        """Forward pass to get last hidden state without computing logits."""
        outputs = get_transformer_backbone_model(model)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_router_logits=output_router_logits,
        )
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]
        return last_hidden_state, outputs

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
        prompt_ids = rollout_batch["prompt_ids"]
        prompt_mask = rollout_batch["prompt_mask"]
        completion_ids = rollout_batch["completion_ids"]
        completion_mask = rollout_batch["completion_mask"]
        old_per_token_logps = rollout_batch["old_per_token_logps"]
        completion_truncated = rollout_batch["completion_truncated"]
        sample_ids = rollout_batch["sample_ids"]

        model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.shape[1]

        last_hidden_state, moe_metrics = self._get_last_hidden_state(
            self.module.policy,
            input_ids=model_input_ids,
            attention_mask=model_attention_mask,
            logits_to_keep=logits_to_keep,
        )
        last_hidden_state = last_hidden_state.contiguous()

        ref_per_token_logps = None
        ref_hidden_state = None
        ref_weight = None
        ref_bias = None
        if self.module.reference_model is not None:
            with torch.no_grad():
                ref_hidden_state, _ = self._get_last_hidden_state(
                    self.module.reference_model,
                    input_ids=model_input_ids,
                    attention_mask=model_attention_mask,
                    logits_to_keep=logits_to_keep,
                )
            ref_hidden_state = ref_hidden_state.contiguous()
            ref_weight, ref_bias = _materialize_liger_lm_head(
                get_lm_head_model(self.module.reference_model),
                loss_parallel_enabled=self.loss_parallel_enabled,
            )

        rewards, rewards_per_func = self.reward_manager.compute_rewards(
            prompts=rollout_batch["prompts"],
            completions=rollout_batch["completions"],
            completion_id_lists=rollout_batch["completion_id_lists"],
            metadata=rollout_batch["metadata"],
        )
        global_rewards_per_func = self.metrics_aggregator.gather_tensor(rewards_per_func.detach())
        global_sample_ids = self.metrics_aggregator.gather_tensor(sample_ids.detach())
        num_generations = self.module.rollout_coordinator.resolve_num_generations(training)
        global_rewards = (global_rewards_per_func * self.module.reward_weights.to(global_rewards_per_func.device).unsqueeze(0)).nansum(dim=-1)
        local_advantages, global_advantages = self.normalize_grouped_advantages(
            local_rewards=rewards.detach(),
            global_rewards=global_rewards,
            local_sample_ids=sample_ids.detach(),
            global_sample_ids=global_sample_ids,
            num_generations=num_generations,
        )
        advantages = local_advantages.to(last_hidden_state.device)

        loss_mask = completion_mask
        if "tool_mask" in rollout_batch:
            loss_mask = completion_mask * rollout_batch["tool_mask"]
        loss_mask = loss_mask.to(last_hidden_state.dtype).contiguous()

        weight, bias = _materialize_liger_lm_head(
            get_lm_head_model(self.module.policy),
            loss_parallel_enabled=self.loss_parallel_enabled,
        )
        loss, liger_metrics = self.liger_grpo_loss(
            last_hidden_state,
            weight,
            completion_ids.contiguous(),
            loss_mask,
            advantages,
            bias,
            ref_per_token_logps,
            old_per_token_logps.contiguous(),
            ref_hidden_state,
            ref_weight,
            ref_bias,
        )

        with torch.no_grad():
            mean_kl = liger_metrics[0] if self.module.config.rollout.kl_beta != 0.0 else completion_ids.new_tensor(0.0, dtype=torch.float32)
            clip_ratio = liger_metrics[-1]

            global_loss_mask = self.metrics_aggregator.gather_tensor(loss_mask.detach())
            mean_kl = mean_kl.detach().to(device=global_loss_mask.device, dtype=global_loss_mask.dtype).mean()
            global_per_token_kl = torch.zeros_like(global_loss_mask) + mean_kl
            global_entropy = torch.zeros_like(global_loss_mask)
            global_is_low_clipped = torch.zeros_like(global_loss_mask)
            global_is_high_clipped = torch.zeros_like(global_loss_mask)
            global_clip_ratio = self.metrics_aggregator.gather_tensor(clip_ratio.detach()).mean()
            global_is_region_clipped = torch.zeros_like(global_loss_mask) + global_clip_ratio
            global_is_cispo_clipped = (
                torch.zeros_like(global_loss_mask) + global_clip_ratio
                if self.module.config.rollout.loss_type == "cispo"
                else torch.zeros_like(global_loss_mask)
            )
            completion_lengths = completion_mask.sum(dim=1).float()
            global_completion_lengths = self.metrics_aggregator.gather_tensor(completion_lengths.detach())
            global_completion_truncated = self.metrics_aggregator.gather_tensor(completion_truncated.to(torch.float32))

        metrics = self.metrics_aggregator.build_training_metrics(
            global_rewards_per_func=global_rewards_per_func,
            reward_weights=self.module.reward_weights,
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
            reward_names=self.module.config.reward.active.reward_funcs,
            moe_outputs=moe_metrics,
        )

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

    def _get_last_hidden_state(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        output_router_logits: bool = True,
    ) -> tuple[torch.Tensor, Any]:
        """Forward pass to get last hidden state without computing logits."""
        outputs = get_transformer_backbone_model(model)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_router_logits=output_router_logits,
        )
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]
        return last_hidden_state, outputs

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> tuple[torch.Tensor, Any]:
        """Compute CE loss using Liger Kernel's fused linear + cross-entropy.

        Instead of materializing the full [batch*seq, vocab] logits tensor, this
        kernel fuses the LM head projection with the cross-entropy computation in
        a chunked manner, reducing peak VRAM by ~30-50% for large vocabularies.

        Args:
            batch: Input batch containing `input_ids` and `attention_mask`.
            labels: Target token IDs, shape [B, S]. Uses ignore_index for masked positions.
            ignore_index: Label value to ignore in loss computation.
            label_smoothing: Label smoothing factor.

        Returns:
            Tuple of scalar loss tensor and MoE metrics.
        """
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        logits_to_keep = labels.shape[1] - 1

        # Shift for next-token prediction: hidden_states[:-1] predicts labels[1:]
        shift_hidden, moe_metrics = self._get_last_hidden_state(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
        )
        shift_hidden = shift_hidden.contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Reshape to 2D for the fused kernel
        batch_seq = shift_hidden.shape[0] * shift_hidden.shape[1]
        hidden_dim = shift_hidden.shape[-1]
        shift_hidden_2d = shift_hidden.reshape(batch_seq, hidden_dim)
        shift_labels_1d = shift_labels.reshape(batch_seq)

        # Get LM head weight (and optional bias)
        weight, bias = _materialize_liger_lm_head(
            get_lm_head_model(self.model),
            loss_parallel_enabled=self.loss_parallel_enabled,
        )

        # Use Liger fused kernel
        loss_fn = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        loss = loss_fn(weight, shift_hidden_2d, shift_labels_1d, bias)
        return loss, moe_metrics
