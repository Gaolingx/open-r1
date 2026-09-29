"""Liger Kernel fused GRPO loss for memory-efficient training.

This module wraps the LigerFusedLinearGRPOLoss kernel which fuses the LM head
linear projection with the GRPO policy gradient loss computation. By avoiding
materialization of the full vocabulary logits tensor, it can reduce peak VRAM
usage by 30-50% for large vocabulary models.

Reference: https://github.com/linkedin/Liger-Kernel
"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate

from lightning_grpo.models.common import get_lm_head_model, get_transformer_backbone_model
from lightning_grpo.models.grpo.loss import GRPOLossComputer
from lightning_grpo.models.grpo.metrics import GRPOMetricsAggregator
from lightning_grpo.models.grpo.reward import GRPORewardManager


def _materialize_liger_lm_head(
    lm_head: torch.nn.Module,
    *,
    loss_parallel_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gather lm_head parameters for Liger from DTensors (FSDP2 only)."""

    if loss_parallel_enabled:
        raise RuntimeError(
            "Liger fused loss is incompatible with Tensor Parallelism (loss_parallel=True). "
            "Please disable Tensor Parallelism when using Liger Kernel."
        )

    tp_plan = getattr(lm_head, "_hf_tp_plan", None)
    if tp_plan is not None:
        raise RuntimeError(
            f"Liger fused loss is incompatible with Tensor Parallelism. "
            f"You are currently using TP plan: {tp_plan}. "
            f"Please disable Tensor Parallelism (set tensor_parallel_size=1) when using Liger Kernel, "
            f"or disable Liger Kernel if you want to use TP."
        )

    weight = lm_head.weight
    bias = getattr(lm_head, "bias", None)

    if isinstance(weight, DTensor):
        weight = weight.full_tensor()
        if bias is not None and isinstance(bias, DTensor):
            bias = bias.full_tensor()

    return weight, bias


def _get_last_hidden_state(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    """Forward pass to get last hidden state without computing logits."""
    outputs = get_transformer_backbone_model(model)(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    last_hidden_state = outputs.last_hidden_state
    last_hidden_state = last_hidden_state[:, :-1, :]
    last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]
    return last_hidden_state


def compute_liger_sft_loss(
    model: torch.nn.Module,
    loss_type: str,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute token-level next-token loss using the patched Liger CE forward.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")

    # liger supports dft loss by just passing use_token_scaling=True
    use_token_scaling = (loss_type == "dft")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        use_token_scaling=use_token_scaling,
    )
    loss = outputs.loss

    return loss, {"_policy_outputs": outputs}


class LigerDPOLossComputer:
    """Compute DPO loss using Liger Kernel's fused linear + DPOLoss kernel."""

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        *,
        beta: float,
        loss_type: str,
        nll_coeff: float = 0.0,
        ignore_index: int = -100,
        loss_parallel_enabled: bool = False,
        compiled: bool = True,
    ) -> None:
        try:
            from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
        except ImportError as e:
            raise ImportError(
                "LigerFusedLinearDPOLoss requires liger-kernel. "
                "Install it with: pip install liger-kernel"
            ) from e

        self.model = model
        self.ref_model = ref_model
        self.nll_coeff = nll_coeff
        self.loss_parallel_enabled = loss_parallel_enabled
        self.loss_fn = LigerFusedLinearDPOLoss(
            beta=beta,
            loss_type=loss_type,
            ignore_index=ignore_index,
            compiled=compiled,
            compute_nll_loss=True,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute DPO loss without materializing full vocabulary logits."""

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        completion_mask = batch["completion_mask"]

        outputs = get_transformer_backbone_model(self.model)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state[:, :-1].contiguous()

        weight, bias = _materialize_liger_lm_head(
            get_lm_head_model(self.model),
            loss_parallel_enabled=self.loss_parallel_enabled,
        )

        with torch.no_grad():
            ref_outputs = get_transformer_backbone_model(self.ref_model)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            ref_hidden_states = ref_outputs.last_hidden_state[:, :-1].contiguous()

        ref_weight, ref_bias = _materialize_liger_lm_head(
            get_lm_head_model(self.ref_model),
            loss_parallel_enabled=self.loss_parallel_enabled,
        )

        shift_completion_mask = completion_mask[:, 1:].contiguous()
        labels = input_ids[:, 1:].clone()
        labels[shift_completion_mask == 0] = -100

        loss, metrics = self.loss_fn(weight, hidden_states, labels, bias, ref_hidden_states, ref_weight, ref_bias)
        (
            chosen_logps,
            rejected_logps,
            chosen_logits_mean,
            rejected_logits_mean,
            nll_loss,
            chosen_rewards,
            rejected_rewards,
        ) = metrics

        # Apply NLL regularization to prevent logps collapse
        if nll_loss is not None:
            loss = (loss - nll_loss) + self.nll_coeff * nll_loss

        return loss, {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits_mean": chosen_logits_mean,
            "rejected_logits_mean": rejected_logits_mean,
            "nll_loss": nll_loss if nll_loss is not None else torch.tensor(0.0, device=loss.device),
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
        }


#: Preset name -> ``LigerFusedLinearGRPOLoss`` ``loss_type``.
#:
#: The kernel implements a fixed family of clipped surrogates. DAPO reuses the GRPO
#: objective and only differs through its asymmetric clip bounds, which are forwarded
#: separately as ``epsilon_low``/``epsilon_high``.
LIGER_LOSS_TYPES: dict[str, str] = {
    "grpo": "grpo",
    "dapo": "grpo",
    "dr_grpo": "dr_grpo",
    "cispo": "cispo",
}


class LigerGRPOLossComputer(GRPOLossComputer):
    """Compute the GRPO objective with Liger Kernel's fused linear + GRPO kernel.

    Instead of materializing the full [batch, seq, vocab] logits tensor, this
    kernel computes the loss in a chunked, fused manner that dramatically
    reduces peak memory usage. The trade-off is slightly higher compute due to
    recomputation, but the memory savings enable larger batch sizes or longer
    sequences.

    Only the objectives in :data:`LIGER_LOSS_TYPES` are implemented by the kernel; the
    factory in ``loss.py`` falls back to the standard PyTorch path for anything else.
    """

    supported_loss_types = frozenset(LIGER_LOSS_TYPES)

    def __init__(
        self,
        module: Any,
        reward_manager: GRPORewardManager,
        metrics_aggregator: GRPOMetricsAggregator,
        *,
        rollout_temperature: float,
        loss_parallel_enabled: bool = False,
        compiled: bool = True,
    ) -> None:
        try:
            from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
        except ImportError as e:
            raise ImportError(
                "LigerFusedLinearGRPOLoss requires liger-kernel. "
                "Install it with: pip install liger-kernel"
            ) from e

        super().__init__(
            module,
            reward_manager,
            metrics_aggregator,
            rollout_temperature=rollout_temperature,
            loss_parallel_enabled=loss_parallel_enabled,
        )

        if self.algorithm.loss_type not in LIGER_LOSS_TYPES:
            raise RuntimeError(
                f"LigerFusedLinearGRPOLoss does not implement loss_type='{self.algorithm.loss_type}'. "
                f"Supported loss types: {sorted(LIGER_LOSS_TYPES)}. "
                "Use the standard PyTorch loss path instead."
            )

        config = module.config
        self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
            beta=config.rollout.kl_beta,
            compiled=compiled,
            epsilon_low=self.algorithm.epsilon_low,
            epsilon_high=self.algorithm.epsilon_high,
            temperature=rollout_temperature,
            use_ref_model=config.rollout.use_reference_model,
            loss_type=LIGER_LOSS_TYPES[self.algorithm.loss_type],
            max_completion_length=config.rollout.max_completion_length,
        )

    def _policy_loss(self, context: GRPOLossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate the objective through the fused kernel (no logits materialization)."""

        rollout_batch = context.rollout_batch
        prompt_ids = rollout_batch["prompt_ids"]
        prompt_mask = rollout_batch["prompt_mask"]
        completion_ids = rollout_batch["completion_ids"]
        completion_mask = rollout_batch["completion_mask"]
        old_per_token_logps = rollout_batch["old_per_token_logps"]

        model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.shape[1]

        last_hidden_state = _get_last_hidden_state(
            self.module.policy,
            input_ids=model_input_ids,
            attention_mask=model_attention_mask,
            logits_to_keep=logits_to_keep,
        )
        last_hidden_state = last_hidden_state.contiguous()

        # The kernel computes the KL term itself (``beta`` + ``use_ref_model``), so the
        # reference model only has to provide its hidden states and LM head.
        ref_hidden_state = None
        ref_weight = None
        ref_bias = None
        if self.module.reference_model is not None:
            with torch.no_grad():
                ref_hidden_state = _get_last_hidden_state(
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

        loss_mask = context.loss_mask.to(last_hidden_state.dtype).contiguous()
        advantages = context.advantages.to(last_hidden_state.device)

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
            None,
            old_per_token_logps.contiguous(),
            ref_hidden_state,
            ref_weight,
            ref_bias,
        )
        return loss, self._kernel_metrics(liger_metrics, loss_mask)

    def _kernel_metrics(
        self,
        liger_metrics: Any,
        loss_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Expand the kernel's scalar statistics into the per-token metric contract.

        ``LigerFusedLinearGRPOLoss`` reports a single mean KL and clip ratio while the
        aggregator consumes per-token tensors, so the scalars are broadcast over the
        completion tokens; the aggregator's masked means then reproduce them exactly.
        ``entropy`` and the directional clip fractions are genuinely unavailable from the
        fused kernel and are reported as zeros.
        """

        zeros = torch.zeros_like(loss_mask)
        if float(self.module.config.rollout.kl_beta) != 0.0:
            mean_kl = liger_metrics[0].detach().to(loss_mask.dtype)
        else:
            mean_kl = zeros.new_zeros(())
        clip_ratio = liger_metrics[-1].detach().to(loss_mask.dtype)
        per_token_kl = zeros + mean_kl
        is_region_clipped = zeros + clip_ratio
        is_cispo = self.algorithm.preset.clip_reporting == "cispo"
        return {
            "loss_mask": loss_mask.detach(),
            "per_token_kl": per_token_kl.detach(),
            "entropy": zeros.detach(),
            "is_low_clipped": zeros.detach(),
            "is_high_clipped": zeros.detach(),
            "is_region_clipped": is_region_clipped.detach(),
            "is_cispo_clipped": is_region_clipped.detach() if is_cispo else zeros.detach(),
        }


