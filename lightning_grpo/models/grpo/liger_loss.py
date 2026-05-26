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
from lightning_grpo.utils.metrics import MoEAuxLossComputer, collect_moe_metrics


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


def _get_last_hidden_state(
    self,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    output_router_logits: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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


def is_liger_kernel_available() -> bool:
    """Check if liger-kernel is installed."""
    try:
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss  # noqa: F401
        return True
    except ImportError:
        return False


class LigerDPOLossComputer:
    """Compute DPO loss using Liger Kernel's fused linear + DPOLoss kernel."""

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        *,
        beta: float,
        loss_type: str,
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
        self.loss_parallel_enabled = loss_parallel_enabled
        self.loss_fn = LigerFusedLinearDPOLoss(
            beta=beta,
            loss_type=loss_type,
            ignore_index=ignore_index,
            compiled=compiled,
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

        return loss, {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits_mean": chosen_logits_mean,
            "rejected_logits_mean": rejected_logits_mean,
            "nll_loss": nll_loss,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
        }

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
        self.aux_loss_computer = MoEAuxLossComputer(model)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        shift_hidden, moe_outputs = _get_last_hidden_state(
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
        metrics = collect_moe_metrics(moe_outputs)
        aux_loss, aux_metrics = self.aux_loss_computer.compute(moe_outputs, attention_mask)
        if aux_loss is not None:
            loss = loss + aux_loss.to(loss.device)
        metrics.update(aux_metrics)
        return loss, metrics
