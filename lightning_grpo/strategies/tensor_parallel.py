"""YAML-configurable tensor parallel plan helpers for Lightning ModelParallelStrategy."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

from lightning.pytorch.utilities import rank_zero_info
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_module
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    ParallelStyle,
)
import torch
import torch.nn as nn

from lightning_grpo.utils.configs.base import DistributedConfig, TensorParallelConfig


def _resolve_tensor_parallel_mesh(device_mesh: Any) -> Any | None:
    """Return the tensor-parallel mesh created by Lightning's ModelParallelStrategy."""

    if device_mesh is None:
        return None
    try:
        return device_mesh["tensor_parallel"]
    except (KeyError, RuntimeError, TypeError):
        return device_mesh


def _mesh_size(mesh: Any) -> int:
    """Best-effort mesh size lookup across PyTorch versions."""

    if mesh is None:
        return 1
    size = getattr(mesh, "size", None)
    if callable(size):
        return int(size())
    return int(size) if size is not None else 1


def _unwrap_base_model(root_module: nn.Module) -> nn.Module:
    """Return the module that owns decoder layers for common Hugging Face CausalLM wrappers."""

    base_model_prefix = getattr(root_module, "base_model_prefix", None)
    if isinstance(base_model_prefix, str) and hasattr(root_module, base_model_prefix):
        candidate = getattr(root_module, base_model_prefix)
        if isinstance(candidate, nn.Module):
            return candidate
    for attribute in ("model", "transformer"):
        candidate = getattr(root_module, attribute, None)
        if isinstance(candidate, nn.Module) and hasattr(candidate, "layers"):
            return candidate
    return root_module


def _tp_enabled(distributed_config: DistributedConfig) -> bool:
    """Check whether tensor parallel should run for this config."""

    tensor_parallel = distributed_config.tensor_parallel
    if distributed_config.strategy not in {"fsdp2", "model_parallel"}:
        return False
    if not tensor_parallel.enabled:
        return False
    return str(distributed_config.tensor_parallel_size) != "1"


def _has_supported_torchtitan_layout(root_module: nn.Module, base_model: nn.Module) -> bool:
    """Return whether the model exposes the module names expected by ``apply_non_moe_tp``."""

    if not hasattr(base_model, "layers"):
        return False
    has_embedding = hasattr(base_model, "tok_embeddings") or hasattr(base_model, "embed_tokens")
    has_lm_head = hasattr(base_model, "lm_head") or hasattr(root_module, "lm_head")
    return has_embedding and has_lm_head


def _attach_wrapper_lm_head_for_tp(root_module: nn.Module, base_model: nn.Module) -> bool:
    """Temporarily expose wrapper-level ``lm_head`` on the base model for TorchTitan-style TP."""

    if hasattr(base_model, "lm_head") or not hasattr(root_module, "lm_head"):
        return False
    base_model.lm_head = root_module.lm_head
    return True


def _detach_wrapper_lm_head_after_tp(base_model: nn.Module, attached: bool) -> None:
    """Remove a temporary ``lm_head`` alias created for TP planning."""

    if attached:
        delattr(base_model, "lm_head")


def _attach_hf_embedding_for_tp(base_model: nn.Module) -> bool:
    """Temporarily expose HF ``embed_tokens`` as TorchTitan-style ``tok_embeddings``."""

    if hasattr(base_model, "tok_embeddings") or not hasattr(base_model, "embed_tokens"):
        return False
    base_model.tok_embeddings = base_model.embed_tokens
    return True


def _detach_hf_embedding_after_tp(base_model: nn.Module, attached: bool) -> None:
    """Remove a temporary ``tok_embeddings`` alias created for TP planning."""

    if attached:
        delattr(base_model, "tok_embeddings")


def _apply_torchtitan_tensor_parallel(
    root_module: nn.Module,
    base_model: nn.Module,
    tp_mesh: DeviceMesh,
    tensor_parallel: TensorParallelConfig,
) -> bool:
    """Apply the TorchTitan-style non-MoE tensor parallel plan when possible."""

    if tensor_parallel.plan in {"none", "config"}:
        return False
    if not _has_supported_torchtitan_layout(root_module, base_model):
        return False

    attached_embedding = _attach_hf_embedding_for_tp(base_model)
    attached_lm_head = _attach_wrapper_lm_head_for_tp(root_module, base_model)
    try:
        apply_non_moe_tp(
            base_model,
            tp_mesh,
            enable_loss_parallel=tensor_parallel.loss_parallel or tensor_parallel.vocab_parallel,
        )
    finally:
        _detach_wrapper_lm_head_after_tp(base_model, attached_lm_head)
        _detach_hf_embedding_after_tp(base_model, attached_embedding)
    rank_zero_info(f"Applied TorchTitan-style tensor parallel plan to {root_module.__class__.__name__}.")
    return True


class NoParallel(ParallelStyle):
    """Replicate computation on the TP mesh without sharding.

    This style does nothing other than:
    (1) setting the module parameters as DTensors on the given mesh, and
    (2) inserting hooks at module boundary to convert torch.Tensor to DTensor and back.

    The reason we need this wrapping is to ensure all parameters are on the same 1D/2D mesh,
    which is assumed by (1) gradient norm clipping, and (2) optimizer fused implementation.

    Used for modules like the MoE router gate that need replicated computation on TP mesh.
    """

    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        local_output_grad_placements: Sequence[Placement] | None = None,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        # If None, output stays as DTensor.
        # If provided, output is cast to local tensor via
        # to_local(grad_placements=local_output_grad_placements).
        self.local_output_grad_placements = local_output_grad_placements

    @staticmethod
    def _prepare_input_fn(
        input_layout: Placement | None,
        desired_input_layout: Placement | None,
        mod: nn.Module,
        inputs: Any,
        device_mesh: DeviceMesh,
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            assert input_layout is not None
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            assert input_layout is not None
            assert desired_input_layout is not None
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(
        output_layout: Placement,
        local_output_grad_placements: Sequence[Placement] | None,
        mod: nn.Module,
        outputs: DTensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor | DTensor:
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        if local_output_grad_placements is not None:
            return outputs.to_local(grad_placements=local_output_grad_placements)
        else:
            return outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn,  # pyrefly: ignore [bad-argument-type]
                self.input_layout,
                self.desired_input_layout,
            ),
            partial(
                self._prepare_output_fn,  # pyrefly: ignore [bad-argument-type]
                self.output_layout,
                self.local_output_grad_placements,
            ),
        )


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer

    # skipping nn.Identity modules (which are added by pipeline parallelism for unused modules)
    root_plan = {}

    if hasattr(model, "tok_embeddings"):
        if isinstance(model.tok_embeddings, nn.Identity):
            root_plan["tok_embeddings"] = NoParallel(
                local_output_grad_placements=(Replicate(),),
            )
        else:
            root_plan["tok_embeddings"] = RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            )

    if hasattr(model, "norm"):
        if isinstance(model.norm, nn.Identity):
            root_plan["norm"] = NoParallel(
                local_output_grad_placements=(Replicate(),),
            )
        else:
            root_plan["norm"] = SequenceParallel()

    if hasattr(model, "lm_head"):
        if isinstance(model.lm_head, nn.Identity):
            root_plan["lm_head"] = NoParallel(
                local_output_grad_placements=(Replicate(),),
            )
        else:
            root_plan["lm_head"] = ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            )
    if root_plan:  # Only call if there's something to parallelize
        parallelize_module(model, tp_mesh, root_plan)

    # Apply tensor + sequence parallelism to every transformer block
    for transformer_block in model.layers:
        layer_plan = {
            "input_layernorm": SequenceParallel(),
            "self_attn": PrepareModuleInput(
                input_kwarg_layouts={"hidden_states": Shard(1)},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
            ),
            "post_attention_layernorm": SequenceParallel(),
        }

        if getattr(transformer_block.self_attn, "q_lora_rank", None) is None:
            layer_plan.update(
                {
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                }
            )
        else:
            layer_plan.update(
                {
                    "self_attn.q_a_proj": NoParallel(
                        local_output_grad_placements=(Replicate(),),
                    ),
                    "self_attn.q_a_layernorm": NoParallel(
                        local_output_grad_placements=(Replicate(),),
                    ),
                    "self_attn.q_b_proj": ColwiseParallel(),
                    "self_attn.kv_a_proj_with_mqa": NoParallel(
                        local_output_grad_placements=(Replicate(),),
                    ),
                    "self_attn.kv_a_layernorm": NoParallel(
                        local_output_grad_placements=(Replicate(),),
                    ),
                    "self_attn.kv_b_proj": ColwiseParallel(),
                }
            )

        # Handle different names for the output projection layer, e.g. o_proj vs dense
        o_proj_name = (
            "o_proj" if hasattr(transformer_block.self_attn, "o_proj") else "dense"
        )
        layer_plan[f"self_attn.{o_proj_name}"] = RowwiseParallel(
            output_layouts=Shard(1)
        )
        # For model that uses RMSNorm on Q and K (i.e. Qwen3)
        if hasattr(transformer_block.self_attn, "q_norm") and hasattr(
            transformer_block.self_attn, "k_norm"
        ):
            layer_plan["self_attn.q_norm"] = SequenceParallel(
                sequence_dim=2, use_local_output=True
            )
            layer_plan["self_attn.k_norm"] = SequenceParallel(
                sequence_dim=2, use_local_output=True
            )

        if not transformer_block.moe_enabled:
            mlp_plan = {
                "mlp": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
            }
            # Handle different names for MLP layers, e.g. gate_proj vs fc1
            gate_proj_name = (
                "gate_proj" if hasattr(transformer_block.mlp, "gate_proj") else "fc1"
            )
            mlp_plan[f"mlp.{gate_proj_name}"] = ColwiseParallel()

            if hasattr(transformer_block.mlp, "up_proj"):
                mlp_plan["mlp.up_proj"] = ColwiseParallel()

            down_proj_name = (
                "down_proj" if hasattr(transformer_block.mlp, "down_proj") else "fc2"
            )
            mlp_plan[f"mlp.{down_proj_name}"] = RowwiseParallel(output_layouts=Shard(1))
            layer_plan.update(mlp_plan)

        # Some models like Phi-2 don't have post_attention_layernorm
        if not hasattr(transformer_block, "post_attention_layernorm"):
            layer_plan.pop("post_attention_layernorm")

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    rank_zero_info("Applied Tensor Parallelism to the model")


def configure_tensor_parallel(
    root_module: nn.Module,
    distributed_config: DistributedConfig,
    device_mesh: Any = None,
) -> None:
    """Apply PyTorch tensor parallelism from YAML/model-config plans before FSDP2 wrapping."""

    if not _tp_enabled(distributed_config):
        return

    tp_mesh = _resolve_tensor_parallel_mesh(device_mesh)
    if _mesh_size(tp_mesh) <= 1:
        return

    base_model = _unwrap_base_model(root_module)
    if _apply_torchtitan_tensor_parallel(root_module, base_model, tp_mesh, distributed_config.tensor_parallel):
        return

    rank_zero_info("Tensor parallel is enabled but the model layout is not supported by the TorchTitan-style TP plan.")
