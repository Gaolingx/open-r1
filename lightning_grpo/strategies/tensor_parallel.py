"""Tensor warpper for the Lightning GRPO pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from torch.distributed.tensor import distribute_tensor, Shard, Replicate, DTensor
from lightning.pytorch.utilities import rank_zero_info
from lightning_grpo.utils.configs.base import DistributedConfig
from lightning_grpo.utils.parallel.tp_utils import (
    ALL_PARALLEL_STYLES,
    _get_parameter_tp_plan,
    add_tensor_parallel_hooks_to_module,
)


def _tp_enabled(distributed_config: DistributedConfig) -> bool:
    tp_cfg = distributed_config.tensor_parallel
    if distributed_config.strategy not in {"fsdp2", "model_parallel"}:
        return False
    if not tp_cfg.enabled:
        return False
    return distributed_config.tensor_parallel_size > 1


def _qualify_base_model_tp_plan(model: nn.Module, plan: Dict[str, str]) -> Dict[str, str]:
    """Rewrite base-model-relative rule names to fully-qualified module names.

    ``config.base_model_tp_plan`` is written against the base model (``layers.*.mlp``),
    while every lookup here walks ``named_modules()`` of the top-level model, where
    those modules live under the base model prefix (``model.layers.*.mlp``). Without
    qualification the rules silently never match. Keys that are already qualified
    (what ``PreTrainedModel.post_init`` puts on ``model._tp_plan``) are left untouched.
    """
    prefix = getattr(model, "base_model_prefix", "") or ""
    base_model = getattr(model, "base_model", None)
    if not prefix or base_model is model:
        return dict(plan)

    prefix = f"{prefix}."
    return {key if key.startswith(prefix) else f"{prefix}{key}": value for key, value in plan.items()}


def get_resolved_tp_plan(
    model: nn.Module,
    tp_config: Any,
    *,
    liger_fused_ce: bool = False,
) -> Dict[str, str]:
    resolved_plan = {}

    if tp_config.plan in ["auto", "default"]:
        base_plan = getattr(model.config, "base_model_tp_plan", None)
        if base_plan:
            resolved_plan.update(_qualify_base_model_tp_plan(model, base_plan))
        instance_plan = getattr(model, "_tp_plan", None)
        if instance_plan:
            resolved_plan.update(instance_plan)

    if tp_config.plan_overrides:
        resolved_plan.update(tp_config.plan_overrides)

    lm_head_key = "lm_head"
    if liger_fused_ce:
        # Liger's fused linear CE consumes ``lm_head.weight`` directly instead of calling
        # the module, so the colwise output all-gather never runs. A vocab-sharded weight
        # would therefore be indexed with global token ids (out-of-bounds inside the
        # Triton kernel), so keep lm_head replicated instead.
        if resolved_plan.pop(lm_head_key, None) is not None:
            rank_zero_info(
                "Liger fused CE is enabled: keeping `lm_head` replicated because the fused "
                "loss bypasses the tensor-parallel gather hook."
            )
    elif lm_head_key in resolved_plan:
        resolved_plan[lm_head_key] = "colwise" if tp_config.loss_parallel else "colwise_gather_output"

    return resolved_plan


def wrap_module_parameters_as_dtensor(module, device_mesh, tp_plan_name):
    """Wrap the already sharded Parameter as a DTensor."""
    tp_mesh = device_mesh["tensor_parallel"] if "tensor_parallel" in device_mesh.mesh_dim_names else device_mesh

    for param_name, param in module.named_parameters(recurse=False):
        # 1. Automatically determine shard dimensions
        # Linear.weight: [out_features, in_features]
        # Linear.bias: [out_features]
        if "colwise" in tp_plan_name:
            # Colwise: Weights are sharded along dim 0, biases are sharded along dim 0
            sharding_dim = 0
        elif "rowwise" in tp_plan_name:
            # Rowwise: Weights are sharded along dim 1, biases are usually not sharded (Replicate)
            sharding_dim = 1 if param.ndim > 1 else -1
        else:
            continue
        dist_spec = [Shard(sharding_dim)] if sharding_dim != -1 else [Replicate()]

        # 2. Use from_local to wrap, without copying memory.
        # Note: At this time, param.data must already be a local shard that has been split by shard tensor.
        dt_param = DTensor.from_local(param.data, tp_mesh, dist_spec)

        new_param = nn.Parameter(dt_param, requires_grad=param.requires_grad)
        setattr(module, param_name, new_param)


def apply_custom_tensor_parallel(
    model: nn.Module,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
    tp_config: Any,
    *,
    liger_fused_ce: bool = False,
):
    final_plan = get_resolved_tp_plan(model, tp_config, liger_fused_ce=liger_fused_ce)
    rank_zero_info(f"Tensor-parallel plan for {type(model).__name__}: {final_plan}")

    tp_mesh = device_mesh["tensor_parallel"] if "tensor_parallel" in device_mesh.mesh_dim_names else device_mesh
    rank = tp_mesh.get_local_rank()

    for name, module in model.named_modules():
        plan_name = _get_parameter_tp_plan(name, final_plan, is_weight=False)
        if plan_name is None:
            continue

        tp_layer = ALL_PARALLEL_STYLES[plan_name]
        tp_layer.device_mesh = tp_mesh
        tp_layer.rank = rank

        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = f"{name}.{param_name}"
            param_plan = _get_parameter_tp_plan(full_param_name, final_plan)

            if param_plan:
                # A module-level rule (e.g. `moe_tp_experts`) only installs the
                # communication hooks; the weights of such a module are sharded by
                # their *own* rules (`experts.gate_up_proj` -> packed_colwise).
                # Resolve the style per parameter, otherwise the module style is
                # applied to it, and `moe_tp_experts` -- a no-op sharder -- would
                # leave the weights replicated while the output hook still sums
                # every rank's copy, inflating the routed-expert output by tp_size.
                shard_layer = ALL_PARALLEL_STYLES[param_plan]
                shard_layer.device_mesh = tp_mesh
                shard_layer.rank = rank
                shard_layer.empty_param = param
                sharded_data = shard_layer.shard_tensor(
                    param.data,
                    device=param.device,
                    dtype=param.dtype
                )
                new_param = nn.Parameter(sharded_data, requires_grad=param.requires_grad)
                setattr(module, param_name, new_param)

                shard_layer.update_module_attributes(module)

        # Handling DTensor conversion (focus on lm_head and loss_parallel)
        if name == "lm_head" and tp_config.loss_parallel:
            # Wrap the locally segmented Tensor into a DTensor
            wrap_module_parameters_as_dtensor(module, tp_mesh, plan_name)

            # Remove the old hook (ColwiseParallel will register all_reduce_backward)
            # because DTensor will handle these internally automatically.
            if hasattr(module, "_forward_pre_hooks"): module._forward_pre_hooks.clear()
            if hasattr(module, "_forward_hooks"): module._forward_hooks.clear()

            # Input into the DTensor field
            module.register_forward_pre_hook(
                lambda mod, inputs: (distribute_tensor(inputs[0], tp_mesh, [Replicate()]),)
            )
        else:
            # Non loss parallel modules use the original Hook
            add_tensor_parallel_hooks_to_module(
                model=model,
                module=module,
                layer_name=name,
                current_module_plan=plan_name,
                device_mesh=tp_mesh
            )
    return model


def configure_tensor_parallel(
    root_module: nn.Module,
    distributed_config: DistributedConfig,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
    *,
    liger_fused_ce: bool = False,
) -> None:
    if not _tp_enabled(distributed_config):
        return

    tp_mesh = device_mesh["tensor_parallel"]
    if tp_mesh.size() <= 1:
        return

    rank_zero_info(
        f"Applying Tensor Parallelism (size: {distributed_config.tensor_parallel_size}, "
        f"loss_parallel: {distributed_config.tensor_parallel.loss_parallel})"
    )

    apply_custom_tensor_parallel(
        root_module,
        device_mesh,
        distributed_config.tensor_parallel,
        liger_fused_ce=liger_fused_ce,
    )
