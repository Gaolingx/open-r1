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


def get_resolved_tp_plan(model: nn.Module, tp_config: Any) -> Dict[str, str]:
    resolved_plan = {}

    if tp_config.plan in ["auto", "default"]:
        if hasattr(model.config, "base_model_tp_plan"):
            resolved_plan.update(model.config.base_model_tp_plan)
        if hasattr(model, "_tp_plan"):
            resolved_plan.update(model._tp_plan)

    if tp_config.plan_overrides:
        resolved_plan.update(tp_config.plan_overrides)

    lm_head_key = "lm_head"
    if tp_config.loss_parallel:
        if lm_head_key in resolved_plan:
            resolved_plan[lm_head_key] = "colwise"
    else:
        if lm_head_key in resolved_plan:
            resolved_plan[lm_head_key] = "colwise_gather_output"

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
    tp_config: Any
):
    final_plan = get_resolved_tp_plan(model, tp_config)

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
                tp_layer.empty_param = param
                sharded_data = tp_layer.shard_tensor(
                    param.data,
                    device=param.device,
                    dtype=param.dtype
                )
                new_param = nn.Parameter(sharded_data, requires_grad=param.requires_grad)
                setattr(module, param_name, new_param)

                tp_layer.update_module_attributes(module)

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
        distributed_config.tensor_parallel
    )
