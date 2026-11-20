"""Expert-parallel wrapper for the Lightning GRPO pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_info
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from lightning_grpo.utils.configs.base import DistributedConfig
from lightning_grpo.utils.parallel.tp.expert_parallel import apply_expert_parallelism
from lightning_grpo.utils.parallel.tp.parallel_utils import (
    ALL_PARALLEL_STYLES,
    _get_parameter_plan,
)

_EXPERT_MESH_CACHE: dict[tuple[str, int, int], DeviceMesh] = {}


def _ep_enabled(distributed_config: DistributedConfig) -> bool:
    if distributed_config.strategy not in {"fsdp2", "model_parallel"}:
        return False
    return distributed_config.expert_parallel_size > 1


def _qualify_base_model_ep_plan(model: nn.Module, plan: dict[str, str]) -> dict[str, str]:
    """Rewrite base-model-relative rule names to fully-qualified module names.

    ``config.base_model_ep_plan`` is written against the base model (``layers.*.mlp``),
    while every lookup here walks ``named_modules()`` of the top-level model, where
    those modules live under the base model prefix (``model.layers.*.mlp``).
    """
    prefix = getattr(model, "base_model_prefix", "") or ""
    base_model = getattr(model, "base_model", None)
    if not prefix or base_model is model:
        return dict(plan)

    prefix = f"{prefix}."
    return {key if key.startswith(prefix) else f"{prefix}{key}": value for key, value in plan.items()}


def get_resolved_ep_plan(model: nn.Module, ep_config: Any) -> dict[str, str]:
    resolved_plan = {}

    if ep_config.plan in ["auto", "default"]:
        base_plan = getattr(model.config, "base_model_ep_plan", None)
        if base_plan:
            resolved_plan.update(_qualify_base_model_ep_plan(model, base_plan))
        instance_plan = getattr(model, "_ep_plan", None)
        if instance_plan:
            resolved_plan.update(instance_plan)

    if ep_config.plan_overrides:
        resolved_plan.update(ep_config.plan_overrides)

    return resolved_plan


def validate_ep_topology(ep_size: int, tp_size: int, dp_size: int) -> None:
    """Check the EP size against the dense ``(data_parallel, tensor_parallel)`` layout.

    The expert view is ``(efsdp, expert_parallel)`` over the same ranks, with
    ``efsdp = dp_size * tp_size / ep_size``, so an expert shard is replicated
    ``efsdp`` times. This mirrors the dense ``(data_parallel, tensor_parallel)``
    ``fsdp * tp`` layout used upstream.
    """
    world_size = dp_size * tp_size
    if ep_size % tp_size != 0:
        raise ValueError(
            f"expert_parallel_size ({ep_size}) must be a multiple of tensor_parallel_size ({tp_size})."
        )
    if world_size % ep_size != 0:
        raise ValueError(
            f"expert_parallel_size ({ep_size}) must divide data_parallel_size * tensor_parallel_size "
            f"({dp_size} * {tp_size} = {world_size})."
        )


def expand_ep_plan(model: nn.Module, plan: dict[str, str]) -> dict[str, str]:
    """Replace the wildcards of the plan's rules with concrete module and parameter names.

    Parameters come before their module so ``apply_expert_parallelism`` shards the
    expert weights (and updates ``num_experts``) before the dispatch forward is
    installed on the module that reads it.
    """
    expanded_plan = {}

    for module_name, module in model.named_modules():
        module_style = _get_parameter_plan(module_name, plan, is_weight=False)
        for param_name, _ in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}" if module_name else param_name
            param_style = _get_parameter_plan(name, plan, is_weight=True)
            if param_style is not None:
                expanded_plan[name] = param_style
        if module_style is not None:
            expanded_plan[module_name] = module_style

    return expanded_plan


def _get_expert_mesh(device_mesh: DeviceMesh, ep_size: int) -> DeviceMesh:
    """Build (or reuse) the ``(efsdp, expert_parallel)`` view over the same ranks."""
    world_size = torch.distributed.get_world_size()
    efsdp_size = world_size // ep_size
    key = (device_mesh.device_type, world_size, ep_size)
    mesh = _EXPERT_MESH_CACHE.get(key)
    if mesh is None:
        mesh = init_device_mesh(
            device_mesh.device_type,
            (efsdp_size, ep_size),
            mesh_dim_names=("efsdp", "expert_parallel"),
        )
        _EXPERT_MESH_CACHE[key] = mesh
    return mesh


def apply_custom_expert_parallel(
    model: nn.Module,
    distributed_config: DistributedConfig,
    device_mesh: DeviceMesh,
) -> DeviceMesh:
    mesh_dim_names = getattr(device_mesh, "mesh_dim_names", None) or ()
    if "data_parallel" not in mesh_dim_names or "tensor_parallel" not in mesh_dim_names:
        raise ValueError(
            "Expert parallelism needs Lightning's (data_parallel, tensor_parallel) device mesh, "
            f"got dimensions {tuple(mesh_dim_names)}."
        )

    ep_size = distributed_config.expert_parallel_size
    tp_mesh = device_mesh["tensor_parallel"]
    dp_mesh = device_mesh["data_parallel"]
    validate_ep_topology(ep_size, tp_mesh.size(), dp_mesh.size())

    ep_config = distributed_config.expert_parallel
    final_plan = get_resolved_ep_plan(model, ep_config)
    if not final_plan:
        raise ValueError(
            "expert_parallel_size > 1 requires an expert-parallel plan: set "
            "`distributed.expert_parallel.plan_overrides` or add `base_model_ep_plan` to the model config."
        )

    if unsupported := set(final_plan.values()) - set(ALL_PARALLEL_STYLES.keys()):
        raise ValueError(
            f"Unsupported expert-parallel styles {sorted(unsupported)}. "
            f"Supported styles are {list(ALL_PARALLEL_STYLES.keys())}"
        )
    if "ep_dispatch_experts" not in final_plan.values():
        raise ValueError(
            "The expert-parallel plan needs an `ep_dispatch_experts` rule so tokens reach the "
            "experts held by another rank."
        )

    rank_zero_info(f"Expert-parallel plan for {type(model).__name__}: {final_plan}")

    expert_mesh = _get_expert_mesh(device_mesh, ep_size)
    expanded_plan = expand_ep_plan(model, final_plan)
    apply_expert_parallelism(
        model,
        expert_mesh["expert_parallel"],
        tp_mesh,
        expanded_plan,
        ep_implementation=ep_config.implementation,
    )

    modules = dict(model.named_modules())
    for name, style_name in expanded_plan.items():
        if style_name == "ep_dispatch_experts":
            modules[name]._ep_dispatch_experts = True

    model._expert_mesh = expert_mesh
    model._ep_size = ep_size
    model._ep_plan_expanded = expanded_plan
    return expert_mesh


def configure_expert_parallel(
    root_module: nn.Module,
    distributed_config: DistributedConfig,
    device_mesh: DeviceMesh,
) -> DeviceMesh | None:
    if not _ep_enabled(distributed_config):
        return None

    return apply_custom_expert_parallel(root_module, distributed_config, device_mesh)
