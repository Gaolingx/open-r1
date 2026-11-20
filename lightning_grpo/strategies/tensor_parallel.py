"""Tensor warpper for the Lightning GRPO pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_info
from torch.distributed.tensor import Replicate, Shard

from lightning_grpo.utils.configs.base import DistributedConfig
from lightning_grpo.utils.parallel.tp.parallel_utils import (
    ALL_PARALLEL_STYLES,
    _get_parameter_plan,
)
from lightning_grpo.utils.parallel.tp.tensor_parallel import (
    ColwiseParallel,
    apply_tensor_parallelism,
)

# Loss parallel computes the cross-entropy on a vocabulary-sharded `DTensor`, while the
# stock `colwise` style turns its output into a local tensor. Keep the shard as a DTensor.
ALL_PARALLEL_STYLES.register(
    "colwise_loss_parallel",
    ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=False),
)


def _tp_enabled(distributed_config: DistributedConfig) -> bool:
    tp_cfg = distributed_config.tensor_parallel
    if distributed_config.strategy not in {"fsdp2", "model_parallel"}:
        return False
    if not tp_cfg.enabled:
        return False
    return distributed_config.tensor_parallel_size > 1


def _qualify_base_model_tp_plan(model: nn.Module, plan: dict[str, str]) -> dict[str, str]:
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
) -> dict[str, str]:
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
        resolved_plan[lm_head_key] = (
            "colwise_loss_parallel" if tp_config.loss_parallel else "colwise_gather_output"
        )

    return resolved_plan


def expand_tp_plan(model: nn.Module, plan: dict[str, str]) -> dict[str, str]:
    """Replace the wildcards of the plan's rules with the model's layer numbers.

    ``apply_tensor_parallelism`` looks entries up by concrete name, and shards a
    module's own parameters before installing that module's forward.
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


def apply_custom_tensor_parallel(
    model: nn.Module,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
    tp_config: Any,
    *,
    liger_fused_ce: bool = False,
):
    final_plan = get_resolved_tp_plan(model, tp_config, liger_fused_ce=liger_fused_ce)
    if not final_plan:
        return model

    if unsupported := set(final_plan.values()) - set(ALL_PARALLEL_STYLES.keys()):
        raise ValueError(
            f"Unsupported tensor-parallel styles {sorted(unsupported)}. "
            f"Supported styles are {list(ALL_PARALLEL_STYLES.keys())}"
        )

    rank_zero_info(f"Tensor-parallel plan for {type(model).__name__}: {final_plan}")

    tp_mesh = device_mesh["tensor_parallel"] if "tensor_parallel" in device_mesh.mesh_dim_names else device_mesh
    return apply_tensor_parallelism(model, tp_mesh, expand_tp_plan(model, final_plan))


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
