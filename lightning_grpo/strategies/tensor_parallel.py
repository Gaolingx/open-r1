"""YAML-configurable tensor parallel plan helpers for Lightning ModelParallelStrategy."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from lightning.pytorch.utilities import rank_zero_info
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module
from torch.nn import Module

from lightning_grpo.utils.configs.base import DistributedConfig, TensorParallelConfig

_PARALLEL_STYLES = {
    "colwise",
    "colwise_gather_output",
    "rowwise",
    "sequence",
    "replicated_with_grad_allreduce",
    "vocab_parallel_lm_head",
}
_UNSUPPORTED_STYLES = {
    "packed_colwise",
    "moe_tp_experts",
}
_DEFAULT_ATTENTION_PLAN = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.o_proj": "rowwise",
}
_DEFAULT_DENSE_MLP_PLAN = {
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
    "layers.*.mlp.shared_expert.gate_proj": "colwise",
    "layers.*.mlp.shared_expert.up_proj": "colwise",
    "layers.*.mlp.shared_expert.down_proj": "rowwise",
    "layers.*.mlp.shared_expert_gate": "colwise",
}
_DEFAULT_EMBEDDING_PLAN = {
    "embed_tokens": "rowwise",
}
_DEFAULT_LM_HEAD_PLAN = {
    "lm_head": "colwise_gather_output",
}


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


def _unwrap_base_model(root_module: Module) -> Module:
    """Return the module that owns decoder layers for common Hugging Face CausalLM wrappers."""

    base_model_prefix = getattr(root_module, "base_model_prefix", None)
    if isinstance(base_model_prefix, str) and hasattr(root_module, base_model_prefix):
        candidate = getattr(root_module, base_model_prefix)
        if isinstance(candidate, Module):
            return candidate
    for attribute in ("model", "transformer"):
        candidate = getattr(root_module, attribute, None)
        if isinstance(candidate, Module) and hasattr(candidate, "layers"):
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


def _module_exists(root_module: Module, pattern: str) -> bool:
    """Return whether at least one module name matches the plan pattern."""

    return any(fnmatch(name, pattern) for name, _ in root_module.named_modules())


def _style_to_parallel_style(style: str, tensor_parallel: TensorParallelConfig) -> Any | None:
    """Map a YAML/string TP style to PyTorch parallel style objects."""

    if style == "colwise":
        return ColwiseParallel()
    if style == "colwise_gather_output":
        return ColwiseParallel(output_layouts=Replicate())
    if style == "vocab_parallel_lm_head":
        return ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
    if style == "rowwise":
        return RowwiseParallel()
    if style in {"sequence", "replicated_with_grad_allreduce"}:
        return SequenceParallel(sequence_dim=tensor_parallel.sequence_dim)
    return None


def _auto_plan_from_model_config(base_model: Module, tensor_parallel: TensorParallelConfig) -> dict[str, str]:
    """Resolve an auto TP plan from the model config, then filter unsupported features."""

    config = getattr(base_model, "config", None)
    plan = dict(getattr(config, "base_model_tp_plan", {}) or {})
    if not plan or tensor_parallel.plan in {"default", "qwen_llama"}:
        plan = dict(_DEFAULT_ATTENTION_PLAN)
        if tensor_parallel.parallelize_mlp:
            plan.update(_DEFAULT_DENSE_MLP_PLAN)
    if tensor_parallel.parallelize_embedding:
        plan.update(_DEFAULT_EMBEDDING_PLAN)

    filtered_plan: dict[str, str] = {}
    for pattern, style in plan.items():
        if style in _UNSUPPORTED_STYLES:
            continue
        if style not in _PARALLEL_STYLES:
            continue
        if not tensor_parallel.parallelize_mlp and ".mlp." in pattern:
            continue
        if not tensor_parallel.parallelize_embedding and "embed" in pattern:
            continue
        if not tensor_parallel.parallelize_lm_head and "lm_head" in pattern:
            continue
        if not tensor_parallel.sequence_parallel and style in {"sequence", "replicated_with_grad_allreduce"}:
            continue
        filtered_plan[pattern] = style
    filtered_plan.update(tensor_parallel.plan_overrides)
    return filtered_plan


def _lm_head_plan_from_model(root_module: Module, tensor_parallel: TensorParallelConfig) -> dict[str, str]:
    """Resolve TP plan entries that live on the CausalLM wrapper, such as ``lm_head``."""

    if not (tensor_parallel.parallelize_lm_head or tensor_parallel.vocab_parallel or tensor_parallel.loss_parallel):
        return {}

    plan = dict(getattr(root_module, "_tp_plan", {}) or {})
    if not plan or tensor_parallel.plan in {"default", "qwen_llama"}:
        plan.update(_DEFAULT_LM_HEAD_PLAN)

    if tensor_parallel.vocab_parallel or tensor_parallel.loss_parallel:
        for pattern in list(plan):
            if "lm_head" in pattern:
                plan[pattern] = "vocab_parallel_lm_head"

    plan.update({pattern: style for pattern, style in tensor_parallel.plan_overrides.items() if _module_exists(root_module, pattern)})
    return {pattern: style for pattern, style in plan.items() if "lm_head" in pattern}


def _build_parallelize_plan(base_model: Module, tensor_parallel: TensorParallelConfig) -> dict[str, Any]:
    """Build a PyTorch ``parallelize_module`` plan from YAML/model-config strings."""

    if tensor_parallel.plan == "none":
        return {}

    string_plan = tensor_parallel.plan_overrides if tensor_parallel.plan == "config" else _auto_plan_from_model_config(base_model, tensor_parallel)
    parallelize_plan: dict[str, Any] = {}
    for pattern, style in string_plan.items():
        if not _module_exists(base_model, pattern):
            continue
        parallel_style = _style_to_parallel_style(style, tensor_parallel)
        if parallel_style is not None:
            parallelize_plan[pattern] = parallel_style
    return parallelize_plan


def _build_lm_head_parallelize_plan(root_module: Module, tensor_parallel: TensorParallelConfig) -> dict[str, Any]:
    """Build wrapper-level TP plan entries for CausalLM heads."""

    if tensor_parallel.plan == "none":
        return {}

    parallelize_plan: dict[str, Any] = {}
    for pattern, style in _lm_head_plan_from_model(root_module, tensor_parallel).items():
        if not _module_exists(root_module, pattern):
            continue
        parallel_style = _style_to_parallel_style(style, tensor_parallel)
        if parallel_style is not None:
            parallelize_plan[pattern] = parallel_style
    return parallelize_plan


def configure_tensor_parallel(
    root_module: Module,
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
    parallelize_plan = _build_parallelize_plan(base_model, distributed_config.tensor_parallel)
    lm_head_plan = _build_lm_head_parallelize_plan(root_module, distributed_config.tensor_parallel)
    if not parallelize_plan and not lm_head_plan:
        rank_zero_info("Tensor parallel is enabled but no matching TP plan entries were found.")
        return

    if parallelize_plan:
        parallelize_module(base_model, tp_mesh, parallelize_plan)
    if lm_head_plan:
        parallelize_module(root_module, tp_mesh, lm_head_plan)
    rank_zero_info(
        f"Applied tensor parallel plan with {len(parallelize_plan) + len(lm_head_plan)} entries "
        f"to {root_module.__class__.__name__}."
    )
