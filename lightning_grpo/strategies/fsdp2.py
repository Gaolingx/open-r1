"""Composable FSDP2 helpers used from LightningModule.configure_model."""

from __future__ import annotations

from fnmatch import fnmatch
from importlib import import_module
from typing import Any

from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.nn import Module

from lightning_grpo.utils.configs.base import DistributedConfig


def _import_module_class(path: str) -> type[Module]:
    """Import a torch module class from a fully-qualified dotted path."""

    module_path, _, class_name = path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(
            "FSDP policy class paths must be fully-qualified, for example "
            "`transformers.models.llama.modeling_llama.LlamaDecoderLayer`."
        )

    module = import_module(module_path)
    class_object = getattr(module, class_name)
    if not isinstance(class_object, type) or not issubclass(class_object, Module):
        raise TypeError(f"FSDP policy target must be a torch.nn.Module subclass: {path}")
    return class_object


def _resolve_policy_classes(class_paths: list[str]) -> tuple[type[Module], ...]:
    """Resolve YAML-configured class paths into module classes."""

    return tuple(_import_module_class(class_path) for class_path in class_paths)


def _resolve_data_parallel_mesh(device_mesh: Any) -> Any | None:
    """Return the data-parallel mesh created by Lightning's ModelParallelStrategy."""

    if device_mesh is None:
        return None
    try:
        return device_mesh["data_parallel"]
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


def _matches_module(name: str, module: Module, name_patterns: list[str], policy_classes: tuple[type[Module], ...]) -> bool:
    """Check whether a module should be fully sharded before the root module."""

    return any(fnmatch(name, pattern) for pattern in name_patterns) or bool(policy_classes and isinstance(module, policy_classes))


def configure_fully_shard(
    root_module: Module,
    distributed_config: DistributedConfig,
    device_mesh: Any = None,
) -> None:
    """Apply PyTorch composable FSDP2 ``fully_shard`` using config-only policies.

    This helper keeps Lightning modules generic: each module only passes its root
    model (for example ``self.model`` or ``self.policy``), while YAML controls
    which child blocks are sharded by class path or by module-name glob.
    """

    if distributed_config.strategy not in {"fsdp2", "model_parallel"}:
        return

    dp_mesh = _resolve_data_parallel_mesh(device_mesh)
    if _mesh_size(dp_mesh) <= 1:
        return

    shard_kwargs = dict(distributed_config.fsdp_fully_shard_kwargs)
    if dp_mesh is not None:
        shard_kwargs.setdefault("mesh", dp_mesh)

    policy_classes = _resolve_policy_classes(distributed_config.fsdp_auto_wrap_policy_classes)
    target_names = distributed_config.fsdp_fully_shard_module_names

    for name, module in root_module.named_modules():
        if not name:
            continue
        if _matches_module(name, module, target_names, policy_classes):
            fully_shard(module, **shard_kwargs)

    if distributed_config.fsdp_fully_shard_root:
        fully_shard(root_module, **shard_kwargs)
