"""Distributed strategy helpers for Lightning trainers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload, ShardingStrategy
from torch.nn import Module

from lightning_grpo.configs.base import DistributedConfig, PrecisionConfig


def _resolve_sharding_strategy(name: str) -> ShardingStrategy:
    """Map string configuration to an FSDP sharding strategy."""

    mapping = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported FSDP sharding strategy: {name}")
    return mapping[name]


def _import_module_class(path: str) -> type[Module]:
    """Import a module class from a fully-qualified dotted path."""

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


def _resolve_policy_classes(class_paths: list[str]) -> set[type[Module]] | None:
    """Resolve YAML-configured class paths into an FSDP policy set."""

    if not class_paths:
        return None
    return {_import_module_class(class_path) for class_path in class_paths}


def configure_cuda_precision(
    precision_config: PrecisionConfig,
    accelerator: str | None,
) -> None:
    """Configure TF32 and float32 matmul precision when CUDA is available."""

    if accelerator not in {None, "auto", "gpu"} or not torch.cuda.is_available():
        return

    torch.backends.cudnn.allow_tf32 = precision_config.tf32
    torch.set_float32_matmul_precision("high" if precision_config.tf32 else "highest")


def build_strategy(config: DistributedConfig) -> str | DDPStrategy | FSDPStrategy:
    """Build the Lightning strategy object from configuration."""

    if config.strategy == "auto":
        return "auto"
    if config.strategy == "ddp":
        return DDPStrategy(
            find_unused_parameters=config.find_unused_parameters,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
        )
    if config.strategy == "fsdp":
        auto_wrap_policy = _resolve_policy_classes(config.fsdp_auto_wrap_policy_classes)
        activation_checkpointing_policy = None
        if config.fsdp_activation_checkpointing:
            activation_checkpointing_policy = _resolve_policy_classes(
                config.fsdp_activation_checkpointing_policy_classes
            )

        return FSDPStrategy(
            cpu_offload=CPUOffload(offload_params=config.fsdp_cpu_offload),
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy=activation_checkpointing_policy,
            sharding_strategy=_resolve_sharding_strategy(config.fsdp_sharding_strategy),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            state_dict_type="full",
        )
    raise ValueError(f"Unknown distributed strategy: {config.strategy}")


def trainer_strategy_kwargs(
    distributed_config: DistributedConfig,
    precision_config: PrecisionConfig | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for [`lightning.pytorch.Trainer`](lightning_grpo/strategies/factory.py:126)."""

    configure_cuda_precision(precision_config, distributed_config.accelerator)

    return {
        "accelerator": distributed_config.accelerator,
        "devices": distributed_config.devices,
        "num_nodes": distributed_config.num_nodes,
        "strategy": build_strategy(distributed_config),
        "precision": precision_config.trainer_precision,
        "sync_batchnorm": distributed_config.sync_batchnorm,
    }
