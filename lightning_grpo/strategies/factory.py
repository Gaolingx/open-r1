"""Distributed strategy helpers for Lightning trainers."""

from __future__ import annotations

from typing import Any

from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload, ShardingStrategy

from lightning_grpo.configs.base import DistributedConfig


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


def build_strategy(config: DistributedConfig) -> str | DDPStrategy | FSDPStrategy:
    """Build the Lightning strategy object from configuration."""

    if config.strategy == "auto":
        return "auto"
    if config.strategy == "ddp":
        return DDPStrategy(
            find_unused_parameters=config.find_unused_parameters,
        )
    if config.strategy == "fsdp":
        return FSDPStrategy(
            cpu_offload=CPUOffload(offload_params=config.fsdp_cpu_offload),
            activation_checkpointing_policy=set(),
            sharding_strategy=_resolve_sharding_strategy(config.fsdp_sharding_strategy),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            state_dict_type="full",
        )
    raise ValueError(f"Unknown distributed strategy: {config.strategy}")


def trainer_strategy_kwargs(config: DistributedConfig) -> dict[str, Any]:
    """Build keyword arguments for [`lightning.pytorch.Trainer`](lightning_grpo/strategies/factory.py:24)."""

    return {
        "accelerator": config.accelerator,
        "devices": config.devices,
        "num_nodes": config.num_nodes,
        "strategy": build_strategy(config),
        "sync_batchnorm": config.sync_batchnorm,
    }
