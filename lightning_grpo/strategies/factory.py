"""Distributed strategy helpers for Lightning trainers."""

from __future__ import annotations

from typing import Any

import torch
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


def resolve_parallel_devices(
    accelerator: str,
    devices: int | str | list[int],
) -> list[torch.device] | None:
    """Resolve explicit parallel devices for DDPStrategy when GPU ids are provided."""

    if accelerator != "gpu":
        return None

    if isinstance(devices, list):
        gpu_ids = devices
    elif isinstance(devices, int):
        gpu_ids = list(range(devices))
    else:
        return None

    return [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]


def build_strategy(
    config: DistributedConfig,
    devices: int | str | list[int] | None = None,
    accelerator: str | None = None,
) -> str | DDPStrategy | FSDPStrategy:
    """Build the Lightning strategy object from configuration."""

    resolved_accelerator = accelerator or config.accelerator
    resolved_devices = config.devices if devices is None else devices

    if config.strategy == "auto":
        return "auto"
    if config.strategy == "ddp":
        return DDPStrategy(
            parallel_devices=resolve_parallel_devices(resolved_accelerator, resolved_devices),
            find_unused_parameters=config.find_unused_parameters,
        )
    if config.strategy == "fsdp":
        return FSDPStrategy(
            parallel_devices=resolve_parallel_devices(resolved_accelerator, resolved_devices),
            cpu_offload=CPUOffload(offload_params=config.fsdp_cpu_offload),
            activation_checkpointing_policy=set(),
            sharding_strategy=_resolve_sharding_strategy(config.fsdp_sharding_strategy),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            state_dict_type="full",
        )
    raise ValueError(f"Unknown distributed strategy: {config.strategy}")


def trainer_strategy_kwargs(
    config: DistributedConfig,
    devices: int | str | list[int] | None = None,
    accelerator: str | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for [`lightning.pytorch.Trainer`](lightning_grpo/strategies/factory.py:42)."""

    resolved_accelerator = accelerator or config.accelerator
    resolved_devices = config.devices if devices is None else devices

    return {
        "accelerator": resolved_accelerator,
        "devices": resolved_devices,
        "num_nodes": config.num_nodes,
        "strategy": build_strategy(config, devices=resolved_devices, accelerator=resolved_accelerator),
        "sync_batchnorm": config.sync_batchnorm,
    }
