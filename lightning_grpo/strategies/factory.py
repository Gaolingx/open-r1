"""Distributed strategy helpers for Lightning trainers."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.strategies import DDPStrategy, ModelParallelStrategy

from lightning_grpo.utils.configs.base import DistributedConfig, PrecisionConfig


def configure_cuda_precision(
    precision_config: PrecisionConfig,
    accelerator: str | None,
) -> None:
    """Configure TF32 and float32 matmul precision when CUDA is available."""

    if accelerator not in {None, "auto", "gpu"} or not torch.cuda.is_available():
        return

    torch.backends.cudnn.allow_tf32 = precision_config.tf32
    torch.set_float32_matmul_precision("high" if precision_config.tf32 else "highest")


def build_strategy(config: DistributedConfig) -> str | DDPStrategy | ModelParallelStrategy:
    """Build the Lightning strategy object from configuration."""

    if config.strategy == "auto":
        return "auto"
    if config.strategy == "ddp":
        return DDPStrategy(
            find_unused_parameters=config.find_unused_parameters,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
        )
    if config.strategy in {"fsdp2", "model_parallel"}:
        return ModelParallelStrategy(
            data_parallel_size=config.data_parallel_size,
            tensor_parallel_size=config.tensor_parallel_size,
            **config.model_parallel_specific_kwargs,
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
        "sync_batchnorm": distributed_config.sync_batchnorm,
    }
