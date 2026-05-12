"""Composable FSDP2 helpers used from LightningModule.configure_model."""

from __future__ import annotations

from typing import Any

from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh
import torch
import torch.nn as nn

from lightning_grpo.utils.configs.base import DistributedConfig, PrecisionConfig
from lightning_grpo.utils.modeling import resolve_torch_dtype
from lightning_grpo.utils.fsdp_helper import get_fsdp_reshard_after_forward_policy


def configure_fully_shard(
    model: nn.Module,
    distributed_config: DistributedConfig,
    precision_config: PrecisionConfig,
    device_mesh: DeviceMesh,
) -> None:
    """Apply PyTorch composable FSDP2 ``fully_shard`` using config-only policies.

    This helper keeps Lightning modules generic: each module only passes its root
    model (for example ``self.model`` or ``self.policy``), while YAML controls
    which child blocks are sharded by class path or by module-name glob.
    """

    if distributed_config.strategy not in {"fsdp2", "model_parallel"}:
        return

    dp_mesh = device_mesh["data_parallel"]
    if dp_mesh.size() <= 1:
        return

    _apply_fsdp(
        model=model,
        dp_mesh=dp_mesh,
        param_dtype=resolve_torch_dtype(precision_config.fsdp_param_dtype),
        reduce_dtype=resolve_torch_dtype(precision_config.fsdp_reduce_dtype),
        pp_enabled=False,
        shard_kwargs=dict(distributed_config.fsdp_fully_shard_kwargs),
        cpu_offload=distributed_config.fsdp_cpu_offload,
        forward_prefetch=distributed_config.fsdp_forward_prefetch,
        backward_prefetch=distributed_config.fsdp_backward_prefetch,
        reshard_after_forward_policy=distributed_config.fsdp_reshard_after_forward,
    )


def _apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    shard_kwargs: dict[str, Any],
    cpu_offload: bool = False,
    forward_prefetch: bool = True,
    backward_prefetch: bool = True,
    reshard_after_forward_policy: str = "default",
) -> None:
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        shard_kwargs (dict[str, Any]): Extra keyword arguments passed to ``fully_shard``.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        forward_prefetch (bool, optional): Whether to configure forward prefetch between FSDP modules. Defaults to True.
        backward_prefetch (bool, optional): Whether to configure backward prefetch between FSDP modules. Defaults to True.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy, **shard_kwargs}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for transformer_block in model.layers:

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.lm_head is not None:
        fully_shard(
            [model.norm, model.lm_head],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)

    # forward
    if forward_prefetch:
        transformer_blocks = list(model.layers.values())
        next_transformer_blocks = transformer_blocks[1:] + [None]

        if model.tok_embeddings is not None and model.layers is not None:
            model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

        for transformer_block, next_transformer_block in zip(
            transformer_blocks, next_transformer_blocks
        ):
            if next_transformer_block is not None:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block]
                )
            elif model.norm is not None and model.lm_head is not None:
                transformer_block.set_modules_to_forward_prefetch(
                    [model.norm, model.lm_head]
                )

    # backward
    if backward_prefetch:
        reversed_transformer_blocks = list(reversed(model.layers.values()))
        prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

        if (
            model.norm is not None
            and model.lm_head is not None
            and model.layers is not None
        ):
            model.lm_head.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

        for transformer_block, prev_transformer_block in zip(
            reversed_transformer_blocks, prev_transformer_blocks
        ):
            if prev_transformer_block is not None:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block]
                )
            elif model.tok_embeddings is not None:
                transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])
