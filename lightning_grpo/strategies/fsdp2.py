"""Composable FSDP2 helpers used from LightningModule.configure_model."""

from __future__ import annotations

from typing import Any

from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh
import torch
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.strategies.hsdp import resolve_hsdp_mesh
from lightning_grpo.utils.modeling_common import get_transformer_backbone_model
from lightning_grpo.utils.configs.base import DistributedConfig, PrecisionConfig
from lightning_grpo.utils.dtype import resolve_torch_dtype
from lightning_grpo.utils.parallel.fsdp_utils import get_fsdp_reshard_after_forward_policy


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
    expert_mesh = getattr(model, "_expert_mesh", None)
    replicate_size = distributed_config.data_parallel_replicate_size
    if dp_mesh.size() <= 1 and expert_mesh is None:
        if replicate_size > 1:
            raise ValueError(
                "data_parallel_replicate_size > 1 requires a data-parallel size > 1; "
                "increase the data-parallel world size or reduce the replication factor."
            )
        return

    hsdp_mesh = resolve_hsdp_mesh(dp_mesh, replicate_size)
    if hsdp_mesh.ndim == 2:
        rank_zero_info(
            f"Applying HSDP: dp_replicate={replicate_size}, dp_shard={hsdp_mesh.size() // replicate_size}"
        )

    _apply_fsdp(
        model=model,
        dp_mesh=hsdp_mesh,
        expert_mesh=expert_mesh,
        expert_replicate_size=replicate_size,
        param_dtype=resolve_torch_dtype(precision_config.fsdp_param_dtype),
        reduce_dtype=resolve_torch_dtype(precision_config.fsdp_reduce_dtype),
        pp_enabled=False,
        shard_kwargs=dict(distributed_config.fsdp_fully_shard_kwargs),
        cpu_offload=distributed_config.fsdp_cpu_offload,
        forward_prefetch=distributed_config.fsdp_forward_prefetch,
        backward_prefetch=distributed_config.fsdp_backward_prefetch,
        reshard_after_forward_policy=distributed_config.fsdp_reshard_after_forward,
    )


def _shard_expert_modules(
    model: nn.Module,
    expert_mesh: DeviceMesh,
    dp_size: int,
    mp_policy: MixedPrecisionPolicy,
    shard_kwargs: dict[str, Any],
    cpu_offload: bool,
    replicate_size: int,
) -> None:
    """Shard the EP dispatch modules on the expert view of the ranks.

    The experts hold one EP shard per rank, so their FSDP axis is ``efsdp``. An EP
    group holds ``ep_size / tp_size`` distinct batches, hence the expert gradient
    spans ``dp_size`` batches while its FSDP group only has ``efsdp`` ranks. Under
    HSDP the ``efsdp`` axis is further split into ``(efsdp_replicate, efsdp_shard)``;
    the divide factor stays ``dp_size`` because the replicate all-reduce sums the
    same distinct batches rather than adding new ones.
    """
    efsdp_mesh = resolve_hsdp_mesh(
        expert_mesh["efsdp"],
        replicate_size,
        mesh_dim_names=("efsdp_replicate", "efsdp_shard"),
    )
    if "mesh" in shard_kwargs:
        raise ValueError("fsdp_fully_shard_kwargs must not set 'mesh'; it is derived from the device mesh.")
    expert_fsdp_config = {"mp_policy": mp_policy, **shard_kwargs, "mesh": efsdp_mesh}
    if cpu_offload:
        expert_fsdp_config["offload_policy"] = CPUOffloadPolicy()

    expert_modules = [
        module for module in model.modules() if getattr(module, "_ep_dispatch_experts", False)
    ]
    for module in expert_modules:
        fully_shard(module, **expert_fsdp_config, reshard_after_forward=True)
        module.set_gradient_divide_factor(float(dp_size))
        if efsdp_mesh.size() > 1:
            # `set_gradient_divide_factor` uses PREMUL_SUM for factors other than the
            # group size, which non-NCCL backends reject (torch#190747).
            module.set_force_sum_reduction_for_comms(True)


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
    expert_mesh: DeviceMesh | None = None,
    expert_replicate_size: int = 1,
) -> None:
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism. A 2D
            ``(replicate, shard)`` mesh enables HSDP.
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
        expert_mesh (DeviceMesh | None, optional): The ``(efsdp, expert_parallel)``
            mesh built by the EP strategy. Defaults to None (EP disabled).
        expert_replicate_size (int, optional): HSDP replicate degree applied to the
            expert ``efsdp`` axis. Defaults to 1 (pure FSDP over experts).

    """
    if "mesh" in shard_kwargs:
        raise ValueError("fsdp_fully_shard_kwargs must not set 'mesh'; it is derived from the device mesh.")
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, **shard_kwargs, "mesh": dp_mesh}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if expert_mesh is not None:
        _shard_expert_modules(
            model,
            expert_mesh,
            dp_mesh.size(),
            mp_policy,
            shard_kwargs,
            cpu_offload,
            expert_replicate_size,
        )

    if dp_mesh.size() <= 1:
        return

    base_model = get_transformer_backbone_model(model)
    tie_embeddings = getattr(model.config, "tie_word_embeddings", False)

    embed_tokens = getattr(base_model, "embed_tokens", None)
    layers = getattr(base_model, "layers", [])
    norm = getattr(base_model, "norm", None)
    lm_head = getattr(model, "lm_head", None)

    if tie_embeddings:
        tied_modules = []
        if embed_tokens is not None:
            tied_modules.append(embed_tokens)
        if norm is not None:
            tied_modules.append(norm)
        if lm_head is not None:
            tied_modules.append(lm_head)

        if tied_modules:
            fully_shard(
                tied_modules,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )
    else:
        if embed_tokens is not None:
            fully_shard(
                embed_tokens,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        if norm is not None and lm_head is not None:
            fully_shard(
                [norm, lm_head],
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )

    for transformer_block in layers:
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config)

    # forward
    if forward_prefetch and isinstance(layers, (list, nn.ModuleList)) and len(layers) > 0:
        transformer_blocks = list(layers)
        next_transformer_blocks = transformer_blocks[1:] + [None]

        if embed_tokens is not None:
            embed_tokens.set_modules_to_forward_prefetch([transformer_blocks[0]])

        for transformer_block, next_transformer_block in zip(
                transformer_blocks, next_transformer_blocks
        ):
            if next_transformer_block is not None:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block]
                )
            elif norm is not None and lm_head is not None:
                transformer_block.set_modules_to_forward_prefetch(
                    [norm, lm_head]
                )

    # backward
    if backward_prefetch and isinstance(layers, (list, nn.ModuleList)) and len(layers) > 0:
        reversed_transformer_blocks = list(reversed(layers))
        prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

        if norm is not None and lm_head is not None:
            lm_head.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

        for transformer_block, prev_transformer_block in zip(
                reversed_transformer_blocks, prev_transformer_blocks
        ):
            if prev_transformer_block is not None:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block]
                )
            elif embed_tokens is not None:
                transformer_block.set_modules_to_backward_prefetch([embed_tokens])
