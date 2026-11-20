"""Materialize DTensors without ``DTensor.full_tensor()`` / ``redistribute()``.

torch 2.10+ rewrites ``_StridedShard`` placements into an explicit shard order and
rejects the ones whose ``split_factor`` is not a product of mesh sizes -- exactly the
case of HF's ``PackedColwiseParallel``, where ``split_factor`` is a packing ratio (2
for fused gate/up), not a mesh dimension. The per-placement gather primitives still
work, so walk the mesh dimensions here instead of going through ``redistribute``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Shard
from torch.distributed.tensor.placement_types import _StridedShard


SHARD_PLACEMENTS = Shard | _StridedShard


def gather_full_tensor(tensor: Any) -> Any:
    """Return the replicated value of ``tensor``, gathering over its mesh dimensions.

    Every rank of the DTensor's mesh must call this. Non-DTensor values pass through.
    """
    if not isinstance(tensor, DTensor):
        return tensor

    mesh: DeviceMesh = tensor.device_mesh
    coordinate = mesh.get_coordinate()
    placements = tensor.placements

    # Shape shared by the ranks of each mesh dim: the tensor before that dim's sharding.
    # Placements are applied left to right, so gather them back right to left.
    logical_shapes: list[list[int]] = []
    current = list(tensor.shape)
    for mesh_dim, placement in enumerate(placements):
        logical_shapes.append(current)
        if isinstance(placement, SHARD_PLACEMENTS) and mesh.size(mesh_dim) > 1:
            current = list(current)
            current[placement.dim] = placement.local_shard_size_and_offset(
                current[placement.dim], mesh.size(mesh_dim), coordinate[mesh_dim]
            )[0]

    local = tensor.to_local()
    for mesh_dim in reversed(range(len(placements))):
        if mesh.size(mesh_dim) == 1:
            continue
        placement = placements[mesh_dim]
        if isinstance(placement, SHARD_PLACEMENTS):
            # Also covers the strided layout of packed (fused) weights.
            logical_shape = list(logical_shapes[mesh_dim])
            local = placement._to_replicate_tensor(local, mesh, mesh_dim, logical_shape)
        elif isinstance(placement, Partial):
            local = placement._reduce_value(local, mesh, mesh_dim)

    # Collectives are async. `Shard._to_replicate_tensor` returns the raw
    # ``AsyncCollectiveTensor`` (the strided one reshapes it, which waits implicitly), and
    # ``.cpu()` on a CPU tensor is a no-op, so the wrapper would reach ``torch.save``
    # and produce a file that ``torch.load(weights_only=True)`` refuses.
    if isinstance(local, AsyncCollectiveTensor):
        local = local.wait()

    return local


def gather_full_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """``gather_full_tensor`` over a state dict; non-tensor entries are kept as-is."""
    return {key: gather_full_tensor(value) for key, value in state_dict.items()}
