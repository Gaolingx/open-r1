"""HSDP mesh helpers for the Lightning pipeline.

Hybrid sharded data parallelism is FSDP over a 2D ``(replicate, shard)`` device
mesh: ``fully_shard`` replicas on dim 0 and shards on dim 1. Lightning's
``ModelParallelStrategy`` only builds a 1D ``data_parallel`` axis, so the
replicate/shard split is recovered by unflattening that axis. The flattened
``data_parallel`` axis is kept as-is for sampling and every other consumer.
"""

from __future__ import annotations

from torch.distributed.device_mesh import DeviceMesh

DEFAULT_MESH_DIM_NAMES = ("dp_replicate", "dp_shard")

_HSDP_MESH_CACHE: dict[tuple, DeviceMesh] = {}


def resolve_hsdp_degrees(
    dp_size: int,
    replicate_size: int,
    *,
    label: str = "data parallel",
) -> tuple[int, int]:
    """Split ``dp_size`` into ``(replicate, shard)`` degrees.

    ``replicate_size=1`` keeps pure FSDP (shard over the whole DP axis).
    """
    if replicate_size < 1:
        raise ValueError(f"data_parallel_replicate_size must be >= 1, got {replicate_size}.")
    if replicate_size == 1:
        return 1, dp_size
    if dp_size % replicate_size:
        raise ValueError(
            f"data_parallel_replicate_size ({replicate_size}) must divide the "
            f"{label} size ({dp_size})."
        )
    return replicate_size, dp_size // replicate_size


def resolve_hsdp_mesh(
    dp_mesh: DeviceMesh,
    replicate_size: int,
    *,
    mesh_dim_names: tuple[str, ...] = DEFAULT_MESH_DIM_NAMES,
) -> DeviceMesh:
    """Return the 1D ``dp_mesh`` unchanged, or its 2D ``(replicate, shard)`` HSDP view.

    The view is cached: the policy and reference models both ask for it, and
    unflattening twice would create duplicate process groups.
    """
    if dp_mesh.ndim != 1:
        raise ValueError(f"HSDP needs a 1D data-parallel mesh, got {dp_mesh.ndim}D.")
    replicate, shard = resolve_hsdp_degrees(dp_mesh.size(), replicate_size)
    if replicate == 1:
        return dp_mesh

    key = (
        dp_mesh.device_type,
        tuple(dp_mesh.mesh.flatten().tolist()),
        replicate,
        tuple(mesh_dim_names),
    )
    mesh = _HSDP_MESH_CACHE.get(key)
    if mesh is None:
        mesh = dp_mesh._unflatten(0, (replicate, shard), tuple(mesh_dim_names))
        _HSDP_MESH_CACHE[key] = mesh
    return mesh
