"""Unit tests for the HSDP mesh helpers (no distributed setup required)."""

from __future__ import annotations

import pytest
import torch

from lightning_grpo.strategies.hsdp import (
    DEFAULT_MESH_DIM_NAMES,
    _HSDP_MESH_CACHE,
    resolve_hsdp_degrees,
    resolve_hsdp_mesh,
)


@pytest.mark.parametrize('nodes,devices', [(1, 8), (2, 4)])
def test_fsdp_without_tp_uses_entire_world(nodes, devices):
    from lightning.fabric.plugins.environments import LightningEnvironment
    from lightning_grpo.strategies.factory import build_strategy
    from lightning_grpo.utils.configs.base import DistributedConfig

    strategy = build_strategy(DistributedConfig(strategy='fsdp2'))
    strategy.cluster_environment = LightningEnvironment()
    strategy.parallel_devices = [torch.device('cpu')] * devices
    strategy.num_nodes = nodes
    strategy.set_world_ranks()
    assert (strategy._data_parallel_size, strategy._tensor_parallel_size) == (nodes * devices, 1)


def test_disabled_tp_rejects_explicit_tp_mesh():
    from lightning_grpo.strategies.factory import build_strategy
    from lightning_grpo.utils.configs.base import DistributedConfig

    with pytest.raises(ValueError, match='tensor_parallel.enabled'):
        build_strategy(DistributedConfig(strategy='model_parallel', tensor_parallel_size=2))


@pytest.fixture(autouse=True)
def _clear_mesh_cache():
    _HSDP_MESH_CACHE.clear()
    yield
    _HSDP_MESH_CACHE.clear()


class _FakeMesh:
    """Minimal stand-in for a 1D ``DeviceMesh`` that records unflatten calls."""

    def __init__(self, ranks: list[int], *, ndim: int = 1) -> None:
        self.device_type = "cpu"
        self.mesh = torch.tensor(ranks)
        self.ndim = ndim
        self.unflatten_calls: list[tuple] = []

    def size(self, mesh_dim: int | None = None) -> int:
        return self.mesh.numel()

    def _unflatten(self, dim, mesh_sizes, mesh_dim_names):
        self.unflatten_calls.append((dim, tuple(mesh_sizes), tuple(mesh_dim_names)))
        return ("unflattened", tuple(mesh_sizes), tuple(mesh_dim_names))


@pytest.mark.parametrize(
    ("dp_size", "replicate", "expected"),
    [
        (8, 1, (1, 8)),
        (8, 2, (2, 4)),
        (8, 8, (8, 1)),
        (1, 1, (1, 1)),
    ],
)
def test_resolve_hsdp_degrees(dp_size, replicate, expected):
    assert resolve_hsdp_degrees(dp_size, replicate) == expected


@pytest.mark.parametrize("replicate", [0, -1])
def test_resolve_hsdp_degrees_rejects_non_positive(replicate):
    with pytest.raises(ValueError, match="must be >= 1"):
        resolve_hsdp_degrees(8, replicate)


@pytest.mark.parametrize(("dp_size", "replicate"), [(8, 3), (1, 2)])
def test_resolve_hsdp_degrees_rejects_non_divisor(dp_size, replicate):
    with pytest.raises(ValueError, match="must divide"):
        resolve_hsdp_degrees(dp_size, replicate)


def test_resolve_hsdp_degrees_uses_label_in_error():
    with pytest.raises(ValueError, match="expert data parallel size"):
        resolve_hsdp_degrees(3, 2, label="expert data parallel")


def test_replicate_one_returns_the_fsdp_mesh():
    dp_mesh = _FakeMesh(list(range(8)))

    assert resolve_hsdp_mesh(dp_mesh, 1) is dp_mesh
    assert dp_mesh.unflatten_calls == []


def test_replicate_gt_one_unflattens_and_caches():
    dp_mesh = _FakeMesh(list(range(8)))

    mesh = resolve_hsdp_mesh(dp_mesh, 2)

    assert mesh == ("unflattened", (2, 4), DEFAULT_MESH_DIM_NAMES)
    assert dp_mesh.unflatten_calls == [(0, (2, 4), DEFAULT_MESH_DIM_NAMES)]
    assert resolve_hsdp_mesh(dp_mesh, 2) is mesh
    assert len(dp_mesh.unflatten_calls) == 1


def test_custom_mesh_dim_names_are_forwarded():
    expert_mesh = _FakeMesh(list(range(8)))

    resolve_hsdp_mesh(expert_mesh, 4, mesh_dim_names=("efsdp_replicate", "efsdp_shard"))

    assert expert_mesh.unflatten_calls == [(0, (4, 2), ("efsdp_replicate", "efsdp_shard"))]


def test_two_dimensional_mesh_is_rejected():
    with pytest.raises(ValueError, match="1D data-parallel mesh"):
        resolve_hsdp_mesh(_FakeMesh(list(range(4)), ndim=2), 2)
