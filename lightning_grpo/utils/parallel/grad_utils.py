"""Mesh-aware parameter grouping and gradient clipping.

A model that mixes sharding strategies does not keep its parameters on one
``DeviceMesh``. With the plan this repository ships:

* TP/FSDP weights are ``DTensor`` on ``(data_parallel, tensor_parallel)``;
* EP experts are ``DTensor`` on ``(efsdp, expert_parallel)``;
* anything created outside ``fully_shard`` (router bias, replicated buffers
  promoted to parameters, ...) stays a plain ``torch.Tensor``.

Two "whole set" kernels reject that mix, which is why enabling TP+EP together
fails in the optimizer step rather than in the model:

* ``torch.nn.utils.clip_grad_norm_`` stacks every per-parameter norm into one
  vector. Those norms are ``DTensor`` scalars carrying the mesh of their own
  gradient, so two meshes raise
  ``All operands in aten.stack.default must have the same mesh``.
* ``AdamW(foreach=True)`` batches every parameter of a device into one
  ``_foreach_*`` call. A plain tensor with ``numel > 1`` next to a ``DTensor``
  raises ``got mixed torch.Tensor and DTensor`` (only ``numel == 1`` tensors are
  implicitly treated as replicated).

Both upstream trainers solve it by grouping on the mesh, running the whole-set
kernel per group, and combining the groups into one scalar: see
``transformers.Trainer._mixed_mesh_grad_norm`` / ``has_mixed_dtensor`` and
torchtitan's ``distributed.utils.clip_grad_norm_``. Grouping in the optimizer
also lets ``foreach`` stay on, which is faster than disabling it globally.

This module only *groups*; it does not change which collective each sharding
strategy performs. Every ``DTensor`` norm is reduced over its own mesh by
``torch.nn.utils.get_total_norm`` before it is merged, and the merged norm is
then used to scale every group, so all parameters keep their relative scale.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

__all__ = [
    "MixedMeshGradientClippingMixin",
    "clip_grad_norm_",
    "get_total_grad_norm",
    "group_by_mesh",
    "has_mixed_mesh",
    "iter_group_tensors",
    "split_param_groups_by_mesh",
]


def _mesh_of(tensor: torch.Tensor) -> DeviceMesh | None:
    """The mesh a sharded tensor lives on, or ``None`` for a plain tensor."""

    return tensor.device_mesh if isinstance(tensor, DTensor) else None


def group_by_mesh(tensors: Iterable[torch.Tensor]) -> dict[DeviceMesh | None, list[torch.Tensor]]:
    """Group tensors by their ``DeviceMesh`` (``None`` groups plain tensors).

    Insertion order follows the order tensors are seen. Every rank sees the same
    model in the same order, so the groups line up and the collectives inside
    ``get_total_norm`` / ``clip_grads_with_norm_`` are issued in the same order.
    """

    grouped: dict[DeviceMesh | None, list[torch.Tensor]] = defaultdict(list)
    for tensor in tensors:
        grouped[_mesh_of(tensor)].append(tensor)
    return grouped


def has_mixed_mesh(tensors: Iterable[torch.Tensor]) -> bool:
    """Whether ``tensors`` span more than one mesh, so whole-set kernels cannot span them."""

    meshes: set[DeviceMesh | None] = set()
    for tensor in tensors:
        meshes.add(_mesh_of(tensor))
        if len(meshes) > 1:
            return True
    return False


def iter_group_tensors(parameters: Iterable[Any]) -> Iterator[torch.Tensor]:
    """Yield the tensors of ``parameters``, accepting a bare iterable or param groups."""

    for entry in parameters:
        if isinstance(entry, dict):
            yield from entry["params"]
        else:
            yield entry


def split_param_groups_by_mesh(parameters: Iterable[Any]) -> list[dict[str, Any]]:
    """Split parameter groups so that each returned group holds a single mesh.

    Group options (``lr``, ``weight_decay``, ...) are copied onto every split so
    the optimizer behaves exactly as before; only the membership changes.
    """

    groups: list[dict[str, Any]] = []
    for entry in parameters:
        if isinstance(entry, dict):
            options = {key: value for key, value in entry.items() if key != "params"}
            params = list(entry["params"])
        else:
            options, params = {}, [entry]
        for mesh_params in group_by_mesh(params).values():
            groups.append({**options, "params": mesh_params})
    return groups


@torch.no_grad()
def get_total_grad_norm(
    parameters: Iterable[torch.Tensor],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Compute one gradient norm over parameters that may live on different meshes.

    Returns a local, 0-dim tensor holding the same value on every rank.
    """

    params = [param for param in parameters if param.grad is not None]
    norms: list[torch.Tensor] = []
    for grads in group_by_mesh(param.grad for param in params).values():
        norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite)
        # A sharded norm is a DTensor carrying a `_NormPartial` placement: the
        # per-mesh reduction already happened, `full_tensor()` is a no-op copy
        # for the value and gives us a materialized local scalar.
        norms.append(norm.full_tensor() if isinstance(norm, DTensor) else norm)

    if not norms:
        return torch.zeros(())

    if len(norms) == 1:
        return norms[0]

    device = norms[0].device
    stacked = torch.stack([norm.to(device) for norm in norms])
    if math.isinf(norm_type):
        return stacked.amax()
    return stacked.pow(norm_type).sum().pow(1.0 / norm_type)


@torch.no_grad()
def clip_grad_norm_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Clip gradients of a multi-mesh parameter set to ``max_norm``, returning the pre-clip norm."""

    params = [param for param in parameters if param.grad is not None]
    total_norm = get_total_grad_norm(params, norm_type, error_if_nonfinite)
    if float(max_norm) > 0:
        # One global coefficient for every group: clipping each mesh separately
        # would change the relative scale of the parameter groups.
        for mesh_params in group_by_mesh(params).values():
            torch.nn.utils.clip_grads_with_norm_(mesh_params, max_norm, total_norm)
    return total_norm


class MixedMeshGradientClippingMixin:
    """``configure_gradient_clipping`` that spans several device meshes.

    Mix into a ``LightningModule`` ahead of the base class. When the gradients
    live on one mesh this delegates to Lightning unchanged; when they do not, it
    clips through :func:`clip_grad_norm_`.
    """

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        clip_val = gradient_clip_val
        if clip_val is None:
            clip_val = getattr(self.trainer, "gradient_clip_val", None) or 0.0

        algorithm = gradient_clip_algorithm
        if algorithm is None:
            algorithm = getattr(self.trainer, "gradient_clip_algorithm", None)
        algorithm = str(getattr(algorithm, "value", algorithm) or "norm").lower()

        params = [
            param
            for group in optimizer.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if float(clip_val) > 0 and algorithm == "norm" and has_mixed_mesh(param.grad for param in params):
            clip_grad_norm_(params, float(clip_val))
            return

        super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
