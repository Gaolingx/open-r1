"""Narrow interfaces shared by the GRPO collaborators.

The GRPO helpers used to receive the whole :class:`~lightning_grpo.models.grpo_module.GRPOLightningModule`
and read back ``module.config`` / ``module.policy`` / ``module.trainer`` / ``module.rollout_coordinator``.
That back-reference formed a cycle (module -> helper -> module) and made the helpers impossible to
unit test without faking a Lightning module. Each helper now receives the smallest thing it needs:

* configuration as plain values (the ``*Config`` dataclasses),
* the policy and reference model as *providers* (:data:`PolicyFn`), because tensor parallelism,
  FSDP2 and ``torch.compile`` re-bind ``module.policy`` after construction,
* distribution as a value object (:class:`DistributedContext`),
* metric publishing as a structural protocol (:class:`MetricsSink`),
* reward scoring as a structural protocol (:class:`RewardScorer`).

This module is deliberately a leaf: it must not import ``grpo_module`` (nor anything that imports
it) so the dependency graph stays acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

#: All-gathers a tensor across ranks and returns the concatenation along dim 0.
GatherFn = Callable[[torch.Tensor], torch.Tensor]
#: Returns the current policy or reference model.
PolicyFn = Callable[[], Any]
#: Builds a rollout engine from the keyword arguments accepted by ``create_rollout_engine``.
EngineFactory = Callable[..., Any]


def _pad_to_shape(tensor: torch.Tensor, shape: list[int]) -> torch.Tensor:
    """Right-pad ``tensor`` with zeros up to ``shape`` (no-op when it already matches)."""

    padding: list[int] = []
    for size, target in zip(reversed(tensor.shape), reversed(shape)):
        padding.extend((0, target - size))
    if not any(padding):
        return tensor
    return F.pad(tensor, padding)


@dataclass(frozen=True)
class DistributedContext:
    """Rank information plus the all-gather service shared by the GRPO collaborators."""

    rank: int = 0
    world_size: int = 1
    all_gather_fn: GatherFn | None = None

    @property
    def is_distributed(self) -> bool:
        """Whether more than one rank participates in the run."""

        return self.world_size > 1

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather ``tensor`` across ranks, tolerating per-rank shape differences.

        Ranks can hold differently shaped completion batches, so the local shapes are
        gathered first and every shard is right-padded to the largest one.
        """

        if not self.is_distributed:
            return tensor
        if self.all_gather_fn is None:
            raise RuntimeError("DistributedContext requires all_gather_fn when world_size > 1.")
        if tensor.dim() == 0:
            return self.all_gather_fn(tensor).reshape(-1)

        local_size = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
        gathered_sizes = self.all_gather_fn(local_size).reshape(-1, tensor.dim())
        max_size = gathered_sizes.max(dim=0).values.tolist()

        gathered = self.all_gather_fn(_pad_to_shape(tensor, max_size))
        return gathered.reshape(-1, *max_size[1:])


@runtime_checkable
class MetricsSink(Protocol):
    """The part of ``LightningModule`` the metrics aggregator needs to publish metrics.

    Parameter types are intentionally permissive so the protocol describes the minimum
    surface of Lightning's logger API instead of re-validating it.
    """

    def log(
        self,
        name: str,
        value: Any,
        *,
        prog_bar: bool = False,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        sync_dist: bool = False,
    ) -> Any: ...

    def log_dict(
        self,
        dictionary: Mapping[str, Any],
        *,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        sync_dist: bool = False,
    ) -> Any: ...


@runtime_checkable
class RewardScorer(Protocol):
    """Scoring surface of the reward manager as consumed by the loss computer."""

    reward_weight_tensor: torch.Tensor
    reward_names: list[str]

    def compute_rewards(
        self,
        *,
        prompts: list[str],
        completions: list[str] | list[list[dict[str, str]]],
        completion_id_lists: list[list[int]],
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
