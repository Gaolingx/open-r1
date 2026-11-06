# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributed training backend abstraction."""

from contextlib import contextmanager


# Process group backing the object collectives below, created on first use.
_OBJECT_GROUP: object | None = None
_OBJECT_GROUP_RESOLVED = False


def _object_group():
    """Process group for object collectives, or ``None`` when running single-process.

    A replacement for `accelerate.utils.gather_object` needs one because Lightning's ``ModelParallelStrategy``
    initializes the default group with ``device_id=<this rank's GPU>``. PyTorch picks the staging device for object
    collectives from the group's backends, so payloads are staged in CUDA tensors, and on this stack that collective
    is rejected with ``Attempt to perform collective on tensor not on device passed to init_process_group``. A gloo
    group has no device binding and carries the same payloads.

    ``new_group`` is itself a collective, so it can only be created once every rank runs the same call site, which
    is why this is resolved lazily rather than in a constructor.
    """

    global _OBJECT_GROUP, _OBJECT_GROUP_RESOLVED
    if not _OBJECT_GROUP_RESOLVED:
        _OBJECT_GROUP_RESOLVED = True

        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            _OBJECT_GROUP = dist.new_group(backend="gloo")
    return _OBJECT_GROUP


def gather_object(obj):
    """Drop-in for `accelerate.utils.gather_object`, flattened the same way."""

    group = _object_group()
    if group is None:
        return obj

    import torch.distributed as dist

    gathered: list = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, obj, group=group)
    return [item for rank_objects in gathered for item in rank_objects]


def broadcast_object_list(object_list, from_process: int = 0) -> None:
    """Drop-in for `accelerate.utils.broadcast_object_list`."""

    group = _object_group()
    if group is None:
        return

    import torch.distributed as dist

    dist.broadcast_object_list(object_list, src=from_process, group=group)


class DistributedBackend:
    """Abstracts distributed backend specifics (DeepSpeed ZeRO, FSDP) behind a uniform API.

    Detects the active backend once at construction from ``accelerator.state``, then provides context managers that are
    no-ops on backends where the operation is not needed.

    Args:
        accelerator ([`~accelerate.Accelerator`]):
            The accelerator instance managing the distributed state.

    Example:

    ```python
    >>> dist = DistributedBackend(accelerator)
    >>> with dist.gather_params(list(model.parameters())):
    ...     model.merge_adapter()
    >>> with dist.summon_full_params(model, recurse=False):
    ...     outputs = model.generate(inputs)
    ```
    """

    def __init__(self, accelerator):
        ds_plugin = accelerator.state.deepspeed_plugin
        fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
        self.zero_stage = ds_plugin.zero_stage if ds_plugin else 0
        self.fsdp_version = getattr(fsdp_plugin, "fsdp_version", None) if fsdp_plugin else None

    @property
    def is_zero3(self) -> bool:
        """Whether DeepSpeed ZeRO Stage 3 is active."""
        return self.zero_stage == 3

    @property
    def is_fsdp(self) -> bool:
        """Whether FSDP (any version) is active."""
        return self.fsdp_version is not None

    @contextmanager
    def gather_params(self, params):
        """Gather sharded parameters under DeepSpeed ZeRO-3; no-op otherwise.

        Args:
            params (iterable of `torch.nn.Parameter`):
                Parameters to gather.
        """
        if self.is_zero3:
            import deepspeed

            with deepspeed.zero.GatheredParameters(params):
                yield
        else:
            yield

    @contextmanager
    def summon_full_params(self, module, **kwargs):
        """Materialize full FSDP v1 parameters; no-op for FSDP v2 or non-FSDP backends.

        FSDP v2 parameters are always accessible and require no explicit materialization.

        Args:
            module (`torch.nn.Module`):
                The FSDP-wrapped module.
            **kwargs:
                Forwarded to ``FSDP.summon_full_params`` (e.g. ``recurse``, ``writeback``).
        """
        if self.fsdp_version == 1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(module, **kwargs):
                yield
        else:
            yield
