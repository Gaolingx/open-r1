# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypeGuard

from transformers.utils import (
    is_torch_available,
    is_torch_distributed_available,
    is_torch_greater_or_equal,
    logging,
)


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor


if is_torch_available():
    import torch


def _check_distributed_checkpointing_available() -> bool:
    # `torch.distributed.checkpoint.hf_storage` (HuggingFaceStorageReader/Writer) first ships in torch 2.8.
    return is_torch_distributed_available() and is_torch_greater_or_equal("2.8")


def _is_torch_distributed_initialized() -> bool:
    if not is_torch_distributed_available():
        return False
    return torch.distributed.is_initialized()


def is_dtensor(obj: object) -> TypeGuard[DTensor]:
    if not is_torch_distributed_available():
        return False
    from torch.distributed.tensor import DTensor

    return isinstance(obj, DTensor)


def _get_torch_distributed_rank() -> int:
    if not _is_torch_distributed_initialized():
        return 0
    return torch.distributed.get_rank()


def _warning_once_on_rank_zero(message: str) -> None:
    """Log a warning once per process, on rank 0 only."""
    if _get_torch_distributed_rank() == 0:
        logger.warning_once(message)


def _get_torch_distributed_world_size() -> int:
    if not _is_torch_distributed_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_local_dist_rank_0() -> bool:
    return _is_torch_distributed_initialized() and int(os.environ.get("LOCAL_RANK", "-1")) == 0


DISTRIBUTED_BACKEND_MAP = {
    "cuda": "nccl",
    "cpu": "gloo",
    "xpu": "xccl",
    "hpu": "hccl",
    "neuron": "neuron",
    "tpu": "tpu_dist",
}


def get_distributed_backend(device_type: str) -> str | None:
    """Return the `torch.distributed` backend for `device_type`, or `None` if it has no dedicated one."""
    dist_backend = DISTRIBUTED_BACKEND_MAP.get(device_type)
    if dist_backend is None:
        raise ValueError(f"No distributed backend found for device type '{device_type}'")
    return dist_backend
