"""Torch dtype helpers shared by the model-building utilities."""

import torch

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def resolve_torch_dtype(model_param_dtype: str) -> torch.dtype:
    """Resolve the parameter dtype from configuration."""

    return DTYPE_MAP[model_param_dtype]
