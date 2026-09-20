"""Metrics helper for the Lightning GRPO pipeline."""

from __future__ import annotations

from typing import Any, Optional

import torch


def _get_output_value(outputs: Any, key: str) -> Any:
    if isinstance(outputs, dict):
        return outputs.get(key)
    return getattr(outputs, key, None)


def format_metric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu().item())
    return float(value)
