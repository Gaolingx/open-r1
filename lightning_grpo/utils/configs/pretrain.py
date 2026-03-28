"""Pretraining-specific configuration for the Lightning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lightning_grpo.utils.configs.base import ExperimentConfig


@dataclass
class PretrainConfig(ExperimentConfig):
    """Configuration for causal language model pretraining."""

    task: Literal["pretrain"] = "pretrain"
