"""Pretraining-specific configuration for the Lightning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lightning_grpo.utils.configs.base import DataConfig, TrainingBaseConfig


@dataclass
class PretrainDataConfig(DataConfig):
    """Dataset configuration used by plain text pretraining."""

    text_column: str = "text"


@dataclass
class PretrainConfig(TrainingBaseConfig):
    """Configuration for causal language model pretraining."""

    task: Literal["pretrain"] = "pretrain"
    data: PretrainDataConfig = field(default_factory=PretrainDataConfig)
