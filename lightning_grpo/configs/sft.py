"""SFT-specific configuration for the Lightning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from lightning_grpo.configs.base import ExperimentConfig


@dataclass
class SFTConfig(ExperimentConfig):
    """Configuration for supervised fine-tuning."""

    task: Literal["sft"] = "sft"
    label_smoothing: float = 0.0
    completion_only_loss: bool = False
    eval_generation_max_new_tokens: int = 256
    system_prompt: Optional[str] = None
    validation_prompts: list[str] = field(default_factory=list)
