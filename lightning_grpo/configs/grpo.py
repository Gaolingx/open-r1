"""GRPO-specific configuration for the Lightning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from lightning_grpo.configs.base import ExperimentConfig


@dataclass
class RewardConfig:
    """Reward function configuration for GRPO."""

    reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format", "tag_count"])
    code_language: str = "python"
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0
    cosine_min_value_wrong: float = 0.0
    cosine_max_value_wrong: float = -0.5
    cosine_min_value_correct: float = 0.5
    cosine_max_value_correct: float = 1.0
    cosine_max_len: int = 1000
    parallel_code_exec_per_proc: int = 1
    code_provider: str = "e2b"
    enforce_same_language: bool = False
    code_eval_test_batch_size: int = 1
    code_eval_scoring_mode: str = "weighted_sum"
    ioi_provider: str = "piston"
    soft_punish_cache: int = 0


@dataclass
class RolloutConfig:
    """Online rollout configuration for GRPO."""

    num_generations: int = 8
    max_prompt_length: int = 2048
    max_completion_length: int = 1024
    temperature: float = 0.8
    top_p: float = 0.95
    kl_beta: float = 0.04
    epsilon: float = 0.2
    advantage_epsilon: float = 1.0e-6
    use_reference_model: bool = True
    generation_batch_size: int = 1


@dataclass
class GRPOConfig(ExperimentConfig):
    """Configuration for online GRPO optimization."""

    task: Literal["grpo"] = "grpo"
    system_prompt: Optional[str] = None
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
