"""GRPO-specific configuration for the Lightning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from lightning_grpo.utils.configs.base import ModelConfig, TrainingBaseConfig
from lightning_grpo.utils.configs.sft import ChatDataConfig


@dataclass
class GRPODataConfig(ChatDataConfig):
    """Dataset configuration for prompt-only RL and agentic tool-use RL."""

    mode: Literal["reasoning", "agentic"] = "reasoning"
    gt_column: str = "gt"
    tools_column: str = "tools"
    add_system_ratio: float = 0.0
    empty_think_ratio: float = 0.2
    thinking_ratio: float = 0.5


@dataclass
class GRPORolloutConfig:
    """Rollout and policy-gradient hyperparameters for GRPO."""

    num_generations: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 1024
    max_total_length: int = 2048
    max_turns: int = 3
    temperature: float = 0.8
    top_p: float = 1.0
    kl_beta: float = 0.1
    loss_type: Literal["grpo", "bnpo", "cispo"] = "cispo"
    epsilon: float = 0.2
    epsilon_high: float = 5.0
    advantage_epsilon: float = 1.0e-4
    use_reference_model: bool = True
    debug_samples: bool = False
    debug_every_n_steps: int = 20


@dataclass
class GRPOActiveRewardConfig:
    """Active reward function names and scalar weights."""

    reward_funcs: list[str] = field(default_factory=lambda: ["format", "gt"])
    weights: list[float] = field(default_factory=lambda: [1.0, 1.0])


@dataclass
class GRPORewardConfig:
    """Reward configuration for reasoning RL and agentic RL."""

    active: GRPOActiveRewardConfig = field(default_factory=GRPOActiveRewardConfig)
    reward_model: Optional[ModelConfig] = None
    repetition_ngram: int = 3
    repetition_cap: float = 0.5
    min_response_chars: int = 5
    max_response_chars: int = 800
    min_think_chars: int = 20
    max_think_chars: int = 300


@dataclass
class GRPOConfig(TrainingBaseConfig):
    """Configuration for Group Relative Policy Optimization training.

    Rollouts are generated locally with ``model.generate`` inside the Lightning
    module. The same module supports plain reasoning RL samples and multi-turn
    agentic tool-call samples.
    """

    task: Literal["grpo"] = "grpo"
    system_prompt: Optional[str] = None
    data: GRPODataConfig = field(default_factory=GRPODataConfig)
    rollout: GRPORolloutConfig = field(default_factory=GRPORolloutConfig)
    reward: GRPORewardConfig = field(default_factory=GRPORewardConfig)
    ref_model: Optional[ModelConfig] = None