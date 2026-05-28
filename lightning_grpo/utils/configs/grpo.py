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
    thinking_ratio: float = 0.5


@dataclass
class GRPORewardConfig:
    """Reward configuration for GRPO training paradigms."""

    reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format", "tag_count"])
    reward_weights: list[float] | None = None
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
    max_completion_len: int = 16384
    soft_punish_cache: int = 0


@dataclass
class GRPORolloutConfig:
    """Rollout and policy-gradient hyperparameters for GRPO."""

    engine: Literal["torch", "sglang"] = "torch"
    sglang_base_url: str = "http://localhost:8998"
    sglang_model_path: Optional[str] = None
    sglang_shared_path: str = "./sglang_ckpt_grpo"
    sglang_timeout: int = 120
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
