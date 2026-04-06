"""GRPO-specific configuration for the Lightning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from lightning_grpo.utils.configs.base import TrainingBaseConfig
from lightning_grpo.utils.configs.sft import ChatDataConfig


@dataclass
class RewardModelTemplateConfig:
    """Formatting controls for reward-model scoring prompts."""

    enabled: bool = False
    chat_template: Optional[str] = None
    add_generation_prompt: bool = False
    include_system_prompt: bool = False
    system_prompt: Optional[str] = None


@dataclass
class RewardModelConfig:
    """Reward model inference configuration for non-programmatic GRPO rewards."""

    enabled: bool = False
    model_name_or_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    model_revision: str = "main"
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    max_length: int = 4096
    batch_size: int = 4
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    score_field: str = "score"
    normalize: bool = False
    bias: float = 0.0
    scale: float = 1.0
    template: RewardModelTemplateConfig = field(default_factory=RewardModelTemplateConfig)


@dataclass
class BaseRewardConfig:
    """Shared reward configuration for GRPO training paradigms."""

    reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format", "tag_count"])
    reward_weights: list[float] | None = None
    format_mode: Literal["strict", "compatible", "no_answer_tag"] = "strict"
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
class RLVRRewardConfig(BaseRewardConfig):
    """Reward configuration for rule-based / verifiable GRPO training."""


@dataclass
class RLHFRewardConfig(BaseRewardConfig):
    """Reward configuration for reward-model-based GRPO training."""

    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)


@dataclass
class RewardConfig:
    """Top-level reward config with explicit RLVR and RLHF sub-configs."""

    training_paradigm: Literal["rlvr", "rlhf"] = "rlvr"
    rlvr: RLVRRewardConfig = field(default_factory=RLVRRewardConfig)
    rlhf: RLHFRewardConfig = field(default_factory=RLHFRewardConfig)

    @property
    def active(self) -> BaseRewardConfig:
        return self.rlhf if self.training_paradigm == "rlhf" else self.rlvr


@dataclass
class RolloutEngineConfig:
    """Pluggable rollout backend configuration."""

    engine_type: Literal["policy", "sglang", "reward_model"] = "policy"
    sglang_base_url: str = "http://localhost:8996"
    sglang_model_path: Optional[str] = None
    sglang_shared_path: str = "./sglang_ckpt_grpo"
    request_timeout: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    retry_max_backoff_seconds: float = 30.0


@dataclass
class DebugConfig:
    """Sample-level debug configuration for GRPO training."""

    enabled: bool = False
    every_n_steps: int = 0
    questions: list[str] = field(default_factory=list)
    generation_config_path: Optional[str] = None


@dataclass
class RolloutConfig:
    """Online rollout configuration for GRPO."""

    num_generations: int = 8
    num_generations_eval: int | None = None
    max_prompt_length: int = 2048
    generation_config_path: Optional[str] = None
    temperature: float = 1.0
    kl_beta: float = 0.04
    epsilon: float = 0.2
    epsilon_high: float = 5.0
    loss_type: Literal["grpo", "cispo"] = "grpo"
    advantage_epsilon: float = 1.0e-6
    use_reference_model: bool = True
    generation_batch_size: int = 0
    engine: RolloutEngineConfig = field(default_factory=RolloutEngineConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class GRPOConfig(TrainingBaseConfig):
    """Configuration for online GRPO optimization."""

    task: Literal["grpo"] = "grpo"
    system_prompt: Optional[str] = None
    data: ChatDataConfig = field(default_factory=ChatDataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
