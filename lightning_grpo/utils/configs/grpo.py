"""GRPO-specific configuration for the Lightning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from lightning_grpo.utils.configs.base import ModelConfig, TrainingBaseConfig
from lightning_grpo.utils.configs.sft import ChatDataConfig


@dataclass
class GRPODataConfig(ChatDataConfig):
    """Dataset configuration for prompt-only RL and agentic tool-use RL."""


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
class ToolCallingConfig:
    """Configuration for multi-turn tool calling during rollout."""

    enabled: bool = False
    max_iterations: int = 5
    tools: list[str] = field(default_factory=list)
    chat_template: Optional[str] = None
    chat_template_kwargs: dict | None = None


@dataclass
class VLLMConfig:
    """vLLM rollout backend configuration supporting server and colocate modes."""

    mode: Literal["server", "colocate"] = "server"
    sampling_config_path: Optional[str] = None
    # Server mode configuration
    server_base_url: Optional[str] = None
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_timeout: float = 240.0
    group_port: int = 51216
    # Colocate mode configuration
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_length: Optional[int] = None
    max_num_seqs: Optional[int] = None
    enable_sleep_mode: bool = False
    model_impl: str = "auto"
    # Generation overrides
    repetition_penalty: float = 1.0
    structured_outputs_regex: Optional[str] = None
    logprobs: int = 0
    generation_kwargs: dict | None = None


@dataclass
class GRPORolloutConfig:
    """Rollout and policy-gradient hyperparameters for GRPO.

    ``loss_type`` selects an *algorithm preset*: it pins the policy objective
    (see :data:`lightning_grpo.models.grpo.loss.GRPO_ALGORITHM_PRESETS`), the
    default advantage estimator and the default loss aggregation mode in one
    go. Setting ``advantage_estimator`` / ``loss_agg_mode`` explicitly overrides
    the preset defaults, which makes it possible to mix e.g. the DAPO objective
    with RLOO advantages.
    """

    engine: Literal["torch", "vllm"] = "torch"
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    tool_calling: ToolCallingConfig = field(default_factory=ToolCallingConfig)
    num_generations: int = 4
    num_generations_eval: int = 1
    max_prompt_length: int = 1024
    max_completion_length: int = 1024
    max_total_length: int = 2048
    temperature: float = 0.8
    top_p: float = 1.0
    kl_beta: float = 0.1
    use_reference_model: bool = True
    debug_samples: bool = False
    debug_every_n_steps: int = 20

    # --- Algorithm selection -------------------------------------------------
    loss_type: Literal[
        "grpo",
        "dapo",
        "dr_grpo",
        "cispo",
        "gspo",
        "sapo",
        "geo_mean",
        "dro",
        "clip_cov",
        "kl_cov",
        "dppo_tv",
        "dppo_kl",
        "gpg",
        "reinforce",
    ] = "cispo"
    # ``bnpo`` was a straight alias of ``grpo`` and is kept only for backwards
    # compatible configs; it resolves to the ``grpo`` preset with a warning.
    advantage_estimator: Optional[str] = None
    loss_agg_mode: Optional[str] = None

    # --- Importance-ratio clipping ------------------------------------------
    epsilon: float = 0.2
    # ``None`` means "use the preset default": ``epsilon`` for symmetric PPO-style
    # clipping, 0.28 for DAPO ("clip-higher") and 5.0 for CISPO (which historically
    # only clipped the upper side).
    epsilon_high: Optional[float] = None
    # Lower bound of the ratio for dual-clip PPO (https://arxiv.org/pdf/1912.09729).
    clip_ratio_c: float = 3.0
    # ``False`` turns the GRPO advantage into the Dr.GRPO advantage (no std scaling).
    # ``None`` means "use the preset default": ``True`` everywhere except
    # ``dr_grpo``, whose whole point is the unscaled advantage.
    norm_adv_by_std_in_grpo: Optional[bool] = None
    advantage_epsilon: float = 1.0e-4
    # Discount factor used by the REINFORCE++ advantage estimators.
    gamma: float = 1.0
    # DAPO dynamic sampling, "light": zero out the loss mask of prompt groups whose
    # rewards are all identical (zero advantage, zero gradient) instead of resampling.
    drop_zero_advantage_groups: bool = False

    # --- DAPO overlong reward shaping ---------------------------------------
    # When ``overlong_buffer_len > 0`` a linear length penalty is added to the
    # reward of completions whose length exceeds
    # ``max_completion_length - overlong_buffer_len``. It is only applied to
    # samples with a non-positive total reward, mirroring DAPO's intent of not
    # punishing correct-but-long answers.
    overlong_buffer_len: int = 0
    overlong_penalty_factor: float = 1.0

    # --- Policy-loss specific knobs ----------------------------------------
    dro_beta: float = 0.1
    sapo_tau_pos: float = 1.0
    sapo_tau_neg: float = 1.0
    clip_cov_ratio: float = 2.0e-4
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 2.0e-4
    ppo_kl_coef: float = 1.0
    gpg_alpha: float = 1.0

    # --- ReMax --------------------------------------------------------------
    # Temperature used for the greedy baseline rollout. 0.0 requests greedy
    # decoding; raise it slightly for engines that reject a zero temperature.
    remax_baseline_temperature: float = 0.0


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
