"""DPO-specific configuration for the Lightning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from lightning_grpo.utils.configs.base import DataConfig
from lightning_grpo.utils.configs.base import TrainingBaseConfig
from lightning_grpo.utils.configs.sft import ChatDataConfig


@dataclass
class DPODataConfig(ChatDataConfig):
    """Dataset configuration specific to Direct Preference Optimization.

    Expects datasets with 'chosen' and 'rejected' columns (conversational format)
    or 'prompt', 'chosen', 'rejected' columns (standard format).
    """

    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    ignore_index: int = -100
    truncation_mode: Literal["keep_start", "keep_end"] = "keep_start"


@dataclass
class DPOConfig(TrainingBaseConfig):
    """Configuration for Direct Preference Optimization training.

    Implements the DPO algorithm from "Direct Preference Optimization:
    Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023).
    Uses LigerFusedLinearDPOLoss for memory-efficient fused computation.
    """

    task: Literal["dpo"] = "dpo"
    system_prompt: Optional[str] = None
    data: DPODataConfig = field(default_factory=DPODataConfig)

    # DPO hyperparameters
    beta: float = 0.1
    """KL penalty coefficient controlling deviation from the reference model."""

    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    """DPO loss variant. 'sigmoid' is the standard DPO loss."""

    # Reference model
    ref_model_name_or_path: Optional[str] = None
    """Path to a separate reference model. If None, uses a frozen copy of the policy model."""

    precompute_ref_log_probs: bool = False
    """Whether to precompute reference log-probs before training to save memory."""

    # Liger kernel is inherited from TrainingBaseConfig.liger_kernel
