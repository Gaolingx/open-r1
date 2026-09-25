"""SFT-specific configuration for the Lightning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from lightning_grpo.utils.configs.base import DataConfig
from lightning_grpo.utils.configs.base import TrainingBaseConfig


@dataclass
class ChatDataConfig(DataConfig):
    """Dataset configuration shared by the chat-style supervised tasks.

    Rows must expose a single OpenAI-style conversation column; there is no support for
    separate prompt/response text columns or ShareGPT `conversations` blobs.
    """

    messages_column: str = "messages"
    """Column holding the conversation as `[{"role": ..., "content": ...}, ...]`."""

    tools_column: str = "tools"
    """Optional column holding the OpenAI `tools` request parameter for that row."""

    add_generation_prompt: bool = True


@dataclass
class SFTDataConfig(ChatDataConfig):
    """Dataset configuration specific to supervised fine-tuning."""

    completion_only_loss: Optional[bool] = None
    assistant_only_loss: bool = False
    assistant_response_template: Optional[str] = None
    assistant_response_template_ids: Optional[list[int]] = None
    instruction_template: Optional[str] = None
    instruction_template_ids: Optional[list[int]] = None
    ignore_index: int = -100
    packing_enabled: bool = True


@dataclass
class SFTConfig(TrainingBaseConfig):
    """Configuration for supervised fine-tuning."""

    task: Literal["sft"] = "sft"
    system_prompt: Optional[str] = None
    data: SFTDataConfig = field(default_factory=SFTDataConfig)
    label_smoothing: float = 0.0
    eval_generation_max_new_tokens: int = 256
    loss_type: Literal["nll", "dft"] = "nll"
    validation_prompts: list[str] = field(default_factory=list)
