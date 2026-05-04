"""SFT data module and collation utilities for Lightning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal, Optional

from dacite import Config, from_dict
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from lightning_grpo.utils.configs.base import ModelConfig, OptimizationConfig
from lightning_grpo.data.features import TOKENIZED_SFT_FEATURES
from lightning_grpo.utils.configs.sft import SFTDataConfig
from lightning_grpo.data.base import (
    ChatTemplateDataModule,
    apply_chat_template,
    postprocess_chat_text,
    preprocess_chat_messages,
)
from lightning_grpo.utils.modeling import load_tokenizer


@dataclass
class SFTToolFunctionSchema:
    """Nested function schema used by chat-template tool definitions."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SFTToolSchema:
    """OpenAI-compatible tool definition with a nested JSON schema."""

    type: str = "function"
    function: SFTToolFunctionSchema = field(default_factory=SFTToolFunctionSchema)


@dataclass
class SFTToolCallFunctionSchema:
    """Nested function call emitted by assistant messages."""

    name: str = ""
    arguments: Any = field(default_factory=dict)


@dataclass
class SFTToolCallSchema:
    """OpenAI-compatible assistant tool-call payload."""

    id: str = ""
    type: str = "function"
    function: Optional[SFTToolCallFunctionSchema] = None
    name: str = ""
    arguments: Any = field(default_factory=dict)


@dataclass
class SFTMessageSchema:
    """Chat message schema with nested tool JSON parsed through dacite."""

    role: str = ""
    content: Any = ""
    reasoning_content: str = ""
    tools: Optional[list[SFTToolSchema]] = None
    tool_calls: Optional[list[SFTToolCallSchema]] = None


DACITE_SFT_JSON_CONFIG = Config(
    cast=[str],
    strict=False,
)


def _json_loads_if_needed(value: Any) -> Any:
    """Parse JSON-looking strings while preserving ordinary strings."""

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _drop_empty_schema_values(value: Any) -> Any:
    """Remove dacite/default placeholders that should not be sent to chat templates."""

    if isinstance(value, dict):
        return {
            key: _drop_empty_schema_values(item)
            for key, item in value.items()
            if item is not None and item != "" and item != {} and item != []
        }
    if isinstance(value, list):
        return [_drop_empty_schema_values(item) for item in value]
    return value


def _normalize_tool_definition(tool: Any) -> Any:
    """Convert one nested tool schema dict into a stable dacite-backed structure."""

    tool = _json_loads_if_needed(tool)
    if not isinstance(tool, dict):
        return tool
    normalized = from_dict(
        data_class=SFTToolSchema,
        data=tool,
        config=DACITE_SFT_JSON_CONFIG,
    )
    return _drop_empty_schema_values(asdict(normalized))


def _normalize_tool_call(tool_call: Any) -> Any:
    """Convert one assistant tool-call payload into a stable dacite-backed structure."""

    tool_call = _json_loads_if_needed(tool_call)
    if not isinstance(tool_call, dict):
        return tool_call
    normalized_call = dict(tool_call)
    if "function" in normalized_call:
        normalized_call["function"] = _json_loads_if_needed(normalized_call["function"])
        if isinstance(normalized_call["function"], dict) and "arguments" in normalized_call["function"]:
            normalized_call["function"]["arguments"] = _json_loads_if_needed(
                normalized_call["function"]["arguments"]
            )
    if "arguments" in normalized_call:
        normalized_call["arguments"] = _json_loads_if_needed(normalized_call["arguments"])
    normalized = from_dict(
        data_class=SFTToolCallSchema,
        data=normalized_call,
        config=DACITE_SFT_JSON_CONFIG,
    )
    return _drop_empty_schema_values(asdict(normalized))


def _normalize_sft_message_schema(message: Any) -> dict[str, Any]:
    """Normalize one SFT chat message and parse nested JSON schema fields."""

    message = _json_loads_if_needed(message)
    if not isinstance(message, dict):
        message = {"content": message}

    normalized = dict(message)
    if "content" not in normalized and "value" in normalized:
        normalized["content"] = normalized["value"]
    if "role" not in normalized and "from" in normalized:
        role = normalized["from"]
        normalized["role"] = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "bot": "assistant",
            "system": "system",
        }.get(str(role).lower(), role)
    normalized.setdefault("content", "")
    if "tools" in normalized:
        tools = _json_loads_if_needed(normalized["tools"])
        normalized["tools"] = [_normalize_tool_definition(tool) for tool in tools] if isinstance(tools, list) else tools
    if "tool_calls" in normalized:
        tool_calls = _json_loads_if_needed(normalized["tool_calls"])
        normalized["tool_calls"] = (
            [_normalize_tool_call(tool_call) for tool_call in tool_calls]
            if isinstance(tool_calls, list)
            else tool_calls
        )

    schema = from_dict(
        data_class=SFTMessageSchema,
        data=normalized,
        config=DACITE_SFT_JSON_CONFIG,
    )
    known = _drop_empty_schema_values(asdict(schema))
    extras = {
        key: _json_loads_if_needed(value)
        for key, value in normalized.items()
        if key not in SFTMessageSchema.__dataclass_fields__
    }
    return {**extras, **known}


def normalize_sft_conversation_messages(messages: Any) -> tuple[list[dict[str, Any]], Any]:
    """Normalize SFT messages and extract parsed tool definitions from system prompts."""

    messages = _json_loads_if_needed(messages)
    if not isinstance(messages, list):
        messages = [messages]

    normalized: list[dict[str, Any]] = []
    tools = None
    for message in messages:
        current = _normalize_sft_message_schema(message)
        if current.get("role") == "system" and current.get("tools"):
            tools = current["tools"]
        normalized.append(current)
    return normalized, tools


class SFTBatchCollator:
    """Causal LM collator for pre-tokenized SFT samples."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of tokenized examples."""

        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        if "attention_mask" in batch[0]:
            attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
        else:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTDataModule(ChatTemplateDataModule):
    """Lightning data module for supervised fine-tuning."""

    def __init__(
        self,
        data_config: SFTDataConfig,
        model_config: ModelConfig,
        optimization_config: OptimizationConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(data_config=data_config, model_config=model_config, system_prompt=system_prompt)
        self.optimization_config = optimization_config
        self.tokenizer = load_tokenizer(model_config)
        self.collator = SFTBatchCollator(self.tokenizer)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and preprocess train and validation datasets."""

        dataset_dict = self.load_dataset_dict()
        formatter = self.build_conversation_template()

        train_split = dataset_dict[self.data_config.train_split]
        self.train_dataset = self._tokenize_dataset(train_split, formatter)

        self.val_dataset = None
        val_split_name = self.resolve_val_split_name(dataset_dict)
        if val_split_name is not None:
            self.val_dataset = self._tokenize_dataset(dataset_dict[val_split_name], formatter)

    @staticmethod
    def _split_prompt_and_completion(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split a chat transcript into prompt messages and the final assistant completion."""

        if messages and messages[-1].get("role") == "assistant":
            return messages[:-1], [messages[-1]]
        return list(messages), []

    def _should_add_generation_prompt(self, messages: list[dict[str, Any]]) -> bool:
        """Decide whether chat rendering should append a generation prompt."""

        if not self.data_config.add_generation_prompt:
            return False
        return not messages or messages[-1].get("role") != "assistant"

    @staticmethod
    def _extract_assistant_mask(processed: dict[str, Any]) -> list[bool]:
        """Normalize assistant-token masks returned by chat templates across tokenizer variants."""

        raw_mask = processed.get("assistant_masks")
        if raw_mask is None:
            raw_mask = processed.get("assistant_mask")
        if raw_mask is None:
            return []
        return [bool(value) for value in raw_mask]

    @staticmethod
    def _labels_from_mask(full_ids: list[int], assistant_mask: list[bool]) -> list[int]:
        """Keep loss only on tokens selected by a tokenizer-provided mask."""

        if len(assistant_mask) != len(full_ids):
            raise RuntimeError("Assistant token mask length does not match tokenized input length.")
        return [token_id if mask else -100 for token_id, mask in zip(full_ids, assistant_mask)]

    @staticmethod
    def _labels_from_last_assistant_mask(
        full_ids: list[int],
        assistant_mask: list[bool],
        prompt_len: int,
    ) -> list[int]:
        """Keep loss only on the final assistant turn using the assistant mask and prompt boundary."""

        if len(assistant_mask) != len(full_ids):
            raise RuntimeError("Assistant token mask length does not match tokenized input length.")
        prompt_len = min(prompt_len, len(full_ids))
        return [
            token_id if mask and index >= prompt_len else -100
            for index, (token_id, mask) in enumerate(zip(full_ids, assistant_mask))
        ]

    def _apply_chat_template_tokenize(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        *,
        add_generation_prompt: bool,
        return_assistant_tokens_mask: bool = False,
    ) -> dict[str, Any] | None:
        """Tokenize chat messages with tokenizer templates when supported."""

        if not hasattr(self.tokenizer, "apply_chat_template"):
            return None

        try:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                truncation=True,
                max_length=self.data_config.max_seq_length,
                return_dict=True,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
            )
        except TypeError:
            return None

        if isinstance(tokenized, Mapping):
            return dict(tokenized)
        if hasattr(tokenized, "keys"):
            return {key: tokenized[key] for key in tokenized.keys()}
        return {"input_ids": list(tokenized), "attention_mask": [1] * len(tokenized)}

    def _render_chat_tokenize(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        *,
        add_generation_prompt: bool,
    ) -> dict[str, list[int]]:
        """Render chat text then tokenize it, matching the shared GRPO/pretrain abstractions."""

        text = apply_chat_template(
            tokenizer=self.tokenizer,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        text = postprocess_chat_text(text)
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.data_config.max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        return {
            "input_ids": list(tokenized["input_ids"]),
            "attention_mask": list(tokenized["attention_mask"]),
        }

    def _prompt_token_count(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
    ) -> int:
        """Tokenize the prompt prefix used by completion-only supervision."""

        prompt_messages, _ = self._split_prompt_and_completion(messages)
        processed = self._apply_chat_template_tokenize(
            prompt_messages,
            tools,
            add_generation_prompt=self._should_add_generation_prompt(prompt_messages),
        )
        if processed is None:
            processed = self._render_chat_tokenize(
                prompt_messages,
                tools,
                add_generation_prompt=self._should_add_generation_prompt(prompt_messages),
            )
        return len(processed["input_ids"])

    def _build_labels(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        full_ids: list[int],
        processed: dict[str, Any] | None,
        label_mode: Literal["all_tokens", "last_assistant", "all_assistant"],
    ) -> list[int]:
        """Build labels according to the configured SFT supervision mode."""

        if label_mode == "all_tokens":
            return list(full_ids)

        assistant_mask = self._extract_assistant_mask(processed or {}) if processed is not None else []
        if assistant_mask and label_mode == "all_assistant":
            return self._labels_from_mask(full_ids, assistant_mask)
        if assistant_mask and label_mode == "last_assistant":
            return self._labels_from_last_assistant_mask(
                full_ids,
                assistant_mask,
                self._prompt_token_count(messages, tools),
            )
        if label_mode == "all_assistant":
            raise RuntimeError(
                "label_mode='all_assistant' requires tokenizer support for assistant token masks."
            )

        prompt_len = min(self._prompt_token_count(messages, tools), len(full_ids))
        return [-100] * prompt_len + full_ids[prompt_len:]

    def _tokenize_sample(
        self,
        sample: dict[str, Any],
        label_mode: Literal["all_tokens", "last_assistant", "all_assistant"],
    ) -> dict[str, list[int]]:
        """Format one dataset row into token ids and SFT labels."""

        messages, tools = normalize_sft_conversation_messages(sample["messages"])
        messages = preprocess_chat_messages(messages)
        add_generation_prompt = self._should_add_generation_prompt(messages)
        needs_assistant_mask = label_mode in {"last_assistant", "all_assistant"}

        processed = self._apply_chat_template_tokenize(
            messages,
            tools,
            add_generation_prompt=add_generation_prompt,
            return_assistant_tokens_mask=needs_assistant_mask,
        )
        if processed is None:
            processed = self._render_chat_tokenize(messages, tools, add_generation_prompt=add_generation_prompt)

        input_ids = list(processed["input_ids"])
        attention_mask = list(processed.get("attention_mask", [1] * len(input_ids)))
        labels = self._build_labels(messages, tools, input_ids, processed, label_mode)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _tokenize_dataset(self, dataset: Dataset, formatter: Any) -> Dataset:
        """Convert dataset rows into tokenized causal language modeling samples."""

        def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            samples = [formatter(sample) for sample in self.iter_batch_samples(batch)]
            label_mode = self.data_config.label_mode
            tokenized_samples = [self._tokenize_sample(sample, label_mode) for sample in samples]

            return {
                "input_ids": [sample["input_ids"] for sample in tokenized_samples],
                "attention_mask": [sample["attention_mask"] for sample in tokenized_samples],
                "labels": [sample["labels"] for sample in tokenized_samples],
            }

        return self.map_dataset(
            dataset,
            preprocess_batch,
            desc="Tokenizing SFT dataset",
            features=TOKENIZED_SFT_FEATURES,
        )

    def train_dataloader(self):
        """Build the training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("SFT train dataset is not initialized. Call setup() first.")
        return self._build_dataloader(
            self.train_dataset,
            batch_size=self.optimization_config.train_micro_batch_size,
            collate_fn=self.collator,
            shuffle=not self.data_config.streaming,
            drop_last=True,
        )

    def val_dataloader(self):
        """Build the validation dataloader when a validation split is configured."""

        if self.val_dataset is None:
            return None
        return self._build_dataloader(
            self.val_dataset,
            batch_size=self.optimization_config.eval_micro_batch_size,
            collate_fn=self.collator,
            shuffle=False,
            drop_last=False,
        )
