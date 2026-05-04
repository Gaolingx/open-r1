"""SFT data module and collation utilities for Lightning."""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from lightning_grpo.utils.configs.base import ModelConfig, OptimizationConfig
from lightning_grpo.data.features import TOKENIZED_SFT_FEATURES
from lightning_grpo.utils.configs.sft import SFTDataConfig
from lightning_grpo.data.base import (
    ChatTemplateProcessor,
    ChatTemplateDataModule,
    preprocess_chat_messages,
)
from lightning_grpo.utils.modeling import load_tokenizer


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
        self.chat_processor = ChatTemplateProcessor(self.tokenizer)
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

        return self.chat_processor.should_add_generation_prompt(messages, self.data_config.add_generation_prompt)

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

        return self.chat_processor.tokenize(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            max_length=self.data_config.max_seq_length,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
        )

    def _render_chat_tokenize(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        *,
        add_generation_prompt: bool,
    ) -> dict[str, list[int]]:
        """Render chat text then tokenize it, matching the shared GRPO/pretrain abstractions."""

        return self.chat_processor.tokenize(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            max_length=self.data_config.max_seq_length,
        )

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

        messages, tools = self.chat_processor.prepare_sample(sample)
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
