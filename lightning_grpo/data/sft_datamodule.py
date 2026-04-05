"""SFT data module and collation utilities for Lightning."""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from lightning_grpo.utils.configs.base import ModelConfig, OptimizationConfig
from lightning_grpo.utils.configs.sft import SFTDataConfig
from lightning_grpo.data.base import (
    ChatTemplateDataModule,
    apply_chat_template,
    normalize_conversation_messages,
    postprocess_chat_text,
    preprocess_chat_messages,
)
from lightning_grpo.utils.modeling import load_tokenizer


class SFTBatchCollator:
    """Pad tokenized SFT samples into dense training batches."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of tokenized examples."""

        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]

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
    def _build_labels_from_assistant_mask(full_ids: list[int], assistant_mask: list[bool]) -> list[int]:
        """Keep loss only on assistant tokens across all assistant turns."""

        if len(assistant_mask) != len(full_ids):
            raise RuntimeError("Assistant token mask length does not match tokenized input length.")
        return [token_id if mask else -100 for token_id, mask in zip(full_ids, assistant_mask)]

    @staticmethod
    def _build_labels_from_last_assistant_mask(
        full_ids: list[int],
        assistant_mask: list[bool],
        prompt_len: int,
    ) -> list[int]:
        """Keep loss only on the final assistant turn using the assistant mask and prompt boundary."""

        if len(assistant_mask) != len(full_ids):
            raise RuntimeError("Assistant token mask length does not match tokenized input length.")
        prompt_len = min(prompt_len, len(full_ids))
        return [token_id if mask and index >= prompt_len else -100 for index, (token_id, mask) in enumerate(zip(full_ids, assistant_mask))]

    def _tokenize_chat_messages(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        *,
        add_generation_prompt: bool,
        return_assistant_tokens_mask: bool = False,
    ) -> dict[str, Any] | None:
        """Tokenize chat messages via tokenizer templates when supported."""

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

        if isinstance(tokenized, dict):
            return tokenized
        return {"input_ids": list(tokenized), "attention_mask": [1] * len(tokenized)}

    def _build_labels_from_prompt_completion(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        full_ids: list[int],
    ) -> list[int]:
        """Mask prompt tokens while keeping completion tokens trainable."""

        prompt_messages, _ = self._split_prompt_and_completion(messages)
        prompt_text = apply_chat_template(
            tokenizer=self.tokenizer,
            messages=prompt_messages,
            add_generation_prompt=self._should_add_generation_prompt(prompt_messages),
            tools=tools,
        )
        prompt_text = postprocess_chat_text(prompt_text)
        prompt_tokenized = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.data_config.max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        prompt_ids = list(prompt_tokenized["input_ids"])
        prompt_len = min(len(prompt_ids), len(full_ids))
        return [-100] * prompt_len + full_ids[prompt_len:]

    def _build_labels_for_mode(
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
        if label_mode == "all_assistant":
            if assistant_mask:
                return self._build_labels_from_assistant_mask(full_ids, assistant_mask)
            raise RuntimeError(
                "label_mode='all_assistant' requires tokenizer support for assistant token masks."
            )

        if assistant_mask:
            prompt_messages, _ = self._split_prompt_and_completion(messages)
            prompt_processed = self._tokenize_chat_messages(
                prompt_messages,
                tools,
                add_generation_prompt=self._should_add_generation_prompt(prompt_messages),
            )
            if prompt_processed is None:
                raise RuntimeError(
                    "label_mode='last_assistant' requires chat-template tokenization support when using assistant token masks."
                )
            prompt_ids = list(prompt_processed["input_ids"])
            return self._build_labels_from_last_assistant_mask(full_ids, assistant_mask, len(prompt_ids))

        return self._build_labels_from_prompt_completion(messages, tools, full_ids)

    def _tokenize_dataset(self, dataset: Dataset, formatter: Any) -> Dataset:
        """Convert dataset rows into tokenized causal language modeling samples."""

        def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            samples = [formatter(sample) for sample in self.iter_batch_samples(batch)]
            label_mode = self.data_config.label_mode

            input_ids_batch: list[list[int]] = []
            attention_mask_batch: list[list[int]] = []
            labels_batch: list[list[int]] = []

            for sample in samples:
                raw_messages = sample["messages"]
                messages, tools = normalize_conversation_messages(raw_messages)
                messages = preprocess_chat_messages(messages)
                add_generation_prompt = self._should_add_generation_prompt(messages)

                processed = None
                if label_mode in {"last_assistant", "all_assistant"}:
                    processed = self._tokenize_chat_messages(
                        messages,
                        tools,
                        add_generation_prompt=add_generation_prompt,
                        return_assistant_tokens_mask=True,
                    )

                if processed is not None:
                    full_ids = list(processed["input_ids"])
                    attention_mask = list(processed.get("attention_mask", [1] * len(full_ids)))
                else:
                    full_text = apply_chat_template(
                        tokenizer=self.tokenizer,
                        messages=messages,
                        add_generation_prompt=add_generation_prompt,
                        tools=tools,
                    )
                    full_text = postprocess_chat_text(full_text)
                    full_tokenized = self.tokenizer(
                        full_text,
                        truncation=True,
                        max_length=self.data_config.max_seq_length,
                        padding=False,
                        add_special_tokens=False,
                    )
                    full_ids = list(full_tokenized["input_ids"])
                    attention_mask = list(full_tokenized["attention_mask"])

                labels = self._build_labels_for_mode(messages, tools, full_ids, processed, label_mode)

                input_ids_batch.append(full_ids)
                attention_mask_batch.append(attention_mask)
                labels_batch.append(labels)

            return {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels_batch,
            }

        return self.map_dataset(dataset, preprocess_batch, desc="Tokenizing SFT dataset")

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
