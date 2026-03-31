"""SFT data module and collation utilities for Lightning."""

from __future__ import annotations

from typing import Any, Optional

import torch
from datasets import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from lightning_grpo.utils.configs.base import DataConfig, ModelConfig, OptimizationConfig
from lightning_grpo.data.base import (
    ConversationTemplate,
    apply_chat_template,
    load_dataset_from_config,
    normalize_conversation_messages,
    postprocess_chat_text,
    preprocess_chat_messages,
    resolve_validation_split_name,
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


class SFTDataModule(LightningDataModule):
    """Lightning data module for supervised fine-tuning."""

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimization_config: OptimizationConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.system_prompt = system_prompt
        self.tokenizer = load_tokenizer(model_config)
        self.collator = SFTBatchCollator(self.tokenizer)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def _build_assistant_mask_from_labels(self, labels: list[int]) -> list[int]:
        """Derive a dense assistant token mask from labels."""

        return [0 if token_id == -100 else 1 for token_id in labels]

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and preprocess train and validation datasets."""

        dataset_dict = load_dataset_from_config(self.data_config)
        formatter = ConversationTemplate(
            prompt_column=self.data_config.prompt_column,
            response_column=self.data_config.response_column,
            messages_column=self.data_config.messages_column,
            system_prompt=self.system_prompt,
        )

        train_split = dataset_dict[self.data_config.train_split]
        self.train_dataset = self._tokenize_dataset(train_split, formatter)

        val_split_name = resolve_validation_split_name(self.data_config, dataset_dict)
        if val_split_name is not None:
            self.val_dataset = self._tokenize_dataset(dataset_dict[val_split_name], formatter)

    @staticmethod
    def _split_prompt_and_completion(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split a chat transcript into prompt messages and the final assistant completion."""

        if messages and messages[-1].get("role") == "assistant":
            return messages[:-1], [messages[-1]]
        return list(messages), []

    def _tokenize_dataset(self, dataset: Dataset, formatter: ConversationTemplate) -> Dataset:
        """Convert dataset rows into tokenized causal language modeling samples."""

        def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            samples = []
            for index in range(len(next(iter(batch.values())))):
                sample = {key: value[index] for key, value in batch.items()}
                samples.append(formatter(sample))

            input_ids_batch: list[list[int]] = []
            attention_mask_batch: list[list[int]] = []
            labels_batch: list[list[int]] = []

            for sample in samples:
                raw_messages = sample["messages"]
                messages, tools = normalize_conversation_messages(raw_messages)
                messages = preprocess_chat_messages(messages)

                processed = None
                if self.data_config.assistant_only_loss and hasattr(self.tokenizer, "apply_chat_template"):
                    try:
                        processed = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=False,
                            tools=tools,
                            return_dict=True,
                            return_assistant_tokens_mask=True,
                        )
                    except TypeError:
                        processed = None

                if processed is not None:
                    full_ids = list(processed["input_ids"])
                    attention_mask = list(processed.get("attention_mask", [1] * len(full_ids)))
                    assistant_mask = list(processed.get("assistant_masks", []))
                    labels = list(full_ids)
                    if self.data_config.mask_prompt_labels or self.data_config.assistant_only_loss:
                        if assistant_mask:
                            labels = [token_id if mask else -100 for token_id, mask in zip(full_ids, assistant_mask)]
                        elif self.data_config.assistant_only_loss:
                            raise RuntimeError(
                                "assistant_only_loss=True but tokenizer did not return assistant token masks."
                            )
                else:
                    full_text = apply_chat_template(
                        tokenizer=self.tokenizer,
                        messages=messages,
                        add_generation_prompt=self.data_config.add_generation_prompt,
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

                    if self.data_config.mask_prompt_labels:
                        prompt_messages, _ = self._split_prompt_and_completion(messages)
                        prompt_text = apply_chat_template(
                            tokenizer=self.tokenizer,
                            messages=prompt_messages,
                            add_generation_prompt=self.data_config.add_generation_prompt,
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
                        labels = [-100] * prompt_len + full_ids[prompt_len:]
                    else:
                        labels = list(full_ids)

                input_ids_batch.append(full_ids)
                attention_mask_batch.append(attention_mask)
                labels_batch.append(labels)

            return {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels_batch,
            }

        columns_to_remove = list(dataset.column_names)
        return dataset.map(
            preprocess_batch,
            batched=True,
            batch_size=self.data_config.preprocessing_batch_size,
            num_proc=None if self.data_config.streaming else self.data_config.num_workers,
            remove_columns=columns_to_remove,
            load_from_cache_file=self.data_config.preprocessing_use_cache,
            keep_in_memory=self.data_config.preprocessing_keep_in_memory,
            desc="Tokenizing SFT dataset",
        )

    def train_dataloader(self) -> DataLoader:
        """Build the training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("SFT train dataset is not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.optimization_config.train_micro_batch_size,
            shuffle=not self.data_config.streaming,
            num_workers=self.data_config.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Build the validation dataloader when a validation split is configured."""

        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.optimization_config.eval_micro_batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=False,
        )
