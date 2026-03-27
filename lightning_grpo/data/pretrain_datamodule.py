"""Pretraining data module and collation utilities for Lightning."""

from __future__ import annotations

from typing import Any, Optional

import torch
from datasets import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_grpo.utils.configs.pretrain import PretrainConfig
from lightning_grpo.data.base import load_dataset_from_config, resolve_validation_split_name
from lightning_grpo.utils.modeling import load_tokenizer


class PretrainBatchCollator:
    """Pad tokenized pretraining samples into dense training batches."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of tokenized examples."""

        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        }


class PretrainDataModule(LightningDataModule):
    """Lightning data module for causal LM pretraining."""

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = load_tokenizer(config.model)
        self.collator = PretrainBatchCollator(self.tokenizer.pad_token_id)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        text_column = self.config.data.text_column
        max_length = self.config.data.max_seq_length
        tokenizer = self.tokenizer

        def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            texts = [str(text) for text in batch[text_column]]
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_length - 2,
                padding=False,
                add_special_tokens=False,
            )

            input_ids_batch: list[list[int]] = []
            attention_mask_batch: list[list[int]] = []
            labels_batch: list[list[int]] = []

            for ids in tokenized["input_ids"]:
                tokens = [tokenizer.bos_token_id] + list(ids) + [tokenizer.eos_token_id]
                attention_mask = [1] * len(tokens)
                labels = list(tokens)
                input_ids_batch.append(tokens)
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
            batch_size=self.config.data.preprocessing_batch_size,
            num_proc=None if self.config.data.streaming else self.config.data.num_workers,
            remove_columns=columns_to_remove,
            load_from_cache_file=self.config.data.preprocessing_use_cache,
            keep_in_memory=self.config.data.preprocessing_keep_in_memory,
            desc="Tokenizing pretraining dataset",
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and tokenize the pretraining dataset."""

        if (
                not self.config.data.train_files
                and not self.config.data.dataset_name
                and not self.config.data.dataset_mixture
        ):
            raise ValueError(
                "One of data.train_files, data.dataset_name, or data.dataset_mixture must be configured for pretraining."
            )
        if stage in (None, "fit"):
            dataset_dict = load_dataset_from_config(self.config.data)
            train_dataset = dataset_dict[self.config.data.train_split]
            self.train_dataset = self._tokenize_dataset(train_dataset)

            self.val_dataset = None
            val_split_name = resolve_validation_split_name(self.config.data, dataset_dict)
            if val_split_name is not None:
                self.val_dataset = self._tokenize_dataset(dataset_dict[val_split_name])

    def train_dataloader(self) -> DataLoader:
        """Build the training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("Pretrain dataset is not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.optimization.train_micro_batch_size,
            shuffle=not self.config.data.streaming,
            num_workers=self.config.data.num_workers,
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
            batch_size=self.config.optimization.eval_micro_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=False,
        )
