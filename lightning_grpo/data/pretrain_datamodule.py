"""Pretraining data module and collation utilities for Lightning."""

from __future__ import annotations

from typing import Any, Optional

import torch
from datasets import Dataset

from lightning_grpo.utils.configs.pretrain import PretrainConfig
from lightning_grpo.data.base import BaseLMDataModule
from lightning_grpo.data.features import TOKENIZED_PRETRAIN_FEATURES
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


class PretrainDataModule(BaseLMDataModule):
    """Lightning data module for causal LM pretraining."""

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__(data_config=config.data, model_config=config.model)
        self.config = config
        self.tokenizer = load_tokenizer(config.model)
        self.collator = PretrainBatchCollator(self.tokenizer.pad_token_id)

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

        return self.map_dataset(
            dataset,
            preprocess_batch,
            desc="Tokenizing pretraining dataset",
            features=TOKENIZED_PRETRAIN_FEATURES,
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
            dataset_dict = self.load_dataset_dict()
            train_dataset = dataset_dict[self.config.data.train_split]
            self.train_dataset = self._tokenize_dataset(train_dataset)

            self.val_dataset = None
            val_split_name = self.resolve_val_split_name(dataset_dict)
            if val_split_name is not None:
                self.val_dataset = self._tokenize_dataset(dataset_dict[val_split_name])

    def train_dataloader(self):
        """Build the training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("Pretrain dataset is not initialized. Call setup() first.")
        return self._build_dataloader(
            self.train_dataset,
            batch_size=self.config.optimization.train_micro_batch_size,
            collate_fn=self.collator,
            shuffle=not self.config.data.streaming,
            drop_last=True,
        )

    def val_dataloader(self):
        """Build the validation dataloader when a validation split is configured."""

        if self.val_dataset is None:
            return None
        return self._build_dataloader(
            self.val_dataset,
            batch_size=self.config.optimization.eval_micro_batch_size,
            collate_fn=self.collator,
            shuffle=False,
            drop_last=False,
        )
