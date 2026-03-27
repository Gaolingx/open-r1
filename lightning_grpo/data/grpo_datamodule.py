"""GRPO data module and rollout collation utilities for Lightning."""

from __future__ import annotations

from typing import Any, Optional

from datasets import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from lightning_grpo.utils.configs.base import DataConfig, ModelConfig, OptimizationConfig
from lightning_grpo.utils.configs.grpo import RolloutConfig
from lightning_grpo.data.base import (
    ConversationTemplate,
    apply_chat_template,
    load_dataset_from_config,
    resolve_validation_split_name,
)
from lightning_grpo.utils.modeling import load_tokenizer


class GRPORolloutCollator:
    """Prepare prompt batches for online rollout generation."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, rollout_config: RolloutConfig) -> None:
        self.tokenizer = tokenizer
        self.rollout_config = rollout_config

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate raw prompt strings and metadata for GRPO rollouts."""

        prompts = [item["prompt_text"] for item in batch]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.rollout_config.max_prompt_length,
                padding=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = original_padding_side
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt_text": prompts,
            "metadata": [item.get("metadata", {}) for item in batch],
        }


class GRPODataModule(LightningDataModule):
    """Lightning data module for GRPO prompt and reward flows."""

    def __init__(
            self,
            data_config: DataConfig,
            model_config: ModelConfig,
            optimization_config: OptimizationConfig,
            rollout_config: RolloutConfig,
            system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.rollout_config = rollout_config
        self.system_prompt = system_prompt
        self.tokenizer = load_tokenizer(model_config)
        self.collator = GRPORolloutCollator(self.tokenizer, rollout_config)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and preprocess prompt-only datasets for GRPO."""

        dataset_dict = load_dataset_from_config(self.data_config)
        formatter = ConversationTemplate(
            prompt_column=self.data_config.prompt_column,
            response_column=self.data_config.response_column,
            messages_column=self.data_config.messages_column,
            system_prompt=self.system_prompt,
        )

        train_dataset = dataset_dict[self.data_config.train_split]
        val_split_name = resolve_validation_split_name(self.data_config, dataset_dict)

        self.train_dataset = self._prepare_prompt_dataset(train_dataset, formatter)
        self.val_dataset = None
        if val_split_name is not None:
            self.val_dataset = self._prepare_prompt_dataset(dataset_dict[val_split_name], formatter)

    def _prepare_prompt_dataset(self, dataset: Dataset, formatter: ConversationTemplate) -> Dataset:
        """Build prompt text plus reward metadata for online optimization."""

        def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            prompt_texts: list[str] = []
            metadata: list[dict[str, Any]] = []
            for index in range(len(next(iter(batch.values())))):
                sample = {key: value[index] for key, value in batch.items()}
                formatted = formatter(sample)
                prompt_texts.append(
                    apply_chat_template(
                        tokenizer=self.tokenizer,
                        messages=formatted["messages"],
                        add_generation_prompt=True,
                    )
                )
                metadata.append({
                    key: value
                    for key, value in sample.items()
                    if key != self.data_config.messages_column
                })
            return {"prompt_text": prompt_texts, "metadata": metadata}

        columns_to_remove = list(dataset.column_names)
        return dataset.map(
            preprocess_batch,
            batched=True,
            batch_size=self.data_config.preprocessing_batch_size,
            num_proc=None if self.data_config.streaming else self.data_config.num_workers,
            remove_columns=columns_to_remove,
            load_from_cache_file=self.data_config.preprocessing_use_cache,
            keep_in_memory=self.data_config.preprocessing_keep_in_memory,
            desc="Formatting GRPO prompts",
        )

    def train_dataloader(self) -> DataLoader:
        """Build the training prompt dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("GRPO train dataset is not initialized. Call setup() first.")
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
        """Build the validation prompt dataloader when a validation split is configured."""

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
