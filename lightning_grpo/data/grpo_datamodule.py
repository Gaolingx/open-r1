"""GRPO data module and rollout collation utilities for Lightning."""

from __future__ import annotations

import json
from typing import Any, Optional

from datasets import Dataset
import torch
from transformers import PreTrainedTokenizerBase

from lightning_grpo.utils.configs.base import DataConfig, ModelConfig, OptimizationConfig
from lightning_grpo.data.features import GRPO_PROMPT_FEATURES
from lightning_grpo.utils.configs.grpo import RolloutConfig
from lightning_grpo.data.base import (
    ChatTemplateDataModule,
    apply_chat_template,
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
        metadata = [
            json.loads(item.get("metadata") or "{}")
            if isinstance(item.get("metadata"), str)
            else item.get("metadata", {})
            for item in batch
        ]
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
            "metadata": metadata,
            "sample_id": torch.tensor([int(item["sample_id"]) for item in batch], dtype=torch.long),
        }


class GRPODataModule(ChatTemplateDataModule):
    """Lightning data module for GRPO prompt and reward flows."""

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimization_config: OptimizationConfig,
        rollout_config: RolloutConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(data_config=data_config, model_config=model_config, system_prompt=system_prompt)
        self.optimization_config = optimization_config
        self.rollout_config = rollout_config
        self.tokenizer = load_tokenizer(model_config)
        self.collator = GRPORolloutCollator(self.tokenizer, rollout_config)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and preprocess prompt-only datasets for GRPO."""

        dataset_dict = self.load_dataset_dict()
        formatter = self.build_conversation_template()

        train_dataset = dataset_dict[self.data_config.train_split]
        val_split_name = self.resolve_val_split_name(dataset_dict)

        self.train_dataset = self._prepare_prompt_dataset(train_dataset, formatter)
        self.val_dataset = None
        if val_split_name is not None:
            self.val_dataset = self._prepare_prompt_dataset(dataset_dict[val_split_name], formatter)

    def _prepare_prompt_dataset(self, dataset: Dataset, formatter: Any) -> Dataset:
        """Build prompt text plus reward metadata for online optimization."""

        def preprocess_batch(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
            prompt_texts: list[str] = []
            metadata: list[str] = []
            for sample in self.iter_batch_samples(batch):
                formatted = formatter(sample)
                prompt_texts.append(
                    apply_chat_template(
                        tokenizer=self.tokenizer,
                        messages=formatted["messages"],
                        add_generation_prompt=self.data_config.add_generation_prompt,
                    )
                )
                metadata.append(json.dumps({
                    key: value
                    for key, value in sample.items()
                    if key != self.data_config.messages_column
                }, ensure_ascii=False))
            return {"prompt_text": prompt_texts, "metadata": metadata, "sample_id": indices}

        return self.map_dataset(
            dataset,
            preprocess_batch,
            desc="Formatting GRPO prompts",
            with_indices=True,
            features=GRPO_PROMPT_FEATURES,
        )

    def train_dataloader(self):
        """Build the training prompt dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("GRPO train dataset is not initialized. Call setup() first.")
        return self._build_dataloader(
            self.train_dataset,
            batch_size=self.optimization_config.train_micro_batch_size,
            collate_fn=self.collator,
            shuffle=not self.data_config.streaming,
            drop_last=True,
        )

    def val_dataloader(self):
        """Build the validation prompt dataloader when a validation split is configured."""

        if self.val_dataset is None:
            return None
        return self._build_dataloader(
            self.val_dataset,
            batch_size=self.optimization_config.eval_micro_batch_size,
            collate_fn=self.collator,
            shuffle=False,
            drop_last=False,
        )
