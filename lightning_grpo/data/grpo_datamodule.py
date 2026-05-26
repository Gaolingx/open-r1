"""GRPO prompt data module and collation utilities for Lightning."""

from __future__ import annotations

import json
from typing import Any, Optional

from datasets import Dataset

from lightning_grpo.data.base import (
    ChatTemplateDataModule,
    ChatTemplateProcessor,
    preprocess_chat_messages,
    resolve_shuffle_state,
)
from lightning_grpo.models.common import load_tokenizer
from lightning_grpo.utils.configs.base import ModelConfig, OptimizationConfig
from lightning_grpo.utils.configs.grpo import GRPODataConfig


class GRPOBatchCollator:
    """Keep prompt metadata as Python objects for on-policy rollout generation."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """Collate prompt-only samples without tensorizing variable chat metadata."""

        keys = batch[0].keys()
        return {key: [sample.get(key) for sample in batch] for key in keys}


class GRPODataModule(ChatTemplateDataModule):
    """Lightning data module for GRPO rollouts.

    Reasoning samples return rendered prompt strings. Agentic samples preserve
    message lists, tool schemas, and ground-truth values so the module can run
    multi-turn tool-call rollouts before computing the policy loss.
    """

    def __init__(
        self,
        data_config: GRPODataConfig,
        model_config: ModelConfig,
        optimization_config: OptimizationConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(data_config=data_config, system_prompt=system_prompt)
        self.optimization_config = optimization_config
        self.tokenizer = load_tokenizer(model_config)
        self.chat_processor = ChatTemplateProcessor(self.tokenizer)
        self.collator = GRPOBatchCollator()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and preprocess train and validation datasets."""

        dataset_dict = self.load_dataset_dict()
        train_split = dataset_dict[self.data_config.train_split]
        self.train_dataset = self._prepare_dataset(train_split)

        self.val_dataset = None
        val_split_name = self.resolve_val_split_name(dataset_dict)
        if val_split_name is not None:
            self.val_dataset = self._prepare_dataset(dataset_dict[val_split_name])

    @staticmethod
    def _loads_if_json(value: Any) -> Any:
        """Decode JSON strings used by jsonl tool schemas or ground-truth lists."""

        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _extract_messages_and_tools(self, sample: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
        """Normalize dataset rows into chat messages plus optional tools."""

        messages, row_tools = self.chat_processor.prepare_sample(sample)
        messages = preprocess_chat_messages(messages, add_system_ratio=self.data_config.add_system_ratio)
        tools = self._loads_if_json(row_tools)

        normalized: list[dict[str, Any]] = []
        for message in messages:
            current = dict(message)
            if current.get("tools") and tools is None:
                tools = self._loads_if_json(current.get("tools"))
            if isinstance(current.get("tool_calls"), str):
                current["tool_calls"] = self._loads_if_json(current["tool_calls"])
            normalized.append(current)
        return normalized, tools

    def _prompt_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop a final assistant answer when datasets contain supervised traces."""

        if messages and messages[-1].get("role") == "assistant":
            return messages[:-1]
        return messages

    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Map raw rows into lightweight rollout samples."""

        iter_batch_samples = self.iter_batch_samples
        mode = self.data_config.mode
        gt_column = self.data_config.gt_column

        def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            prompts: list[str] = []
            messages_batch: list[list[dict[str, Any]]] = []
            tools_batch: list[Any] = []
            gt_batch: list[Any] = []

            for sample in iter_batch_samples(batch):
                messages, tools = self._extract_messages_and_tools(sample)
                prompt_messages = self._prompt_messages(messages)
                prompt = self.chat_processor.render(
                    prompt_messages,
                    add_generation_prompt=True,
                    tools=tools,
                    open_thinking=False,
                )
                prompts.append(prompt)
                messages_batch.append(prompt_messages)
                tools_batch.append(tools)
                gt_batch.append(self._loads_if_json(sample.get(gt_column, sample.get("answer", ""))))

            result: dict[str, list[Any]] = {
                "prompt": prompts,
                "messages": messages_batch,
                "tools": tools_batch,
                "gt": gt_batch,
            }
            result["mode"] = [mode] * len(prompts)
            return result

        return self.map_dataset(dataset, preprocess_batch, desc="Preparing GRPO rollout dataset")

    def train_dataloader(self):
        """Build the training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("GRPO train dataset is not initialized. Call setup() first.")
        return self._build_dataloader(
            self.train_dataset,
            batch_size=self.optimization_config.train_micro_batch_size,
            collate_fn=self.collator,
            shuffle=resolve_shuffle_state(self.data_config),
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