"""DPO data module and collation utilities for Lightning."""

from __future__ import annotations

from typing import Any, Optional

import torch
from datasets import Dataset
from lightning.pytorch.utilities import rank_zero_warn
from transformers import PreTrainedTokenizerBase

from lightning_grpo.utils.configs.base import ModelConfig, OptimizationConfig
from lightning_grpo.utils.configs.dpo import DPODataConfig
from lightning_grpo.data.base import (
    ChatTemplateProcessor,
    ChatTemplateDataModule,
    preprocess_chat_messages,
    resolve_shuffle_state,
    iter_batch_samples,
)
from lightning_grpo.data.converter import json_loads_if_needed
from lightning_grpo.models.common import load_tokenizer


class DPOBatchCollator:
    """Collator for DPO preference pairs.

    Each sample contains prompt_ids, chosen_ids, and rejected_ids.
    The collator concatenates prompt+chosen and prompt+rejected, then pads
    to form a batch where the first half is chosen and the second half is rejected.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, ignore_index: int = -100) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of DPO preference samples into a padded batch.

        Returns a dict with:
            - input_ids: [2*B, max_len] (chosen first, rejected second)
            - attention_mask: [2*B, max_len]
            - completion_mask: [2*B, max_len] indicating completion token positions
        """

        chosen_input_ids = []
        chosen_attention_mask = []
        chosen_completion_mask = []
        rejected_input_ids = []
        rejected_attention_mask = []
        rejected_completion_mask = []

        for item in batch:
            prompt_ids = item["prompt_ids"]
            c_ids = item["chosen_ids"]
            r_ids = item["rejected_ids"]

            # Concatenate prompt + chosen
            chosen_seq = prompt_ids + c_ids
            chosen_input_ids.append(torch.tensor(chosen_seq, dtype=torch.long))
            chosen_attention_mask.append(torch.ones(len(chosen_seq), dtype=torch.long))
            # Completion mask: 0 for prompt tokens, 1 for completion tokens
            chosen_comp = [0] * len(prompt_ids) + [1] * len(c_ids)
            chosen_completion_mask.append(torch.tensor(chosen_comp, dtype=torch.long))

            # Concatenate prompt + rejected
            rejected_seq = prompt_ids + r_ids
            rejected_input_ids.append(torch.tensor(rejected_seq, dtype=torch.long))
            rejected_attention_mask.append(torch.ones(len(rejected_seq), dtype=torch.long))
            rejected_comp = [0] * len(prompt_ids) + [1] * len(r_ids)
            rejected_completion_mask.append(torch.tensor(rejected_comp, dtype=torch.long))

        # Combine chosen and rejected into a single batch [chosen_0, ..., chosen_B, rejected_0, ..., rejected_B]
        all_input_ids = chosen_input_ids + rejected_input_ids
        all_attention_mask = chosen_attention_mask + rejected_attention_mask
        all_completion_mask = chosen_completion_mask + rejected_completion_mask

        # Pad sequences to the same length
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            all_attention_mask, batch_first=True, padding_value=0,
        )
        completion_mask = torch.nn.utils.rnn.pad_sequence(
            all_completion_mask, batch_first=True, padding_value=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }


class DPODataModule(ChatTemplateDataModule):
    """Lightning data module for Direct Preference Optimization.

    Expects datasets in one of two formats:
    1. Conversational: 'chosen' and 'rejected' columns with message lists,
       optionally with a 'prompt' column.
    2. Standard: 'prompt', 'chosen', 'rejected' columns with plain text.
    """

    def __init__(
        self,
        data_config: DPODataConfig,
        model_config: ModelConfig,
        optimization_config: OptimizationConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(data_config=data_config, system_prompt=system_prompt)
        self.optimization_config = optimization_config
        self.tokenizer = load_tokenizer(model_config)
        self.chat_processor = ChatTemplateProcessor(self.tokenizer)
        self.collator = DPOBatchCollator(self.tokenizer, ignore_index=self.data_config.ignore_index)

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
    def _split_messages_for_dpo(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split canonical messages into prompt and assistant completion messages."""

        if messages and messages[-1].get("role") == "assistant":
            return messages[:-1], [messages[-1]]
        return messages, []

    def _tokenize_dataset(self, dataset: Dataset, formatter: Any) -> Dataset:
        """Convert dataset rows into tokenized DPO preference pairs."""

        chat_processor = self.chat_processor
        max_seq_length = self.data_config.max_seq_length
        chosen_column = self.data_config.chosen_column
        rejected_column = self.data_config.rejected_column
        split_messages_for_dpo = self._split_messages_for_dpo

        def tokenize_messages(messages: list[dict[str, Any]]) -> list[int]:
            """Tokenize a list of chat messages using the chat template."""
            processed = chat_processor.tokenize(
                messages,
                add_generation_prompt=False,
                tools=None,
                max_length=max_seq_length,
            )
            return list(processed["input_ids"])

        def normalize_preference_sample(raw_sample: dict[str, Any], completion_column: str) -> dict[str, Any]:
            """Build one canonical prompt+completion sample for a DPO side."""

            raw_completion = json_loads_if_needed(raw_sample[completion_column])
            if isinstance(raw_completion, list):
                sample = dict(raw_sample)
                sample["messages"] = raw_completion
                return formatter(sample)

            sample = dict(raw_sample)
            sample["response"] = raw_completion
            return formatter(sample)

        def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            """Tokenize a batch of preference samples into prompt/chosen/rejected ids."""

            all_prompt_ids: list[list[int]] = []
            all_chosen_ids: list[list[int]] = []
            all_rejected_ids: list[list[int]] = []

            samples = iter_batch_samples(batch)
            for raw_sample in samples:
                chosen_sample = normalize_preference_sample(raw_sample, chosen_column)
                rejected_sample = normalize_preference_sample(raw_sample, rejected_column)
                chosen_messages, chosen_tools = chat_processor.prepare_sample(chosen_sample)
                rejected_messages, rejected_tools = chat_processor.prepare_sample(rejected_sample)
                chosen_messages = preprocess_chat_messages(chosen_messages, self.data_config.add_system_ratio)
                rejected_messages = preprocess_chat_messages(rejected_messages, self.data_config.add_system_ratio)

                prompt_messages, chosen_completion = split_messages_for_dpo(chosen_messages)
                _, rejected_completion = split_messages_for_dpo(rejected_messages)
                prompt_ids = tokenize_messages(prompt_messages) if prompt_messages else []
                chosen_all_ids = tokenize_messages(prompt_messages + chosen_completion)
                rejected_all_ids = tokenize_messages(prompt_messages + rejected_completion)
                chosen_ids = chosen_all_ids[len(prompt_ids):]
                rejected_ids = rejected_all_ids[len(prompt_ids):]

                # Truncate to max_seq_length
                max_completion_len = max_seq_length - len(prompt_ids)
                if max_completion_len <= 0:
                    # Truncate prompt if it exceeds max length
                    prompt_ids = prompt_ids[:max_seq_length // 2]
                    max_completion_len = max_seq_length - len(prompt_ids)

                chosen_ids = chosen_ids[:max_completion_len]
                rejected_ids = rejected_ids[:max_completion_len]

                if chosen_ids and rejected_ids:
                    all_prompt_ids.append(prompt_ids)
                    all_chosen_ids.append(chosen_ids)
                    all_rejected_ids.append(rejected_ids)

            return {
                "prompt_ids": all_prompt_ids,
                "chosen_ids": all_chosen_ids,
                "rejected_ids": all_rejected_ids,
            }

        return self.map_dataset(
            dataset,
            preprocess_batch,
            desc="Tokenizing DPO dataset",
        )

    def train_dataloader(self):
        """Build the training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("DPO train dataset is not initialized. Call setup() first.")
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
