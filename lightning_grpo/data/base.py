"""Shared dataset and formatting helpers for the Lightning GRPO pipeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from lightning_grpo.configs.base import DataConfig, DatasetSource, ModelConfig


class ConversationTemplate:
    """Convert row-based samples into chat-style message lists."""

    def __init__(
            self,
            prompt_column: str,
            response_column: str,
            messages_column: str = "messages",
            system_prompt: Optional[str] = None,
    ) -> None:
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.messages_column = messages_column
        self.system_prompt = system_prompt

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Create a canonical message list for one sample."""

        if self.messages_column in sample and sample[self.messages_column] is not None:
            return {"messages": sample[self.messages_column]}

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if self.prompt_column not in sample:
            raise KeyError(f"Prompt column '{self.prompt_column}' not found in sample.")

        messages.append({"role": "user", "content": str(sample[self.prompt_column])})
        if self.response_column in sample and sample[self.response_column] is not None:
            messages.append({"role": "assistant", "content": str(sample[self.response_column])})
        return {"messages": messages}


DEFAULT_VAL_SPLIT_NAME = "test"


def _load_single_dataset(source: DatasetSource, seed: int) -> Dataset:
    """Load one dataset source from the Hugging Face hub."""

    dataset = load_dataset(source.path, source.config_name, split=source.split)
    if source.columns:
        dataset = dataset.select_columns(source.columns)
    if source.weight < 1.0:
        sample_size = max(1, int(len(dataset) * source.weight))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    return dataset


def _with_optional_validation_split(dataset_dict: DatasetDict, data_config: DataConfig) -> DatasetDict:
    """Add a derived validation split when requested and one is not already present."""

    existing_val_split = resolve_validation_split_name(data_config, dataset_dict)
    if existing_val_split is not None or data_config.val_split_size is None:
        return dataset_dict

    if data_config.train_split not in dataset_dict:
        raise KeyError(f"Train split '{data_config.train_split}' not found in dataset.")

    split_dataset = dataset_dict[data_config.train_split].train_test_split(
        test_size=data_config.val_split_size,
        seed=data_config.split_seed,
    )
    dataset_dict[data_config.train_split] = split_dataset["train"]
    dataset_dict[DEFAULT_VAL_SPLIT_NAME] = split_dataset["test"]
    return dataset_dict


def resolve_validation_split_name(data_config: DataConfig, dataset_dict: DatasetDict) -> Optional[str]:
    """Resolve which validation split should be used for the current dataset config."""

    if data_config.val_split and data_config.val_split in dataset_dict:
        return data_config.val_split
    if data_config.val_split_size is not None and DEFAULT_VAL_SPLIT_NAME in dataset_dict:
        return DEFAULT_VAL_SPLIT_NAME
    return None


def load_dataset_from_config(data_config: DataConfig) -> DatasetDict:
    """Load a dataset or dataset mixture from configuration."""

    if data_config.dataset_mixture:
        datasets = [_load_single_dataset(source, seed=data_config.split_seed) for source in data_config.dataset_mixture]
        combined_dataset = concatenate_datasets(datasets).shuffle(seed=data_config.split_seed)
        dataset_dict = DatasetDict({data_config.train_split: combined_dataset})
    elif data_config.dataset_name:
        dataset_dict = load_dataset(data_config.dataset_name, data_config.dataset_config)
    else:
        raise ValueError("Either dataset_name or dataset_mixture must be configured.")

    return _with_optional_validation_split(dataset_dict, data_config)


def apply_chat_template(
        tokenizer: Any,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
) -> str:
    """Render messages into a training string with the tokenizer template."""

    if tokenizer.chat_template is None:
        rendered = []
        for message in messages:
            rendered.append(f"<{message['role']}>\n{message['content']}")
        if add_generation_prompt:
            rendered.append("<assistant>\n")
        return "\n".join(rendered)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def dump_data_config(data_config: DataConfig) -> dict[str, Any]:
    """Serialize the data configuration for logging and checkpoint metadata."""

    return asdict(data_config)
