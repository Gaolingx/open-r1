"""Shared dataset, chat formatting, and sequence-packing helpers for Lightning pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import random
from typing import Any, Callable, Optional

import torch
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    IterableDataset,
    IterableDatasetDict,
    Sequence as ArrowSequence,
    Value,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_grpo.utils.configs.base import DataConfig, DatasetConfig


SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是nekomind，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是nekomind，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are nekomind, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are nekomind, a small but useful language model.",
]


def iter_batch_samples(batch: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Convert a dict-of-lists batch into a list of row dictionaries."""

    if not batch:
        return []
    batch_size = len(next(iter(batch.values())))
    return [
        {key: value[index] for key, value in batch.items()}
        for index in range(batch_size)
    ]


def sample_system_prompt(system_prompt: Optional[str], add_system_ratio: float) -> Optional[str]:
    """Resolve the system prompt for one row.

    A configured `system_prompt` always wins; otherwise `add_system_ratio` randomly picks
    one of `SYSTEM_PROMPTS`. Resolving this once per row keeps both DPO preference sides
    and every document packed into a row consistent with each other.
    """

    if system_prompt:
        return system_prompt
    if add_system_ratio > 0.0 and random.random() < add_system_ratio:
        return random.choice(SYSTEM_PROMPTS)
    return None


def preprocess_chat_messages(
    messages: Any,
    *,
    tools: Any = None,
    system_prompt: Optional[str] = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Validate one OpenAI-style message list and optionally prepend a system prompt.

    Datasets must already store OpenAI messages: a non-empty list of
    `{"role": ..., "content": ...}` dicts. JSON strings, ShareGPT `conversations` blobs,
    bare prompt strings and messages without a role are rejected instead of coerced, so a
    mis-formatted dataset fails during preprocessing instead of quietly producing wrong
    training data.
    """

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise TypeError(
            "Chat samples must store messages as a non-empty list of {'role', 'content'} "
            f"dicts, got {type(messages).__name__}."
        )

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping) or not message.get("role"):
            raise TypeError(f"Every chat message must be a dict with a 'role', got {message!r}.")
        current = dict(message)
        if current.get("content") is None:
            current["content"] = ""
        normalized.append(current)

    if system_prompt and normalized[0].get("role") != "system":
        normalized.insert(0, {"role": "system", "content": system_prompt})
    return normalized, tools


class ChatTemplateProcessor:
    """Shared renderer/tokenizer for converter-normalized chat samples."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def should_add_generation_prompt(messages: list[dict[str, Any]], enabled: bool) -> bool:
        """Decide whether chat rendering should append a generation prompt."""

        return enabled and (not messages or messages[-1].get("role") != "assistant")

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: Any = None,
        **chat_template_kwargs: Any,
    ) -> str:
        """Render messages to text with the tokenizer chat template."""

        return apply_chat_template(
            tokenizer=self.tokenizer,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **chat_template_kwargs,
        )

    def tokenize(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: Any = None,
        max_length: int,
        return_assistant_tokens_mask: bool = False,
    ) -> dict[str, Any]:
        """Tokenize chat messages through tokenizer templates with a text fallback."""

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                    tools=tools,
                    truncation=True,
                    max_length=max_length,
                    return_dict=True,
                    return_assistant_tokens_mask=return_assistant_tokens_mask,
                )
                if isinstance(tokenized, Mapping):
                    return dict(tokenized)
                if hasattr(tokenized, "keys"):
                    return {key: tokenized[key] for key in tokenized.keys()}
                return {"input_ids": list(tokenized), "attention_mask": [1] * len(tokenized)}
            except TypeError:
                pass

        text = self.render(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )
        return {
            "input_ids": list(tokenized["input_ids"]),
            "attention_mask": list(tokenized["attention_mask"]),
        }


DEFAULT_VAL_SPLIT_NAME = "test"


def resolve_shuffle_state(data_config: DataConfig) -> bool:
    """Resolve DataLoader shuffle state from data configuration."""

    return bool(data_config.shuffle and not data_config.streaming)


def rename_dataset_columns(
    dataset: Dataset | IterableDataset,
    column_mapping: Optional[Mapping[str, str]],
) -> Dataset | IterableDataset:
    """Rename source columns to the canonical names the pipelines expect.

    Datasets on the Hub disagree about what to call the same content (`messages`,
    `message`, `conversations`, ...). Renaming here, at load time, resolves that naming
    question once per dataset while every downstream transform keeps consuming a single
    canonical name -- the row *format* stays strictly validated, only the *label* on the
    column is allowed to differ.

    `column_mapping` maps the dataset's own column name to the canonical one, so it is a
    per-dataset setting: a mixture can freely combine sources that name things
    differently.
    """

    if not column_mapping:
        return dataset

    existing = getattr(dataset, "column_names", None)
    if existing is not None:
        available = set(existing)
        for source, target in column_mapping.items():
            if source not in available:
                raise ValueError(
                    f"column_mapping renames '{source}', but the dataset has no such column. "
                    f"Available columns: {sorted(available)}."
                )
            if target in available and target != source:
                raise ValueError(
                    f"column_mapping would rename '{source}' to '{target}', but '{target}' "
                    "already exists in the dataset."
                )

    for source, target in column_mapping.items():
        dataset = dataset.rename_column(source, target)
    return dataset


def prepare_dataset_split(
    dataset: Dataset | IterableDataset,
    dataset_config: DatasetConfig,
) -> Dataset | IterableDataset:
    """Canonicalise one split's column names, then keep only the configured columns.

    The rename has to happen first: `columns` selects canonical names, so a dataset that
    spells `messages` as `message` would otherwise have the column dropped before the
    rename had a chance to run.
    """

    renamed = rename_dataset_columns(dataset, dataset_config.column_mapping)
    return select_dataset_columns(renamed, dataset_config.columns)


def select_dataset_columns(
    dataset: Dataset | IterableDataset,
    columns: Optional[list[str]],
) -> Dataset | IterableDataset:
    """Keep only the configured columns that the dataset actually provides."""

    if not columns:
        return dataset
    existing_columns = getattr(dataset, "column_names", None)
    if not existing_columns:
        return dataset
    existing = set(existing_columns)
    selected = [name for name in columns if name in existing]
    if not selected or set(selected) == existing:
        return dataset
    if isinstance(dataset, IterableDataset):
        # Streaming datasets cannot select columns, only drop the ones we do not want.
        dropped = [name for name in existing_columns if name not in existing.intersection(selected)]
        return dataset.remove_columns(dropped) if dropped else dataset
    return dataset.select_columns(selected)


def load_dataset_from_config(data_config: DataConfig) -> DatasetDict | IterableDatasetDict:
    """Load a dataset or dataset mixture from configuration."""

    def _shuffle_streaming_dataset(dataset: IterableDataset) -> IterableDataset:
        """Apply buffer-based shuffle to a streaming dataset."""
        return dataset.shuffle(seed=data_config.split_seed, buffer_size=data_config.shuffle_buffer_size)

    def _load_single_dataset(dataset_config: DatasetConfig) -> DatasetDict | IterableDatasetDict:
        """Load one dataset from a nested dataset configuration."""
        kwargs: dict[str, Any] = {
            "cache_dir": data_config.cache_dir,
            "streaming": data_config.streaming,
        }
        if dataset_config.data_files is not None:
            kwargs["data_files"] = dataset_config.data_files
        return load_dataset(dataset_config.id, dataset_config.config, **kwargs)

    if data_config.dataset_list is not None:
        datasets_train: list[Dataset | IterableDataset] = []
        datasets_val: list[Dataset | IterableDataset] = []
        weights: list[float] = []

        train_split = data_config.train_split
        val_split = data_config.val_split

        for dataset_config in data_config.dataset_list:
            ds_dict = _load_single_dataset(dataset_config)

            has_train = train_split in ds_dict
            has_val = val_split in ds_dict if val_split else False

            if has_train:
                datasets_train.append(prepare_dataset_split(ds_dict[train_split], dataset_config))
                weights.append(dataset_config.weight if dataset_config.weight is not None else 1.0)

            if has_val:
                datasets_val.append(prepare_dataset_split(ds_dict[val_split], dataset_config))

            # If neither train nor val split matched, fall back to the configured split or first available
            if not has_train and not has_val:
                split_name = dataset_config.split or train_split
                if split_name in ds_dict:
                    ds = ds_dict[split_name]
                else:
                    available = list(ds_dict.keys())
                    ds = ds_dict[available[0]]
                datasets_train.append(prepare_dataset_split(ds, dataset_config))
                weights.append(dataset_config.weight if dataset_config.weight is not None else 1.0)

        # Combine training datasets
        all_default_weights = all(w == 1.0 for w in weights)

        if len(datasets_train) == 0:
            combined_train = None
        elif len(datasets_train) == 1:
            combined_train = datasets_train[0]
        elif all_default_weights:
            combined_train = concatenate_datasets(datasets_train)
        else:
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            combined_train = interleave_datasets(
                datasets_train,
                probabilities=probabilities,
                seed=data_config.split_seed,
                stopping_strategy="all_exhausted",
            )

        # Combine validation datasets
        combined_val = None
        if datasets_val:
            combined_val = datasets_val[0] if len(datasets_val) == 1 else concatenate_datasets(datasets_val)

        # Build the final dataset dict
        splits: dict[str, Dataset | IterableDataset] = {}
        if combined_train is not None:
            if data_config.streaming and all_default_weights and data_config.shuffle:
                combined_train = _shuffle_streaming_dataset(combined_train)
            splits[train_split] = combined_train
        if combined_val is not None and val_split:
            splits[val_split] = combined_val

        if data_config.streaming:
            dataset_dict = IterableDatasetDict(splits)
        else:
            dataset_dict = DatasetDict(splits)

    elif data_config.dataset_name:
        dataset_dict = _load_single_dataset(
            DatasetConfig(
                id=data_config.dataset_name,
                config=data_config.dataset_config,
                data_files=data_config.data_files,
            )
        )
        if data_config.train_split in dataset_dict and data_config.streaming and data_config.shuffle:
            dataset_dict[data_config.train_split] = _shuffle_streaming_dataset(
                dataset_dict[data_config.train_split]
            )
    else:
        raise ValueError("Either data.dataset_name or data.dataset_list must be configured.")

    return dataset_dict


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    tools: Any = None,
    return_assistant_tokens_mask: bool = False,
    **chat_template_kwargs: Any,
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
        tools=tools,
        return_assistant_tokens_mask=return_assistant_tokens_mask,
        **chat_template_kwargs,
    )


DOC_STARTS_COLUMN = "doc_starts"
"""Column recording where each packed document starts inside a packed row."""


class SequencePacker:
    """Pack whole token sequences into fixed-length rows.

    Packing exists to turn padding tokens into training tokens: instead of one padded
    row per sample, several samples share a row and the collator pads only the tail of
    each batch. A sequence is never split across rows and never shrunk to make room
    for a neighbour, so packing cannot silently drop training signal.

    ``lookahead`` controls how far the packer may reach past the sequence it is
    currently holding:

    * ``0`` (default) -- strict order-preserving next-fit. The held sequence is placed
      while it fits and the row is closed the moment it does not, exactly reproducing
      the source order. Closing a row early leaves a gap, so rows average roughly
      75-85% full on a realistic length distribution.
    * ``>0`` -- bounded lookahead. When the held sequence no longer fits, the largest
      sequence among the next ``lookahead`` that does fit is placed instead, and the
      skipped sequence is kept for a later row. Sequences move at most ``lookahead``
      positions and the fill ratio rises to ~95%+.

    ``Dataset.map`` requires the output to keep the input row count, so ``pack`` emits
    at most one row per input sequence and fills the remainder with placeholder rows
    that ``has_packed_documents`` filters out afterwards. Streaming pipelines have no
    such invariant and can set ``emit_placeholders=False``.
    """

    def __init__(
        self,
        max_seq_length: int,
        *,
        lookahead: int = 0,
        enabled: bool = True,
        emit_placeholders: bool = True,
    ) -> None:
        self.max_seq_length = int(max_seq_length)
        self.lookahead = max(0, int(lookahead))
        self.enabled = bool(enabled)
        self.emit_placeholders = bool(emit_placeholders)

    def plan(self, lengths: Sequence[int]) -> list[list[int]]:
        """Group source indices into rows that each fit ``max_seq_length``.

        Returns one inner list per output row holding the indices of the sequences it
        contains; an empty inner list is a placeholder row.
        """

        row_count = len(lengths)
        if not self.enabled:
            # Packing disabled: every sequence keeps its own row.
            return [[index] for index in range(row_count)]

        rows: list[list[int]] = []
        queue = list(range(row_count))
        buffer: list[int] = []
        buffer_length = 0

        while queue:
            window = min(len(queue), 1 + self.lookahead)
            candidate = -1
            for position in range(window):
                index = queue[position]
                if buffer_length + lengths[index] > self.max_seq_length:
                    continue
                # Largest sequence that still fits keeps rows full.
                if candidate < 0 or lengths[index] > lengths[queue[candidate]]:
                    candidate = position

            if candidate < 0:
                if not buffer:
                    # Defensive: a sequence longer than max_seq_length (a tokenizer that
                    # ignores its own max_length) still goes in rather than looping.
                    candidate = 0
                else:
                    # Nothing in the window fits, so close the row and try again.
                    rows.append(buffer)
                    buffer, buffer_length = [], 0
                    continue

            index = queue.pop(candidate)
            buffer.append(index)
            buffer_length += lengths[index]

            if buffer_length >= self.max_seq_length:
                rows.append(buffer)
                buffer, buffer_length = [], 0

        if buffer:
            rows.append(buffer)

        if self.emit_placeholders:
            rows.extend([] for _ in range(row_count - len(rows)))
        return rows

    def pack(self, columns: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, list[list[int]]]:
        """Concatenate the planned rows of every parallel column.

        ``columns`` holds the per-sequence token columns to pack (for example
        ``{"input_ids": ids, "labels": labels}``). ``doc_starts`` is added to the
        result because it is what lets the collator rebuild the boundary loss mask.
        """

        if not columns:
            raise ValueError("SequencePacker.pack requires at least one column to pack.")
        lengths = [len(sequence) for sequence in next(iter(columns.values()))]
        plan = self.plan(lengths)
        packed = {
            name: [[token for index in row for token in sequences[index]] for row in plan]
            for name, sequences in columns.items()
        }
        packed[DOC_STARTS_COLUMN] = compute_doc_starts(plan, lengths)
        return packed


def compute_doc_starts(plan: Sequence[Sequence[int]], lengths: Sequence[int]) -> list[list[int]]:
    """Return the offset of every packed document inside its row."""

    starts_by_row: list[list[int]] = []
    for row in plan:
        starts: list[int] = []
        offset = 0
        for index in row:
            starts.append(offset)
            offset += lengths[index]
        starts_by_row.append(starts)
    return starts_by_row


def has_packed_documents(row: Mapping[str, Any]) -> bool:
    """Return True for real packed rows and False for packing placeholders."""

    return len(row[DOC_STARTS_COLUMN]) > 0


def packed_features(*token_columns: str, dtype: str = "int32") -> Features:
    """Arrow schema for packed rows: `int32` token columns plus `doc_starts`.

    Declaring ``int32`` matters: ``datasets`` infers ``int64`` for plain Python integer
    lists, which doubles the Arrow footprint, while ``int32`` covers any realistic
    vocabulary and any ``ignore_index``. ``doc_starts`` costs O(rows + documents)
    rather than O(tokens), so it is essentially free next to the tokens it describes.
    """

    columns = {name: ArrowSequence(Value(dtype)) for name in token_columns}
    columns[DOC_STARTS_COLUMN] = ArrowSequence(Value(dtype))
    return Features(columns)


def resolve_pad_token_id(tokenizer: Any) -> int:
    """Return the tokenizer padding id, falling back to EOS for pad-less models."""

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError(
            f"Tokenizer {type(tokenizer).__name__} defines neither pad_token_id nor "
            "eos_token_id, so batches cannot be padded."
        )
    return int(pad_token_id)


class PackedCausalLMCollator:
    """Pad packed token rows and rebuild the columns that are not cached on disk.

    Cached rows carry ``input_ids``, an optional ``labels``, and ``doc_starts``. Both
    ``attention_mask`` and (when absent) ``labels`` are reconstructed here, which
    removes the redundant all-ones mask column from the Arrow cache without changing
    what the model sees.

    ``labels`` is ignored on padding and -- when ``boundary_loss_mask`` is on -- on the
    first token of every packed document after the first one. That stops the loss from
    being asked to predict the opening token of a different document right after the
    previous document's ``<eos>``.
    """

    def __init__(
        self,
        pad_token_id: int,
        *,
        ignore_index: int = -100,
        boundary_loss_mask: bool = True,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.boundary_loss_mask = boundary_loss_mask

    @staticmethod
    def _pad(sequences: Sequence[Sequence[int]], padding_value: int) -> torch.Tensor:
        """Right-pad a batch of token sequences into a single long tensor.

        ``torch.nn.utils.rnn.pad_sequence`` cannot pad a zero-length sequence, and a
        packing placeholder row is exactly that, so pad explicitly instead.
        """

        max_length = max((len(sequence) for sequence in sequences), default=0)
        padded = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            if sequence:
                padded[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        return padded

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of packed tokenized examples."""

        lengths = [len(item["input_ids"]) for item in batch]
        input_ids = self._pad([item["input_ids"] for item in batch], self.pad_token_id)

        # Derive the mask from the recorded lengths instead of from `input_ids`:
        # `pad_token_id` is often the same id as `eos_token_id`, and the real `<eos>`
        # tokens inside a packed row must stay visible to the model.
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            attention_mask[row, :length] = 1

        if "labels" in batch[0]:
            labels = self._pad([item["labels"] for item in batch], self.ignore_index)
        else:
            labels = input_ids.clone()
            labels[attention_mask == 0] = self.ignore_index

        if self.boundary_loss_mask:
            self._mask_document_boundaries(labels, batch)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_document_boundaries(self, labels: torch.Tensor, batch: list[dict[str, Any]]) -> None:
        """Ignore the token that starts a new document inside a packed row."""

        sequence_length = labels.shape[1]
        for row, item in enumerate(batch):
            for start in item.get(DOC_STARTS_COLUMN) or ():
                # Offset 0 is never a prediction target after the internal shift, and
                # anything past the real sequence belongs to a placeholder's tail.
                if 0 < start < sequence_length:
                    labels[row, start] = self.ignore_index


class BaseDataModule(LightningDataModule):
    """Reusable base class for dataset-driven Lightning data modules."""

    def __init__(self, data_config: DataConfig) -> None:
        super().__init__()
        self.data_config = data_config
        self.train_dataset: Optional[Dataset | IterableDataset] = None
        self.val_dataset: Optional[Dataset | IterableDataset] = None

    def load_dataset_dict(self) -> DatasetDict | IterableDatasetDict:
        """Load the configured dataset dictionary."""

        return load_dataset_from_config(self.data_config)

    def resolve_val_split_name(self, dataset_dict: DatasetDict | IterableDatasetDict) -> Optional[str]:
        """Resolve the validation split name for the current dataset dictionary."""

        if self.data_config.val_split and self.data_config.val_split in dataset_dict:
            return self.data_config.val_split
        if DEFAULT_VAL_SPLIT_NAME in dataset_dict:
            return DEFAULT_VAL_SPLIT_NAME
        return None

    def map_dataset(
        self,
        dataset: Dataset | IterableDataset,
        preprocess_fn: Callable[..., dict[str, Any]],
        desc: str,
        **kwargs: Any,
    ) -> Dataset | IterableDataset:
        """Apply a shared batched dataset preprocessing transform."""

        if isinstance(dataset, IterableDataset):
            remove_columns = list(dataset.column_names) if dataset.column_names else None
            return dataset.map(
                preprocess_fn,
                batched=True,
                batch_size=self.data_config.preprocessing_batch_size,
                remove_columns=remove_columns,
                **kwargs,
            )

        return dataset.map(
            preprocess_fn,
            batched=True,
            batch_size=self.data_config.preprocessing_batch_size,
            num_proc=self.data_config.num_workers,
            remove_columns=list(dataset.column_names),
            load_from_cache_file=self.data_config.preprocessing_use_cache,
            keep_in_memory=self.data_config.preprocessing_keep_in_memory,
            desc=desc,
            **kwargs,
        )

    def map_packed_dataset(
        self,
        dataset: Dataset | IterableDataset,
        preprocess_fn: Callable[..., dict[str, Any]],
        desc: str,
        *,
        features: Optional[Features] = None,
        drop_placeholders: bool = True,
    ) -> Dataset | IterableDataset:
        """Map a packing transform and drop the placeholder rows it emits.

        `Dataset.map` cannot change the row count, so a packing transform emits at most
        one row per input row and pads the rest with empty placeholders. Filtering them
        out is index based, so the packed Arrow file is reused rather than rewritten.
        """

        if isinstance(dataset, IterableDataset):
            packed = self.map_dataset(dataset, preprocess_fn, desc=desc)
            return packed.filter(has_packed_documents) if drop_placeholders else packed

        packed = self.map_dataset(dataset, preprocess_fn, desc=desc, features=features)
        if not drop_placeholders:
            return packed
        return packed.filter(
            has_packed_documents,
            num_proc=self.data_config.num_workers,
            load_from_cache_file=self.data_config.preprocessing_use_cache,
            keep_in_memory=self.data_config.preprocessing_keep_in_memory,
            desc=f"{desc} (dropping pack placeholders)",
        )

    def build_sequence_packer(
        self,
        max_seq_length: int,
        *,
        emit_placeholders: bool = True,
    ) -> SequencePacker:
        """Construct a packer from the shared data configuration."""

        return SequencePacker(
            max_seq_length,
            lookahead=self.data_config.packing_lookahead,
            enabled=self.data_config.packing_enabled,
            emit_placeholders=emit_placeholders,
        )

    def _build_dataloader(
        self,
        dataset: Dataset | IterableDataset,
        batch_size: int,
        collate_fn: Callable[[list[dict[str, Any]]], Any],
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        """Build a dataloader with shared worker and memory settings."""

        is_iterable = isinstance(dataset, IterableDataset)
        num_workers = 0 if is_iterable else self.data_config.num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if is_iterable else shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=drop_last,
        )


class ChatTemplateDataModule(BaseDataModule):
    """Base class for chat-style pipelines reading OpenAI-style message columns.

    Subclasses are expected to set `self.chat_processor` before calling
    `prepare_messages`.
    """

    def __init__(
        self,
        data_config: DataConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(data_config=data_config)
        self.system_prompt = system_prompt

    def reject_sequence_packing(self, task: str, reason: str) -> None:
        """Fail loudly when a pipeline that cannot pack is configured to pack.

        Packing is a data-level optimisation with a real semantic cost (tokens from
        different samples attend to each other), and for some pipelines it is outright
        incorrect. Raising here turns a silently ignored config key into a clear error.
        """

        if self.data_config.packing_enabled:
            raise ValueError(f"data.packing_enabled is not supported for {task}: {reason}")

    def prepare_messages(self, sample: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Any]:
        """Read one dataset row as OpenAI messages plus tool definitions.

        Only the OpenAI shape is supported, so this validates the row and applies the
        system-prompt policy; it never rewrites one format into another.
        """

        messages = sample.get(self.data_config.messages_column)
        if not messages:
            raise KeyError(
                f"Sample carries no non-empty '{self.data_config.messages_column}' column. "
                "Chat datasets must provide OpenAI messages: a list of "
                "{'role', 'content'} dicts."
            )
        return preprocess_chat_messages(
            messages,
            tools=sample.get(self.data_config.tools_column),
            system_prompt=sample_system_prompt(self.system_prompt, self.data_config.add_system_ratio),
        )
