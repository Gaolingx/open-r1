"""Pretraining data module: tokenize raw text, pack documents, and collate batches."""

from __future__ import annotations

from typing import Any, Optional

from datasets import Dataset, IterableDataset

from lightning_grpo.data.base import (
    BaseDataModule,
    PackedCausalLMCollator,
    SequencePacker,
    packed_features,
    resolve_pad_token_id,
    resolve_shuffle_state,
)
from lightning_grpo.models.common import load_tokenizer
from lightning_grpo.utils.configs.pretrain import PretrainConfig


PACKED_FEATURES = packed_features("input_ids")
"""On-disk schema for packed pretraining rows: `input_ids` plus `doc_starts`.

`input_ids` stores every token exactly once; the all-ones `attention_mask` and the
`labels` copy of the previous pipeline are rebuilt by the collator instead of being
cached, which removed two of the three token arrays from the Arrow files.
"""


class PretrainTokenizeAndPack:
    """Tokenize raw text and pack whole documents into max-length rows.

    Documents are never split across rows and never shrunk to make room, so the only
    short rows are the tails of a packing batch, which the collator pads. That turns
    most padding tokens into real training tokens. `SequencePacker` documents the
    lookahead and placeholder-row semantics.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        text_column: str,
        max_seq_length: int,
        packer: SequencePacker,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.packer = packer
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        # Reserve room for the BOS/EOS wrappers that the tokenizer does not add itself.
        special_tokens = int(self.bos_token_id is not None) + int(self.eos_token_id is not None)
        self.inner_max_length = max(1, max_seq_length - special_tokens)

    def _wrap_document(self, token_ids: list[int]) -> list[int]:
        """Add the document delimiters used both for packing and for the LM loss."""

        tokens = list(token_ids)
        if self.bos_token_id is not None:
            tokens.insert(0, self.bos_token_id)
        if self.eos_token_id is not None:
            tokens.append(self.eos_token_id)
        return tokens

    def __call__(self, batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        """Tokenize one batch of raw texts and pack them into full-length rows."""

        texts = [str(text) for text in batch[self.text_column]]
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.inner_max_length,
            padding=False,
            add_special_tokens=False,
        )
        documents = [self._wrap_document(token_ids) for token_ids in encoded["input_ids"]]
        return self.packer.pack({"input_ids": documents})


class PretrainDataModule(BaseDataModule):
    """Lightning data module for causal LM pretraining."""

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__(data_config=config.data)
        self.config = config
        self.tokenizer = load_tokenizer(config.model)
        self.collator = PackedCausalLMCollator(
            resolve_pad_token_id(self.tokenizer),
            ignore_index=config.data.ignore_index,
            boundary_loss_mask=config.data.packing_boundary_loss_mask,
        )

    def _build_dataset(self, dataset: Dataset | IterableDataset, *, desc: str) -> Dataset | IterableDataset:
        """Tokenize raw text, pack whole documents into max-length rows, drop placeholders."""

        packer = self.build_sequence_packer(
            self.config.data.max_seq_length,
            # Streaming datasets have no fixed row count that must be preserved.
            emit_placeholders=not isinstance(dataset, IterableDataset),
        )
        transform = PretrainTokenizeAndPack(
            self.tokenizer,
            text_column=self.config.data.text_column,
            max_seq_length=self.config.data.max_seq_length,
            packer=packer,
        )

        return self.map_packed_dataset(
            dataset,
            transform,
            desc=desc,
            features=PACKED_FEATURES,
            drop_placeholders=self.config.data.packing_enabled,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load, tokenize, and pack the pretraining dataset."""

        dataset_dict = self.load_dataset_dict()
        train_dataset = dataset_dict[self.config.data.train_split]
        self.train_dataset = self._build_dataset(
            train_dataset,
            desc="Tokenizing and packing pretraining dataset",
        )

        self.val_dataset = None
        val_split_name = self.resolve_val_split_name(dataset_dict)
        if val_split_name is not None:
            self.val_dataset = self._build_dataset(
                dataset_dict[val_split_name],
                desc="Tokenizing and packing pretraining validation dataset",
            )

    def train_dataloader(self):
        """Build the training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("Pretrain dataset is not initialized. Call setup() first.")
        return self._build_dataloader(
            self.train_dataset,
            batch_size=self.config.optimization.train_micro_batch_size,
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
            batch_size=self.config.optimization.eval_micro_batch_size,
            collate_fn=self.collator,
            shuffle=False,
            drop_last=False,
        )
