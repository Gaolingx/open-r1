"""SFT data module and collation utilities for Lightning."""

from __future__ import annotations

from functools import partial
import re
from typing import Any, Callable, Optional

import pyarrow as pa
from datasets import Dataset, IterableDataset
from lightning.pytorch.utilities import rank_zero_warn

from lightning_grpo.utils.configs.base import ModelConfig, OptimizationConfig
from lightning_grpo.utils.configs.sft import SFTDataConfig
from lightning_grpo.data.base import (
    ChatTemplateProcessor,
    ChatTemplateDataModule,
    PackedCausalLMCollator,
    SequencePacker,
    iter_batch_samples,
    resolve_pad_token_id,
    resolve_shuffle_state,
)
from lightning_grpo.utils.modeling_common import load_tokenizer


class SkipSFTSampleError(ValueError):
    """Raised when a malformed or non-trainable SFT sample should be skipped."""


class SFTConfigurationError(RuntimeError):
    """Raised for label-masking misconfiguration instead of silently skipping samples."""


class SFTDataModule(ChatTemplateDataModule):
    """Lightning data module for supervised fine-tuning.

    Chat samples are tokenized into `input_ids`/`labels`, packed into max-length rows,
    and collated with the shared `PackedCausalLMCollator`, which rebuilds the attention
    mask instead of caching it.

    Labels are built on the untruncated conversation and only then cut to `max_seq_length`,
    keeping the tail, so truncation never silently drops assistant supervision.
    """

    def __init__(
        self,
        data_config: SFTDataConfig,
        model_config: ModelConfig,
        optimization_config: OptimizationConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(data_config=data_config, system_prompt=system_prompt)
        self.optimization_config = optimization_config
        self.tokenizer = load_tokenizer(model_config)
        self.chat_processor = ChatTemplateProcessor(self.tokenizer)
        self.collator = PackedCausalLMCollator(
            resolve_pad_token_id(self.tokenizer),
            ignore_index=self.data_config.ignore_index,
            boundary_loss_mask=self.data_config.packing_boundary_loss_mask,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and preprocess train and validation datasets."""

        self._validate_assistant_only_loss_support()
        dataset_dict = self.load_dataset_dict()

        train_split = dataset_dict[self.data_config.train_split]
        self.train_dataset = self._build_dataset(
            train_split,
            desc="Tokenizing and packing SFT dataset",
        )

        self.val_dataset = None
        val_split_name = self.resolve_val_split_name(dataset_dict)
        if val_split_name is not None:
            self.val_dataset = self._build_dataset(
                dataset_dict[val_split_name],
                desc="Tokenizing and packing SFT validation dataset",
            )

    def _validate_assistant_only_loss_support(self) -> None:
        """Fail fast when assistant-only labels cannot be built for this tokenizer/config.

        The assistant mask comes from the chat template's ``{% generation %}`` blocks. Without
        a usable mask every sample raises during preprocessing and is turned into a warning, so
        training would silently run on an empty dataset.
        """

        if not self.data_config.assistant_only_loss:
            return

        has_generation_block = bool(
            re.search(r"\{\%-?\s*generation\s*-?\%\}", getattr(self.tokenizer, "chat_template", None) or "")
        )
        if has_generation_block:
            # A generation block can still yield an all-false mask, so probe one conversation.
            try:
                processed = self.chat_processor.tokenize(
                    [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}],
                    add_generation_prompt=False,
                    tools=None,
                    max_length=None,
                    return_assistant_tokens_mask=True,
                )
            except Exception as exc:  # re-raised with configuration guidance
                raise SFTConfigurationError(
                    "data.assistant_only_loss=True could not obtain an assistant token mask from the tokenizer. "
                    f"Tokenizer error: {type(exc).__name__}: {exc}. Fix the tokenizer chat template or set "
                    "data.assistant_only_loss=False."
                ) from exc

            if any(self._extract_assistant_mask(processed)):
                return

        reason = (
            "the tokenizer returned an all-false mask for a minimal chat sample"
            if has_generation_block
            else "the tokenizer chat template has no '{% generation %}' block"
        )
        raise SFTConfigurationError(
            f"data.assistant_only_loss=True needs an assistant token mask, but {reason}. Add "
            "{% generation %}...{% endgeneration %} around the assistant output in the chat template, "
            "or set data.assistant_only_loss=False."
        )

    def _build_dataset(
        self,
        dataset: Dataset | IterableDataset,
        *,
        desc: str,
    ) -> Dataset | IterableDataset:
        """Tokenize chat samples and pack them into max-length rows."""

        packer = self.build_sequence_packer(self.data_config.max_seq_length)
        transform = self._build_transform(packer)
        return self.map_dataset(dataset, transform, desc=desc, table_output=True)

    @staticmethod
    def _extract_assistant_mask(processed: dict[str, Any]) -> list[bool]:
        """Read the assistant-token mask emitted by `return_assistant_tokens_mask=True`."""

        return [bool(value) for value in processed.get("assistant_masks") or []]

    @staticmethod
    def _message_has_text(message: dict[str, Any]) -> bool:
        """Return whether a message carries trainable content or a tool call.

        Tool-call-only assistant turns have ``content: None`` but are still trainable: the
        template renders their ``tool_calls`` and ends the turn with EOS.
        """

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
        return bool(message.get("tool_calls"))

    def _build_transform(
        self,
        packer: SequencePacker,
    ) -> Callable[[dict[str, list[Any]]], pa.Table]:
        """Build the row -> packed token table transform."""

        chat_processor = self.chat_processor
        data_config = self.data_config

        prepare_messages = partial(
            self.prepare_messages,
            data_config=data_config,
            system_prompt=self.system_prompt,
        )

        extract_assistant_mask = self._extract_assistant_mask
        message_has_text = self._message_has_text

        def should_add_generation_prompt(messages: list[dict[str, Any]]) -> bool:
            return chat_processor.should_add_generation_prompt(messages, data_config.add_generation_prompt)

        def apply_chat_template_tokenize(
            messages: list[dict[str, Any]],
            tools: Any,
            *,
            add_generation_prompt: bool,
            return_assistant_tokens_mask: bool = False,
        ) -> dict[str, Any]:
            # Labels must be built on the full conversation: truncating here cuts the target tail.
            return chat_processor.tokenize(
                messages,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                max_length=None,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
            )

        def prompt_messages_only(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if messages and messages[-1].get("role") == "assistant":
                return messages[:-1]
            return list(messages)

        def prompt_token_count(messages: list[dict[str, Any]], tools: Any) -> int:
            prompt_messages = prompt_messages_only(messages)
            processed = apply_chat_template_tokenize(
                prompt_messages,
                tools,
                add_generation_prompt=should_add_generation_prompt(prompt_messages),
            )
            return len(processed["input_ids"])

        def build_labels(
            messages: list[dict[str, Any]],
            tools: Any,
            full_ids: list[int],
            processed: dict[str, Any],
        ) -> list[int]:
            labels = list(full_ids)

            if data_config.completion_only_loss:
                prompt_len = prompt_token_count(messages, tools)
                labels = [
                    token_id if index >= prompt_len else data_config.ignore_index
                    for index, token_id in enumerate(labels)
                ]

            if data_config.assistant_only_loss:
                assistant_mask = extract_assistant_mask(processed)
                if not assistant_mask or not any(assistant_mask):
                    raise SFTConfigurationError(
                        "assistant_only_loss=True requires {% generation %}...{% endgeneration %} blocks around the "
                        "assistant output in the chat template."
                    )
                if len(assistant_mask) != len(full_ids):
                    raise SFTConfigurationError("Assistant token mask length does not match tokenized input length.")
                labels = [label if mask else data_config.ignore_index for label, mask in zip(labels, assistant_mask)]

            return labels

        def validate_messages_for_training(messages: list[dict[str, Any]]) -> None:
            if not any(message.get("role") == "user" and message_has_text(message) for message in messages):
                raise SkipSFTSampleError("no user message with text")
            if not any(message.get("role") == "assistant" and message_has_text(message) for message in messages):
                raise SkipSFTSampleError("no assistant message with content or tool_calls")

        def tokenize_sample(sample: dict[str, Any]) -> dict[str, list[int]]:
            messages, tools = prepare_messages(sample)
            validate_messages_for_training(messages)
            add_generation_prompt = should_add_generation_prompt(messages)

            processed = apply_chat_template_tokenize(
                messages,
                tools,
                add_generation_prompt=add_generation_prompt,
                return_assistant_tokens_mask=data_config.assistant_only_loss,
            )

            input_ids = list(processed["input_ids"])
            labels = build_labels(messages, tools, input_ids, processed)
            if not any(label != data_config.ignore_index for label in labels):
                raise SkipSFTSampleError("assistant mask produced no trainable labels")

            max_length = data_config.max_seq_length
            if len(input_ids) > max_length:
                # The SFT goal sits at the conversation end, so drop history from the front.
                # `SequencePacker` never truncates, so the length invariant must hold here.
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]
                if not any(label != data_config.ignore_index for label in labels):
                    raise SkipSFTSampleError("supervised target cut off by max_seq_length")

            return {"input_ids": input_ids, "labels": labels}

        def preprocess_batch(batch: dict[str, list[Any]]) -> pa.Table:
            tokenized_samples: list[dict[str, list[int]]] = []
            skipped: list[str] = []

            for raw_sample in iter_batch_samples(batch):
                try:
                    tokenized_samples.append(tokenize_sample(raw_sample))
                except SFTConfigurationError:
                    # Masking misconfiguration must fail the run, not silently shrink the corpus.
                    raise
                except Exception as exc:
                    skipped.append(f"{type(exc).__name__}: {exc}")

            if skipped:
                rank_zero_warn(f"Skipped {len(skipped)} SFT sample(s): {sorted(set(skipped))}")

            return packer.pack(
                {
                    "input_ids": [sample["input_ids"] for sample in tokenized_samples],
                    "labels": [sample["labels"] for sample in tokenized_samples],
                }
            )

        return preprocess_batch

    def train_dataloader(self):
        """Build the training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("SFT train dataset is not initialized. Call setup() first.")
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
