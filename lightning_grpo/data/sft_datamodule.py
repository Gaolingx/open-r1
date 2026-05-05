"""SFT data module and collation utilities for Lightning."""

from __future__ import annotations

from typing import Any, Optional

import torch
from datasets import Dataset
from lightning.pytorch.utilities import rank_zero_warn
from transformers import PreTrainedTokenizerBase

from lightning_grpo.utils.configs.base import ModelConfig, OptimizationConfig
from lightning_grpo.utils.configs.sft import SFTDataConfig
from lightning_grpo.data.base import (
    ChatTemplateProcessor,
    ChatTemplateDataModule,
    preprocess_chat_messages,
)
from lightning_grpo.utils.modeling import load_tokenizer


class SkipSFTSampleError(ValueError):
    """Raised when a malformed or non-trainable SFT sample should be skipped."""


class SFTBatchCollator:
    """Causal LM collator for pre-tokenized SFT samples."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, ignore_index: int = -100) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of tokenized examples."""

        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        if "attention_mask" in batch[0]:
            attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
        else:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.ignore_index,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTDataModule(ChatTemplateDataModule):
    """Lightning data module for supervised fine-tuning."""

    def __init__(
        self,
        data_config: SFTDataConfig,
        model_config: ModelConfig,
        optimization_config: OptimizationConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(data_config=data_config, model_config=model_config, system_prompt=system_prompt)
        self.optimization_config = optimization_config
        self.tokenizer = load_tokenizer(model_config)
        self.chat_processor = ChatTemplateProcessor(self.tokenizer)
        self.collator = SFTBatchCollator(self.tokenizer, ignore_index=self.data_config.ignore_index)

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
    def _extract_assistant_mask(processed: dict[str, Any]) -> list[bool]:
        """Normalize assistant-token masks returned by chat templates across tokenizer variants."""

        raw_mask = processed.get("assistant_masks")
        if raw_mask is None:
            raw_mask = processed.get("assistant_mask")
        if raw_mask is None:
            return []
        return [bool(value) for value in raw_mask]

    @staticmethod
    def _find_subsequence_starts(sequence: list[int], subsequence: list[int]) -> list[int]:
        """Return all start offsets where a token subsequence appears."""

        if not subsequence or len(subsequence) > len(sequence):
            return []
        template_len = len(subsequence)
        return [
            index
            for index in range(len(sequence) - template_len + 1)
            if sequence[index: index + template_len] == subsequence
        ]

    @staticmethod
    def _next_marker_start(marker_starts: list[int], after: int, sequence_len: int) -> int:
        """Return the next marker offset after a position, or the sequence end."""

        return next((start for start in marker_starts if start > after), sequence_len)

    @staticmethod
    def _message_has_text(message: dict[str, Any]) -> bool:
        """Return whether a message has non-empty text content."""

        content = message.get("content", "")
        if isinstance(content, str):
            return bool(content.strip())
        if content is None:
            return False
        return bool(content)

    def _tokenize_dataset(self, dataset: Dataset, formatter: Any) -> Dataset:
        """Convert dataset rows into tokenized causal language modeling samples."""

        tokenizer = self.tokenizer
        chat_processor = self.chat_processor
        max_seq_length = self.data_config.max_seq_length
        completion_only_loss = self.data_config.completion_only_loss
        assistant_only_loss = self.data_config.assistant_only_loss
        ignore_index = self.data_config.ignore_index
        add_generation_prompt_config = self.data_config.add_generation_prompt
        assistant_response_template_ids_config = self.data_config.assistant_response_template_ids
        assistant_response_template = self.data_config.assistant_response_template
        instruction_template_ids_config = self.data_config.instruction_template_ids
        instruction_template = self.data_config.instruction_template

        iter_batch_samples = self.iter_batch_samples
        extract_assistant_mask = self._extract_assistant_mask
        find_subsequence_starts = self._find_subsequence_starts
        next_marker_start = self._next_marker_start
        message_has_text = self._message_has_text

        def assistant_response_template_ids() -> list[int]:
            if assistant_response_template_ids_config is not None:
                return list(assistant_response_template_ids_config)
            if assistant_response_template is None:
                return []
            return list(tokenizer.encode(assistant_response_template, add_special_tokens=False))

        def instruction_template_ids() -> list[int]:
            if instruction_template_ids_config is not None:
                return list(instruction_template_ids_config)
            if instruction_template is None:
                return []
            return list(tokenizer.encode(instruction_template, add_special_tokens=False))

        def should_add_generation_prompt(messages: list[dict[str, Any]]) -> bool:
            return chat_processor.should_add_generation_prompt(messages, add_generation_prompt_config)

        def apply_chat_template_tokenize(
            messages: list[dict[str, Any]],
            tools: Any,
            *,
            add_generation_prompt: bool,
            return_assistant_tokens_mask: bool = False,
        ) -> dict[str, Any] | None:
            return chat_processor.tokenize(
                messages,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                max_length=max_seq_length,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
            )

        def split_prompt_and_completion(
            messages: list[dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            if messages and messages[-1].get("role") == "assistant":
                return messages[:-1], [messages[-1]]
            return list(messages), []

        def prompt_token_count(messages: list[dict[str, Any]], tools: Any) -> int:
            prompt_messages, _ = split_prompt_and_completion(messages)
            processed = apply_chat_template_tokenize(
                prompt_messages,
                tools,
                add_generation_prompt=should_add_generation_prompt(prompt_messages),
            )
            return len(processed["input_ids"])

        def labels_from_mask(full_ids: list[int], assistant_mask: list[bool]) -> list[int]:
            if len(assistant_mask) != len(full_ids):
                raise RuntimeError("Assistant token mask length does not match tokenized input length.")
            return [token_id if mask else ignore_index for token_id, mask in zip(full_ids, assistant_mask)]

        def completion_mask_from_prompt_len(full_ids: list[int], prompt_len: int) -> list[bool]:
            """Build a completion mask from tokenized prompt length."""

            prompt_len = min(prompt_len, len(full_ids))
            return [index >= prompt_len for index in range(len(full_ids))]

        def labels_from_response_template(full_ids: list[int]) -> list[int]:
            response_template_ids = assistant_response_template_ids()
            if not response_template_ids:
                raise RuntimeError(
                    "Tokenizer did not provide assistant token masks. Configure data.assistant_response_template "
                    "or data.assistant_response_template_ids to mask SFT labels from assistant response markers."
                )

            response_starts = find_subsequence_starts(full_ids, response_template_ids)
            if not response_starts:
                raise RuntimeError(
                    "Configured assistant response template was not found in the tokenized sample. Check that "
                    "data.assistant_response_template exactly matches the model chat template output, or configure "
                    "data.assistant_response_template_ids."
                )

            current_instruction_template_ids = instruction_template_ids()
            instruction_starts = (
                find_subsequence_starts(full_ids, current_instruction_template_ids)
                if current_instruction_template_ids
                else []
            )
            if current_instruction_template_ids and not instruction_starts:
                raise RuntimeError(
                    "Configured instruction template was not found in the tokenized sample. Check that "
                    "data.instruction_template exactly matches the model chat template output, or configure "
                    "data.instruction_template_ids."
                )
            if len(response_starts) > 1 and not instruction_starts:
                raise RuntimeError(
                    "Multi-turn assistant-only masking with response templates requires data.instruction_template or "
                    "data.instruction_template_ids so user spans between assistant responses can be ignored."
                )

            all_marker_starts = sorted({*response_starts, *instruction_starts})
            labels = [ignore_index] * len(full_ids)
            for response_start in response_starts:
                label_start = response_start + len(response_template_ids)
                label_end = next_marker_start(all_marker_starts, response_start, len(full_ids))
                labels[label_start:label_end] = full_ids[label_start:label_end]
            return labels

        def build_labels(
            messages: list[dict[str, Any]],
            tools: Any,
            full_ids: list[int],
            processed: dict[str, Any] | None,
        ) -> list[int]:
            labels = list(full_ids)

            if completion_only_loss:
                completion_mask = completion_mask_from_prompt_len(full_ids, prompt_token_count(messages, tools))
                labels = [token_id if mask else ignore_index for token_id, mask in zip(labels, completion_mask)]

            if assistant_only_loss:
                assistant_mask = extract_assistant_mask(processed or {}) if processed is not None else []
                if assistant_mask and any(assistant_mask):
                    if len(assistant_mask) != len(full_ids):
                        raise RuntimeError("Assistant token mask length does not match tokenized input length.")
                    labels = [label if mask else ignore_index for label, mask in zip(labels, assistant_mask)]
                elif assistant_response_template_ids():
                    assistant_labels = labels_from_response_template(full_ids)
                    labels = [label if assistant_label != ignore_index else ignore_index for label, assistant_label in zip(labels, assistant_labels)]
                else:
                    raise RuntimeError(
                        "assistant_only_loss=True requires tokenizer support for assistant token masks or a configured "
                        "data.assistant_response_template/data.assistant_response_template_ids."
                    )

            return labels

        def validate_messages_for_training(messages: list[dict[str, Any]]) -> None:
            user_messages = [message for message in messages if message.get("role") == "user"]
            assistant_messages = [message for message in messages if message.get("role") == "assistant"]
            if not user_messages:
                raise SkipSFTSampleError("missing user message")
            if not assistant_messages:
                raise SkipSFTSampleError("missing assistant message")
            if not any(message_has_text(message) for message in user_messages):
                raise SkipSFTSampleError("empty prompt")
            if not any(message_has_text(message) for message in assistant_messages):
                raise SkipSFTSampleError("empty response")

        def tokenize_sample(sample: dict[str, Any]) -> dict[str, list[int]]:
            messages, tools = chat_processor.prepare_sample(sample)
            messages = preprocess_chat_messages(messages)
            validate_messages_for_training(messages)
            add_generation_prompt = should_add_generation_prompt(messages)
            needs_assistant_mask = assistant_only_loss

            processed = apply_chat_template_tokenize(
                messages,
                tools,
                add_generation_prompt=add_generation_prompt,
                return_assistant_tokens_mask=needs_assistant_mask,
            )

            input_ids = list(processed["input_ids"])
            attention_mask = list(processed.get("attention_mask", [1] * len(input_ids)))
            labels = build_labels(messages, tools, input_ids, processed)
            if not any(label != ignore_index for label in labels):
                raise SkipSFTSampleError("assistant mask produced no trainable labels")
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            tokenized_samples: list[dict[str, list[int]]] = []
            skipped_reasons: dict[str, int] = {}

            for raw_sample in iter_batch_samples(batch):
                try:
                    sample = formatter(raw_sample)
                    tokenized_samples.append(tokenize_sample(sample))
                except Exception as exc:
                    exc_message = str(exc)
                    reason = f"preprocessing error: {type(exc).__name__}"
                    if exc_message:
                        reason = f"{reason}: {exc_message}"
                    skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

            if skipped_reasons:
                rank_zero_warn(
                    "Skipped SFT samples: "
                    + ", ".join(f"{reason}={count}" for reason, count in sorted(skipped_reasons.items()))
                )

            return {
                "input_ids": [sample["input_ids"] for sample in tokenized_samples],
                "attention_mask": [sample["attention_mask"] for sample in tokenized_samples],
                "labels": [sample["labels"] for sample in tokenized_samples],
            }

        return self.map_dataset(
            dataset,
            preprocess_batch,
            desc="Tokenizing SFT dataset",
        )

    def train_dataloader(self):
        """Build the training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("SFT train dataset is not initialized. Call setup() first.")
        return self._build_dataloader(
            self.train_dataset,
            batch_size=self.optimization_config.train_micro_batch_size,
            collate_fn=self.collator,
            shuffle=not self.data_config.streaming,
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
