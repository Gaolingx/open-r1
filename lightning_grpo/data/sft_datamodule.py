"""SFT data module and collation utilities for Lightning."""

from __future__ import annotations

from typing import Any, Literal, Optional

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

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

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
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class RenderedAssistantSpanError(RuntimeError):
    """Raised when assistant spans cannot be aligned in rendered chat tokens."""


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
        self.collator = SFTBatchCollator(self.tokenizer)

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
    def _split_prompt_and_completion(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split a chat transcript into prompt messages and the final assistant completion."""

        if messages and messages[-1].get("role") == "assistant":
            return messages[:-1], [messages[-1]]
        return list(messages), []

    def _should_add_generation_prompt(self, messages: list[dict[str, Any]]) -> bool:
        """Decide whether chat rendering should append a generation prompt."""

        return self.chat_processor.should_add_generation_prompt(messages, self.data_config.add_generation_prompt)

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
    def _labels_from_mask(full_ids: list[int], assistant_mask: list[bool]) -> list[int]:
        """Keep loss only on tokens selected by a tokenizer-provided mask."""

        if len(assistant_mask) != len(full_ids):
            raise RuntimeError("Assistant token mask length does not match tokenized input length.")
        return [token_id if mask else -100 for token_id, mask in zip(full_ids, assistant_mask)]

    @staticmethod
    def _labels_from_last_assistant_mask(
        full_ids: list[int],
        assistant_mask: list[bool],
        prompt_len: int,
    ) -> list[int]:
        """Keep loss only on the final assistant turn using the assistant mask and prompt boundary."""

        if len(assistant_mask) != len(full_ids):
            raise RuntimeError("Assistant token mask length does not match tokenized input length.")
        prompt_len = min(prompt_len, len(full_ids))
        return [
            token_id if mask and index >= prompt_len else -100
            for index, (token_id, mask) in enumerate(zip(full_ids, assistant_mask))
        ]

    @staticmethod
    def _find_token_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
        """Return the first token-subsequence match at or after start, or -1 when absent."""

        if not needle:
            return start
        max_start = len(haystack) - len(needle)
        for index in range(start, max_start + 1):
            if haystack[index:index + len(needle)] == needle:
                return index
        return -1

    @staticmethod
    def _tokenize_text_without_specials(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
        """Tokenize rendered chat text without adding extra tokenizer special tokens."""

        if not text:
            return []
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def _render_chat_text(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        *,
        add_generation_prompt: bool,
    ) -> str:
        """Render chat messages to text without tokenizing."""

        return self.chat_processor.render(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _build_rendered_turn_spans(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        full_ids: list[int],
    ) -> list[tuple[str, int, int]]:
        """Align each rendered chat turn to token spans without relying on template generation blocks."""

        spans: list[tuple[str, int, int]] = []
        rendered_prefix = ""
        token_offset = 0

        for index, message in enumerate(messages):
            prefix_messages = messages[:index]
            current_messages = messages[:index + 1]
            prefix_text = self._render_chat_text(prefix_messages, tools, add_generation_prompt=False) if prefix_messages else ""
            current_text = self._render_chat_text(current_messages, tools, add_generation_prompt=False)

            if rendered_prefix and current_text.startswith(rendered_prefix):
                turn_text = current_text[len(rendered_prefix):]
            elif prefix_text and current_text.startswith(prefix_text):
                turn_text = current_text[len(prefix_text):]
            else:
                raise RenderedAssistantSpanError("rendered chat template is not prefix-stable")

            turn_ids = self._tokenize_text_without_specials(self.tokenizer, turn_text)
            start = self._find_token_subsequence(full_ids, turn_ids, token_offset)
            if start < 0:
                raise RenderedAssistantSpanError("rendered chat turn tokens could not be aligned")
            end = min(start + len(turn_ids), len(full_ids))
            spans.append((str(message.get("role", "")), start, end))
            rendered_prefix = current_text
            token_offset = end

        return spans

    def _assistant_target_span(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        full_ids: list[int],
        assistant_index: int,
        span_start: int,
        span_end: int,
    ) -> tuple[int, int]:
        """Return assistant target span: generated content plus the assistant turn-closing EOS."""

        assistant_prompt = self._render_chat_text(
            messages[:assistant_index],
            tools,
            add_generation_prompt=True,
        )
        assistant_prompt_ids = self._tokenize_text_without_specials(self.tokenizer, assistant_prompt)
        prompt_start = self._find_token_subsequence(full_ids, assistant_prompt_ids, 0)
        if prompt_start < 0:
            return span_start, span_end
        target_start = min(max(prompt_start + len(assistant_prompt_ids), span_start), span_end)
        return target_start, span_end

    def _labels_from_rendered_assistant_spans(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        full_ids: list[int],
        label_mode: Literal["last_assistant", "all_assistant"],
    ) -> list[int]:
        """Mask labels by assistant turns at the data-processing layer."""

        labels = [-100] * len(full_ids)
        spans = self._build_rendered_turn_spans(messages, tools, full_ids)
        assistant_spans = [
            (index, start, end)
            for index, (role, start, end) in enumerate(spans)
            if role == "assistant"
        ]
        if label_mode == "last_assistant":
            assistant_spans = assistant_spans[-1:]
        for index, start, end in assistant_spans:
            target_start, target_end = self._assistant_target_span(messages, tools, full_ids, index, start, end)
            labels[target_start:target_end] = full_ids[target_start:target_end]
        return labels

    def _tokenize_render_input_only_sample(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        label_mode: Literal["last_assistant", "all_assistant"],
    ) -> dict[str, list[int]]:
        """Tokenize SFT data by separately rendering prompt and assistant answer segments.

        This supports modern chat templates that do not expose Hugging Face
        ``{% generation %}`` blocks.  For each assistant turn, the prompt segment is
        rendered with ``add_generation_prompt=True`` and masked out, while the
        assistant answer suffix is rendered from the completed turn and kept as
        labels.  This mirrors the render-input-only strategy used by LLaMA-Factory
        style preprocessing.
        """

        input_ids: list[int] = []
        labels: list[int] = []
        consumed_text = ""
        assistant_answer_ranges: list[tuple[int, int]] = []

        for assistant_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            prompt_text = self._render_chat_text(
                messages[:assistant_index],
                tools,
                add_generation_prompt=True,
            )
            completed_text = self._render_chat_text(
                messages[:assistant_index + 1],
                tools,
                add_generation_prompt=False,
            )
            if not prompt_text.startswith(consumed_text) or not completed_text.startswith(prompt_text):
                raise RenderedAssistantSpanError("render-input-only chat template is not prefix-stable")

            prompt_segment_text = prompt_text[len(consumed_text):]
            answer_segment_text = completed_text[len(prompt_text):]
            prompt_ids = self._tokenize_text_without_specials(self.tokenizer, prompt_segment_text)
            answer_ids = self._tokenize_text_without_specials(self.tokenizer, answer_segment_text)

            input_ids.extend(prompt_ids)
            labels.extend([-100] * len(prompt_ids))
            answer_start = len(input_ids)
            input_ids.extend(answer_ids)
            labels.extend(answer_ids)
            assistant_answer_ranges.append((answer_start, len(input_ids)))
            consumed_text = completed_text

        if label_mode == "last_assistant" and assistant_answer_ranges:
            last_start, last_end = assistant_answer_ranges[-1]
            labels = [
                token_id if last_start <= index < last_end else -100
                for index, token_id in enumerate(input_ids)
            ]

        input_ids = input_ids[:self.data_config.max_seq_length]
        labels = labels[:self.data_config.max_seq_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def _apply_chat_template_tokenize(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        *,
        add_generation_prompt: bool,
        return_assistant_tokens_mask: bool = False,
    ) -> dict[str, Any] | None:
        """Tokenize chat messages with tokenizer templates when supported."""

        return self.chat_processor.tokenize(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            max_length=self.data_config.max_seq_length,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
        )

    def _render_chat_tokenize(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        *,
        add_generation_prompt: bool,
    ) -> dict[str, list[int]]:
        """Render chat text then tokenize it, matching the shared GRPO/pretrain abstractions."""

        return self.chat_processor.tokenize(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            max_length=self.data_config.max_seq_length,
        )

    def _prompt_token_count(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
    ) -> int:
        """Tokenize the prompt prefix used by completion-only supervision."""

        prompt_messages, _ = self._split_prompt_and_completion(messages)
        processed = self._apply_chat_template_tokenize(
            prompt_messages,
            tools,
            add_generation_prompt=self._should_add_generation_prompt(prompt_messages),
        )
        if processed is None:
            processed = self._render_chat_tokenize(
                prompt_messages,
                tools,
                add_generation_prompt=self._should_add_generation_prompt(prompt_messages),
            )
        return len(processed["input_ids"])

    def _build_labels(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        full_ids: list[int],
        processed: dict[str, Any] | None,
        label_mode: Literal["all_tokens", "last_assistant", "all_assistant"],
    ) -> list[int]:
        """Build labels according to the configured SFT supervision mode."""

        if label_mode == "all_tokens":
            return list(full_ids)

        assistant_mask = self._extract_assistant_mask(processed or {}) if processed is not None else []
        has_assistant_mask = len(assistant_mask) == len(full_ids) and any(assistant_mask)
        if has_assistant_mask and label_mode == "all_assistant":
            return self._labels_from_mask(full_ids, assistant_mask)
        if has_assistant_mask and label_mode == "last_assistant":
            return self._labels_from_last_assistant_mask(
                full_ids,
                assistant_mask,
                self._prompt_token_count(messages, tools),
            )
        return self._labels_from_rendered_assistant_spans(messages, tools, full_ids, label_mode)

    @staticmethod
    def _message_has_text(message: dict[str, Any]) -> bool:
        """Return whether a message has non-empty text content."""

        content = message.get("content", "")
        if isinstance(content, str):
            return bool(content.strip())
        if content is None:
            return False
        return bool(content)

    @staticmethod
    def _validate_messages_for_training(messages: list[dict[str, Any]]) -> None:
        """Skip samples that do not contain a usable SFT prompt/completion pair."""

        user_messages = [message for message in messages if message.get("role") == "user"]
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        if not user_messages:
            raise SkipSFTSampleError("missing user message")
        if not assistant_messages:
            raise SkipSFTSampleError("missing assistant message")
        if not any(SFTDataModule._message_has_text(message) for message in user_messages):
            raise SkipSFTSampleError("empty prompt")
        if not any(SFTDataModule._message_has_text(message) for message in assistant_messages):
            raise SkipSFTSampleError("empty response")

    @staticmethod
    def _has_trainable_labels(labels: list[int]) -> bool:
        """Return whether at least one token contributes to the SFT loss."""

        return any(label != -100 for label in labels)

    def _tokenize_sample(
        self,
        sample: dict[str, Any],
        label_mode: Literal["all_tokens", "last_assistant", "all_assistant"],
    ) -> dict[str, list[int]]:
        """Format one dataset row into token ids and SFT labels."""

        messages, tools = self.chat_processor.prepare_sample(sample)
        messages = preprocess_chat_messages(messages)
        self._validate_messages_for_training(messages)
        add_generation_prompt = self._should_add_generation_prompt(messages)
        needs_assistant_mask = label_mode in {"last_assistant", "all_assistant"}

        processed = self._apply_chat_template_tokenize(
            messages,
            tools,
            add_generation_prompt=add_generation_prompt,
            return_assistant_tokens_mask=needs_assistant_mask,
        )
        if processed is None:
            processed = self._render_chat_tokenize(messages, tools, add_generation_prompt=add_generation_prompt)

        assistant_mask = self._extract_assistant_mask(processed) if needs_assistant_mask else []
        if needs_assistant_mask and not any(assistant_mask):
            return self._tokenize_render_input_only_sample(messages, tools, label_mode)

        input_ids = list(processed["input_ids"])
        attention_mask = list(processed.get("attention_mask", [1] * len(input_ids)))
        labels = self._build_labels(messages, tools, input_ids, processed, label_mode)
        if not self._has_trainable_labels(labels):
            raise SkipSFTSampleError("assistant mask produced no trainable labels")
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _tokenize_dataset(self, dataset: Dataset, formatter: Any) -> Dataset:
        """Convert dataset rows into tokenized causal language modeling samples."""

        def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            label_mode = self.data_config.label_mode
            tokenized_samples: list[dict[str, list[int]]] = []
            skipped_reasons: dict[str, int] = {}

            for raw_sample in self.iter_batch_samples(batch):
                try:
                    sample = formatter(raw_sample)
                    tokenized_samples.append(self._tokenize_sample(sample, label_mode))
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
