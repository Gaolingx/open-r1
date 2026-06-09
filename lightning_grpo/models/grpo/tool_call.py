"""Tool calling integration for GRPO training.

This module provides a mixin class that adds multi-turn tool calling support
to the GRPOLightningModule. It handles ToolCallExecutor initialization,
the tool calling loop during rollouts, and policy-based generation for
tool calling when vLLM chat generation is not available.
"""

from __future__ import annotations

import copy
from typing import Any

import asyncio
import importlib
import inspect
import threading

import torch

from lightning_grpo.utils.chat_template.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
    supports_tool_calling,
)
from lightning_grpo.models.common import compute_per_token_logps
from lightning_grpo.models.grpo.rollout_module.utils import pad_sequences


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _resolve_callable(path: str) -> Any:
    """Import a callable from a fully-qualified dotted path."""

    module_path, _, attr_name = path.rpartition(".")
    if not module_path or not attr_name:
        raise ValueError(f"Tool path must be fully-qualified, got: {path!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, attr_name)
    if not callable(value):
        raise TypeError(f"Configured tool is not callable: {path}")
    return value


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------


class GRPOToolCallMixin:
    """Mixin that adds tool calling capabilities to GRPOLightningModule.

    Expects the host class to provide:
        - self.config (GRPOConfig)
        - self.tokenizer
        - self.policy
        - self.rollout_coordinator.rollout_engine
        - self.config.rollout.temperature
        - self.device
        - self.trainer
        - self.log(...)
    """

    @property
    def rollout_engine(self):
        """Shortcut to the rollout coordinator's engine."""
        return self.rollout_coordinator.rollout_engine

    def setup_tool_calling(self) -> None:
        """Initialize tool-calling state from ``config.rollout.tool_calling``."""

        tool_config = self.config.rollout.tool_calling
        self.tool_executor = None
        self.max_tool_calling_iterations = int(tool_config.max_iterations)
        self.processing_class = self.tokenizer
        self._tokenizer = self.tokenizer
        self._sync_tools: dict[str, Any] = {}
        self._async_tools: dict[str, Any] = {}
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None
        self.chat_template = tool_config.chat_template or getattr(self.tokenizer, "chat_template", None)
        self.chat_template_kwargs = tool_config.chat_template_kwargs or {}

        if not tool_config.enabled:
            return

        if tool_config.chat_template is not None:
            self.tokenizer.chat_template = tool_config.chat_template

        if not supports_tool_calling(self.processing_class):
            raise ValueError(
                "Tool calling is enabled, but the tokenizer chat template does not support tool-calling "
                "conversations. Please provide a tool-calling chat template via rollout.tool_calling.chat_template."
            )

        if getattr(self._tokenizer, "response_schema", None) is None:
            try:
                self.processing_class = add_response_schema(self.processing_class)
                self.tokenizer = self.processing_class
                self._tokenizer = self.tokenizer
            except Exception as exc:
                raise ValueError(
                    "Tool calling is enabled, but response schema could not be added. "
                    "Please configure tokenizer.response_schema manually."
                ) from exc

        if not is_chat_template_prefix_preserving(self.processing_class):
            self.chat_template = get_training_chat_template(self.processing_class)
            if self.chat_template is None:
                self.chat_template = self.processing_class.chat_template
            else:
                self.processing_class.chat_template = self.chat_template
                self.tokenizer = self.processing_class
                self._tokenizer = self.tokenizer
        else:
            self.chat_template = self.processing_class.chat_template

        for tool_path in tool_config.tools:
            tool = _resolve_callable(tool_path)
            name = getattr(tool, "__name__", tool_path.rsplit(".", 1)[-1])
            if inspect.iscoroutinefunction(tool):
                self._async_tools[name] = tool
            else:
                self._sync_tools[name] = tool

        if not self._sync_tools and not self._async_tools:
            raise ValueError("rollout.tool_calling.enabled=True requires at least one callable in rollout.tool_calling.tools.")

        if self._async_tools:
            self._start_tool_async_loop()
        self.tool_executor = self

    def _start_tool_async_loop(self) -> None:
        """Start a background event loop for async tools."""

        if self._async_loop is not None:
            return
        loop = asyncio.new_event_loop()

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, name="grpo-tool-calling-loop", daemon=True)
        thread.start()
        self._async_loop = loop
        self._async_thread = thread

    @property
    def async_loop(self) -> asyncio.AbstractEventLoop:
        if self._async_loop is None:
            self._start_tool_async_loop()
        if self._async_loop is None:
            raise RuntimeError("Failed to initialize async tool loop.")
        return self._async_loop

    def shutdown_tool_calling(self) -> None:
        """Release resources used by the tool-calling mixin."""

        if self._async_loop is not None:
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._async_thread is not None:
            self._async_thread.join(timeout=1.0)
        self._async_loop = None
        self._async_thread = None

    def shutdown(self) -> None:
        """Compatibility hook used by ``GRPOLightningModule.on_train_end``."""

        self.shutdown_tool_calling()

    def _generate_single_turn(self, prompt_id_lists: list[list[int]]) -> tuple[list[list[int]], list[list[float]] | None]:
        """Generate one assistant turn from already-tokenized prompt states."""

        prompt_ids = pad_sequences(prompt_id_lists, self.tokenizer.pad_token_id, self.device)
        attention_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
        result = self.rollout_engine.generate(
            prompt_ids=prompt_ids,
            attention_mask=attention_mask,
            num_generations=1,
            max_new_tokens=self.max_completion_length,
            temperature=self.config.rollout.temperature,
            top_p=self.config.rollout.top_p,
        )
        completion_ids = result.completion_id_lists
        logprobs = None
        if result.per_token_logps.numel() > 0:
            logprobs = [row[: len(ids)].detach().cpu().tolist() for row, ids in zip(result.per_token_logps, completion_ids, strict=True)]
        return completion_ids, logprobs
    
    def _get_tool_suffix_ids(self, tool_messages):
        """Get token IDs for tool result formatting by using a minimal dummy conversation."""
        # Use the real tool name instead of a dummy: some templates (e.g. GPT-OSS) derive the tool response
        # header from the assistant's tool call name.
        dummy_tool_calls = [{"type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}}}]
        dummy_messages = [
            {"role": "user", "content": "dummy"},
            {
                "role": "assistant",
                # "content" is required here because VLM processors crash on tokenize=True without it
                # (KeyError in processing_utils.py). See huggingface/transformers#45290.
                "content": "",
                "tool_calls": dummy_tool_calls,
            },
        ]

        prefix_ids = self.processing_class.apply_chat_template(
            dummy_messages,
            add_generation_prompt=False,
            tokenize=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        full_ids = self.processing_class.apply_chat_template(
            dummy_messages + tool_messages,
            add_generation_prompt=True,
            tokenize=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        prefix_ids = prefix_ids.tolist() if hasattr(prefix_ids, "tolist") else list(prefix_ids)
        full_ids = full_ids.tolist() if hasattr(full_ids, "tolist") else list(full_ids)

        # Some chat templates (notably Qwen3/Qwen3.5) render "...<|im_end|>\n" after an assistant/tool block.
        # When we compute `suffix_ids` by slicing `full_ids`, we must align the slicing boundary to
        # EOS (not EOS + newline). Templates that don't use EOS as end-of-turn (e.g. Gemma uses
        # <turn|>) skip this trimming.
        eos_positions = [i for i, tok_id in enumerate(prefix_ids) if tok_id == self._tokenizer.eos_token_id]
        if eos_positions:
            prefix_ids = prefix_ids[: eos_positions[-1] + 1]

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Unexpected tokenization: the EOS-trimmed prefix IDs are not a prefix of the full IDs.")
        return full_ids[len(prefix_ids) :]

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs):
        # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt
        tool_calls = [completion[0].get("tool_calls") for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_mask = [[1] * len(ids) for ids in completion_ids]  # 0 for tool result tokens, 1 elsewhere
        # Collect images from multimodal tool responses for the forward pass
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0

        while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]  # select only prompts that need tool calls
            # Snapshot state so we can rollback tool results that would exceed max_completion_length
            completions_len_before = [len(completions[i]) for i in idxs_with_tool]
            prompts_len_before = [len(prompts[i]) for i in idxs_with_tool]

            # Call the tools, and build the new prompt for generation
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
                async_tool_dict = self._async_tool_dicts[idx_with_tool]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_call_count += 1
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            if name in sync_tool_dict:
                                tool_call_results.append((name, sync_tool_dict[name](**function["arguments"])))
                            elif name in async_tool_dict:
                                async_coros.append((name, async_tool_dict[name](**function["arguments"])))
                            else:
                                raise ValueError(f"Tool {name} not found.")
                        except Exception as e:
                            tool_failure_count += 1
                            result = {"error": str(e)}
                            tool_call_results.append((name, result))
                    else:
                        tool_failure_count += 1
                        name = tool_call.get("name", "unknown")
                        tool_call_results.append((name, {"error": f"Unsupported tool call type: {tool_call['type']}"}))

                if async_coros:

                    async def _run_async_tools(async_coros):
                        coros = [coro for _, coro in async_coros]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [(name, result) for (name, _), result in zip(async_coros, results, strict=False)]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            tool_call_results.append((name, {"error": str(result)}))
                        else:
                            tool_call_results.append((name, result))

                for name, result in tool_call_results:
                    content = result if isinstance(result, list) else str(result)
                    tool_message = {"role": "tool", "name": name, "content": content}
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

            # Build token IDs by concatenation: prompt + completion + tool_suffix.
            prompt_completion_tool_ids = []
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                # Extract trailing tool messages from completions
                tool_messages = []
                for message in reversed(completions[idx_with_tool]):
                    if message["role"] == "tool":
                        tool_messages.insert(0, message)
                    else:
                        break
                suffix_ids = self._get_tool_suffix_ids(tool_messages)
                prompt_completion_tool_ids.append(
                    prompt_ids[idx_with_tool] + completion_ids[idx_with_tool] + suffix_ids
                )

            # Drop tool results whose addition would push the sequence past max_completion_length (the completion
            # budget) or past the backend context ceiling (vLLM and transformers will error out on inputs longer than
            # the model's max length). The sample exits the loop with its completion as-is, and the tool
            # messages/images appended this iteration are rolled back so completions and tool_images stay consistent
            # with completion_ids.
            if self.config.rollout.engine == "vllm" and self.config.rollout.vllm.mode == "colocate":
                max_model_len = self.rollout_engine.engine.llm.llm_engine.model_config.max_model_len
            else:
                config = self.policy.config
                max_model_len = config.max_position_embeddings
            overlong = [
                len(pct) - len(prompt_ids[i]) > self.max_completion_length or len(pct) >= max_model_len
                for i, pct in zip(idxs_with_tool, prompt_completion_tool_ids, strict=True)
            ]
            for idx in range(len(idxs_with_tool)):
                if overlong[idx]:
                    idx_with_tool = idxs_with_tool[idx]
                    del completions[idx_with_tool][completions_len_before[idx] :]
                    del prompts[idx_with_tool][prompts_len_before[idx] :]
            # Keep only non-overlong items for further processing
            idxs_with_tool = [idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o]
            prompt_completion_tool_ids = [
                pct for pct, o in zip(prompt_completion_tool_ids, overlong, strict=True) if not o
            ]
            if not idxs_with_tool:
                break  # all overlong, exit tool loop

            # Generate new completions after tool execution (using concatenated IDs, no re-tokenization)
            post_tool_ids, post_tool_logprobs = self._generate_single_turn(prompt_completion_tool_ids)

            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length.
            # The pre-regen check guarantees len(completion_tool_ids) <= max_completion_length, so any
            # excess can only come from post_tool_ids. post_tool_ids is model-generated text and never
            # contains image tokens, so a plain slice is safe.
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                completion_tool_length = len(prompt_completion_tool_ids[idx]) - len(prompt_ids[idx_with_tool])
                excess_length = completion_tool_length + len(post_tool_ids[idx]) - self.max_completion_length
                if excess_length > 0:
                    new_len = len(post_tool_ids[idx]) - excess_length
                    post_tool_ids[idx] = post_tool_ids[idx][:new_len]
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][:new_len]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]

            # Update completion_ids with the new completions (after tool execution)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            # Decode post-tool completions.
            post_tool_completions = [parse_response(self._tokenizer, ids) if ids else {} for ids in post_tool_ids]

            # Add post-tool completions to the existing completions
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:  # {} if post-tool completions completely truncated
                    completions[idx_with_tool].append(post_tool_completions[idx])

            # Check for further tool calls
            tool_calls = [completion.get("tool_calls") for completion in post_tool_completions]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            iteration_num += 1

        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count

    @staticmethod
    def _coerce_prompt_messages(prompt: Any) -> list[dict[str, Any]]:
        """Return a mutable chat-message list, preserving structured prompts when available."""

        if isinstance(prompt, list):
            messages: list[dict[str, Any]] = []
            for message in prompt:
                if isinstance(message, dict):
                    current = copy.deepcopy(message)
                    current.setdefault("content", "")
                    messages.append(current)
                else:
                    messages.append({"role": "user", "content": str(message)})
            return messages
        if isinstance(prompt, dict):
            current = copy.deepcopy(prompt)
            current.setdefault("content", "")
            return [current]
        return [{"role": "user", "content": "" if prompt is None else str(prompt)}]

    def _prompt_messages_for_tool_loop(self, rollout_batch: dict[str, Any], completion_count: int) -> list[list[dict[str, Any]]]:
        """Reconstruct per-completion prompts for tool calling without losing chat structure."""

        metadata = rollout_batch.get("metadata") or []
        prompt_texts = rollout_batch.get("prompts") or [""] * completion_count
        sample_ids = rollout_batch.get("sample_ids")
        prompts: list[list[dict[str, Any]]] = []

        for index in range(completion_count):
            prompt_index = int(sample_ids[index].item()) if isinstance(sample_ids, torch.Tensor) else index
            prompt_source: Any = None

            # Prefer the original structured prompt captured by GRPODataModule. This mirrors TRL's
            # `prompts = [x["prompt"] for x in inputs]` behavior and preserves system messages,
            # multi-turn history, and structured chat content during post-tool regeneration.
            metadata_candidates = []
            if index < len(metadata):
                metadata_candidates.append(metadata[index])
            if prompt_index < len(metadata):
                metadata_candidates.append(metadata[prompt_index])
            for item in metadata_candidates:
                if isinstance(item, dict) and item.get("prompt_messages") is not None:
                    prompt_source = item["prompt_messages"]
                    break

            if prompt_source is None:
                prompt_source = prompt_texts[prompt_index] if prompt_index < len(prompt_texts) else ""
            prompts.append(self._coerce_prompt_messages(prompt_source))

        return prompts

    def _run_tool_calling(self, rollout_batch: dict[str, Any]) -> dict[str, Any]:
        """Execute model-requested tools and rebuild rollout tensors for GRPO loss."""

        if self.tool_executor is None:
            return rollout_batch

        prompt_ids_tensor = rollout_batch["prompt_ids"]
        prompt_mask_tensor = rollout_batch["prompt_mask"]
        prompt_ids = [ids[mask.bool()].detach().cpu().tolist() for ids, mask in zip(prompt_ids_tensor, prompt_mask_tensor, strict=True)]
        completion_ids = [list(ids) for ids in rollout_batch["completion_id_lists"]]
        completions = [[parse_response(self.tokenizer, ids)] for ids in completion_ids]
        prompts = self._prompt_messages_for_tool_loop(rollout_batch, len(completion_ids))

        self._sync_tool_dicts = [self._sync_tools for _ in completion_ids]
        self._async_tool_dicts = [self._async_tools for _ in completion_ids]
        prompt_budget = self.config.rollout.max_total_length - prompt_ids_tensor.size(1)
        self.max_completion_length = max(1, min(self.config.rollout.max_completion_length, prompt_budget))

        tool_mask, completions, completion_ids, _, tool_call_count, tool_failure_count = self._tool_call_loop(
            prompts,
            prompt_ids,
            completion_ids,
            completions,
            logprobs=None,
        )

        device = prompt_ids_tensor.device
        completion_ids_tensor = pad_sequences(completion_ids, self.tokenizer.pad_token_id, device)
        if completion_ids_tensor.size(1) > self.max_completion_length:
            completion_ids_tensor = completion_ids_tensor[:, : self.max_completion_length]
            completion_ids = [row[: self.max_completion_length] for row in completion_ids]
            tool_mask = [row[: self.max_completion_length] for row in tool_mask]
        completion_mask = (completion_ids_tensor != self.tokenizer.pad_token_id).long()
        tool_mask_tensor = pad_sequences(tool_mask, 0, device)
        if tool_mask_tensor.size(1) > completion_ids_tensor.size(1):
            tool_mask_tensor = tool_mask_tensor[:, : completion_ids_tensor.size(1)]
        elif tool_mask_tensor.size(1) < completion_ids_tensor.size(1):
            pad_width = completion_ids_tensor.size(1) - tool_mask_tensor.size(1)
            tool_mask_tensor = torch.nn.functional.pad(tool_mask_tensor, (0, pad_width), value=0)
        completion_id_lists = [ids[mask.bool()].detach().cpu().tolist() for ids, mask in zip(completion_ids_tensor, completion_mask, strict=True)]

        with torch.no_grad():
            old_per_token_logps = compute_per_token_logps(
                self,
                prompt_ids_tensor,
                prompt_mask_tensor,
                completion_ids_tensor,
                completion_mask,
                self.config.rollout.temperature,
            )

        completion_truncated = torch.tensor(
            [len(ids) >= self.max_completion_length and (not ids or ids[-1] != self.tokenizer.eos_token_id) for ids in completion_id_lists],
            device=device,
            dtype=torch.bool,
        )
        text_completions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_id_lists]
        updated = dict(rollout_batch)
        updated.update(
            {
                "completion_ids": completion_ids_tensor,
                "completion_mask": completion_mask,
                "old_per_token_logps": old_per_token_logps.detach(),
                "completion_truncated": completion_truncated,
                "completion_id_lists": completion_id_lists,
                "completions": completions,
                "completion_texts": text_completions,
                "tool_mask": tool_mask_tensor.to(completion_mask.dtype),
                "tool_call_count": tool_call_count,
                "tool_failure_count": tool_failure_count,
            }
        )
        return updated
