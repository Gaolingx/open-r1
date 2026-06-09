"""Tool calling integration for GRPO training.

This module provides a mixin class that adds multi-turn tool calling support
to the GRPOLightningModule. It handles ToolCallExecutor initialization,
the tool calling loop during rollouts, and policy-based generation for
tool calling when vLLM chat generation is not available.
"""

from __future__ import annotations

from typing import Any

import asyncio
import importlib
import inspect
import threading

import torch
from transformers import PreTrainedTokenizerBase

from lightning_grpo.models.common import compute_per_token_logps
from lightning_grpo.models.grpo.rollout_module.utils import pad_sequences


# ---------------------------------------------------------------------------
# Tensor utility helpers
# ---------------------------------------------------------------------------


def _validate_tool_calls(tool_calls: list | None) -> None:
    """
    Validate tool_calls to ensure all required fields exist with valid values.

    Raises ValueError when the model generates malformed tool calls (e.g., missing 'arguments' field) that are
    partially parsed.

    Args:
        tool_calls: List of tool call dictionaries, or None.
    """
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list or None.")

    for idx, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            raise ValueError(f"tool_calls[{idx}] must be a dict.")

        # Handle nested function structure: {"type": "function", "function": {"name": ..., "arguments": ...}}
        if "function" in tool_call:
            func = tool_call["function"]
            if not isinstance(func, dict):
                raise ValueError(f"tool_calls[{idx}]['function'] must be a dict.")
            if not isinstance(func.get("name"), str):
                raise ValueError(f"tool_calls[{idx}]['function']['name'] must be a string.")
            # Some templates (e.g. Qwen3.5) omit arguments for valid no-arg calls; normalize to {}.
            if "arguments" not in func or func["arguments"] is None:
                func["arguments"] = {}
        else:
            # Handle flat structure: {"name": ..., "arguments": ...}
            if not isinstance(tool_call.get("name"), str):
                raise ValueError(f"tool_calls[{idx}]['name'] must be a string.")
            # Some templates (e.g. Qwen3.5) omit arguments for valid no-arg calls; normalize to {}.
            if "arguments" not in tool_call or tool_call["arguments"] is None:
                tool_call["arguments"] = {}


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Normalize flat or nested tool-call payloads to OpenAI-style function calls."""

    if "function" in tool_call:
        normalized = dict(tool_call)
        normalized.setdefault("type", "function")
        function = dict(normalized["function"])
        function.setdefault("arguments", {})
        normalized["function"] = function
        return normalized

    return {
        "type": tool_call.get("type", "function"),
        "function": {
            "name": tool_call["name"],
            "arguments": tool_call.get("arguments") or {},
        },
    }


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


def parse_response(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> dict:
    r"""
    Parse a token sequence into structured response dictionaries with fallback handling.

    Attempts to parse the sequence using `tokenizer.parse_response()`. If parsing fails (e.g., due to malformed tool
    calls like `<tool_call>{"type":"function"</tool_call>`), falls back to decoding as plain text.

    Also removes incorrectly appended EOS tokens from tool call content when present, and validates tool_calls to
    ensure all required fields exist.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer with a `parse_response()` method.
        ids (`list[int]`):
            List of token sequences.

    Returns:
        `dict`:
            Response dictionary.

    Example:
    ```python
    >>> from trl.chat_template_utils import parse_response, add_response_schema
    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    >>> tokenizer = add_response_schema(tokenizer)  # temporary until built-in support
    >>> text = '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
    >>> ids = tokenizer(text)["input_ids"]
    >>> parse_response(tokenizer, ids)
    {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {'name': 'multiply', 'arguments': {'a': 3, 'b': 4}}}]}
    ```
    """
    try:
        parsed = tokenizer.parse_response(ids)
        if parsed is None:  # this can happen if the response is heavily truncated and even the content is lost
            raise ValueError("parse_response returned None")
        # Hotfix: remove incorrectly appended EOS token from tool calls
        # See https://github.com/huggingface/transformers/issues/42249
        if isinstance(parsed.get("content"), str) and tokenizer.eos_token:
            parsed["content"] = parsed["content"].removesuffix(tokenizer.eos_token)
        # Normalize: ensure content is always a string (some models omit it or set it to None)
        if not parsed.get("content"):
            parsed["content"] = ""
        # Validate tool_calls to prevent Jinja2 Undefined errors when fields are missing
        if "tool_calls" in parsed:
            _validate_tool_calls(parsed["tool_calls"])
            parsed["tool_calls"] = [_normalize_tool_call(tool_call) for tool_call in parsed["tool_calls"]]
    except (AttributeError, ValueError, TypeError):
        # Fallback: decode as plain text if parsing fails. This happens if the model outputs malformed tool calls.
        content = tokenizer.decode(ids, skip_special_tokens=True)
        parsed = {"role": "assistant", "content": content}
    return parsed


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
        self.chat_template = tool_config.chat_template or getattr(self.tokenizer, "chat_template", None)
        self.chat_template_kwargs = tool_config.chat_template_kwargs or {}
        self._sync_tools: dict[str, Any] = {}
        self._async_tools: dict[str, Any] = {}
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None

        if not tool_config.enabled:
            return

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

    def _run_tool_calling(self, rollout_batch: dict[str, Any]) -> dict[str, Any]:
        """Execute model-requested tools and rebuild rollout tensors for GRPO loss."""

        if self.tool_executor is None:
            return rollout_batch

        prompt_ids_tensor = rollout_batch["prompt_ids"]
        prompt_mask_tensor = rollout_batch["prompt_mask"]
        prompt_ids = [ids[mask.bool()].detach().cpu().tolist() for ids, mask in zip(prompt_ids_tensor, prompt_mask_tensor, strict=True)]
        completion_ids = [list(ids) for ids in rollout_batch["completion_id_lists"]]
        completions = [[parse_response(self.tokenizer, ids)] for ids in completion_ids]

        prompt_texts = rollout_batch.get("prompts") or [""] * len(completion_ids)
        sample_ids = rollout_batch.get("sample_ids")
        prompts: list[list[dict[str, Any]]] = []
        for index in range(len(completion_ids)):
            prompt_index = int(sample_ids[index].item()) if isinstance(sample_ids, torch.Tensor) else index
            prompt_text = prompt_texts[prompt_index] if prompt_index < len(prompt_texts) else ""
            prompts.append([{"role": "user", "content": prompt_text}])

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
