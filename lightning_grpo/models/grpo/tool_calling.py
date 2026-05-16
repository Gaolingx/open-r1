"""Multi-turn tool calling support for GRPO rollout.

This module implements a tool execution loop that allows the model to invoke
tools during generation, receive results, and continue generating. It follows
the same pattern as TRL's _tool_call_loop but adapted for the Lightning GRPO
architecture with support for both sync and async tool functions.

The tool_mask mechanism ensures that tokens from tool results (injected by the
environment) are excluded from the policy gradient loss, since the model did
not generate those tokens.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from collections.abc import Callable
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _start_async_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start an event loop on a daemon thread for async tool execution."""
    loop = asyncio.new_event_loop()

    def _run(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run, args=(loop,), daemon=True)
    thread.start()
    return loop, thread


def _shutdown_async_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    """Gracefully shut down the async event loop."""
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


def load_tools_from_names(tool_names: list[str]) -> list[Callable]:
    """Load tool callables from dotted module paths or known tool registry.

    Each tool_name can be:
    - A dotted path like 'my_module.my_tool' which will be imported
    - A name from a built-in tool registry (extensible)
    """
    tools: list[Callable] = []
    for name in tool_names:
        if "." in name:
            # Import from dotted path
            module_path, func_name = name.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            if not callable(func):
                raise ValueError(f"Tool '{name}' is not callable.")
            tools.append(func)
        else:
            raise ValueError(
                f"Tool '{name}' not found. Use a dotted import path like 'module.function'."
            )
    return tools


class ToolCallExecutor:
    """Manages tool execution during multi-turn rollout generation.

    Supports both synchronous and asynchronous tool functions. Async tools are
    executed concurrently via a daemon thread event loop for efficiency.
    """

    def __init__(
        self,
        tools: list[Callable],
        tokenizer: PreTrainedTokenizerBase,
        max_iterations: int = 5,
        max_completion_length: int = 8192,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.max_completion_length = max_completion_length
        self.chat_template = chat_template
        self.chat_template_kwargs = chat_template_kwargs or {}

        # Separate sync and async tools
        self._sync_tools: dict[str, Callable] = {}
        self._async_tools: dict[str, Callable] = {}
        for tool in tools:
            name = tool.__name__
            if asyncio.iscoroutinefunction(tool):
                self._async_tools[name] = tool
            else:
                self._sync_tools[name] = tool

        # Start async loop if we have async tools
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None
        if self._async_tools:
            self._async_loop, self._async_thread = _start_async_loop()

        # Build tool schemas for chat template
        self.tool_schemas = self._build_tool_schemas(tools)

    @staticmethod
    def _build_tool_schemas(tools: list[Callable]) -> list[dict[str, Any]]:
        """Build JSON tool schemas from function signatures and docstrings."""
        schemas = []
        for tool in tools:
            sig = inspect.signature(tool)
            params = {}
            for param_name, param in sig.parameters.items():
                param_info: dict[str, Any] = {"type": "string"}
                if param.annotation != inspect.Parameter.empty:
                    type_map = {
                        str: "string",
                        int: "integer",
                        float: "number",
                        bool: "boolean",
                        list: "array",
                        dict: "object",
                    }
                    param_info["type"] = type_map.get(param.annotation, "string")
                params[param_name] = param_info

            schema = {
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": (tool.__doc__ or "").strip().split("\n")[0],
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": list(params.keys()),
                    },
                },
            }
            schemas.append(schema)
        return schemas

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> tuple[str, str]:
        """Execute a single tool call and return (name, result_str)."""
        if tool_call.get("type") != "function":
            name = tool_call.get("function", {}).get("name", "unknown")
            return name, json.dumps({"error": f"Unsupported tool call type: {tool_call.get('type')}"})

        function = tool_call["function"]
        name = function["name"]
        arguments = function.get("arguments", {})

        # Parse arguments if they're a JSON string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return name, json.dumps({"error": f"Invalid JSON arguments: {arguments}"})

        try:
            if name in self._sync_tools:
                result = self._sync_tools[name](**arguments)
            elif name in self._async_tools:
                if self._async_loop is None:
                    return name, json.dumps({"error": "Async loop not initialized"})
                future = asyncio.run_coroutine_threadsafe(
                    self._async_tools[name](**arguments), self._async_loop
                )
                result = future.result(timeout=60.0)
            else:
                return name, json.dumps({"error": f"Tool '{name}' not found"})
        except Exception as e:
            logger.warning("Tool '%s' execution failed: %s", name, e)
            return name, json.dumps({"error": str(e)})

        return name, str(result)

    def execute_tool_calls_batch(
        self, tool_calls_batch: list[list[dict[str, Any]]]
    ) -> list[list[dict[str, str]]]:
        """Execute tool calls for a batch of completions.

        Args:
            tool_calls_batch: List of tool call lists, one per completion.

        Returns:
            List of tool message lists, one per completion.
        """
        results: list[list[dict[str, str]]] = []
        for tool_calls in tool_calls_batch:
            messages = []
            # Collect async calls for concurrent execution
            async_coros = []
            sync_results = []

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name", "unknown")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        sync_results.append((name, json.dumps({"error": "Invalid JSON"})))
                        continue

                if name in self._async_tools and self._async_loop is not None:
                    async_coros.append((name, self._async_tools[name](**arguments)))
                elif name in self._sync_tools:
                    try:
                        result = self._sync_tools[name](**arguments)
                        sync_results.append((name, str(result)))
                    except Exception as e:
                        sync_results.append((name, json.dumps({"error": str(e)})))
                else:
                    sync_results.append((name, json.dumps({"error": f"Tool '{name}' not found"})))

            # Execute async tools concurrently
            if async_coros and self._async_loop is not None:
                async def _gather(coros):
                    tasks = [coro for _, coro in coros]
                    return await asyncio.gather(*tasks, return_exceptions=True)

                future = asyncio.run_coroutine_threadsafe(_gather(async_coros), self._async_loop)
                async_results = future.result(timeout=120.0)
                for (name, _), result in zip(async_coros, async_results):
                    if isinstance(result, Exception):
                        sync_results.append((name, json.dumps({"error": str(result)})))
                    else:
                        sync_results.append((name, str(result)))

            for name, content in sync_results:
                messages.append({"role": "tool", "name": name, "content": content})
            results.append(messages)

        return results

    def run_tool_loop(
        self,
        *,
        prompts: list[list[dict[str, str]]],
        prompt_ids: list[list[int]],
        completion_ids: list[list[int]],
        completions: list[list[dict[str, Any]]],
        logprobs: Optional[list[list[float]]],
        generate_fn: Callable,
    ) -> dict[str, Any]:
        """Run the multi-turn tool calling loop.

        This iteratively:
        1. Detects tool calls in completions
        2. Executes the tools
        3. Appends tool results to the conversation
        4. Regenerates completions
        5. Repeats until no more tool calls or max_iterations reached

        Args:
            prompts: Conversational prompts (list of message dicts per sample).
            prompt_ids: Token IDs for each prompt.
            completion_ids: Token IDs for each completion.
            completions: Parsed completion messages.
            logprobs: Per-token log-probs for completions.
            generate_fn: Callable to regenerate after tool execution.
                Signature: generate_fn(conversations) -> (new_completion_ids, new_logprobs)

        Returns:
            Dict with updated completion_ids, logprobs, completions, and tool_mask.
        """
        # Initialize tool_mask: 1 for model-generated tokens, 0 for tool-injected tokens
        tool_mask = [[1] * len(ids) for ids in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0

        # Detect initial tool calls
        tool_calls_per_sample = []
        for completion in completions:
            last_msg = completion[-1] if completion else {}
            calls = last_msg.get("tool_calls", [])
            tool_calls_per_sample.append(calls)

        idxs_with_tool = [i for i, calls in enumerate(tool_calls_per_sample) if calls]
        iteration = 0

        while idxs_with_tool and iteration < self.max_iterations:
            # Build conversations with tool results
            active_tool_calls = [tool_calls_per_sample[i] for i in idxs_with_tool]

            # Execute tools
            tool_results = self.execute_tool_calls_batch(active_tool_calls)
            tool_call_count += sum(len(calls) for calls in active_tool_calls)

            # Build updated conversations for regeneration
            conversations_for_regen = []
            for idx_pos, idx in enumerate(idxs_with_tool):
                # Append assistant message with tool calls
                conv = list(prompts[idx])
                # Add all completion messages so far
                for msg in completions[idx]:
                    conv.append(msg)
                # Add tool results
                for tool_msg in tool_results[idx_pos]:
                    conv.append(tool_msg)
                    completions[idx].append(tool_msg)
                conversations_for_regen.append(conv)

            # Check length constraints before regenerating
            tokenized_convs = []
            for conv in conversations_for_regen:
                try:
                    tokens = self.tokenizer.apply_chat_template(
                        conv,
                        tools=self.tool_schemas if self.tool_schemas else None,
                        chat_template=self.chat_template,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=False,
                        **self.chat_template_kwargs,
                    )
                    tokenized_convs.append(tokens)
                except Exception:
                    tokenized_convs.append(None)

            # Filter out overlong conversations
            valid_idxs = []
            valid_convs = []
            for pos, (idx, tokens) in enumerate(zip(idxs_with_tool, tokenized_convs)):
                if tokens is None:
                    continue
                prompt_len = len(prompt_ids[idx])
                current_completion_len = len(completion_ids[idx])
                # Check if adding tool results + new generation would exceed max
                tool_token_len = len(tokens) - prompt_len - current_completion_len
                remaining = self.max_completion_length - current_completion_len - tool_token_len
                if remaining > 0:
                    valid_idxs.append(idx)
                    valid_convs.append(conversations_for_regen[pos])

            if not valid_convs:
                break

            # Regenerate completions after tool execution
            new_completion_ids, new_logprobs = generate_fn(valid_convs)

            # Update completion_ids, logprobs, and tool_mask
            for pos, idx in enumerate(valid_idxs):
                prompt_len = len(prompt_ids[idx])
                # Tokenize the full conversation up to the new generation
                conv_tokens = tokenized_convs[idxs_with_tool.index(idx)]
                if conv_tokens is None:
                    continue

                # Tool result tokens (between old completion end and new generation start)
                tool_token_count = len(conv_tokens) - prompt_len - len(completion_ids[idx])

                # Extend completion_ids with tool tokens + new completion
                tool_tokens = conv_tokens[prompt_len + len(completion_ids[idx]):]
                new_comp = new_completion_ids[pos] if pos < len(new_completion_ids) else []

                # Truncate to max_completion_length
                total_new = tool_tokens + new_comp
                max_new = self.max_completion_length - len(completion_ids[idx])
                total_new = total_new[:max_new]

                completion_ids[idx] = completion_ids[idx] + total_new
                tool_mask[idx] += [0] * min(tool_token_count, len(total_new))
                remaining_new = len(total_new) - tool_token_count
                if remaining_new > 0:
                    tool_mask[idx] += [1] * remaining_new

                if logprobs is not None and new_logprobs is not None:
                    new_lp = new_logprobs[pos] if pos < len(new_logprobs) else []
                    logprobs[idx] += [0.0] * tool_token_count + new_lp[:remaining_new]

                # Parse new completion for further tool calls
                if new_comp:
                    new_text = self.tokenizer.decode(new_comp, skip_special_tokens=True)
                    # Try to parse as structured message
                    new_msg = {"role": "assistant", "content": new_text}
                    # Check for tool calls in the new completion
                    # This is model-specific; we rely on the tokenizer's chat template parsing
                    completions[idx].append(new_msg)

            # Detect new tool calls
            tool_calls_per_sample = []
            for completion in completions:
                last_msg = completion[-1] if completion else {}
                calls = last_msg.get("tool_calls", [])
                tool_calls_per_sample.append(calls)

            idxs_with_tool = [i for i in valid_idxs if tool_calls_per_sample[i]]
            iteration += 1

        logger.debug(
            "Tool loop completed: %d iterations, %d calls, %d failures",
            iteration, tool_call_count, tool_failure_count,
        )

        return {
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "completions": completions,
            "tool_mask": tool_mask,
            "tool_call_count": tool_call_count,
            "tool_failure_count": tool_failure_count,
        }

    def shutdown(self) -> None:
        """Clean up async resources."""
        if self._async_loop is not None and self._async_thread is not None:
            _shutdown_async_loop(self._async_loop, self._async_thread)
            self._async_loop = None
            self._async_thread = None
