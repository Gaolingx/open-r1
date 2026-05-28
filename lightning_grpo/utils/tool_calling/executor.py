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

import logging
from collections.abc import Callable
from typing import Any, Optional

from transformers import PreTrainedTokenizerBase

from lightning_grpo.utils.tool_calling.parser import ToolCallRunner, load_tools_from_names

logger = logging.getLogger(__name__)


class ToolCallExecutor(ToolCallRunner):
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
        super().__init__(tools)
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.max_completion_length = max_completion_length
        self.chat_template = chat_template
        self.chat_template_kwargs = chat_template_kwargs or {}

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
            calls = self.extract_tool_calls(last_msg) if isinstance(last_msg, dict) else []
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
                    new_msg = self.parse_assistant_message(new_text)
                    completions[idx].append(new_msg)

            # Detect new tool calls
            tool_calls_per_sample = []
            for completion in completions:
                last_msg = completion[-1] if completion else {}
                calls = self.extract_tool_calls(last_msg) if isinstance(last_msg, dict) else []
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
