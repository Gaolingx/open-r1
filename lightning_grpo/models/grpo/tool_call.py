"""Tool calling integration for GRPO training.

This module provides a mixin class that adds multi-turn tool calling support
to the GRPOLightningModule. It handles ToolCallExecutor initialization,
the tool calling loop during rollouts, and policy-based generation for
tool calling when vLLM chat generation is not available.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.models.common import compute_per_token_logps
from lightning_grpo.utils.tool_calling.executor import ToolCallExecutor
from lightning_grpo.utils.tool_calling.parser import load_tools_from_names


# ---------------------------------------------------------------------------
# Tensor utility helpers
# ---------------------------------------------------------------------------


def pad_sequences(
    sequences: list[list[int]] | list[torch.Tensor],
    pad_value: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Right-pad a list of variable-length sequences into a 2-D tensor."""

    tensors = [torch.tensor(s, device=device) if not isinstance(s, torch.Tensor) else s.to(device) for s in sequences]
    max_len = max(t.size(0) for t in tensors)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype, device=device)
    for i, t in enumerate(tensors):
        padded[i, : t.size(0)] = t
    return padded


def pad_float_sequences(
    sequences: list[list[float]],
    target_length: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Pad float sequences to *target_length* with zeros."""

    padded = torch.zeros(len(sequences), target_length, device=device)
    for i, seq in enumerate(sequences):
        length = min(len(seq), target_length)
        padded[i, :length] = torch.tensor(seq[:length], device=device)
    return padded


def truncate_completions(
    completion_ids: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int | list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
    """Truncate completions at the first EOS token and build masks.

    Returns:
        completion_ids: Truncated tensor (unchanged shape, trailing set to pad).
        completion_mask: Binary mask of valid (non-pad) tokens.
        completion_truncated: Boolean tensor indicating if each sample was truncated.
        completion_id_lists: Python lists of token ids (without padding).
    """

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    batch_size, seq_len = completion_ids.shape
    completion_truncated = torch.zeros(batch_size, dtype=torch.bool, device=completion_ids.device)
    completion_id_lists: list[list[int]] = []

    for i in range(batch_size):
        row = completion_ids[i]
        eos_positions = []
        if eos_token_id:
            for eos_id in eos_token_id:
                positions = (row == eos_id).nonzero(as_tuple=True)[0]
                if positions.numel() > 0:
                    eos_positions.append(positions[0].item())
        if eos_positions:
            first_eos = min(eos_positions)
            # Include the EOS token itself
            end_pos = first_eos + 1
            completion_ids[i, end_pos:] = pad_token_id
            completion_truncated[i] = True
        # Build id list without padding
        valid_mask = row != pad_token_id
        completion_id_lists.append(row[valid_mask].tolist())

    completion_mask = (completion_ids != pad_token_id).long()
    return completion_ids, completion_mask, completion_truncated, completion_id_lists


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

    tool_executor: ToolCallExecutor | None

    def _init_tool_executor(self) -> None:
        """Initialize the ToolCallExecutor if tool calling is configured."""

        self.tool_executor = None
        if not self.config.rollout.tool_calling.enabled:
            return

        tools = load_tools_from_names(self.config.rollout.tool_calling.tools)
        self.tool_executor = ToolCallExecutor(
            tools=tools,
            tokenizer=self.tokenizer,
            max_iterations=self.config.rollout.tool_calling.max_iterations,
            max_completion_length=self.config.rollout.max_completion_length,
            chat_template=self.config.rollout.tool_calling.chat_template,
            chat_template_kwargs=self.config.rollout.tool_calling.chat_template_kwargs,
        )
        rank_zero_info(
            "Tool calling enabled with %d tools, max %d iterations",
            len(tools),
            self.config.rollout.tool_calling.max_iterations,
        )

    def _run_tool_calling(self, rollout_batch: dict[str, Any]) -> dict[str, Any]:
        """Execute multi-turn tool calling loop on rollout results.

        Detects tool calls in completions, executes them, regenerates, and
        produces a tool_mask that excludes tool-injected tokens from the loss.
        """
        completions = rollout_batch.get("completions", [])
        if not completions:
            return rollout_batch

        # Check if any completion has tool calls, including text-formatted tool calls.
        has_tool_calls = any(
            isinstance(c, list) and c and isinstance(c[-1], dict) and self.tool_executor.extract_tool_calls(c[-1])
            for c in completions
        )
        if not has_tool_calls:
            return rollout_batch

        if not hasattr(self.rollout_engine, "generate_chat") and self.config.rollout.engine != "torch":
            raise RuntimeError(
                "Tool calling requires a rollout engine with generate_chat support or the local policy rollout engine."
            )

        # Build generate function for regeneration after tool execution
        def _generate_fn(conversations):
            """Regenerate completions for conversations with tool results."""
            engine = self.rollout_engine
            # Use vLLM chat generation if available
            if hasattr(engine, "generate_chat"):
                _, new_ids, new_logprobs = engine.generate_chat(
                    conversations,
                    num_generations=1,
                    tools=self.tool_executor.tool_schemas,
                    chat_template=self.config.rollout.tool_calling.chat_template,
                    chat_template_kwargs=self.config.rollout.tool_calling.chat_template_kwargs,
                )
                return new_ids, new_logprobs
            return self._generate_tool_calling_with_policy(conversations)

        # Run the tool loop
        prompt_ids_list = rollout_batch.get("prompt_id_lists", [
            ids[mask.bool()].tolist()
            for ids, mask in zip(rollout_batch["prompt_ids"], rollout_batch["prompt_mask"])
        ])
        completion_ids_list = rollout_batch.get("completion_id_lists", [
            ids.tolist() for ids in rollout_batch["completion_ids"]
        ])
        logprobs_list = rollout_batch.get("old_per_token_logps_list", None)

        # Get prompts as conversations
        prompts = rollout_batch.get("prompts_structured", rollout_batch.get("prompts", []))

        tool_result = self.tool_executor.run_tool_loop(
            prompts=prompts if isinstance(prompts[0], list) else [[{"role": "user", "content": p}] for p in prompts],
            prompt_ids=prompt_ids_list,
            completion_ids=completion_ids_list,
            completions=completions,
            logprobs=logprobs_list,
            generate_fn=_generate_fn,
        )

        # Update rollout_batch with tool calling results
        device = rollout_batch["completion_ids"].device
        completion_id_lists = tool_result["completion_ids"]
        completion_ids = pad_sequences(completion_id_lists, self.tokenizer.pad_token_id, device)
        completion_ids, completion_mask, completion_truncated, completion_id_lists = truncate_completions(
            completion_ids,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )
        rollout_batch["completion_ids"] = completion_ids
        rollout_batch["completion_mask"] = completion_mask
        rollout_batch["completion_truncated"] = completion_truncated
        rollout_batch["completion_id_lists"] = completion_id_lists
        rollout_batch["completions"] = tool_result["completions"]
        rollout_batch["completions_text"] = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_id_lists
        ]

        logprobs = tool_result.get("logprobs")
        if logprobs is not None:
            rollout_batch["old_per_token_logps"] = pad_float_sequences(logprobs, completion_ids.shape[1], device)
        else:
            model_input_ids = torch.cat([rollout_batch["prompt_ids"], completion_ids], dim=1)
            model_attention_mask = torch.cat([rollout_batch["prompt_mask"], completion_mask], dim=1)
            rollout_batch["old_per_token_logps"] = compute_per_token_logps(
                self.policy,
                model_input_ids,
                completion_ids.shape[1],
                attention_mask=model_attention_mask,
                temperature=self.config.rollout.temperature,
            ).detach()

        if tool_result["tool_mask"]:
            tool_masks = [mask[:len(ids)] for mask, ids in zip(tool_result["tool_mask"], completion_id_lists, strict=True)]
            rollout_batch["tool_mask"] = pad_sequences(tool_masks, 0, device)

        # Log tool calling metrics
        if self.trainer.is_global_zero:
            self.log("train/tool_call_count", float(tool_result["tool_call_count"]), on_step=True)
            self.log("train/tool_failure_count", float(tool_result["tool_failure_count"]), on_step=True)

        return rollout_batch

    def _generate_tool_calling_with_policy(
        self,
        conversations: list[list[dict[str, Any]]],
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Generate one assistant turn for tool calling with the local policy engine."""
        if not conversations:
            return [], []

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            prompt_id_lists = [
                self.tokenizer.apply_chat_template(
                    conversation,
                    tools=self.tool_executor.tool_schemas,
                    chat_template=self.config.rollout.tool_calling.chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **self.config.rollout.tool_calling.chat_template_kwargs,
                )
                for conversation in conversations
            ]
            prompt_ids = pad_sequences(prompt_id_lists, self.tokenizer.pad_token_id, self.device)
            attention_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
            model = self.policy.module if hasattr(self.policy, "module") else self.policy
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                generation_config=self.policy.generation_config,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        prompt_width = prompt_ids.shape[1]
        completion_ids = [row[prompt_width:].tolist() for row in generated]
        padded_completion_ids = pad_sequences(completion_ids, self.tokenizer.pad_token_id, self.device)
        _, completion_mask, _, completion_id_lists = truncate_completions(
            padded_completion_ids,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )
        model_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)
        model_attention_mask = torch.cat([attention_mask, completion_mask], dim=1)
        logprobs = compute_per_token_logps(
            self.policy,
            model_input_ids,
            padded_completion_ids.shape[1],
            attention_mask=model_attention_mask,
            temperature=self.config.rollout.temperature,
        )
        return completion_id_lists, [row[:len(ids)].detach().cpu().tolist() for row, ids in zip(logprobs, completion_id_lists, strict=True)]
