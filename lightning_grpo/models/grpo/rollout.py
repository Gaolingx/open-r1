"""Rollout coordination helpers for GRPO training."""

from __future__ import annotations

import json
import random
from typing import Any

import torch

from lightning_grpo.models.common import compute_per_token_logps
from lightning_grpo.models.grpo.rollout_engine import create_rollout_engine
from lightning_grpo.models.grpo.tools import execute_tool, parse_tool_calls
from lightning_grpo.data.base import apply_chat_template

class LocalGenerateRolloutCoordinator:
    """Coordinate GRPO rollouts through the configured inference engine."""

    def __init__(self, module: Any) -> None:
        self.module = module
        rollout_config = module.config.rollout
        sglang_model_path = rollout_config.sglang_model_path or module.config.model.tokenizer_name_or_path or module.config.model.model_name_or_path
        self.rollout_engine = create_rollout_engine(
            engine_type=rollout_config.engine,
            policy_model=module.policy,
            tokenizer=module.tokenizer,
            device=str(module.device),
            sglang_base_url=rollout_config.sglang_base_url,
            sglang_model_path=sglang_model_path,
            sglang_shared_path=rollout_config.sglang_shared_path,
            sglang_timeout=rollout_config.sglang_timeout,
        )

    def update_policy(self) -> None:
        """Refresh the rollout engine's policy weights or model handle."""

        self.rollout_engine.update_policy(self.module.policy)

    def resolve_num_generations(self, training: bool) -> int:
        """Return the number of completions sampled per prompt."""

        return self.module.config.rollout.num_generations if training else 1

    def _max_prompt_length(self) -> int:
        """Return a prompt window that leaves room for at least one completion token."""

        max_total_length = self.module.config.rollout.max_total_length
        if max_total_length <= 1:
            raise ValueError("rollout.max_total_length must be greater than 1 for GRPO loss computation.")
        return min(self.module.config.rollout.max_prompt_length, max_total_length - 1)

    def _tokenize_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        """Left-pad prompts and truncate to the configured prompt window."""

        tokenizer = self.module.tokenizer
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_prompt_length(),
                add_special_tokens=False,
                return_token_type_ids=False,
            )
        finally:
            tokenizer.padding_side = old_padding_side
        return {key: value.to(self.module.device) for key, value in encoded.items()}

    def _generate(self, prompts: list[str], *, num_generations: int = 1) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Generate completions with the configured rollout engine."""

        encoded = self._tokenize_prompts(prompts)
        result = self.rollout_engine.generate(
            prompt_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            num_generations=num_generations,
            max_new_tokens=self.module.config.rollout.max_completion_length,
            temperature=self.module.config.rollout.temperature,
        )
        return result.completion_ids, result.completion_mask, result.completions

    @torch.no_grad()
    def _generate_once(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Generate one completion for each prompt and return padded completion ids."""

        completion_ids, completion_mask, completion_texts = self._generate(prompts, num_generations=1)
        return completion_ids, completion_mask, completion_texts

    def _batch_metadata(self, batch: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """Return per-sample metadata dictionaries from the collated batch."""

        raw_metadata = batch.get("metadata") or [{} for _ in batch.get("prompt_text", [])]
        metadata: list[dict[str, Any]] = []
        for item in raw_metadata:
            if isinstance(item, str):
                try:
                    parsed = json.loads(item)
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = item or {}
            metadata.append(dict(parsed) if isinstance(parsed, dict) else {})
        return metadata

    def _metadata_for_generation(self, base_metadata: dict[str, Any], completion: str, turn_outputs: list[str] | None = None, unfinished: bool = False) -> dict[str, Any]:
        """Preserve dataset metadata and add rollout-specific fields."""

        metadata = dict(base_metadata)
        if "solution" not in metadata:
            for alias in ("answer", "response", "output", "gold_answer", "gold_solution", "gt"):
                if alias in metadata and metadata[alias] is not None:
                    metadata["solution"] = metadata[alias]
                    break
        metadata.setdefault("turn_outputs", turn_outputs or [completion])
        metadata["unfinished"] = unfinished
        return metadata

    def _render_messages(self, messages: list[dict[str, Any]], tools: Any, *, add_generation_prompt: bool, open_thinking: bool) -> str:
        """Render a chat state while tolerating tokenizers without custom kwargs."""

        try:
            return apply_chat_template(
                self.module.tokenizer,
                messages,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                open_thinking=open_thinking,
            )
        except TypeError:
            return apply_chat_template(
                self.module.tokenizer,
                messages,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
            )

    def _rollout_reasoning(self, batch: dict[str, list[Any]], *, num_generations: int) -> dict[str, Any]:
        """Generate single-turn reasoning RL completions."""

        prompts = list(batch["prompt_text"])
        expanded_prompts = [prompt for prompt in prompts for _ in range(num_generations)]
        completion_ids, completion_mask, completions = self._generate(prompts, num_generations=num_generations)
        batch_metadata = self._batch_metadata(batch)
        metadata = []
        for index, completion in enumerate(completions):
            sample_index = index // num_generations
            base_metadata = batch_metadata[sample_index] if sample_index < len(batch_metadata) else {}
            metadata.append(self._metadata_for_generation(base_metadata, completion, [completion], unfinished=False))
        return self._pack_rollout(
            expanded_prompts,
            completion_ids,
            completion_mask,
            completions,
            metadata,
            num_generations=num_generations,
        )

    def _rollout_agentic(self, batch: dict[str, list[Any]], *, num_generations: int) -> dict[str, Any]:
        """Run multi-turn local tool-call rollouts and mask tool observations from loss."""

        tokenizer = self.module.tokenizer
        all_prompts: list[str] = []
        response_ids_batch: list[list[int]] = []
        response_masks_batch: list[list[int]] = []
        completions: list[str] = []
        metadata: list[dict[str, Any]] = []
        batch_metadata = self._batch_metadata(batch)
        batch_messages = batch.get("messages") or [None] * len(batch.get("prompt_text", []))

        for sample_index, messages in enumerate(batch_messages):
            base_metadata = batch_metadata[sample_index] if sample_index < len(batch_metadata) else {}
            tools = base_metadata.get("tools") or (batch.get("tools", [None] * len(batch_messages))[sample_index] if batch.get("tools") else None)
            if messages is None:
                messages = base_metadata.get("messages")
            if messages is None:
                prompt_text = batch.get("prompt_text", [""] * len(batch_messages))[sample_index]
                messages = [{"role": "user", "content": prompt_text}]
            for _ in range(num_generations):
                chat_state = [dict(message) for message in messages]
                open_thinking = random.random() < self.module.config.data.thinking_ratio
                initial_prompt = self._render_messages(chat_state, tools, add_generation_prompt=True, open_thinking=open_thinking)
                response_ids: list[int] = []
                response_mask: list[int] = []
                turn_outputs: list[str] = []
                unfinished = False

                for turn in range(self.module.config.rollout.max_turns):
                    context = self._render_messages(chat_state, tools, add_generation_prompt=True, open_thinking=open_thinking)
                    turn_ids, _, turn_texts = self._generate_once([context])
                    generated_ids = [token for token in turn_ids[0].tolist() if token != tokenizer.pad_token_id]
                    turn_text = turn_texts[0]
                    turn_outputs.append(turn_text)
                    response_ids.extend(generated_ids)
                    response_mask.extend([1] * len(generated_ids))

                    calls = parse_tool_calls(turn_text)
                    if not calls:
                        break
                    unfinished = turn == self.module.config.rollout.max_turns - 1
                    chat_state.append({"role": "assistant", "content": turn_text})
                    for call in calls:
                        name = call.get("name", "")
                        raw_args = call.get("arguments", {})
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                raw_args = {}
                        result = execute_tool(name, raw_args)
                        result_str = (json.dumps(result, ensure_ascii=False) if result is not None else '{"error": "tool not found"}')[:2048]
                        chat_state.append({"role": "tool", "content": result_str})
                    observation = self._render_messages(chat_state, tools, add_generation_prompt=not unfinished, open_thinking=open_thinking)
                    observation_ids = tokenizer(observation, add_special_tokens=False).input_ids
                    current_len = len(tokenizer(initial_prompt, add_special_tokens=False).input_ids) + len(response_ids)
                    obs_delta = observation_ids[current_len:]
                    response_ids.extend(obs_delta)
                    response_mask.extend([0] * len(obs_delta))

                all_prompts.append(initial_prompt)
                if not response_ids:
                    fallback_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                    response_ids.append(fallback_token_id)
                    response_mask.append(1)
                response_ids_batch.append(response_ids)
                response_masks_batch.append(response_mask)
                completion = turn_outputs[-1] if turn_outputs else ""
                completions.append(completion)
                rollout_metadata = self._metadata_for_generation(base_metadata, completion, turn_outputs, unfinished)
                if tools is not None:
                    rollout_metadata["tools"] = tools
                metadata.append(rollout_metadata)

        padded_ids = [torch.tensor(ids, dtype=torch.long, device=self.module.device) for ids in response_ids_batch]
        padded_masks = [torch.tensor(mask, dtype=torch.long, device=self.module.device) for mask in response_masks_batch]
        completion_ids = torch.nn.utils.rnn.pad_sequence(padded_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        completion_mask = torch.nn.utils.rnn.pad_sequence(padded_masks, batch_first=True, padding_value=0)
        return self._pack_rollout(all_prompts, completion_ids, completion_mask, completions, metadata, num_generations=num_generations)

    def _pack_rollout(
        self,
        prompts: list[str],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        completions: list[str],
        metadata: list[dict[str, Any]],
        num_generations: int,
        old_per_token_logps: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Pack generated samples into tensors consumed by the GRPO loss."""

        tokenizer = self.module.tokenizer
        prompt_encoded = self._tokenize_prompts(prompts)
        prompt_ids = prompt_encoded["input_ids"]
        prompt_mask = prompt_encoded["attention_mask"]
        completion_budget = self.module.config.rollout.max_total_length - prompt_ids.size(1)
        if completion_budget <= 0:
            raise RuntimeError(
                "Tokenized prompts leave no room for completion tokens. "
                "Reduce rollout.max_prompt_length or increase rollout.max_total_length."
            )
        max_completion = min(completion_ids.size(1), completion_budget)
        if max_completion <= 0:
            fallback_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            completion_ids = completion_ids.new_full((completion_ids.size(0), 1), fallback_token_id)
            completion_mask = completion_mask.new_ones((completion_mask.size(0), 1))
            max_completion = 1
        completion_ids = completion_ids[:, :max_completion]
        completion_mask = completion_mask[:, :max_completion]
        if old_per_token_logps is not None:
            old_logps = old_per_token_logps[:, :max_completion].to(self.module.device) * completion_mask.to(old_per_token_logps.dtype)
        else:
            with torch.no_grad():
                old_logps = compute_per_token_logps(self.module, prompt_ids, prompt_mask, completion_ids, completion_mask)
        completion_truncated = (completion_mask.sum(dim=1) >= max_completion).to(torch.long)
        sample_ids = torch.arange(completion_ids.size(0), device=self.module.device) // num_generations

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_logps.detach(),
            "completion_truncated": completion_truncated,
            "sample_ids": sample_ids,
            "prompts": prompts[::num_generations],
            "completions": completions,
            "completion_id_lists": [row[row != tokenizer.pad_token_id].tolist() for row in completion_ids.detach().cpu()],
            "metadata": metadata,
        }

    def rollout(self, batch: dict[str, list[Any]], *, training: bool) -> dict[str, Any]:
        """Dispatch to reasoning or agentic rollout mode."""

        num_generations = self.resolve_num_generations(training)
        modes = set(batch.get("mode", [self.module.config.data.mode]))
        if "agentic" in modes or self.module.config.data.mode == "agentic":
            return self._rollout_agentic(batch, num_generations=num_generations)
        return self._rollout_reasoning(batch, num_generations=num_generations)
