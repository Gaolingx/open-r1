"""Rollout coordination helpers for GRPO training."""

from __future__ import annotations

import json
import random
from typing import Any

import torch

from lightning_grpo.models.grpo.reward import execute_tool, parse_tool_calls
from lightning_grpo.data.base import apply_chat_template

class LocalGenerateRolloutCoordinator:
    """Generate on-policy rollouts locally with ``model.generate``."""

    def __init__(self, module: Any) -> None:
        self.module = module

    def resolve_num_generations(self, training: bool) -> int:
        """Return the number of completions sampled per prompt."""

        return self.module.config.rollout.num_generations if training else 1

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
                max_length=self.module.config.rollout.max_prompt_length,
                add_special_tokens=False,
                return_token_type_ids=False,
            )
        finally:
            tokenizer.padding_side = old_padding_side
        return {key: value.to(self.module.device) for key, value in encoded.items()}

    @torch.no_grad()
    def _generate_once(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Generate one completion for each prompt and return padded completion ids."""

        tokenizer = self.module.tokenizer
        encoded = self._tokenize_prompts(prompts)
        prompt_lens = encoded["attention_mask"].sum(dim=1)
        generation_kwargs = {
            "max_new_tokens": self.module.config.rollout.max_completion_length,
            "do_sample": self.module.config.rollout.temperature > 0,
            "temperature": self.module.config.rollout.temperature,
            "top_p": self.module.config.rollout.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        outputs = self.module.policy.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            **generation_kwargs,
        )

        input_width = encoded["input_ids"].size(1)
        completion_rows: list[torch.Tensor] = []
        completion_texts: list[str] = []
        for row in outputs:
            completion = row[input_width:]
            if completion.numel() == 0:
                completion = row.new_tensor([tokenizer.eos_token_id or tokenizer.pad_token_id])
            completion_rows.append(completion)
            completion_texts.append(tokenizer.decode(completion, skip_special_tokens=True))
        completion_ids = torch.nn.utils.rnn.pad_sequence(
            completion_rows,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        completion_mask = (completion_ids != tokenizer.pad_token_id).long()
        return completion_ids, completion_mask, completion_texts

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

        prompts = list(batch["prompt"])
        expanded_prompts = [prompt for prompt in prompts for _ in range(num_generations)]
        completion_ids, completion_mask, completions = self._generate_once(expanded_prompts)
        metadata = []
        for index, completion in enumerate(completions):
            sample_index = index // num_generations
            metadata.append(
                {
                    "gt": batch.get("gt", [None] * len(prompts))[sample_index],
                    "tools": batch.get("tools", [None] * len(prompts))[sample_index],
                    "turn_outputs": [completion],
                    "unfinished": False,
                }
            )
        return self._pack_rollout(expanded_prompts, completion_ids, completion_mask, completions, metadata, num_generations=num_generations)

    def _rollout_agentic(self, batch: dict[str, list[Any]], *, num_generations: int) -> dict[str, Any]:
        """Run multi-turn local tool-call rollouts and mask tool observations from loss."""

        tokenizer = self.module.tokenizer
        all_prompts: list[str] = []
        response_ids_batch: list[list[int]] = []
        response_masks_batch: list[list[int]] = []
        completions: list[str] = []
        metadata: list[dict[str, Any]] = []

        for sample_index, messages in enumerate(batch["messages"]):
            tools = batch.get("tools", [None] * len(batch["messages"]))[sample_index]
            gt = batch.get("gt", [None] * len(batch["messages"]))[sample_index]
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
                    if tokenizer.eos_token_id is not None:
                        generated_ids = [token for token in generated_ids if token != tokenizer.eos_token_id]
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
                response_ids_batch.append(response_ids)
                response_masks_batch.append(response_mask)
                completions.append(turn_outputs[-1] if turn_outputs else "")
                metadata.append({"gt": gt, "tools": tools, "turn_outputs": turn_outputs, "unfinished": unfinished})

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
    ) -> dict[str, Any]:
        """Pack generated samples into tensors consumed by the GRPO loss."""

        tokenizer = self.module.tokenizer
        prompt_encoded = self._tokenize_prompts(prompts)
        prompt_ids = prompt_encoded["input_ids"]
        prompt_mask = prompt_encoded["attention_mask"]
        max_completion = min(completion_ids.size(1), self.module.config.rollout.max_total_length - prompt_ids.size(1))
        completion_ids = completion_ids[:, :max_completion]
        completion_mask = completion_mask[:, :max_completion]
        with torch.no_grad():
            old_logps = self.module.compute_per_token_logps(prompt_ids, prompt_mask, completion_ids, completion_mask)
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