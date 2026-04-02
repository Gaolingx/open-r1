"""Reward management helpers for GRPO training."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from open_r1.rewards import get_reward_funcs
from lightning_grpo.data.base import apply_chat_template


class GRPORewardManager:
    """Build and evaluate reward functions for GRPO rollouts."""

    def __init__(self, config: Any, tokenizer: Any, rollout_engine: Any, reward_model_engine: Any, device: torch.device) -> None:
        self.config = config
        self.reward_config = config.reward.active
        self.tokenizer = tokenizer
        self.rollout_engine = rollout_engine
        self.reward_model_engine = reward_model_engine
        self.device = device
        self.reward_funcs = get_reward_funcs(self._build_reward_script_args())
        reward_weights = self.reward_config.reward_weights
        if reward_weights is not None:
            if len(reward_weights) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(reward_weights)}) must match number of reward functions ({len(self.reward_funcs)})"
                )
            self.reward_weight_tensor = torch.tensor(reward_weights, dtype=torch.float32)
        else:
            self.reward_weight_tensor = torch.ones(len(self.reward_funcs), dtype=torch.float32)

    def _build_reward_script_args(self) -> SimpleNamespace:
        reward = self.reward_config
        return SimpleNamespace(
            reward_funcs=reward.reward_funcs,
            format_mode=getattr(reward, "format_mode", "strict"),
            code_language=reward.code_language,
            repetition_n_grams=reward.repetition_n_grams,
            repetition_max_penalty=reward.repetition_max_penalty,
            cosine_min_value_wrong=reward.cosine_min_value_wrong,
            cosine_max_value_wrong=reward.cosine_max_value_wrong,
            cosine_min_value_correct=reward.cosine_min_value_correct,
            cosine_max_value_correct=reward.cosine_max_value_correct,
            cosine_max_len=reward.cosine_max_len,
            parallel_code_exec_per_proc=getattr(reward, "parallel_code_exec_per_proc", 1),
            code_provider=getattr(reward, "code_provider", "e2b"),
            enforce_same_language=getattr(reward, "enforce_same_language", False),
            code_eval_test_batch_size=getattr(reward, "code_eval_test_batch_size", 1),
            code_eval_scoring_mode=getattr(reward, "code_eval_scoring_mode", "weighted_sum"),
            ioi_provider=getattr(reward, "ioi_provider", "piston"),
            max_completion_len=getattr(self.rollout_engine.generation_config, "max_new_tokens", None),
            soft_punish_cache=getattr(reward, "soft_punish_cache", 0),
            reward_model_engine=self.reward_model_engine,
        )

    def format_reward_model_text(self, prompt: str, completion: str) -> str:
        reward_model_config = self.config.reward.rlhf.reward_model
        template_config = reward_model_config.template
        if not template_config.enabled:
            return f"{prompt}{completion}"

        messages: list[dict[str, str]] = []
        if template_config.include_system_prompt and template_config.system_prompt:
            messages.append({"role": "system", "content": template_config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": completion})

        original_template = self.tokenizer.chat_template
        if template_config.chat_template:
            self.tokenizer.chat_template = template_config.chat_template
        try:
            return apply_chat_template(
                tokenizer=self.tokenizer,
                messages=messages,
                add_generation_prompt=template_config.add_generation_prompt,
            )
        finally:
            self.tokenizer.chat_template = original_template

    def compute_rewards(
        self,
        prompts: list[str],
        completions: list[list[dict[str, str]]],
        completion_id_lists: list[list[int]],
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not metadata:
            metadata = [{} for _ in completions]

        reward_kwargs: dict[str, list[Any]] = {}
        for sample in metadata:
            for key in sample:
                reward_kwargs.setdefault(key, [])

        for key in reward_kwargs:
            reward_kwargs[key] = [sample.get(key) for sample in metadata]

        reward_matrix: list[torch.Tensor] = []
        reward_model_texts = [
            self.format_reward_model_text(prompt, completion[0]["content"])
            for prompt, completion in zip(prompts, completions)
        ]
        for reward_fn in self.reward_funcs:
            reward_values = reward_fn(
                prompts=prompts,
                completions=completions,
                completion_ids=completion_id_lists,
                reward_model_texts=reward_model_texts,
                **reward_kwargs,
            )
            reward_values = [value if value is not None else torch.nan for value in reward_values]
            reward_tensor = torch.tensor(reward_values, device=self.device, dtype=torch.float32)
            reward_matrix.append(reward_tensor)

        rewards_per_func = torch.stack(reward_matrix, dim=-1)
        rewards = (rewards_per_func * self.reward_weight_tensor.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=-1)
        return rewards, rewards_per_func
