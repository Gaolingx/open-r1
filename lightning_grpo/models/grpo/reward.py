"""Reward management helpers for GRPO training."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from open_r1.rewards import get_reward_funcs


class GRPORewardManager:
    """Build and evaluate reward functions for GRPO rollouts."""

    def __init__(self, config: Any, tokenizer: Any, device: torch.device) -> None:
        self.config = config
        self.reward_config = config.reward
        self.tokenizer = tokenizer
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
            code_language=reward.code_language,
            repetition_n_grams=reward.repetition_n_grams,
            repetition_max_penalty=reward.repetition_max_penalty,
            cosine_min_value_wrong=reward.cosine_min_value_wrong,
            cosine_max_value_wrong=reward.cosine_max_value_wrong,
            cosine_min_value_correct=reward.cosine_min_value_correct,
            cosine_max_value_correct=reward.cosine_max_value_correct,
            cosine_max_len=reward.cosine_max_len,
            parallel_code_exec_per_proc=reward.parallel_code_exec_per_proc,
            code_provider=reward.code_provider,
            enforce_same_language=reward.enforce_same_language,
            code_eval_test_batch_size=reward.code_eval_test_batch_size,
            code_eval_scoring_mode=reward.code_eval_scoring_mode,
            ioi_provider=reward.ioi_provider,
            max_completion_len=reward.max_completion_len,
            soft_punish_cache=reward.soft_punish_cache,
        )

    def compute_rewards(
        self,
        prompts: list[str],
        completions: list[str] | list[list[dict[str, str]]],
        completion_id_lists: list[list[int]],
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted rewards, adapting local rollout text to Open-R1 reward inputs."""

        if completions and isinstance(completions[0], str):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        if not metadata:
            metadata = [{} for _ in completions]

        reward_kwargs: dict[str, list[Any]] = {}
        for sample in metadata:
            for key in sample:
                reward_kwargs.setdefault(key, [])

        for key in reward_kwargs:
            reward_kwargs[key] = [sample.get(key) for sample in metadata]

        if "solution" not in reward_kwargs:
            for alias in ("answer", "response", "output", "gold_answer", "gold_solution"):
                if alias in reward_kwargs:
                    reward_kwargs["solution"] = reward_kwargs[alias]
                    break

        reward_matrix: list[torch.Tensor] = []
        for reward_fn in self.reward_funcs:
            reward_values = reward_fn(
                prompts=prompts,
                completions=completions,
                completion_ids=completion_id_lists,
                **reward_kwargs,
            )
            reward_values = [value if value is not None else torch.nan for value in reward_values]
            reward_tensor = torch.tensor(reward_values, device=torch.device(self.device), dtype=torch.float32)
            reward_matrix.append(reward_tensor)

        rewards_per_func = torch.stack(reward_matrix, dim=-1)
        rewards = (rewards_per_func * self.reward_weight_tensor.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=-1)
        return rewards, rewards_per_func
