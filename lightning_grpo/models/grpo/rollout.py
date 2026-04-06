"""Rollout coordination helpers for GRPO training."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.models.rollout_engine import create_rollout_engine, load_generation_config


class GRPORolloutCoordinator:
    """Own rollout backends and generation helpers."""

    def __init__(self, config: Any, policy: torch.nn.Module, tokenizer: Any) -> None:
        self.config = config
        self.policy = policy
        self.tokenizer = tokenizer
        self.rollout_engine = create_rollout_engine(
            engine_type=config.rollout.engine.engine_type,
            policy_model=policy,
            tokenizer=tokenizer,
            generation_config_path=config.rollout.generation_config_path,
            generation_batch_size=config.rollout.generation_batch_size,
            sglang_base_url=config.rollout.engine.sglang_base_url,
            sglang_model_path=config.rollout.engine.sglang_model_path,
            sglang_shared_path=config.rollout.engine.sglang_shared_path,
            request_timeout=config.rollout.engine.request_timeout,
            max_retries=config.rollout.engine.max_retries,
            retry_backoff_seconds=config.rollout.engine.retry_backoff_seconds,
            retry_max_backoff_seconds=config.rollout.engine.retry_max_backoff_seconds,
            reward_model_config=config.reward.rlhf.reward_model,
        )
        self.reward_model_engine = None
        if config.reward.rlhf.reward_model.enabled:
            self.reward_model_engine = create_rollout_engine(
                engine_type="reward_model",
                policy_model=policy,
                tokenizer=tokenizer,
                generation_config_path=None,
                reward_model_config=config.reward.rlhf.reward_model,
            )

    def resolve_num_generations(self, training: bool) -> int:
        if training:
            return self.config.rollout.num_generations
        return self.config.rollout.num_generations_eval or self.config.rollout.num_generations

    def decode_completion_ids(self, completion_texts: list[str]) -> tuple[list[str], list[list[dict[str, str]]]]:
        structured_completions = [[{"role": "assistant", "content": text}] for text in completion_texts]
        return completion_texts, structured_completions

    @torch.no_grad()
    def generate(self, batch: dict[str, Any], *, training: bool) -> dict[str, torch.Tensor | list[Any]]:
        num_generations = self.resolve_num_generations(training)
        rollout = self.rollout_engine.rollout(
            prompt_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_generations=num_generations,
        )
        completion_texts, structured_completions = self.decode_completion_ids(rollout.completions_text)

        repeated_prompts = [prompt for prompt in batch["prompt_text"] for _ in range(num_generations)]
        repeated_metadata = [meta for meta in batch["metadata"] for _ in range(num_generations)]

        return {
            "prompt_ids": rollout.prompt_ids,
            "prompt_mask": rollout.prompt_mask,
            "completion_ids": rollout.completion_ids,
            "completion_mask": rollout.completion_mask,
            "completion_truncated": rollout.completion_truncated,
            "old_per_token_logps": rollout.per_token_logps,
            "prompts": repeated_prompts,
            "completions_text": completion_texts,
            "completions": structured_completions,
            "completion_id_lists": rollout.completion_id_lists,
            "metadata": repeated_metadata,
        }

    @torch.no_grad()
    def emit_debug_samples(self, trainer: Any, device: torch.device) -> None:
        debug_config = self.config.rollout.debug
        if not debug_config.enabled or not debug_config.questions:
            return
        if not trainer.is_global_zero:
            return

        self.policy.eval()
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            generation_config = self.rollout_engine.generation_config
            if debug_config.generation_config_path is not None:
                generation_config = load_generation_config(debug_config.generation_config_path)

            for index, question in enumerate(debug_config.questions):
                tokenized = self.tokenizer([question], return_tensors="pt", padding=True, truncation=True).to(device)
                generated = self.policy.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    num_return_sequences=1,
                    use_cache=True,
                    generation_config=generation_config,
                )
                completion = generated[:, tokenized["input_ids"].shape[1]:]
                text = self.tokenizer.batch_decode(completion, skip_special_tokens=True)[0]
                rank_zero_info(f"[GRPO DEBUG][{index}] question: {question}")
                rank_zero_info(f"[GRPO DEBUG][{index}] completion: {text}")
        finally:
            self.tokenizer.padding_side = original_padding_side
            self.policy.train()

    def sync_policy(self, policy: torch.nn.Module) -> None:
        if self.config.rollout.engine.engine_type == "policy":
            self.rollout_engine.update_policy(policy)
