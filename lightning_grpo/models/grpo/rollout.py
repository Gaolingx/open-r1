"""Rollout coordination helpers for GRPO training."""

from __future__ import annotations

import os
from typing import Any

import torch
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.models.rollout_engine import create_rollout_engine


class GRPORolloutCoordinator:
    """Own rollout backends and generation helpers."""

    def __init__(self, config: Any, policy: torch.nn.Module, tokenizer: Any) -> None:
        self.config = config
        self.policy = policy
        self.tokenizer = tokenizer

        # Resolve distributed info for vLLM engine
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.rollout_engine = create_rollout_engine(
            engine_type=config.rollout.engine.engine_type,
            policy_model=policy,
            tokenizer=tokenizer,
            sampling_config_path=config.rollout.sampling_config_path,
            generation_batch_size=config.rollout.generation_batch_size,
            reward_model_config=config.reward.rlhf.reward_model,
            vllm_config=getattr(config.rollout.engine, "vllm", None),
            model_name_or_path=getattr(config.model, "model_name_or_path", None),
            world_size=world_size,
            local_rank=local_rank,
            global_rank=global_rank,
        )
        self.reward_model_engine = None
        if config.reward.rlhf.reward_model.enabled:
            self.reward_model_engine = create_rollout_engine(
                engine_type="reward_model",
                policy_model=policy,
                tokenizer=tokenizer,
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
        repeated_sample_ids = batch["sample_id"].to(rollout.prompt_ids.device).repeat_interleave(num_generations)

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
            "sample_ids": repeated_sample_ids,
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

            for index, question in enumerate(debug_config.questions):
                tokenized = self.tokenizer([question], return_tensors="pt", padding=True, truncation=True).to(device)
                generated = self.policy.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    num_return_sequences=1,
                    generation_config=self.policy.generation_config,
                )
                completion = generated[:, tokenized["input_ids"].shape[1]:]
                text = self.tokenizer.batch_decode(completion, skip_special_tokens=False)[0]
                rank_zero_info(f"[GRPO DEBUG][{index}] question: {question}")
                rank_zero_info(f"[GRPO DEBUG][{index}] completion: {text}")
        finally:
            self.tokenizer.padding_side = original_padding_side
            self.policy.train()

    def sync_policy(self, policy: torch.nn.Module) -> None:
        if self.config.rollout.engine.engine_type in ("policy", "sglang", "vllm"):
            self.rollout_engine.update_policy(policy)
