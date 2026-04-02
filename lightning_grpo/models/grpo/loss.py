"""Loss computation helpers for GRPO training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from lightning_grpo.models.common import entropy_from_logits, masked_mean
from lightning_grpo.models.rollout_engine import compute_per_token_logps
from lightning_grpo.utils.modeling import collect_moe_metrics


class GRPOLossComputer:
    """Compute GRPO loss and derived metrics."""

    def __init__(self, module: Any, reward_manager: Any, metrics_aggregator: Any) -> None:
        self.module = module
        self.reward_manager = reward_manager
        self.metrics_aggregator = metrics_aggregator

    def selective_log_softmax(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    def get_per_token_logps(self, model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
        return compute_per_token_logps(
            model,
            input_ids,
            logits_to_keep,
            attention_mask=attention_mask,
            temperature=float(getattr(self.module.rollout_coordinator.rollout_engine.generation_config, "temperature", 1.0) or 1.0),
        )

    def compute_advantages(self, rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
        grouped_rewards = rewards.view(-1, num_generations)
        grouped_mean = grouped_rewards.mean(dim=1, keepdim=True)
        grouped_std = grouped_rewards.std(dim=1, keepdim=True)
        grouped_advantages = (grouped_rewards - grouped_mean) / (grouped_std + self.module.config.rollout.advantage_epsilon)
        return grouped_advantages.reshape(-1)

    def compute_loss(self, rollout_batch: dict[str, torch.Tensor | list[Any]], *, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prompt_ids = rollout_batch["prompt_ids"]
        prompt_mask = rollout_batch["prompt_mask"]
        completion_ids = rollout_batch["completion_ids"]
        completion_mask = rollout_batch["completion_mask"]
        old_per_token_logps = rollout_batch["old_per_token_logps"]
        completion_truncated = rollout_batch["completion_truncated"]

        model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.shape[1]

        outputs = self.module.policy(input_ids=model_input_ids, attention_mask=model_attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :] / self.module.config.rollout.temperature
        per_token_logps = self.selective_log_softmax(logits, completion_ids)
        moe_metrics = collect_moe_metrics(outputs)

        with torch.no_grad():
            if self.module.reference_model is not None:
                ref_per_token_logps = self.get_per_token_logps(
                    self.module.reference_model,
                    model_input_ids,
                    model_attention_mask,
                    logits_to_keep,
                )
            else:
                ref_per_token_logps = None

        rewards, rewards_per_func = self.reward_manager.compute_rewards(
            prompts=rollout_batch["prompts"],
            completions=rollout_batch["completions"],
            completion_id_lists=rollout_batch["completion_id_lists"],
            metadata=rollout_batch["metadata"],
        )
        global_rewards_per_func = self.metrics_aggregator.gather_tensor(rewards_per_func.detach())
        num_generations = self.module.rollout_coordinator.resolve_num_generations(training)
        global_rewards = (global_rewards_per_func * self.module.reward_weights.to(global_rewards_per_func.device).unsqueeze(0)).nansum(dim=-1)
        global_advantages = self.compute_advantages(global_rewards, num_generations)
        advantages = self.compute_advantages(rewards, num_generations).unsqueeze(1)

        log_ratio = per_token_logps - old_per_token_logps
        importance_ratio = torch.exp(log_ratio)
        clip_eps = getattr(self.module.config.rollout, "epsilon", 0.2)
        is_low_clipped = torch.zeros_like(per_token_logps, dtype=torch.bool)
        is_high_clipped = torch.zeros_like(per_token_logps, dtype=torch.bool)
        is_region_clipped = torch.zeros_like(per_token_logps, dtype=torch.bool)
        is_cispo_clipped = torch.zeros_like(per_token_logps, dtype=torch.bool)
        if self.module.config.rollout.loss_type == "cispo":
            clipped_ratio = torch.clamp(importance_ratio, max=self.module.config.rollout.epsilon_high).detach()
            surrogate = clipped_ratio * advantages * per_token_logps
            is_cispo_clipped = (importance_ratio > self.module.config.rollout.epsilon_high) & (advantages > 0)
        else:
            clipped_ratio = torch.clamp(importance_ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            surrogate_unclipped = importance_ratio * advantages
            surrogate_clipped = clipped_ratio * advantages
            surrogate = torch.minimum(surrogate_unclipped, surrogate_clipped)
            is_low_clipped = (importance_ratio < 1.0 - clip_eps) & (advantages < 0)
            is_high_clipped = (importance_ratio > 1.0 + clip_eps) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

        if ref_per_token_logps is not None:
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1.0
        else:
            per_token_kl = torch.zeros_like(per_token_logps)

        loss_mask = completion_mask.to(per_token_logps.dtype)
        per_token_loss = -(surrogate - self.module.config.rollout.kl_beta * per_token_kl)
        loss = masked_mean(per_token_loss, loss_mask)

        with torch.no_grad():
            entropy = entropy_from_logits(logits)
            completion_lengths = completion_mask.sum(dim=1).float()
            global_loss_mask = self.metrics_aggregator.gather_tensor(loss_mask.detach())
            global_per_token_kl = self.metrics_aggregator.gather_tensor(per_token_kl.detach())
            global_entropy = self.metrics_aggregator.gather_tensor(entropy.detach())
            global_completion_lengths = self.metrics_aggregator.gather_tensor(completion_lengths.detach())
            global_completion_truncated = self.metrics_aggregator.gather_tensor(completion_truncated.to(torch.float32))
            global_is_low_clipped = self.metrics_aggregator.gather_tensor(is_low_clipped.to(per_token_logps.dtype))
            global_is_high_clipped = self.metrics_aggregator.gather_tensor(is_high_clipped.to(per_token_logps.dtype))
            global_is_region_clipped = self.metrics_aggregator.gather_tensor(is_region_clipped.to(per_token_logps.dtype))
            global_is_cispo_clipped = self.metrics_aggregator.gather_tensor(is_cispo_clipped.to(per_token_logps.dtype))

        metrics = self.metrics_aggregator.build_training_metrics(
            global_rewards_per_func=global_rewards_per_func,
            reward_weights=self.module.reward_weights,
            num_generations=num_generations,
            global_per_token_kl=global_per_token_kl,
            global_loss_mask=global_loss_mask,
            global_entropy=global_entropy,
            global_completion_lengths=global_completion_lengths,
            global_completion_truncated=global_completion_truncated,
            global_is_low_clipped=global_is_low_clipped,
            global_is_high_clipped=global_is_high_clipped,
            global_is_region_clipped=global_is_region_clipped,
            global_is_cispo_clipped=global_is_cispo_clipped,
            global_advantages=global_advantages,
            moe_metrics=moe_metrics,
            reward_names=self.module.config.reward.active.reward_funcs,
        )
        return loss, metrics
