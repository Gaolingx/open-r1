"""Lightning module for GRPO-style online RL fine-tuning."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from open_r1.rewards import get_reward_funcs
from lightning_grpo.configs.grpo import GRPOConfig
from lightning_grpo.models.common import (
    approx_kl_divergence,
    build_optimizer,
    build_scheduler,
    entropy_from_logits,
    masked_mean,
)
from lightning_grpo.utils.modeling import count_trainable_parameters, load_causal_lm, load_tokenizer


class GRPOLightningModule(L.LightningModule):
    """A simplified Lightning-native implementation of the GRPO training loop."""

    def __init__(self, config: GRPOConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = load_causal_lm(config.model, config.precision)
        self.reference_model = load_causal_lm(config.model, config.precision) if config.rollout.use_reference_model else None
        if self.reference_model is not None:
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad = False

        self.tokenizer = load_tokenizer(config.model)
        self.reward_funcs = get_reward_funcs(config.reward)
        self.save_hyperparameters(config.to_dict())

        trainable, total = count_trainable_parameters(self.policy)
        self.trainable_parameter_count = trainable
        self.total_parameter_count = total

    def on_fit_start(self) -> None:
        """Log static parameter counts once training starts."""

        self.log("model/trainable_parameters", float(self.trainable_parameter_count), rank_zero_only=True)
        self.log("model/total_parameters", float(self.total_parameter_count), rank_zero_only=True)

    def forward(self, **batch: torch.Tensor) -> Any:
        """Forward prompts and completions through the policy model."""

        return self.policy(**batch)

    @torch.no_grad()
    def _generate(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | list[str]]:
        """Generate completions for online GRPO optimization."""

        generated = self.policy.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=True,
            temperature=self.config.rollout.temperature,
            top_p=self.config.rollout.top_p,
            max_new_tokens=self.config.rollout.max_completion_length,
            num_return_sequences=self.config.rollout.num_generations,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = batch["input_ids"].shape[1]
        completion_ids = generated[:, prompt_length:]
        completion_mask = (completion_ids != self.tokenizer.pad_token_id).long()
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        repeated_prompts = [prompt for prompt in batch["prompt_text"] for _ in range(self.config.rollout.num_generations)]
        repeated_metadata = [meta for meta in batch["metadata"] for _ in range(self.config.rollout.num_generations)]
        return {
            "prompt_input_ids": generated[:, :prompt_length],
            "prompt_attention_mask": (generated[:, :prompt_length] != self.tokenizer.pad_token_id).long(),
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "prompts": repeated_prompts,
            "completions": completions,
            "metadata": repeated_metadata,
        }

    def _compute_rewards(self, prompts: list[str], completions: list[str], metadata: list[dict[str, Any]]) -> torch.Tensor:
        """Aggregate registered reward functions."""

        total_reward = torch.zeros(len(completions), device=self.device, dtype=torch.float32)
        for reward_fn in self.reward_funcs:
            reward_values = reward_fn(prompts=prompts, completions=completions, metadata=metadata)
            total_reward = total_reward + torch.as_tensor(reward_values, device=self.device, dtype=torch.float32)
        return total_reward

    def _gather_log_probs(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Gather token log-probabilities for sampled tokens."""

        log_probs = F.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    def _compute_loss(self, rollout_batch: dict[str, torch.Tensor | list[str]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute policy gradient, KL, and entropy terms for GRPO."""

        prompt_ids = rollout_batch["prompt_input_ids"]
        prompt_mask = rollout_batch["prompt_attention_mask"]
        completion_ids = rollout_batch["completion_ids"]
        completion_mask = rollout_batch["completion_mask"]

        model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        outputs = self.policy(input_ids=model_input_ids, attention_mask=model_attention_mask)
        completion_logits = outputs.logits[:, -completion_ids.shape[1] - 1 : -1, :]
        log_probs = self._gather_log_probs(completion_logits, completion_ids)

        with torch.no_grad():
            if self.reference_model is not None:
                ref_outputs = self.reference_model(input_ids=model_input_ids, attention_mask=model_attention_mask)
                ref_logits = ref_outputs.logits[:, -completion_ids.shape[1] - 1 : -1, :]
                ref_log_probs = self._gather_log_probs(ref_logits, completion_ids)
            else:
                ref_log_probs = log_probs.detach()

        rewards = self._compute_rewards(
            prompts=rollout_batch["prompts"],
            completions=rollout_batch["completions"],
            metadata=rollout_batch["metadata"],
        )
        grouped_rewards = rewards.view(-1, self.config.rollout.num_generations)
        advantages = grouped_rewards - grouped_rewards.mean(dim=-1, keepdim=True)
        advantages = advantages / (grouped_rewards.std(dim=-1, keepdim=True) + self.config.rollout.advantage_epsilon)
        advantages = advantages.reshape(-1).unsqueeze(-1)

        kl = approx_kl_divergence(log_probs, ref_log_probs)
        entropy = entropy_from_logits(completion_logits)
        policy_loss = -(advantages * log_probs)
        masked_policy_loss = masked_mean(policy_loss, completion_mask)
        masked_kl = masked_mean(kl, completion_mask)
        masked_entropy = masked_mean(entropy, completion_mask)
        total_loss = masked_policy_loss + self.config.rollout.kl_beta * masked_kl

        metrics = {
            "reward": rewards.mean(),
            "kl": masked_kl,
            "entropy": masked_entropy,
        }
        return total_loss, metrics

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one online rollout and optimization step."""

        rollout_batch = self._generate(batch)
        loss, metrics = self._compute_loss(rollout_batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/reward", metrics["reward"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/kl", metrics["kl"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/entropy", metrics["entropy"], on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Evaluate the current policy with a rollout batch."""

        rollout_batch = self._generate(batch)
        loss, metrics = self._compute_loss(rollout_batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/reward", metrics["reward"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and scheduler for Lightning."""

        optimizer = build_optimizer(self.policy.parameters(), self.config.optimization)
        scheduler = build_scheduler(optimizer, self.config.optimization, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
