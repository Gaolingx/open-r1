"""Lightning module for GRPO-style online RL fine-tuning."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from open_r1.rewards import get_reward_funcs
from lightning_grpo.configs.grpo import GRPOConfig
from lightning_grpo.models.common import build_optimizer, build_scheduler, entropy_from_logits, masked_mean
from lightning_grpo.utils.modeling import count_trainable_parameters, load_causal_lm, load_tokenizer


class GRPOLightningModule(L.LightningModule):
    """Lightning-native GRPO implementation aligned with the core TRL training flow."""

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
        self.reward_funcs = get_reward_funcs(self._build_reward_script_args())
        self.save_hyperparameters(config.to_dict())

        trainable, total = count_trainable_parameters(self.policy)
        self.trainable_parameter_count = trainable
        self.total_parameter_count = total

    def _build_reward_script_args(self) -> SimpleNamespace:
        """Build an args-like object expected by [`open_r1.rewards.get_reward_funcs()`](src/open_r1/rewards.py:646)."""

        reward = self.config.reward
        rollout = self.config.rollout
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
            parallel_code_exec_per_proc=getattr(reward, "parallel_code_exec_per_proc", 1),
            code_provider=getattr(reward, "code_provider", "e2b"),
            enforce_same_language=getattr(reward, "enforce_same_language", False),
            code_eval_test_batch_size=getattr(reward, "code_eval_test_batch_size", 1),
            code_eval_scoring_mode=getattr(reward, "code_eval_scoring_mode", "weighted_sum"),
            ioi_provider=getattr(reward, "ioi_provider", "piston"),
            max_completion_len=rollout.max_completion_length,
            soft_punish_cache=getattr(reward, "soft_punish_cache", 0),
        )

    def on_fit_start(self) -> None:
        """Log static parameter counts once training starts."""

        self.log("model/trainable_parameters", float(self.trainable_parameter_count), rank_zero_only=True)
        self.log("model/total_parameters", float(self.total_parameter_count), rank_zero_only=True)

    def forward(self, **batch: torch.Tensor) -> Any:
        """Forward prompts and completions through the policy model."""

        return self.policy(**batch)

    def _repeat_interleave_rows(self, tensor: torch.Tensor, repeats: int) -> torch.Tensor:
        """Repeat each row in a batch contiguously."""

        return tensor.repeat_interleave(repeats, dim=0)

    def _selective_log_softmax(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Gather token log-probabilities for the sampled tokens."""

        log_probs = F.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    def _get_per_token_logps(self, model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
        """Compute per-token log-probabilities over the completion region only."""

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        logits = logits / self.config.rollout.temperature
        completion_ids = input_ids[:, -logits_to_keep:]
        return self._selective_log_softmax(logits, completion_ids)

    def _truncate_completions(self, completion_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask tokens after the first EOS token, following the TRL GRPO generation path."""

        is_eos = completion_ids == self.tokenizer.eos_token_id
        eos_idx = torch.full(
            (completion_ids.size(0),),
            completion_ids.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        token_positions = torch.arange(completion_ids.size(1), device=completion_ids.device).expand(completion_ids.size(0), -1)
        completion_mask = (token_positions <= eos_idx.unsqueeze(1)).long()
        completion_ids = completion_ids.masked_fill(completion_mask == 0, self.tokenizer.pad_token_id)
        return completion_ids, completion_mask

    def _decode_completion_ids(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> tuple[list[str], list[list[dict[str, str]]], list[list[int]]]:
        """Decode completions into both plain-text and reward-compatible chat formats."""

        completion_id_lists: list[list[int]] = []
        for ids, mask in zip(completion_ids, completion_mask, strict=True):
            valid_ids = ids[mask.bool()].tolist()
            completion_id_lists.append(valid_ids)

        completion_texts = self.tokenizer.batch_decode(completion_id_lists, skip_special_tokens=True)
        structured_completions = [[{"role": "assistant", "content": text}] for text in completion_texts]
        return completion_texts, structured_completions, completion_id_lists

    @torch.no_grad()
    def _generate(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | list[Any]]:
        """Generate grouped completions for online GRPO optimization."""

        num_generations = self.config.rollout.num_generations
        prompt_ids = batch["input_ids"]
        prompt_mask = batch["attention_mask"]

        original_padding_side = self.tokenizer.padding_side
        if original_padding_side != "left":
            self.tokenizer.padding_side = "left"

        try:
            generated = self.policy.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                do_sample=True,
                temperature=self.config.rollout.temperature,
                top_p=self.config.rollout.top_p,
                max_new_tokens=self.config.rollout.max_completion_length,
                num_return_sequences=num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        repeated_prompt_ids = self._repeat_interleave_rows(prompt_ids, num_generations)
        repeated_prompt_mask = self._repeat_interleave_rows(prompt_mask, num_generations)
        prompt_length = repeated_prompt_ids.shape[1]

        completion_ids = generated[:, prompt_length:]
        completion_ids, completion_mask = self._truncate_completions(completion_ids)
        completion_texts, structured_completions, completion_id_lists = self._decode_completion_ids(completion_ids, completion_mask)

        repeated_prompts = [prompt for prompt in batch["prompt_text"] for _ in range(num_generations)]
        repeated_metadata = [meta for meta in batch["metadata"] for _ in range(num_generations)]

        model_input_ids = torch.cat([repeated_prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([repeated_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.shape[1]
        old_per_token_logps = self._get_per_token_logps(
            self.policy,
            model_input_ids,
            model_attention_mask,
            logits_to_keep,
        )

        return {
            "prompt_ids": repeated_prompt_ids,
            "prompt_mask": repeated_prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "prompts": repeated_prompts,
            "completions_text": completion_texts,
            "completions": structured_completions,
            "completion_id_lists": completion_id_lists,
            "metadata": repeated_metadata,
        }

    def _compute_rewards(
        self,
        prompts: list[str],
        completions: list[list[dict[str, str]]],
        completion_id_lists: list[list[int]],
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-reward and aggregated rewards using TRL-style reward kwargs propagation."""

        if not metadata:
            metadata = [{} for _ in completions]

        reward_kwargs: dict[str, list[Any]] = {}
        for sample in metadata:
            for key in sample:
                reward_kwargs.setdefault(key, [])

        for key in reward_kwargs:
            reward_kwargs[key] = [sample.get(key) for sample in metadata]

        reward_matrix: list[torch.Tensor] = []
        for reward_fn in self.reward_funcs:
            reward_values = reward_fn(
                prompts=prompts,
                completions=completions,
                completion_ids=completion_id_lists,
                **reward_kwargs,
            )
            reward_tensor = torch.as_tensor(reward_values, device=self.device, dtype=torch.float32)
            reward_tensor = torch.nan_to_num(reward_tensor, nan=0.0)
            reward_matrix.append(reward_tensor)

        rewards_per_func = torch.stack(reward_matrix, dim=-1)
        rewards = rewards_per_func.sum(dim=-1)
        return rewards, rewards_per_func

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards within each prompt group, as in GRPO."""

        num_generations = self.config.rollout.num_generations
        grouped_rewards = rewards.view(-1, num_generations)
        grouped_mean = grouped_rewards.mean(dim=1, keepdim=True)
        grouped_std = grouped_rewards.std(dim=1, keepdim=True)
        grouped_advantages = (grouped_rewards - grouped_mean) / (grouped_std + self.config.rollout.advantage_epsilon)
        return grouped_advantages.reshape(-1)

    def _compute_loss(self, rollout_batch: dict[str, torch.Tensor | list[Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the GRPO objective with PPO-style ratio clipping and optional KL penalty."""

        prompt_ids = rollout_batch["prompt_ids"]
        prompt_mask = rollout_batch["prompt_mask"]
        completion_ids = rollout_batch["completion_ids"]
        completion_mask = rollout_batch["completion_mask"]
        old_per_token_logps = rollout_batch["old_per_token_logps"]

        model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.shape[1]

        outputs = self.policy(input_ids=model_input_ids, attention_mask=model_attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :] / self.config.rollout.temperature
        per_token_logps = self._selective_log_softmax(logits, completion_ids)

        with torch.no_grad():
            if self.reference_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.reference_model,
                    model_input_ids,
                    model_attention_mask,
                    logits_to_keep,
                )
            else:
                ref_per_token_logps = None

        rewards, rewards_per_func = self._compute_rewards(
            prompts=rollout_batch["prompts"],
            completions=rollout_batch["completions"],
            completion_id_lists=rollout_batch["completion_id_lists"],
            metadata=rollout_batch["metadata"],
        )
        advantages = self._compute_advantages(rewards)
        advantages = advantages.unsqueeze(1)

        log_ratio = per_token_logps - old_per_token_logps
        importance_ratio = torch.exp(log_ratio)
        clip_eps = getattr(self.config.rollout, "epsilon", 0.2)
        clipped_ratio = torch.clamp(importance_ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate_unclipped = importance_ratio * advantages
        surrogate_clipped = clipped_ratio * advantages
        surrogate = torch.minimum(surrogate_unclipped, surrogate_clipped)

        if ref_per_token_logps is not None:
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1.0
        else:
            per_token_kl = torch.zeros_like(per_token_logps)

        loss_mask = completion_mask.to(per_token_logps.dtype)
        per_token_loss = -(surrogate - self.config.rollout.kl_beta * per_token_kl)
        loss = masked_mean(per_token_loss, loss_mask)

        with torch.no_grad():
            entropy = entropy_from_logits(logits)
            completion_lengths = completion_mask.sum(dim=1).float()
            clipped_sequences = (completion_ids[:, -1] != self.tokenizer.eos_token_id).to(torch.float32)
            terminated_lengths = completion_lengths[clipped_sequences == 0]
            if terminated_lengths.numel() == 0:
                terminated_lengths = completion_lengths.new_zeros(1)

            is_low_clipped = (importance_ratio < 1.0 - clip_eps) & (advantages < 0)
            is_high_clipped = (importance_ratio > 1.0 + clip_eps) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

        metrics = {
            "reward": rewards.mean(),
            "reward_std": rewards.std(unbiased=False),
            "advantage_mean": advantages.mean(),
            "advantage_std": advantages.std(unbiased=False),
            "frac_reward_zero_std": (rewards.view(-1, self.config.rollout.num_generations).std(dim=1) < 1.0e-6).float().mean(),
            "kl": masked_mean(per_token_kl, loss_mask),
            "entropy": masked_mean(entropy, loss_mask),
            "completion_length": completion_lengths.mean(),
            "completion_length_min": completion_lengths.min(),
            "completion_length_max": completion_lengths.max(),
            "completion_clipped_ratio": clipped_sequences.mean(),
            "terminated_length_mean": terminated_lengths.mean(),
            "terminated_length_min": terminated_lengths.min(),
            "terminated_length_max": terminated_lengths.max(),
            "clip_ratio_low": masked_mean(is_low_clipped.to(per_token_logps.dtype), loss_mask),
            "clip_ratio_high": masked_mean(is_high_clipped.to(per_token_logps.dtype), loss_mask),
            "clip_ratio_region": masked_mean(is_region_clipped.to(per_token_logps.dtype), loss_mask),
        }
        for index, reward_name in enumerate(self.config.reward.reward_funcs):
            metrics[f"reward/{reward_name}"] = rewards_per_func[:, index].mean()
            metrics[f"reward_std/{reward_name}"] = rewards_per_func[:, index].std(unbiased=False)

        return loss, metrics

    def _log_metrics(self, prefix: str, loss: torch.Tensor, metrics: dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool) -> None:
        """Log the standard GRPO optimization metrics."""

        self.log(f"{prefix}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/reward", metrics["reward"], prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/reward_std", metrics["reward_std"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(
            f"{prefix}/frac_reward_zero_std",
            metrics["frac_reward_zero_std"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(f"{prefix}/advantage_mean", metrics["advantage_mean"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/advantage_std", metrics["advantage_std"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/kl", metrics["kl"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/entropy", metrics["entropy"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(
            f"{prefix}/completions/mean_length",
            metrics["completion_length"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/completions/min_length",
            metrics["completion_length_min"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/completions/max_length",
            metrics["completion_length_max"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/completions/clipped_ratio",
            metrics["completion_clipped_ratio"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/completions/mean_terminated_length",
            metrics["terminated_length_mean"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/completions/min_terminated_length",
            metrics["terminated_length_min"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/completions/max_terminated_length",
            metrics["terminated_length_max"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/clip_ratio/low",
            metrics["clip_ratio_low"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/clip_ratio/high",
            metrics["clip_ratio_high"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/clip_ratio/region",
            metrics["clip_ratio_region"],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        for reward_name in self.config.reward.reward_funcs:
            self.log(
                f"{prefix}/rewards/{reward_name}/mean",
                metrics[f"reward/{reward_name}"],
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
            )
            self.log(
                f"{prefix}/rewards/{reward_name}/std",
                metrics[f"reward_std/{reward_name}"],
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one online rollout and optimization step."""

        rollout_batch = self._generate(batch)
        loss, metrics = self._compute_loss(rollout_batch)
        self._log_metrics("train", loss, metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Evaluate the current policy with a rollout batch."""

        rollout_batch = self._generate(batch)
        loss, metrics = self._compute_loss(rollout_batch)
        self._log_metrics("val", loss, metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and scheduler for Lightning."""

        optimizer = build_optimizer(self.policy.parameters(), self.config.optimization)
        scheduler = build_scheduler(optimizer, self.config.optimization, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
