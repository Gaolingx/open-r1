"""Lightning module for GRPO-style online RL fine-tuning."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_info

from open_r1.rewards import get_reward_funcs
from lightning_grpo.models.rollout_engine import create_rollout_engine, compute_per_token_logps
from lightning_grpo.utils.configs.grpo import GRPOConfig
from lightning_grpo.models.common import build_optimizer, build_scheduler, entropy_from_logits, masked_mean
from lightning_grpo.utils.modeling import collect_moe_metrics, count_trainable_parameters, export_configured_model, load_causal_lm, load_tokenizer, log_moe_metrics


class GRPOLightningModule(L.LightningModule):
    """Lightning-native GRPO implementation aligned with the core TRL training flow."""

    def __init__(self, config: GRPOConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = load_causal_lm(config.model, config.precision)
        self.reference_model = load_causal_lm(config.model, config.precision) if config.rollout.use_reference_model else None
        if self.reference_model is not None:
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()

        self.tokenizer = load_tokenizer(config.model)
        self.rollout_engine = create_rollout_engine(
            engine_type=config.rollout.engine.engine_type,
            policy_model=self.policy,
            tokenizer=self.tokenizer,
            generation_config_path=config.rollout.generation_config_path,
            generation_batch_size=config.rollout.generation_batch_size,
            sglang_base_url=config.rollout.engine.sglang_base_url,
            sglang_model_path=config.rollout.engine.sglang_model_path,
            sglang_shared_path=config.rollout.engine.sglang_shared_path,
            request_timeout=config.rollout.engine.request_timeout,
        )
        self.reward_funcs = get_reward_funcs(self._build_reward_script_args())
        reward_weights = config.reward.reward_weights
        if reward_weights is not None:
            if len(reward_weights) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(reward_weights)}) must match number of reward functions ({len(self.reward_funcs)})"
                )
            reward_weight_tensor = torch.tensor(reward_weights, dtype=torch.float32)
        else:
            reward_weight_tensor = torch.ones(len(self.reward_funcs), dtype=torch.float32)
        self.register_buffer("reward_weights", reward_weight_tensor, persistent=False)
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
        )

    def on_fit_start(self) -> None:
        """Log static parameter counts once training starts."""

        if self.logger is None or not self.trainer.is_global_zero:
            return

        self.logger.log_metrics(
            {
                "model/trainable_parameters": float(self.trainable_parameter_count),
                "model/total_parameters": float(self.total_parameter_count),
            },
            step=self.global_step,
        )

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

        return compute_per_token_logps(
            model,
            input_ids,
            logits_to_keep,
            attention_mask=attention_mask,
            temperature=float(getattr(self.rollout_engine.generation_config, "temperature", 1.0) or 1.0),
        )

    def _decode_completion_ids(self, completion_texts: list[str]) -> tuple[list[str], list[list[dict[str, str]]]]:
        """Convert decoded completions into reward-compatible chat formats."""

        structured_completions = [[{"role": "assistant", "content": text}] for text in completion_texts]
        return completion_texts, structured_completions

    def _gather_tensor_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather same-shaped tensors across ranks for exact distributed metrics."""

        if self.trainer is None or getattr(self.trainer, "world_size", 1) <= 1:
            return tensor

        gathered = self.all_gather(tensor)
        if tensor.dim() == 0:
            return gathered.reshape(-1)
        return gathered.reshape(-1, *tensor.shape[1:])

    def _resolve_num_generations(self, training: bool) -> int:
        """Resolve the rollout multiplicity for train versus eval."""

        if training:
            return self.config.rollout.num_generations
        return self.config.rollout.num_generations_eval or self.config.rollout.num_generations

    @torch.no_grad()
    def _generate(self, batch: dict[str, Any], *, training: bool) -> dict[str, torch.Tensor | list[Any]]:
        """Generate grouped completions for online GRPO optimization."""

        num_generations = self._resolve_num_generations(training)
        rollout = self.rollout_engine.rollout(
            prompt_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_generations=num_generations,
        )
        completion_texts, structured_completions = self._decode_completion_ids(rollout.completions_text)

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
    def _emit_debug_samples(self) -> None:
        """Emit sample-level debug generations for configured prompts."""

        debug_config = self.config.rollout.debug
        if not debug_config.enabled or not debug_config.questions:
            return
        if not self.trainer.is_global_zero:
            return

        self.policy.eval()
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            generation_config = self.rollout_engine.generation_config
            if debug_config.generation_config_path is not None:
                generation_config = self.rollout_engine._load_generation_config(debug_config.generation_config_path)

            for index, question in enumerate(debug_config.questions):
                tokenized = self.tokenizer([question], return_tensors="pt", padding=True, truncation=True).to(self.device)
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
            reward_values = [value if value is not None else torch.nan for value in reward_values]
            reward_tensor = torch.tensor(reward_values, device=self.device, dtype=torch.float32)
            reward_matrix.append(reward_tensor)

        rewards_per_func = torch.stack(reward_matrix, dim=-1)
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=-1)
        return rewards, rewards_per_func

    def _compute_advantages(self, rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
        """Normalize rewards within each prompt group, as in GRPO."""

        grouped_rewards = rewards.view(-1, num_generations)
        grouped_mean = grouped_rewards.mean(dim=1, keepdim=True)
        grouped_std = grouped_rewards.std(dim=1, keepdim=True)
        grouped_advantages = (grouped_rewards - grouped_mean) / (grouped_std + self.config.rollout.advantage_epsilon)
        return grouped_advantages.reshape(-1)

    def _compute_loss(self, rollout_batch: dict[str, torch.Tensor | list[Any]], *, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the GRPO objective with PPO-style ratio clipping and optional KL penalty."""

        prompt_ids = rollout_batch["prompt_ids"]
        prompt_mask = rollout_batch["prompt_mask"]
        completion_ids = rollout_batch["completion_ids"]
        completion_mask = rollout_batch["completion_mask"]
        old_per_token_logps = rollout_batch["old_per_token_logps"]
        completion_truncated = rollout_batch["completion_truncated"]

        model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.shape[1]

        outputs = self.policy(input_ids=model_input_ids, attention_mask=model_attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :] / self.config.rollout.temperature
        per_token_logps = self._selective_log_softmax(logits, completion_ids)
        moe_metrics = collect_moe_metrics(outputs)

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
        global_rewards_per_func = self._gather_tensor_for_metrics(rewards_per_func.detach())
        global_rewards = (
                global_rewards_per_func * self.reward_weights.to(global_rewards_per_func.device).unsqueeze(0)
        ).nansum(dim=-1)
        num_generations = self._resolve_num_generations(training)
        global_advantages = self._compute_advantages(global_rewards, num_generations)
        advantages = self._compute_advantages(rewards, num_generations).unsqueeze(1)

        log_ratio = per_token_logps - old_per_token_logps
        importance_ratio = torch.exp(log_ratio)
        clip_eps = getattr(self.config.rollout, "epsilon", 0.2)
        is_low_clipped = torch.zeros_like(per_token_logps, dtype=torch.bool)
        is_high_clipped = torch.zeros_like(per_token_logps, dtype=torch.bool)
        is_region_clipped = torch.zeros_like(per_token_logps, dtype=torch.bool)
        is_cispo_clipped = torch.zeros_like(per_token_logps, dtype=torch.bool)
        if self.config.rollout.loss_type == "cispo":
            clipped_ratio = torch.clamp(importance_ratio, max=self.config.rollout.epsilon_high).detach()
            surrogate = clipped_ratio * advantages * per_token_logps
            is_cispo_clipped = (importance_ratio > self.config.rollout.epsilon_high) & (advantages > 0)
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
        per_token_loss = -(surrogate - self.config.rollout.kl_beta * per_token_kl)
        loss = masked_mean(per_token_loss, loss_mask)

        with torch.no_grad():
            entropy = entropy_from_logits(logits)
            completion_lengths = completion_mask.sum(dim=1).float()
            global_reward_group_std = global_rewards.view(-1, num_generations).std(dim=1)

            global_loss_mask = self._gather_tensor_for_metrics(loss_mask.detach())
            global_per_token_kl = self._gather_tensor_for_metrics(per_token_kl.detach())
            global_entropy = self._gather_tensor_for_metrics(entropy.detach())
            global_completion_lengths = self._gather_tensor_for_metrics(completion_lengths.detach())
            global_completion_truncated = self._gather_tensor_for_metrics(completion_truncated.to(torch.float32))

            terminated_lengths = global_completion_lengths[global_completion_truncated == 0]
            if terminated_lengths.numel() == 0:
                terminated_lengths = global_completion_lengths.new_zeros(1)

            global_is_low_clipped = self._gather_tensor_for_metrics(is_low_clipped.to(per_token_logps.dtype))
            global_is_high_clipped = self._gather_tensor_for_metrics(is_high_clipped.to(per_token_logps.dtype))
            global_is_region_clipped = self._gather_tensor_for_metrics(is_region_clipped.to(per_token_logps.dtype))
            global_is_cispo_clipped = self._gather_tensor_for_metrics(is_cispo_clipped.to(per_token_logps.dtype))

        metrics = {
            "reward": global_rewards.mean(),
            "reward_std": global_rewards.std(unbiased=False),
            "advantage_mean": global_advantages.mean(),
            "advantage_std": global_advantages.std(unbiased=False),
            "frac_reward_zero_std": (global_reward_group_std < 1.0e-6).float().mean(),
            "kl": masked_mean(global_per_token_kl, global_loss_mask),
            "entropy": masked_mean(global_entropy, global_loss_mask),
            "completion_length": global_completion_lengths.mean(),
            "completion_length_min": global_completion_lengths.min(),
            "completion_length_max": global_completion_lengths.max(),
            "completion_clipped_ratio": global_completion_truncated.mean(),
            "terminated_length_mean": terminated_lengths.mean(),
            "terminated_length_min": terminated_lengths.min(),
            "terminated_length_max": terminated_lengths.max(),
            "clip_ratio_low": masked_mean(global_is_low_clipped, global_loss_mask),
            "clip_ratio_high": masked_mean(global_is_high_clipped, global_loss_mask),
            "clip_ratio_region": masked_mean(global_is_region_clipped, global_loss_mask),
            "cispo_clip_ratio": masked_mean(global_is_cispo_clipped, global_loss_mask),
        }
        metrics.update(moe_metrics)
        for index, reward_name in enumerate(self.config.reward.reward_funcs):
            metrics[f"reward/{reward_name}"] = global_rewards_per_func[:, index].mean()
            metrics[f"reward_std/{reward_name}"] = global_rewards_per_func[:, index].std(unbiased=False)

        return loss, metrics

    def _log_metrics(self, prefix: str, loss: torch.Tensor, metrics: dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool) -> None:
        """Log the standard GRPO optimization metrics."""

        self.log(f"{prefix}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/reward", metrics["reward"], prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/reward_std", metrics["reward_std"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/frac_reward_zero_std", metrics["frac_reward_zero_std"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/advantage_mean", metrics["advantage_mean"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/advantage_std", metrics["advantage_std"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/kl", metrics["kl"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/entropy", metrics["entropy"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        log_moe_metrics(self, metrics, prefix, on_step=on_step, on_epoch=on_epoch)
        self.log(f"{prefix}/completions/mean_length", metrics["completion_length"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/completions/min_length", metrics["completion_length_min"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/completions/max_length", metrics["completion_length_max"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/completions/clipped_ratio", metrics["completion_clipped_ratio"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/completions/mean_terminated_length", metrics["terminated_length_mean"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/completions/min_terminated_length", metrics["terminated_length_min"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        self.log(f"{prefix}/completions/max_terminated_length", metrics["terminated_length_max"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        if self.config.rollout.loss_type == "cispo":
            self.log(f"{prefix}/cispo_clip_ratio", metrics["cispo_clip_ratio"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        else:
            self.log(f"{prefix}/clip_ratio/low", metrics["clip_ratio_low"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
            self.log(f"{prefix}/clip_ratio/high", metrics["clip_ratio_high"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
            self.log(f"{prefix}/clip_ratio/region", metrics["clip_ratio_region"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        for reward_name in self.config.reward.reward_funcs:
            self.log(f"{prefix}/rewards/{reward_name}/mean", metrics[f"reward/{reward_name}"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
            self.log(f"{prefix}/rewards/{reward_name}/std", metrics[f"reward_std/{reward_name}"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one online rollout and optimization step."""

        debug_every = self.config.rollout.debug.every_n_steps
        if self.config.rollout.debug.enabled and debug_every > 0 and self.global_step % debug_every == 0:
            self._emit_debug_samples()

        rollout_batch = self._generate(batch, training=True)
        loss, metrics = self._compute_loss(rollout_batch, training=True)
        self._log_metrics("train", loss, metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Evaluate the current policy with a rollout batch."""

        rollout_batch = self._generate(batch, training=False)
        loss, metrics = self._compute_loss(rollout_batch, training=False)
        self._log_metrics("val", loss, metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and scheduler for Lightning."""

        optimizer = build_optimizer(self.policy.parameters(), self.config.optimization)
        scheduler = build_scheduler(optimizer, self.config.optimization, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_batch_end(self, outputs: Any, batch: dict[str, Any], batch_idx: int) -> None:
        """Sync rollout backend after optimizer updates."""

        if self.config.rollout.engine.engine_type == "policy":
            self.rollout_engine.update_policy(self.policy)

    def on_train_end(self) -> None:
        """Export a Hugging Face-compatible model directory after training."""

        if not self.trainer.is_global_zero:
            return

        export_dir = self.config.output_dir + "/hf_final"
        exported_paths = export_configured_model(self.policy, self.config.model, export_dir, tokenizer=self.tokenizer)
        if exported_paths:
            rank_zero_info(f"Exported model artifacts to {export_dir}: {sorted(exported_paths)}")
