"""Metric aggregation helpers for GRPO training."""

from __future__ import annotations

from typing import Any

import torch

from lightning_grpo.models.common import masked_mean
from lightning_grpo.utils.modeling import log_moe_metrics


class GRPOMetricsAggregator:
    """Aggregate distributed metrics and log them through Lightning."""

    def __init__(self, module: Any) -> None:
        self.module = module

    def gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        trainer = self.module.trainer
        if trainer is None or getattr(trainer, "world_size", 1) <= 1:
            return tensor

        gathered = self.module.all_gather(tensor)
        if tensor.dim() == 0:
            return gathered.reshape(-1)
        return gathered.reshape(-1, *tensor.shape[1:])

    def build_training_metrics(
        self,
        *,
        global_rewards_per_func: torch.Tensor,
        reward_weights: torch.Tensor,
        num_generations: int,
        global_per_token_kl: torch.Tensor,
        global_loss_mask: torch.Tensor,
        global_entropy: torch.Tensor,
        global_completion_lengths: torch.Tensor,
        global_completion_truncated: torch.Tensor,
        global_is_low_clipped: torch.Tensor,
        global_is_high_clipped: torch.Tensor,
        global_is_region_clipped: torch.Tensor,
        global_is_cispo_clipped: torch.Tensor,
        global_advantages: torch.Tensor,
        moe_metrics: dict[str, torch.Tensor],
        reward_names: list[str],
    ) -> dict[str, torch.Tensor]:
        global_rewards = (global_rewards_per_func * reward_weights.to(global_rewards_per_func.device).unsqueeze(0)).nansum(dim=-1)
        global_reward_group_std = global_rewards.view(-1, num_generations).std(dim=1)

        terminated_lengths = global_completion_lengths[global_completion_truncated == 0]
        if terminated_lengths.numel() == 0:
            terminated_lengths = global_completion_lengths.new_zeros(1)

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
        for index, reward_name in enumerate(reward_names):
            metrics[f"reward/{reward_name}"] = global_rewards_per_func[:, index].mean()
            metrics[f"reward_std/{reward_name}"] = global_rewards_per_func[:, index].std(unbiased=False)
        return metrics

    def log_metrics(self, prefix: str, loss: torch.Tensor, metrics: dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool) -> None:
        module = self.module
        module.log(f"{prefix}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/reward", metrics["reward"], prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/reward_std", metrics["reward_std"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/frac_reward_zero_std", metrics["frac_reward_zero_std"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/advantage_mean", metrics["advantage_mean"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/advantage_std", metrics["advantage_std"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/kl", metrics["kl"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/entropy", metrics["entropy"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        log_moe_metrics(module, metrics, prefix, on_step=on_step, on_epoch=on_epoch)
        module.log(f"{prefix}/completions/mean_length", metrics["completion_length"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/completions/min_length", metrics["completion_length_min"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/completions/max_length", metrics["completion_length_max"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/completions/clipped_ratio", metrics["completion_clipped_ratio"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/completions/mean_terminated_length", metrics["terminated_length_mean"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/completions/min_terminated_length", metrics["terminated_length_min"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        module.log(f"{prefix}/completions/max_terminated_length", metrics["terminated_length_max"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        if module.config.rollout.loss_type == "cispo":
            module.log(f"{prefix}/cispo_clip_ratio", metrics["cispo_clip_ratio"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        else:
            module.log(f"{prefix}/clip_ratio/low", metrics["clip_ratio_low"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
            module.log(f"{prefix}/clip_ratio/high", metrics["clip_ratio_high"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
            module.log(f"{prefix}/clip_ratio/region", metrics["clip_ratio_region"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        for reward_name in module.config.reward.active.reward_funcs:
            module.log(f"{prefix}/rewards/{reward_name}/mean", metrics[f"reward/{reward_name}"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
            module.log(f"{prefix}/rewards/{reward_name}/std", metrics[f"reward_std/{reward_name}"], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
