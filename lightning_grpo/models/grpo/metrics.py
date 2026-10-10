"""Metric aggregation helpers for GRPO training."""

from __future__ import annotations

from typing import Sequence

import torch

from lightning_grpo.models.grpo.contracts import MetricsSink
from lightning_grpo.models.grpo.loss import masked_mean


def _missing_reward_to_zero(values: torch.Tensor) -> torch.Tensor:
    """Map missing (NaN) reward entries to 0."""

    return torch.nan_to_num(values, nan=0.0)


class GRPOMetricsAggregator:
    """Aggregate GRPO metrics and publish them through an injected sink.

    Only the two things it actually needs are injected: a :class:`MetricsSink` to publish
    the metrics and the clip/reward names that shape the logged dict. Distributed gathering
    lives on :class:`~lightning_grpo.models.grpo.contracts.DistributedContext` and is
    performed by the loss computer, which is what owns the gathered tensors.
    """

    def __init__(
        self,
        sink: MetricsSink,
        *,
        clip_reporting: str,
        reward_names: Sequence[str],
    ) -> None:
        self.sink = sink
        self.clip_reporting = clip_reporting
        self.reward_names = list(reward_names)

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
        global_shaped_rewards: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # ``global_shaped_rewards`` carries reward-shaping terms that are not part of any
        # individual reward function (e.g. DAPO's overlong length penalty), so the loss
        # and the logged reward agree.
        if global_shaped_rewards is None:
            global_shaped_rewards = (
                global_rewards_per_func * reward_weights.to(global_rewards_per_func.device).unsqueeze(0)
            ).nansum(dim=-1)
        global_rewards = global_shaped_rewards
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
        for index, reward_name in enumerate(self.reward_names):
            # A missing reward is NaN for that (sample, function) pair; scoring it as 0 here
            # matches the ``nansum`` used for the total reward above.
            per_func = _missing_reward_to_zero(global_rewards_per_func[:, index])
            metrics[f"reward/{reward_name}"] = per_func.mean()
            metrics[f"reward_std/{reward_name}"] = per_func.std(unbiased=False)
        return metrics

    def log_metrics(self, prefix: str, loss: torch.Tensor, metrics: dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool) -> None:
        """Log the loss plus the already-aggregated metrics."""

        sink = self.sink
        sink.log(f"{prefix}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        sink.log(f"{prefix}/reward", metrics["reward"], prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=False)

        # Bulk-log the remaining metrics through a single call to minimize logger overhead.
        logged: dict[str, torch.Tensor] = {
            f"{prefix}/reward_std": metrics["reward_std"],
            f"{prefix}/frac_reward_zero_std": metrics["frac_reward_zero_std"],
            f"{prefix}/advantage_mean": metrics["advantage_mean"],
            f"{prefix}/advantage_std": metrics["advantage_std"],
            f"{prefix}/kl": metrics["kl"],
            f"{prefix}/entropy": metrics["entropy"],
            f"{prefix}/completions/mean_length": metrics["completion_length"],
            f"{prefix}/completions/min_length": metrics["completion_length_min"],
            f"{prefix}/completions/max_length": metrics["completion_length_max"],
            f"{prefix}/completions/clipped_ratio": metrics["completion_clipped_ratio"],
            f"{prefix}/completions/mean_terminated_length": metrics["terminated_length_mean"],
            f"{prefix}/completions/min_terminated_length": metrics["terminated_length_min"],
            f"{prefix}/completions/max_terminated_length": metrics["terminated_length_max"],
        }
        if self.clip_reporting == "cispo":
            logged[f"{prefix}/cispo_clip_ratio"] = metrics["cispo_clip_ratio"]
        else:
            logged[f"{prefix}/clip_ratio/low"] = metrics["clip_ratio_low"]
            logged[f"{prefix}/clip_ratio/high"] = metrics["clip_ratio_high"]
            logged[f"{prefix}/clip_ratio/region"] = metrics["clip_ratio_region"]
        for reward_name in self.reward_names:
            logged[f"{prefix}/rewards/{reward_name}/mean"] = metrics[f"reward/{reward_name}"]
            logged[f"{prefix}/rewards/{reward_name}/std"] = metrics[f"reward_std/{reward_name}"]

        sink.log_dict(logged, on_step=on_step, on_epoch=on_epoch, sync_dist=False)
