"""Unit tests for GRPO metric aggregation and logging."""

from __future__ import annotations

import pytest
import torch

from lightning_grpo.models.grpo.metrics import GRPOMetricsAggregator
from lightning_grpo.tests.support import RecordingSink

BATCH = 4
SEQ = 2


def build_inputs(**overrides) -> dict:
    """Inputs for ``build_training_metrics`` with hand-checkable values."""

    inputs = {
        "global_rewards_per_func": torch.tensor(
            [[1.0, 0.5], [0.5, float("nan")], [0.0, 0.25], [1.0, 0.5]]
        ),
        "reward_weights": torch.tensor([2.0, 1.0]),
        "num_generations": 2,
        "global_per_token_kl": torch.tensor([[0.1, 0.3], [0.2, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        "global_loss_mask": torch.tensor([[1, 1], [1, 0], [1, 0], [1, 0]]),
        "global_entropy": torch.full((BATCH, SEQ), 2.0),
        "global_completion_lengths": torch.tensor([4.0, 2.0, 3.0, 5.0]),
        "global_completion_truncated": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "global_is_low_clipped": torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        "global_is_high_clipped": torch.zeros(BATCH, SEQ),
        "global_is_region_clipped": torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        "global_is_cispo_clipped": torch.zeros(BATCH, SEQ),
        "global_advantages": torch.tensor([1.0, -1.0, 0.5, -0.5]),
        "global_shaped_rewards": torch.tensor([1.0, 1.0, 0.5, 2.5]),
    }
    inputs.update(overrides)
    return inputs


def make_aggregator(sink: RecordingSink, *, clip_reporting: str = "ratio") -> GRPOMetricsAggregator:
    return GRPOMetricsAggregator(sink, clip_reporting=clip_reporting, reward_names=["accuracy", "format"])


class TestBuildTrainingMetrics:
    def test_reward_statistics_come_from_the_shaped_rewards(self) -> None:
        metrics = make_aggregator(RecordingSink()).build_training_metrics(**build_inputs())

        assert metrics["reward"].item() == pytest.approx(1.25)
        assert metrics["reward_std"].item() == pytest.approx(0.75)
        # The first prompt group has an identical reward, the second one does not.
        assert metrics["frac_reward_zero_std"].item() == pytest.approx(0.5)

    def test_per_function_rewards_treat_missing_values_as_zero(self) -> None:
        metrics = make_aggregator(RecordingSink()).build_training_metrics(**build_inputs())

        # accuracy: [1.0, 0.5, 0.0, 1.0]; format: [0.5, nan -> 0.0, 0.25, 0.5]
        assert metrics["reward/accuracy"].item() == pytest.approx(0.625)
        assert metrics["reward/format"].item() == pytest.approx(0.3125)

    def test_falls_back_to_re_aggregating_the_per_function_rewards(self) -> None:
        inputs = build_inputs(global_shaped_rewards=None)

        metrics = make_aggregator(RecordingSink()).build_training_metrics(**inputs)

        # nansum(per_func * weights) per row == [2.5, 1.0, 0.25, 2.5]
        assert metrics["reward"].item() == pytest.approx(1.5625)

    def test_masked_means_ignore_padding(self) -> None:
        metrics = make_aggregator(RecordingSink()).build_training_metrics(**build_inputs())

        assert metrics["kl"].item() == pytest.approx((0.1 + 0.3 + 0.2) / 5)
        assert metrics["entropy"].item() == pytest.approx(2.0)
        # One of the five valid tokens carries the clipping indicator.
        assert metrics["clip_ratio_low"].item() == pytest.approx(1.0 / 5.0)
        assert metrics["clip_ratio_region"].item() == pytest.approx(1.0 / 5.0)
        assert metrics["clip_ratio_high"].item() == 0.0

    def test_completion_length_statistics_separate_truncated_sequences(self) -> None:
        metrics = make_aggregator(RecordingSink()).build_training_metrics(**build_inputs())

        assert metrics["completion_length"].item() == pytest.approx(3.5)
        assert metrics["completion_length_min"].item() == pytest.approx(2.0)
        assert metrics["completion_length_max"].item() == pytest.approx(5.0)
        assert metrics["completion_clipped_ratio"].item() == pytest.approx(0.25)
        assert metrics["terminated_length_mean"].item() == pytest.approx(4.0)
        assert metrics["terminated_length_min"].item() == pytest.approx(3.0)
        assert metrics["terminated_length_max"].item() == pytest.approx(5.0)

    def test_fully_truncated_batch_reports_zero_terminated_lengths(self) -> None:
        inputs = build_inputs(global_completion_truncated=torch.ones(BATCH))

        metrics = make_aggregator(RecordingSink()).build_training_metrics(**inputs)

        assert metrics["terminated_length_mean"].item() == 0.0
        assert metrics["completion_clipped_ratio"].item() == pytest.approx(1.0)

    def test_advantage_statistics(self) -> None:
        metrics = make_aggregator(RecordingSink()).build_training_metrics(**build_inputs())

        assert metrics["advantage_mean"].item() == pytest.approx(0.0)
        assert metrics["advantage_std"].item() == pytest.approx(0.790569, rel=1e-5)


class TestLogMetrics:
    def test_loss_is_logged_with_sync_and_prog_bar(self) -> None:
        sink = RecordingSink()
        aggregator = make_aggregator(sink)
        metrics = aggregator.build_training_metrics(**build_inputs())

        aggregator.log_metrics("train", torch.tensor(1.5), metrics, on_step=True, on_epoch=False)

        name, value, kwargs = sink.log_calls[0]
        assert name == "train/loss"
        assert value.item() == pytest.approx(1.5)
        assert kwargs["prog_bar"] is True
        assert kwargs["sync_dist"] is True
        # The aggregated metrics are already global, so they must not be synced again.
        assert sink.dict_calls[0][1]["sync_dist"] is False
        assert "train/advantage_mean" in sink.logged
        assert "train/rewards/accuracy/mean" in sink.logged
        assert "train/rewards/format/std" in sink.logged

    def test_ratio_reporting_logs_directional_clipping(self) -> None:
        sink = RecordingSink()
        aggregator = make_aggregator(sink, clip_reporting="ratio")
        metrics = aggregator.build_training_metrics(**build_inputs())

        aggregator.log_metrics("train", torch.tensor(0.0), metrics, on_step=True, on_epoch=True)

        assert "train/clip_ratio/low" in sink.logged
        assert "train/clip_ratio/high" in sink.logged
        assert "train/clip_ratio/region" in sink.logged
        assert "train/cispo_clip_ratio" not in sink.logged

    def test_cispo_reporting_logs_a_single_ratio(self) -> None:
        sink = RecordingSink()
        aggregator = make_aggregator(sink, clip_reporting="cispo")
        metrics = aggregator.build_training_metrics(**build_inputs())

        aggregator.log_metrics("train", torch.tensor(0.0), metrics, on_step=True, on_epoch=True)

        assert "train/cispo_clip_ratio" in sink.logged
        assert "train/clip_ratio/low" not in sink.logged
