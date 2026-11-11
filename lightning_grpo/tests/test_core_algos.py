"""Unit tests for the vendored GRPO/PPO math in ``lightning_grpo.utils.loss``.

These cover the pieces the training objective is built from: the advantage estimators,
the policy-loss surrogates and the token/sequence loss aggregation. They are pure tensor
tests, so they run on CPU in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lightning_grpo.utils.loss import core_algos as algos
from lightning_grpo.utils.loss import functional as loss_functional


def outcome_rewards(values: list[float]) -> torch.Tensor:
    """One scalar reward per completion, as the loss computer feeds the estimators."""

    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)


class TestGrpoAdvantage:
    def test_normalizes_within_the_prompt_group(self) -> None:
        rewards = outcome_rewards([1.0, 1.0, 0.0, 4.0])
        mask = torch.ones_like(rewards)
        index = np.array([0, 0, 1, 1])

        advantages, returns = algos.compute_grpo_outcome_advantage(rewards, mask, index, epsilon=1e-6)

        expected = torch.tensor([[0.0], [0.0], [-2.0], [2.0]]) / torch.sqrt(torch.tensor(8.0))
        assert torch.allclose(advantages, expected, atol=1e-5)
        assert torch.allclose(returns, advantages)

    def test_std_scaling_can_be_disabled_for_dr_grpo(self) -> None:
        rewards = outcome_rewards([1.0, 1.0, 0.0, 4.0])
        mask = torch.ones_like(rewards)
        index = np.array([0, 0, 1, 1])

        advantages, _ = algos.compute_grpo_outcome_advantage(
            rewards, mask, index, norm_adv_by_std_in_grpo=False
        )

        # Group mean is subtracted but the group std no longer divides the advantage.
        assert torch.allclose(advantages, torch.tensor([[0.0], [0.0], [-2.0], [2.0]]))

    def test_single_sample_group_keeps_its_raw_reward(self) -> None:
        rewards = outcome_rewards([1.5])
        advantages, _ = algos.compute_grpo_outcome_advantage(
            rewards, torch.ones_like(rewards), np.array([0])
        )

        assert torch.allclose(advantages, torch.tensor([[1.5]]))

    def test_response_mask_zeros_padding_tokens(self) -> None:
        rewards = torch.tensor([[1.0, 0.0], [0.5, 0.0]])
        mask = torch.tensor([[1, 0], [1, 0]])
        index = np.array([0, 0])

        advantages, _ = algos.compute_grpo_outcome_advantage(rewards, mask, index, epsilon=1e-6)

        assert torch.allclose(advantages[:, 1], torch.zeros(2))
        assert advantages[0, 0] > 0 > advantages[1, 0]
        assert torch.allclose(advantages[:, 0], torch.tensor([0.7071068, -0.7071068]), atol=1e-5)


class TestOtherOutcomeEstimators:
    def test_rloo_uses_the_leave_one_out_mean(self) -> None:
        rewards = outcome_rewards([1.0, 2.0, 3.0, 4.0])

        advantages, _ = algos.compute_rloo_outcome_advantage(
            rewards, torch.ones_like(rewards), np.array([0, 0, 0, 0])
        )

        # r_i - mean(others) = (r_i - group_mean) * k / (k - 1) with mean 2.5.
        expected = torch.tensor([[-2.0], [-2.0 / 3.0], [2.0 / 3.0], [2.0]])
        assert torch.allclose(advantages, expected, atol=1e-6)

    def test_reinforce_plus_plus_baseline_centers_and_whitens(self) -> None:
        rewards = outcome_rewards([1.0, 2.0, 3.0, 4.0])

        advantages, _ = algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            rewards, torch.ones_like(rewards), index=np.array([0, 0, 0, 0])
        )

        centered = torch.tensor([-1.5, -0.5, 0.5, 1.5])
        assert torch.allclose(advantages[:, 0], centered / torch.sqrt(torch.tensor(1.25)), atol=1e-5)
        assert advantages.mean().abs() < 1e-6

    def test_reinforce_plus_plus_propagates_discounted_returns(self) -> None:
        rewards = torch.tensor([[0.0, 1.0, 0.0]])
        returns_holder = algos.compute_reinforce_plus_plus_outcome_advantage(
            rewards,
            torch.ones_like(rewards),
            config=algos.ActorConfig(gamma=0.5),
        )[1]

        assert torch.allclose(returns_holder, torch.tensor([[0.5, 1.0, 0.0]]))

    def test_remax_subtracts_the_greedy_baseline(self) -> None:
        rewards = torch.tensor([[0.0, 0.0, 2.0]])

        advantages, returns = algos.compute_remax_outcome_advantage(
            rewards, torch.tensor([1.0]), torch.ones_like(rewards)
        )

        assert torch.allclose(returns, torch.tensor([[2.0, 2.0, 2.0]]))
        assert torch.allclose(advantages, torch.tensor([[1.0, 1.0, 1.0]]))

    def test_grpo_passk_rewards_only_the_best_completion(self) -> None:
        rewards = outcome_rewards([1.0, 2.0, 3.0])

        advantages, _ = algos.compute_grpo_passk_outcome_advantage(
            rewards, torch.ones_like(rewards), np.array([0, 0, 0]), config=algos.ActorConfig()
        )

        # std([1, 2, 3]) == 1, so the best completion gets r_max - r_second_max.
        assert torch.allclose(advantages, torch.tensor([[0.0], [0.0], [1.0]]), atol=1e-5)

    def test_grpo_passk_requires_two_samples_per_group(self) -> None:
        rewards = outcome_rewards([1.0])

        with pytest.raises(ValueError, match="at least 2"):
            algos.compute_grpo_passk_outcome_advantage(
                rewards, torch.ones_like(rewards), np.array([0]), config=algos.ActorConfig()
            )

    def test_gpg_scales_by_f_norm(self) -> None:
        rewards = outcome_rewards([1.0, 3.0])

        advantages, _ = algos.compute_gpg_outcome_advantage(
            rewards, torch.ones_like(rewards), np.array([0, 0]), f_norm=4.0
        )

        # (r - group_mean) / f_norm; alpha is re-derived from the zero-reward ratio (= 1).
        assert torch.allclose(advantages, torch.tensor([[-0.25], [0.25]]))

    def test_gpg_upscales_when_some_rewards_are_zero(self) -> None:
        rewards = outcome_rewards([0.0, 2.0])

        advantages, _ = algos.compute_gpg_outcome_advantage(
            rewards, torch.ones_like(rewards), np.array([0, 0])
        )

        # alpha = batch_size / non_zero_count = 2, so the signal is scaled up twice.
        assert torch.allclose(advantages, torch.tensor([[-2.0], [2.0]]))

    def test_opo_weights_the_baseline_by_sequence_length(self) -> None:
        rewards = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        mask = torch.tensor([[1, 0], [1, 1]])

        advantages, _ = algos.compute_opo_outcome_advantage(rewards, mask, np.array([0, 0]))

        # baseline = (1 * 1 + 2 * 4) / (1 + 2) = 3
        assert torch.allclose(advantages, torch.tensor([[-2.0, 0.0], [1.0, 1.0]]))


class TestAggLoss:
    loss_mat = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    loss_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    def test_token_mean_divides_by_the_valid_token_count(self) -> None:
        loss = algos.agg_loss(self.loss_mat, self.loss_mask, "token-mean")

        assert loss == pytest.approx(4.0 / 3.0)

    def test_token_sum_keeps_the_raw_sum(self) -> None:
        assert algos.agg_loss(self.loss_mat, self.loss_mask, "token-sum") == pytest.approx(4.0)

    def test_seq_mean_token_sum_averages_sequence_sums(self) -> None:
        loss = algos.agg_loss(self.loss_mat, self.loss_mask, "seq-mean-token-sum")

        assert loss == pytest.approx(2.0)

    def test_seq_mean_token_sum_norm_divides_by_the_horizon(self) -> None:
        default = algos.agg_loss(self.loss_mat, self.loss_mask, "seq-mean-token-sum-norm")
        explicit = algos.agg_loss(
            self.loss_mat, self.loss_mask, "seq-mean-token-sum-norm", loss_scale_factor=4
        )

        assert default == pytest.approx(2.0 / 3.0)
        assert explicit == pytest.approx(0.5)

    def test_seq_mean_token_mean_averages_per_sequence_token_means(self) -> None:
        loss = algos.agg_loss(self.loss_mat, self.loss_mask, "seq-mean-token-mean")

        # First sequence averages to 1.0, the second (a single valid token) to 2.0.
        assert loss == pytest.approx(1.5)

    def test_fully_masked_sequences_are_excluded(self) -> None:
        loss_mat = torch.cat([self.loss_mat, torch.zeros(1, 3)])
        loss_mask = torch.cat([self.loss_mask, torch.zeros(1, 3, dtype=torch.long)])

        assert algos.agg_loss(loss_mat, loss_mask, "seq-mean-token-sum") == pytest.approx(2.0)
        assert algos.agg_loss(loss_mat, loss_mask, "token-mean") == pytest.approx(4.0 / 3.0)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid loss_agg_mode"):
            algos.agg_loss(self.loss_mat, self.loss_mask, "mean")

    def test_data_parallel_scaling_requires_global_counts(self) -> None:
        with pytest.raises(ValueError, match="batch_num_tokens"):
            algos.agg_loss(self.loss_mat, self.loss_mask, "token-mean", dp_size=2)

        scaled = algos.agg_loss(
            self.loss_mat, self.loss_mask, "token-mean", dp_size=2, batch_num_tokens=3
        )
        assert scaled == pytest.approx(4.0 / 3.0 * 2)


class TestVanillaPolicyLoss:
    def test_ratio_one_reduces_to_the_advantage(self) -> None:
        log_prob = torch.full((1, 2), -0.7)
        advantages = torch.tensor([[2.0, 1.0]])

        loss, metrics = algos.compute_policy_loss_vanilla(
            old_log_prob=log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=torch.ones_like(log_prob),
            config=algos.ActorConfig(),
        )

        assert loss == pytest.approx(-1.5, rel=1e-6)
        assert metrics["actor/pg_clipfrac"] == 0.0

    def test_upper_clip_bounds_the_surrogate(self) -> None:
        old_log_prob = torch.zeros(1, 1)
        log_prob = torch.full((1, 1), float(np.log(2.0)))

        loss, metrics = algos.compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=torch.ones(1, 1),
            response_mask=torch.ones(1, 1),
            config=algos.ActorConfig(clip_ratio=0.2),
        )

        assert loss == pytest.approx(-1.2, rel=1e-6)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0)

    def test_dual_clip_bounds_the_negative_advantage(self) -> None:
        old_log_prob = torch.zeros(1, 1)
        log_prob = torch.full((1, 1), float(np.log(5.0)))

        loss, _ = algos.compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=torch.full((1, 1), -1.0),
            response_mask=torch.ones(1, 1),
            config=algos.ActorConfig(clip_ratio=0.2, clip_ratio_c=3.0),
        )

        # -A * ratio would be 5; dual-clip PPO caps it at -A * clip_ratio_c = 3.
        assert loss == pytest.approx(3.0, rel=1e-6)

    def test_masked_tokens_are_ignored(self) -> None:
        log_prob = torch.full((1, 2), -0.5)
        advantages = torch.tensor([[4.0, 100.0]])
        mask = torch.tensor([[1, 0]])

        loss, _ = algos.compute_policy_loss_vanilla(
            old_log_prob=log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=mask,
            config=algos.ActorConfig(),
        )

        assert loss == pytest.approx(-4.0, rel=1e-6)

    def test_clip_ratio_c_must_exceed_one(self) -> None:
        tensor = torch.zeros(1, 1)
        with pytest.raises(AssertionError, match="clip_ratio_c"):
            algos.compute_policy_loss_vanilla(
                old_log_prob=tensor,
                log_prob=tensor,
                advantages=tensor,
                response_mask=torch.ones(1, 1),
                config=algos.ActorConfig(clip_ratio_c=0.5),
            )


class TestPolicyLossVariants:
    def test_gspo_with_an_unchanged_policy_matches_the_advantage(self) -> None:
        log_prob = torch.full((2, 3), -0.4)
        advantages = torch.tensor([[1.0, 1.0, 1.0], [-0.5, -0.5, -0.5]])

        loss, metrics = algos.compute_policy_loss_gspo(
            old_log_prob=log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=torch.ones_like(log_prob),
            loss_agg_mode="seq-mean-token-mean",
            config=algos.ActorConfig(),
        )

        assert loss == pytest.approx(-0.25, rel=1e-6)
        assert metrics["actor/pg_clipfrac"] == 0.0

    def test_cispo_detaches_the_importance_weight(self) -> None:
        old_log_prob = torch.full((1, 2), -0.5)
        log_prob = torch.full((1, 2), -0.5, requires_grad=True)
        advantages = torch.tensor([[2.0, 2.0]])

        loss, metrics = algos.compute_policy_loss_cispo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=torch.ones_like(log_prob),
            config=algos.ActorConfig(clip_ratio=0.2),
        )

        # L = -sg(clip(ratio)) * A * log_prob, and ratio == 1 here.
        assert loss.item() == pytest.approx(1.0, rel=1e-6)
        assert metrics["actor/pg_clipfrac"] == 0.0

        loss.backward()
        # The gradient flows only through log_prob: dL/dlogpi = -A / token_count.
        assert torch.allclose(log_prob.grad, torch.full((1, 2), -1.0), atol=1e-6)

    def test_cispo_clips_the_detached_weight(self) -> None:
        old_log_prob = torch.zeros(1, 1)
        log_prob = torch.zeros(1, 1, requires_grad=True)

        loss, metrics = algos.compute_policy_loss_cispo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=torch.ones(1, 1),
            response_mask=torch.ones(1, 1),
            config=algos.ActorConfig(clip_ratio=0.2),
        )

        # log_prob == 0 makes the objective vanish, but the ratio (1.0) is inside the clip.
        assert loss.item() == pytest.approx(0.0)
        assert metrics["actor/pg_clipfrac"] == 0.0

    def test_reinforce_uses_log_probability_directly(self) -> None:
        log_prob = torch.full((1, 2), -0.5)
        advantages = torch.tensor([[2.0, 2.0]])

        loss, _ = algos.compute_policy_loss_reinforce(
            rollout_log_prob=log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=torch.ones_like(log_prob),
            loss_agg_mode="seq-mean-token-sum",
            config=algos.ActorConfig(),
        )

        assert loss == pytest.approx(2.0, rel=1e-6)


class TestRegistries:
    def test_policy_loss_lookup(self) -> None:
        assert algos.get_policy_loss_fn("vanilla") is algos.compute_policy_loss_vanilla

    def test_unknown_policy_loss_lists_the_supported_ones(self) -> None:
        with pytest.raises(ValueError, match="Supported modes"):
            algos.get_policy_loss_fn("nope")

    def test_advantage_estimator_lookup(self) -> None:
        assert algos.get_adv_estimator_fn("grpo") is algos.compute_grpo_outcome_advantage
        assert algos.get_adv_estimator_fn(algos.AdvantageEstimator.RLOO) is algos.compute_rloo_outcome_advantage

    def test_unknown_advantage_estimator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown advantage estimator"):
            algos.get_adv_estimator_fn("nope")

    def test_actor_config_none_attributes_fall_back_to_defaults(self) -> None:
        config = algos.ActorConfig(clip_ratio_high=None, extra={"f_norm": 2.0})

        assert config.get("clip_ratio_high", 0.28) == 0.28
        assert config.get("f_norm", 1.0) == 2.0
        assert config.get("missing", "fallback") == "fallback"


class TestMaskedHelpers:
    def test_masked_mean_of_a_fully_masked_tensor_is_zero(self) -> None:
        values = torch.tensor([[1.0, 2.0]])

        assert loss_functional.masked_mean(values, torch.zeros_like(values)) == 0.0

    def test_masked_whiten_standardizes_under_the_mask(self) -> None:
        values = torch.tensor([[1.0, 2.0, 100.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        whitened = loss_functional.masked_whiten(values, mask)

        valid = whitened[mask.bool()]
        assert valid.mean().abs() < 1e-6
        assert valid.var(unbiased=False) == pytest.approx(1.0, rel=1e-5)

    def test_clip_by_value_broadcasts_tensor_bounds(self) -> None:
        values = torch.tensor([[0.5, 2.0], [3.0, 4.0]])
        lower = torch.tensor([[1.0], [2.0]])
        upper = torch.tensor([[2.5], [3.5]])

        clipped = loss_functional.clip_by_value(values, lower, upper)

        assert torch.allclose(clipped, torch.tensor([[1.0, 2.0], [3.0, 3.5]]))

    def test_entropy_from_logits_matches_a_manual_computation(self) -> None:
        logits = torch.tensor([[0.0, 0.0], [10.0, 0.0]])

        entropy = loss_functional.entropy_from_logits(logits)

        probabilities = torch.softmax(logits, dim=-1)
        expected = -(probabilities * torch.log(probabilities)).sum(dim=-1)
        assert entropy[0] == pytest.approx(float(np.log(2.0)), rel=1e-6)
        assert torch.allclose(entropy, expected, atol=1e-6)
