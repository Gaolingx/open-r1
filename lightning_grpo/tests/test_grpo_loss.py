"""Unit tests for the GRPO loss orchestration in ``lightning_grpo.models.grpo.loss``.

The tests cover the three responsibilities that module owns: resolving a
``loss_type`` into a concrete (objective, advantage estimator, aggregation) triple,
shaping/gathering the rewards, and driving the objective for one rollout batch.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from lightning_grpo.models.grpo.contracts import DistributedContext
from lightning_grpo.models.grpo.loss import (
    GRPO_ALGORITHM_PRESETS,
    approx_kl_divergence,
    entropy_from_logits,
    masked_mean,
    materialize_vocab_parallel_logits,
    resolve_grpo_algorithm,
    selective_log_softmax,
)
from lightning_grpo.tests.support import (
    COMPLETION_LEN,
    NUM_GENERATIONS,
    FakeRewardManager,
    build_e2e_computer,
    build_loss_computer,
    make_rollout_batch,
    two_rank_context,
)
from lightning_grpo.utils.configs.grpo import GRPORolloutConfig
from lightning_grpo.utils.loss import core_algos as algos


def rollout_config(**overrides) -> GRPORolloutConfig:
    return GRPORolloutConfig(**overrides)


def completion_mask(lengths: list[int], horizon: int) -> torch.Tensor:
    return (torch.arange(horizon)[None, :] < torch.tensor(lengths)[:, None]).long()


class TestResolveGrpoAlgorithm:
    def test_every_preset_names_a_registered_objective_and_estimator(self) -> None:
        for name, preset in GRPO_ALGORITHM_PRESETS.items():
            assert preset.policy_loss in algos.POLICY_LOSS_REGISTRY, name
            assert preset.advantage_estimator in algos.ADV_ESTIMATOR_REGISTRY, name

    def test_default_config_resolves_to_cispo(self) -> None:
        algorithm = resolve_grpo_algorithm(rollout_config())

        assert algorithm.loss_type == "cispo"
        assert algorithm.preset.policy_loss == "cispo"
        assert algorithm.epsilon_low == pytest.approx(0.2)
        assert algorithm.epsilon_high == pytest.approx(5.0)
        assert algorithm.preset.clip_reporting == "cispo"

    def test_dapo_uses_clip_higher(self) -> None:
        algorithm = resolve_grpo_algorithm(rollout_config(loss_type="dapo"))

        assert algorithm.preset.policy_loss == "vanilla"
        assert algorithm.epsilon_low == pytest.approx(0.2)
        assert algorithm.epsilon_high == pytest.approx(0.28)
        assert algorithm.loss_agg_mode == "token-mean"
        assert algorithm.advantage_estimator == "grpo"

    def test_dr_grpo_drops_the_std_scaling(self) -> None:
        algorithm = resolve_grpo_algorithm(rollout_config(loss_type="dr_grpo"))

        assert algorithm.norm_adv_by_std_in_grpo is False
        assert algorithm.loss_agg_mode == "seq-mean-token-sum-norm"

    def test_gspo_aggregates_at_the_sequence_level(self) -> None:
        algorithm = resolve_grpo_algorithm(rollout_config(loss_type="gspo"))

        assert algorithm.loss_agg_mode == "seq-mean-token-mean"
        assert algorithm.preset.liger_supported is False

    def test_reinforce_pairs_with_a_batch_whitened_estimator(self) -> None:
        algorithm = resolve_grpo_algorithm(rollout_config(loss_type="reinforce"))

        assert algorithm.preset.policy_loss == "reinforce"
        assert algorithm.advantage_estimator == "reinforce_plus_plus"
        assert algorithm.loss_agg_mode == "seq-mean-token-sum"

    def test_explicit_epsilon_high_wins_over_the_preset(self) -> None:
        algorithm = resolve_grpo_algorithm(rollout_config(loss_type="dapo", epsilon_high=0.35))

        assert algorithm.epsilon_high == pytest.approx(0.35)
        assert algorithm.actor_config.clip_ratio_high == pytest.approx(0.35)

    def test_explicit_estimator_and_aggregation_override_the_preset(self) -> None:
        algorithm = resolve_grpo_algorithm(
            rollout_config(loss_type="dapo", advantage_estimator="rloo", loss_agg_mode="token-sum")
        )

        assert algorithm.advantage_estimator == "rloo"
        assert algorithm.loss_agg_mode == "token-sum"

    def test_actor_config_carries_the_algorithm_knobs(self) -> None:
        algorithm = resolve_grpo_algorithm(rollout_config(loss_type="dro", dro_beta=0.25, gpg_alpha=2.0))

        assert algorithm.actor_config.policy_loss.dro_beta == pytest.approx(0.25)
        assert algorithm.actor_config.extra["alpha"] == pytest.approx(2.0)

    def test_unknown_loss_type_lists_the_supported_algorithms(self) -> None:
        with pytest.raises(ValueError, match="Supported algorithms"):
            resolve_grpo_algorithm(rollout_config(loss_type="ppo"))

    def test_deprecated_bnpo_alias_resolves_to_grpo_with_a_warning(self) -> None:
        from lightning_grpo.models.grpo import loss as loss_module

        # The alias warning is emitted once per process; make the test order-independent.
        loss_module._WARNED_LOSS_TYPES.discard("bnpo")

        with pytest.warns(DeprecationWarning, match="bnpo"):
            algorithm = resolve_grpo_algorithm(rollout_config(loss_type="bnpo"))

        assert algorithm.loss_type == "grpo"

    def test_unknown_advantage_estimator_raises(self) -> None:
        with pytest.raises(ValueError, match="Supported:"):
            resolve_grpo_algorithm(rollout_config(advantage_estimator="not-an-estimator"))

    def test_norm_override_that_contradicts_the_preset_warns(self) -> None:
        with pytest.warns(UserWarning, match="expects norm_adv_by_std_in_grpo"):
            algorithm = resolve_grpo_algorithm(
                rollout_config(loss_type="dr_grpo", norm_adv_by_std_in_grpo=True)
            )

        assert algorithm.norm_adv_by_std_in_grpo is True


class TestOverlongPenalty:
    def test_disabled_by_default(self) -> None:
        computer, _ = build_loss_computer(rollout_config(max_completion_length=10))
        rewards = torch.tensor([-1.0, 0.0, 1.0])
        batch = {"completion_mask": completion_mask([10, 10, 10], horizon=10)}

        assert torch.equal(computer._apply_overlong_penalty(batch, rewards), rewards)

    def test_penalises_only_non_positive_rewards_beyond_the_buffer(self) -> None:
        computer, _ = build_loss_computer(
            rollout_config(max_completion_length=10, overlong_buffer_len=4, overlong_penalty_factor=1.0)
        )
        # Threshold is 10 - 4 = 6 tokens.
        batch = {"completion_mask": completion_mask([10, 8, 10, 10], horizon=10)}
        rewards = torch.tensor([1.0, -0.2, 0.0, 0.5])

        penalised = computer._apply_overlong_penalty(batch, rewards)

        # A correct-but-long answer keeps its reward; shorter/positive rows are untouched.
        assert penalised.tolist() == pytest.approx([1.0, -0.7, -1.0, 0.5])

    def test_penalty_scales_with_the_factor_and_is_clamped(self) -> None:
        computer, _ = build_loss_computer(
            rollout_config(max_completion_length=10, overlong_buffer_len=4, overlong_penalty_factor=2.0)
        )
        batch = {"completion_mask": completion_mask([10, 8], horizon=10)}

        penalised = computer._apply_overlong_penalty(batch, torch.tensor([0.0, -1.0]))

        assert penalised.tolist() == pytest.approx([-2.0, -2.0])


class TestRewardShapingAndGathering:
    def test_shaped_rewards_are_gathered_instead_of_re_aggregating_per_function(self) -> None:
        rollout = rollout_config(
            loss_type="dapo",
            max_completion_length=COMPLETION_LEN,
            overlong_buffer_len=2,
            kl_beta=0.0,
        )
        distributed, _ = two_rank_context()
        reward_manager = FakeRewardManager()
        computer, _ = build_loss_computer(rollout, reward_manager=reward_manager, distributed=distributed)
        batch = make_rollout_batch()
        # Every completion exceeds the buffer, so the penalty is visible.
        batch["completion_mask"] = torch.ones_like(batch["completion_mask"])

        rewards = computer._compute_rewards(batch)

        local_rewards, per_func = FakeRewardManager().compute_rewards(
            prompts=batch["prompts"],
            completions=batch["completions"],
            completion_id_lists=batch["completion_id_lists"],
            metadata=batch["metadata"],
        )
        local_shaped = computer._apply_overlong_penalty(batch, local_rewards)
        assert torch.allclose(rewards.global_shaped_rewards, torch.cat([local_shaped, local_shaped]))

        # The old formulation re-aggregated the per-function rewards, dropping the penalty.
        re_aggregated = (
            torch.cat([per_func, per_func]) * FakeRewardManager().reward_weight_tensor
        ).nansum(dim=-1)
        assert not torch.allclose(rewards.global_shaped_rewards, re_aggregated)
        assert (rewards.global_shaped_rewards - re_aggregated).abs().max() > 0.5

    def test_without_shaping_the_total_equals_the_weighted_per_function_sum(self) -> None:
        computer, _ = build_loss_computer(rollout_config(max_completion_length=COMPLETION_LEN))
        batch = make_rollout_batch()

        rewards = computer._compute_rewards(batch)

        expected = (rewards.global_per_func * rewards.weights.unsqueeze(0)).nansum(dim=-1)
        assert torch.allclose(rewards.global_shaped_rewards, expected)

    def test_each_completion_is_scored_once(self) -> None:
        reward_manager = FakeRewardManager()
        computer, _ = build_loss_computer(
            rollout_config(max_completion_length=COMPLETION_LEN), reward_manager=reward_manager
        )

        computer._compute_rewards(make_rollout_batch())

        assert len(reward_manager.calls) == 1
        assert len(reward_manager.calls[0]["completions"]) == 4


class TestAdvantageComputation:
    def test_grpo_advantages_are_normalized_within_the_group(self) -> None:
        computer, _ = build_loss_computer(rollout_config(loss_type="grpo", max_completion_length=COMPLETION_LEN))
        batch = make_rollout_batch()
        rewards = computer._compute_rewards(batch)
        rewards.global_shaped_rewards = torch.tensor([2.0, 0.0, 1.0, 3.0])

        advantages = computer._compute_advantages(batch, rewards, num_generations=NUM_GENERATIONS)

        # Group 0 has mean 2 / std sqrt(2); group 1 has mean 2 / std sqrt(2). The
        # advantage epsilon (1e-4) shrinks the normalized values slightly.
        assert torch.allclose(
            advantages.local, torch.tensor([0.7071068, -0.7071068, -0.7071068, 0.7071068]), atol=1e-4
        )
        assert torch.equal(advantages.local, advantages.global_advantages)

    def test_dr_grpo_skips_the_std_scaling(self) -> None:
        computer, _ = build_loss_computer(
            rollout_config(loss_type="dr_grpo", max_completion_length=COMPLETION_LEN)
        )
        batch = make_rollout_batch()
        rewards = computer._compute_rewards(batch)
        rewards.global_shaped_rewards = torch.tensor([4.0, 0.0, 1.0, 3.0])

        advantages = computer._compute_advantages(batch, rewards, num_generations=NUM_GENERATIONS)

        assert torch.allclose(advantages.local, torch.tensor([2.0, -2.0, -1.0, 1.0]), atol=1e-5)

    def test_two_rank_run_recovers_the_local_slice(self) -> None:
        distributed, gather = two_rank_context()
        computer, _ = build_loss_computer(
            rollout_config(loss_type="grpo", max_completion_length=COMPLETION_LEN), distributed=distributed
        )
        batch = make_rollout_batch()
        rewards = computer._compute_rewards(batch)
        # Overwrite the gathered rewards with a hand-checkable 2-rank tensor.
        rewards.global_shaped_rewards = torch.tensor([2.0, 0.0, 1.0, 3.0]).repeat(2)

        advantages = computer._compute_advantages(batch, rewards, num_generations=NUM_GENERATIONS)

        assert gather.calls > 0
        assert advantages.global_advantages.numel() == 8
        assert torch.allclose(advantages.local, advantages.global_advantages[:4])

    def test_misaligned_ranks_raise_a_clear_error(self) -> None:
        def reordering_gather(tensor: torch.Tensor) -> torch.Tensor:
            return torch.cat([tensor.flip(0), tensor], dim=0)

        distributed = DistributedContext(rank=0, world_size=2, all_gather_fn=reordering_gather)
        computer, _ = build_loss_computer(
            rollout_config(loss_type="grpo", max_completion_length=COMPLETION_LEN), distributed=distributed
        )
        batch = make_rollout_batch()
        rewards = computer._compute_rewards(batch)

        with pytest.raises(RuntimeError, match="Failed to recover this rank's advantages"):
            computer._compute_advantages(batch, rewards, num_generations=NUM_GENERATIONS)

    def test_zero_advantage_groups_are_masked_out(self) -> None:
        computer, _ = build_loss_computer(
            rollout_config(loss_type="dapo", max_completion_length=COMPLETION_LEN, drop_zero_advantage_groups=True)
        )
        advantages = torch.tensor([0.0, 0.0, 1.0, -1.0])

        keep = computer._zero_advantage_group_mask(advantages, num_generations=2)

        assert keep.tolist() == [False, False, True, True]

    def test_zero_advantage_mask_is_opt_in(self) -> None:
        computer, _ = build_loss_computer(rollout_config(max_completion_length=COMPLETION_LEN))

        assert computer._zero_advantage_group_mask(torch.zeros(4), num_generations=2) is None


class TestEndToEnd:
    @pytest.mark.parametrize("loss_type", sorted(GRPO_ALGORITHM_PRESETS))
    def test_every_preset_produces_a_finite_loss_and_gradient(self, loss_type: str) -> None:
        rollout = rollout_config(
            loss_type=loss_type, max_completion_length=COMPLETION_LEN, kl_beta=0.1
        )
        computer, policy, _ = build_e2e_computer(rollout)

        loss, metrics = computer.compute_loss(make_rollout_batch(), num_generations=NUM_GENERATIONS)

        assert torch.isfinite(loss)
        assert metrics["reward"].item() == pytest.approx(0.75)
        assert "advantage_mean" in metrics

        loss.backward()
        gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)

    def test_reward_metric_matches_the_shaped_rewards(self) -> None:
        rollout = rollout_config(loss_type="grpo", max_completion_length=COMPLETION_LEN, kl_beta=0.0)
        computer, _, _ = build_e2e_computer(rollout)

        _, metrics = computer.compute_loss(make_rollout_batch(), num_generations=NUM_GENERATIONS)

        assert metrics["reward"].item() == pytest.approx(0.75)
        assert metrics["reward/accuracy"].item() == pytest.approx(0.5)
        assert metrics["reward/format"].item() == pytest.approx(0.5)

    def test_reference_kl_is_added_to_the_objective(self) -> None:
        base = rollout_config(loss_type="grpo", max_completion_length=COMPLETION_LEN, kl_beta=0.0)
        with_kl = replace(base, kl_beta=0.5)
        computer_without_kl, _, _ = build_e2e_computer(base)
        computer_with_kl, _, _ = build_e2e_computer(with_kl)

        loss_without_kl, _ = computer_without_kl.compute_loss(
            make_rollout_batch(), num_generations=NUM_GENERATIONS
        )
        loss_with_kl, metrics = computer_with_kl.compute_loss(
            make_rollout_batch(), num_generations=NUM_GENERATIONS
        )

        assert metrics["kl"].item() > 0.0
        assert (loss_with_kl - loss_without_kl).item() == pytest.approx(0.5 * metrics["kl"].item(), rel=1e-5)

    def test_no_reference_model_means_zero_kl(self) -> None:
        rollout = rollout_config(
            loss_type="grpo", max_completion_length=COMPLETION_LEN, kl_beta=0.1, use_reference_model=False
        )
        computer, _, _ = build_e2e_computer(rollout)

        _, metrics = computer.compute_loss(make_rollout_batch(), num_generations=NUM_GENERATIONS)

        assert metrics["kl"].item() == pytest.approx(0.0)

    def test_overlong_penalty_reaches_the_objective_on_a_two_rank_run(self) -> None:
        rollout = rollout_config(
            loss_type="dapo",
            max_completion_length=COMPLETION_LEN,
            overlong_buffer_len=2,
            kl_beta=0.0,
        )
        distributed, _ = two_rank_context()
        computer, _, _ = build_e2e_computer(rollout, distributed=distributed)
        batch = make_rollout_batch()
        batch["completion_mask"] = torch.ones_like(batch["completion_mask"])

        loss, metrics = computer.compute_loss(batch, num_generations=NUM_GENERATIONS)

        assert torch.isfinite(loss)
        # Unshaped rewards average to 0.75; the zero-reward row drops to -1.0, so the
        # logged reward (and the advantage the gradient is built from) averages to 0.5.
        assert metrics["reward"].item() == pytest.approx(0.5)


class TestLossHelpers:
    def test_masked_mean_clamps_the_denominator(self) -> None:
        values = torch.tensor([1.0, 2.0])

        assert masked_mean(values, torch.zeros(2)).item() == 0.0
        assert masked_mean(values, torch.tensor([1.0, 0.0])).item() == pytest.approx(1.0)

    def test_selective_log_softmax_matches_the_naive_gather(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 5)
        index = torch.randint(0, 5, (2, 3))

        selected = selective_log_softmax(logits, index)

        expected = torch.gather(logits.log_softmax(-1), -1, index.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(selected, expected, atol=1e-6)

    def test_selective_log_softmax_supports_top_k_indices(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 5)
        index = torch.randint(0, 5, (2, 3, 2))

        selected = selective_log_softmax(logits, index)

        expected = torch.gather(logits.log_softmax(-1), -1, index)
        assert selected.shape == index.shape
        assert torch.allclose(selected, expected, atol=1e-6)

    def test_approx_kl_is_zero_for_an_unchanged_policy(self) -> None:
        log_probs = torch.tensor([[-0.5, -1.0]])

        assert torch.allclose(approx_kl_divergence(log_probs, log_probs), torch.zeros(1, 2), atol=1e-7)

    def test_entropy_matches_log_softmax_definition(self) -> None:
        logits = torch.randn(2, 4)

        entropy = entropy_from_logits(logits)

        expected = -(logits.log_softmax(-1).exp() * logits.log_softmax(-1)).sum(-1)
        assert torch.allclose(entropy, expected, atol=1e-6)

    def test_materialize_passes_plain_tensors_through(self) -> None:
        logits = torch.zeros(1, 2, 3)

        assert materialize_vocab_parallel_logits(logits) is logits
