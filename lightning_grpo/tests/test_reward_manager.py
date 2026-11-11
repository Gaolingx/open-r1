"""Unit tests for the GRPO reward manager (reward-function wiring and weighting)."""

from __future__ import annotations

import pytest
import torch

from lightning_grpo.models.grpo.reward import GRPORewardManager
from lightning_grpo.utils.configs.grpo import GRPORewardConfig


def make_manager(**overrides) -> GRPORewardManager:
    config = GRPORewardConfig(**overrides)
    return GRPORewardManager(config, tokenizer=None, device=torch.device("cpu"))


def constant(value: float):
    """A reward function that ignores its inputs and returns ``value`` per completion."""

    def reward_fn(*, completions, **kwargs):
        return [value] * len(completions)

    return reward_fn


class TestWeightConfiguration:
    def test_weights_default_to_one_per_reward_function(self) -> None:
        manager = make_manager(reward_funcs=["format", "length"])

        assert manager.reward_names == ["format", "length"]
        assert torch.equal(manager.reward_weight_tensor, torch.ones(2))

    def test_explicit_weights_are_kept(self) -> None:
        manager = make_manager(reward_funcs=["format", "length"], reward_weights=[2.0, 0.25])

        assert torch.equal(manager.reward_weight_tensor, torch.tensor([2.0, 0.25]))

    def test_mismatched_weight_count_raises(self) -> None:
        with pytest.raises(ValueError, match="must match number of reward functions"):
            make_manager(reward_funcs=["format", "length"], reward_weights=[1.0])


class TestComputeRewards:
    def test_scores_each_function_and_applies_the_weights(self) -> None:
        manager = make_manager(reward_funcs=["format", "length"], reward_weights=[2.0, 1.0])
        manager.reward_funcs = [constant(1.0), constant(0.5)]

        rewards, per_func = manager.compute_rewards(
            prompts=["p0", "p1"],
            completions=["c0", "c1", "c2", "c3"],
            completion_id_lists=[[1], [2], [3], [4]],
            metadata=[{}, {}, {}, {}],
        )

        assert per_func.shape == (4, 2)
        assert torch.allclose(per_func, torch.tensor([[1.0, 0.5]] * 4))
        assert torch.allclose(rewards, torch.full((4,), 2.5))

    def test_missing_rewards_are_skipped_instead_of_poisoning_the_total(self) -> None:
        manager = make_manager(reward_funcs=["format", "length"], reward_weights=[1.0, 1.0])
        manager.reward_funcs = [lambda *, completions, **kwargs: [1.0, None], constant(0.5)]

        rewards, per_func = manager.compute_rewards(
            prompts=["p0"],
            completions=["c0", "c1"],
            completion_id_lists=[[1], [2]],
            metadata=[{}, {}],
        )

        assert torch.isnan(per_func[1, 0])
        # The missing value contributes 0, so row 1 keeps only the second function's score.
        assert torch.allclose(rewards, torch.tensor([1.5, 0.5]))

    def test_string_completions_are_wrapped_as_assistant_messages(self) -> None:
        received: list[list[dict[str, str]]] = []
        manager = make_manager(reward_funcs=["format"])

        def capture(*, completions, **kwargs):
            received.extend(completions)
            return [0.0] * len(completions)

        manager.reward_funcs = [capture]
        manager.compute_rewards(
            prompts=["p0"],
            completions=["hello", "world"],
            completion_id_lists=[[1], [2]],
            metadata=[{}, {}],
        )

        assert received == [
            [{"role": "assistant", "content": "hello"}],
            [{"role": "assistant", "content": "world"}],
        ]

    def test_prompts_stay_prompt_major_while_completions_are_per_sample(self) -> None:
        received: dict = {}
        manager = make_manager(reward_funcs=["format"])

        def capture(*, prompts, completions, completion_ids, **kwargs):
            received.update(prompts=prompts, completions=completions, completion_ids=completion_ids)
            return [0.0] * len(completions)

        manager.reward_funcs = [capture]
        manager.compute_rewards(
            prompts=["p0", "p1"],
            completions=["c0", "c1", "c2", "c3"],
            completion_id_lists=[[1], [2], [3], [4]],
            metadata=[{}, {}, {}, {}],
        )

        assert received["prompts"] == ["p0", "p1"]
        assert len(received["completions"]) == 4
        assert received["completion_ids"] == [[1], [2], [3], [4]]

    @pytest.mark.parametrize("alias", ["answer", "response", "output", "gold_answer", "gold_solution"])
    def test_solution_aliases_are_forwarded_as_solution(self, alias: str) -> None:
        received: dict = {}
        manager = make_manager(reward_funcs=["format"])

        def capture(*, completions, **kwargs):
            received.update(kwargs)
            return [0.0] * len(completions)

        manager.reward_funcs = [capture]
        manager.compute_rewards(
            prompts=["p0"],
            completions=["c0"],
            completion_id_lists=[[1]],
            metadata=[{alias: "42"}],
        )

        assert received["solution"] == ["42"]

    def test_explicit_solution_is_not_overridden_by_an_alias(self) -> None:
        received: dict = {}
        manager = make_manager(reward_funcs=["format"])

        def capture(*, completions, **kwargs):
            received.update(kwargs)
            return [0.0] * len(completions)

        manager.reward_funcs = [capture]
        manager.compute_rewards(
            prompts=["p0"],
            completions=["c0"],
            completion_id_lists=[[1]],
            metadata=[{"solution": "gold", "answer": "ignored"}],
        )

        assert received["solution"] == ["gold"]
