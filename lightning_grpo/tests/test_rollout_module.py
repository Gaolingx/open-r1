"""Unit tests for the GRPO rollout coordinator's batch packing.

The coordinator is built with an injected engine factory and a stub tokenizer, so these
tests exercise the prompt-major expansion, budget cropping and metadata plumbing without
an inference server or a real model.
"""

from __future__ import annotations

import json

import pytest
import torch

from lightning_grpo.models.grpo.rollout import LocalGenerateRolloutCoordinator
from lightning_grpo.tests.support import (
    FakeRolloutEngine,
    StubTokenizer,
    build_rollout_context,
    make_engine_factory,
)
from lightning_grpo.utils.configs.grpo import GRPORolloutConfig


def make_coordinator(
    rollout: GRPORolloutConfig | None = None,
    *,
    engine: FakeRolloutEngine | None = None,
    tokenizer: StubTokenizer | None = None,
) -> tuple[LocalGenerateRolloutCoordinator, FakeRolloutEngine, StubTokenizer]:
    rollout = rollout or GRPORolloutConfig(num_generations=3, max_total_length=16, max_prompt_length=4)
    engine = engine or FakeRolloutEngine()
    tokenizer = tokenizer or StubTokenizer()
    coordinator = LocalGenerateRolloutCoordinator(
        build_rollout_context(rollout, tokenizer=tokenizer),
        lambda: "policy",
        engine_factory=make_engine_factory(engine),
    )
    return coordinator, engine, tokenizer


class TestConstruction:
    def test_engine_receives_the_rollout_settings(self) -> None:
        rollout = GRPORolloutConfig(
            num_generations=2, max_total_length=16, max_prompt_length=4, max_completion_length=5
        )
        _, engine, _ = make_coordinator(rollout)

        kwargs = engine.init_kwargs
        assert kwargs["engine_type"] == "torch"
        assert kwargs["policy_model"] == "policy"
        assert kwargs["model_name_or_path"] == "toy-model"
        assert kwargs["max_completion_length"] == 5
        assert kwargs["world_size"] == 1

    def test_policy_updates_reach_the_engine(self) -> None:
        coordinator, engine, _ = make_coordinator()

        coordinator.update_policy()

        assert engine.policy_updates == ["policy"]

    def test_num_generations_depends_on_the_stage(self) -> None:
        rollout = GRPORolloutConfig(num_generations=3, num_generations_eval=1, max_total_length=16)
        coordinator, _, _ = make_coordinator(rollout)

        assert coordinator.resolve_num_generations(training=True) == 3
        assert coordinator.resolve_num_generations(training=False) == 1

    def test_max_total_length_must_leave_room_for_a_completion(self) -> None:
        coordinator, _, _ = make_coordinator(GRPORolloutConfig(max_total_length=1))

        with pytest.raises(ValueError, match="max_total_length"):
            coordinator._max_prompt_length()


class TestRolloutPacking:
    def test_prompts_are_expanded_in_prompt_major_order(self) -> None:
        rollout = GRPORolloutConfig(num_generations=3, max_total_length=16, max_prompt_length=4)
        coordinator, engine, tokenizer = make_coordinator(rollout)

        packed = coordinator.rollout({"prompt_text": ["a", "bb"]}, training=True)

        # Two prompts, three generations each -> six completion rows.
        assert engine.calls[0]["num_generations"] == 3
        assert len(packed["completions"]) == 6
        assert packed["prompts"] == ["a", "bb"]
        assert packed["sample_ids"].tolist() == [0, 0, 0, 1, 1, 1]
        # The stub tokenizes "a" as 2 and "bb" as 3 (its second token is always 2).
        assert packed["prompt_ids"][:, 0].tolist() == [2, 2, 2, 3, 3, 3]
        # The prompt batch is tokenized exactly once for both generation and packing.
        assert tokenizer.calls == 1

    def test_completion_lists_drop_padded_tokens(self) -> None:
        engine = FakeRolloutEngine(width=4, truncate_rows=2)
        coordinator, _, _ = make_coordinator(engine=engine)

        packed = coordinator.rollout({"prompt_text": ["a"]}, training=True)

        # The first two rows only hold two real tokens.
        assert packed["completion_id_lists"][0] == [1, 2]
        assert packed["completion_id_lists"][1] == [1, 2]
        assert packed["completion_mask"][0].tolist() == [1, 1, 0, 0]

    def test_cropping_to_the_budget_marks_the_completion_as_truncated(self) -> None:
        rollout = GRPORolloutConfig(num_generations=1, max_total_length=6, max_prompt_length=4)
        engine = FakeRolloutEngine(width=6)
        coordinator, _, _ = make_coordinator(rollout, engine=engine)

        packed = coordinator.rollout({"prompt_text": ["a"]}, training=True)

        # The prompt occupies 2 tokens, so only 4 completion tokens fit.
        assert packed["completion_ids"].shape == (1, 4)
        assert packed["completion_mask"].tolist() == [[1, 1, 1, 1]]
        assert packed["completion_truncated"].tolist() == [1]

    def test_engine_truncation_flags_are_forwarded(self) -> None:
        rollout = GRPORolloutConfig(num_generations=2, max_total_length=16, max_prompt_length=4)
        engine = FakeRolloutEngine(truncated_rows=1)
        coordinator, _, _ = make_coordinator(rollout, engine=engine)

        packed = coordinator.rollout({"prompt_text": ["a"]}, training=True)

        assert packed["completion_truncated"].tolist() == [1, 0]

    def test_engine_log_probs_are_cropped_and_masked(self) -> None:
        rollout = GRPORolloutConfig(num_generations=2, max_total_length=16, max_prompt_length=4)
        engine = FakeRolloutEngine(width=4, truncate_rows=1)
        coordinator, _, _ = make_coordinator(rollout, engine=engine)

        packed = coordinator.rollout({"prompt_text": ["a"]}, training=True)

        assert torch.allclose(packed["old_per_token_logps"][0], torch.tensor([-0.5, -0.5, 0.0, 0.0]))
        assert torch.allclose(packed["old_per_token_logps"][1], torch.full((4,), -0.5))

    def test_prompts_without_a_completion_budget_raise(self, ) -> None:
        rollout = GRPORolloutConfig(max_total_length=6, max_prompt_length=4)
        coordinator, _, _ = make_coordinator(rollout)
        prompt_encoded = {
            "input_ids": torch.ones(1, 6, dtype=torch.long),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
        }

        with pytest.raises(RuntimeError, match="no room for completion tokens"):
            coordinator._pack_rollout(
                prompts=["a"],
                completion_ids=torch.ones(1, 2, dtype=torch.long),
                completion_mask=torch.ones(1, 2, dtype=torch.long),
                completions=["c"],
                metadata=[{}],
                num_generations=1,
                old_per_token_logps=torch.zeros(1, 2),
                prompt_encoded=prompt_encoded,
            )


class TestMetadata:
    def test_prompt_metadata_is_repeated_over_its_completions(self) -> None:
        rollout = GRPORolloutConfig(num_generations=2, max_total_length=16, max_prompt_length=4)
        coordinator, _, _ = make_coordinator(rollout)

        packed = coordinator.rollout(
            {
                "prompt_text": ["a", "bb"],
                "metadata": [json.dumps({"answer": "42"}), json.dumps({"solution": "x"})],
            },
            training=True,
        )

        metadata = packed["metadata"]
        assert len(metadata) == 4
        # ``answer`` is aliased to ``solution`` so the reward functions find the gold label.
        assert metadata[0]["solution"] == "42"
        assert metadata[1]["solution"] == "42"
        assert metadata[2]["solution"] == "x"
        assert metadata[2]["turn_outputs"] == [packed["completions"][2]]
        assert metadata[2]["unfinished"] is False

    def test_invalid_metadata_json_falls_back_to_an_empty_mapping(self) -> None:
        coordinator, _, _ = make_coordinator(GRPORolloutConfig(num_generations=1, max_total_length=16))

        packed = coordinator.rollout({"prompt_text": ["a"], "metadata": ["not json"]}, training=True)

        assert "solution" not in packed["metadata"][0]


class TestRemaxBaseline:
    def test_remax_generates_a_single_greedy_baseline_per_prompt(self) -> None:
        rollout = GRPORolloutConfig(
            num_generations=2,
            max_total_length=16,
            max_prompt_length=4,
            advantage_estimator="remax",
        )
        coordinator, engine, _ = make_coordinator(rollout)

        packed = coordinator.rollout({"prompt_text": ["a", "bb"]}, training=True)

        assert len(engine.calls) == 2
        assert engine.calls[0]["num_generations"] == 2
        assert engine.calls[0]["greedy"] is False
        assert engine.calls[1]["num_generations"] == 1
        assert engine.calls[1]["greedy"] is True
        # One baseline per prompt, repeated later by the loss computer.
        assert len(packed["baseline_completions"]) == 2
        assert len(packed["baseline_metadata"]) == 2

    def test_no_baseline_keys_without_remax(self) -> None:
        coordinator, engine, _ = make_coordinator(GRPORolloutConfig(num_generations=2, max_total_length=16))

        packed = coordinator.rollout({"prompt_text": ["a"]}, training=True)

        assert len(engine.calls) == 1
        assert "baseline_completions" not in packed

    def test_non_greedy_baseline_temperature_is_forwarded(self) -> None:
        rollout = GRPORolloutConfig(
            num_generations=1,
            max_total_length=16,
            advantage_estimator="remax",
            remax_baseline_temperature=0.5,
        )
        coordinator, engine, _ = make_coordinator(rollout)

        coordinator.rollout({"prompt_text": ["a"]}, training=True)

        assert engine.calls[1]["greedy"] is False
        assert engine.calls[1]["temperature"] == pytest.approx(0.5)
