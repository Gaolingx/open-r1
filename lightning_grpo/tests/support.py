"""Test doubles shared by the ``lightning_grpo`` unit tests.

Everything here is deliberately tiny: the GRPO loss computer, the metrics aggregator and
the rollout coordinator are all built around injected collaborators (see
:mod:`lightning_grpo.models.grpo.contracts`), so the tests can drive them with a
``(batch, seq, vocab)`` logits table instead of a transformer, and with a canned rollout
instead of an inference server.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from lightning_grpo.models.grpo.contracts import DistributedContext
from lightning_grpo.models.grpo.loss import create_grpo_loss_computer, resolve_grpo_algorithm
from lightning_grpo.models.grpo.metrics import GRPOMetricsAggregator
from lightning_grpo.models.grpo.rollout_module.utils import RolloutResult
from lightning_grpo.utils.configs.base import ModelConfig
from lightning_grpo.utils.configs.grpo import GRPORolloutConfig

VOCAB_SIZE = 7
PROMPT_LEN = 3
COMPLETION_LEN = 4
NUM_GENERATIONS = 2
NUM_PROMPTS = 2


class ToyCausalLM(nn.Module):
    """Causal-LM stand-in whose logits are a learnable ``token -> logits`` table.

    The loss only needs ``logits`` of shape ``(batch, seq, vocab)`` plus parameters that
    receive gradients, so a real transformer would only slow the tests down.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size, generator=generator) * 0.5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(logits=self.table[input_ids])


class FakeRewardManager:
    """Two-function reward manager with deterministic, group-separated rewards.

    Row ``i`` gets ``accuracy = 1`` for even ``i`` and ``format = 1`` for the first half of
    the batch, so the completions within a prompt group never share the same reward (the
    advantage would be zero and clipping would never be exercised).
    """

    def __init__(self, weights: tuple[float, ...] = (1.0, 0.5)) -> None:
        self.reward_names = ["accuracy", "format"]
        self.reward_weight_tensor = torch.tensor(weights, dtype=torch.float32)
        self.calls: list[dict[str, Any]] = []

    def compute_rewards(
        self,
        *,
        prompts: list[str],
        completions: list[str] | list[list[dict[str, str]]],
        completion_id_lists: list[list[int]],
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(
            {
                "prompts": list(prompts),
                "completions": list(completions),
                "completion_id_lists": list(completion_id_lists),
                "metadata": list(metadata),
            }
        )
        count = len(completions)
        rows = torch.arange(count)
        accuracy = (rows % 2 == 0).float()
        fmt = (rows < count // 2).float()
        per_func = torch.stack([accuracy, fmt], dim=-1)
        return (per_func * self.reward_weight_tensor).nansum(dim=-1), per_func


class RecordingSink:
    """Minimal :class:`~lightning_grpo.models.grpo.contracts.MetricsSink` implementation."""

    def __init__(self) -> None:
        self.logged: dict[str, torch.Tensor] = {}
        self.log_calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.dict_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def log(self, name: str, value: Any, **kwargs: Any) -> None:
        self.log_calls.append((name, value, kwargs))
        self.logged[name] = value

    def log_dict(self, dictionary: dict[str, Any], **kwargs: Any) -> None:
        self.dict_calls.append((dict(dictionary), kwargs))
        self.logged.update(dictionary)


def make_rollout_batch(
    *,
    batch: int = NUM_GENERATIONS * NUM_PROMPTS,
    num_generations: int = NUM_GENERATIONS,
    prompt_len: int = PROMPT_LEN,
    completion_len: int = COMPLETION_LEN,
    vocab_size: int = VOCAB_SIZE,
) -> dict[str, Any]:
    """Build a rollout batch shaped like the one ``LocalGenerateRolloutCoordinator`` emits.

    ``prompts`` holds one entry per prompt (prompt-major) while ``completions`` and
    ``metadata`` hold one row per completion, which is what the loss expects.
    """

    generator = torch.Generator().manual_seed(1234)
    num_prompts = batch // num_generations
    completion_ids = torch.randint(1, vocab_size, (batch, completion_len), generator=generator)
    lengths = torch.tensor([completion_len, completion_len - 2] * (batch // 2), dtype=torch.long)
    completion_mask = (torch.arange(completion_len) < lengths[:, None]).long()
    old_logps = torch.full((batch, completion_len), -1.0) + 0.05 * torch.randn(
        (batch, completion_len), generator=generator
    )
    sample_ids = torch.arange(batch) // num_generations
    baseline_rows = torch.arange(0, batch, num_generations)

    return {
        "prompts": [f"prompt {index}" for index in range(num_prompts)],
        "completions": [f"completion {index}" for index in range(batch)],
        "completion_id_lists": completion_ids.tolist(),
        "metadata": [{"solution": str(index)} for index in range(batch)],
        "prompt_ids": torch.randint(1, vocab_size, (batch, prompt_len), generator=generator),
        "prompt_mask": torch.ones((batch, prompt_len), dtype=torch.long),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completion_truncated": torch.zeros(batch, dtype=torch.long),
        "sample_ids": sample_ids,
        "old_per_token_logps": old_logps,
        # ReMax scores one greedily decoded baseline per prompt.
        "baseline_completions": [f"baseline {index}" for index in range(num_prompts)],
        "baseline_completion_id_lists": completion_ids[baseline_rows].tolist(),
        "baseline_metadata": [{"solution": str(index)} for index in range(num_prompts)],
    }


class TwoRankGather:
    """Emulates ``all_gather`` across two ranks that hold the same local batch.

    :meth:`DistributedContext.gather` all-gathers the local shapes first and the payload
    second, so the fake distinguishes the two calls by parity and concatenates the rank
    shards in rank order.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        is_descriptor = self.calls % 2 == 1
        if is_descriptor:
            if tensor.dim() != 1:
                raise AssertionError("gather should all-gather the shape descriptor first")
            return torch.cat([tensor, tensor])
        return torch.cat([tensor, tensor], dim=0)


def two_rank_context() -> tuple[DistributedContext, TwoRankGather]:
    gather = TwoRankGather()
    return DistributedContext(rank=0, world_size=2, all_gather_fn=gather), gather


def build_loss_computer(
    rollout: GRPORolloutConfig,
    *,
    policy: nn.Module | None = None,
    reference: nn.Module | None = None,
    reward_manager: FakeRewardManager | None = None,
    sink: RecordingSink | None = None,
    distributed: DistributedContext | None = None,
    rollout_temperature: float = 1.0,
):
    """Build a standard (non-Liger) loss computer wired to the test doubles."""

    reward_manager = reward_manager or FakeRewardManager()
    sink = sink or RecordingSink()
    algorithm = resolve_grpo_algorithm(rollout)
    aggregator = GRPOMetricsAggregator(
        sink,
        clip_reporting=algorithm.preset.clip_reporting,
        reward_names=reward_manager.reward_names,
    )
    computer = create_grpo_loss_computer(
        rollout,
        reward_manager,
        aggregator,
        distributed or DistributedContext(),
        policy_fn=lambda: policy,
        reference_policy_fn=lambda: reference,
        use_liger_kernel=False,
        rollout_temperature=rollout_temperature,
    )
    return computer, sink


def build_e2e_computer(
    rollout: GRPORolloutConfig,
    *,
    distributed: DistributedContext | None = None,
    seed: int = 0,
):
    """Build a loss computer whose policy/reference are trainable toy models."""

    policy = ToyCausalLM(seed=seed)
    reference = ToyCausalLM(seed=seed + 100) if rollout.use_reference_model else None
    computer, sink = build_loss_computer(
        rollout,
        policy=policy,
        reference=reference,
        distributed=distributed,
    )
    return computer, policy, sink


class StubTokenizer:
    """One-token-per-word tokenizer that left-pads, like the rollout coordinator expects."""

    pad_token_id = 0
    eos_token_id = 9

    def __init__(self) -> None:
        self.padding_side = "right"
        self.calls = 0

    def __call__(
        self,
        prompts: list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        add_special_tokens: bool = False,
        return_token_type_ids: bool = False,
    ) -> dict[str, torch.Tensor]:
        self.calls += 1
        rows = [torch.tensor([(len(prompt) % 5) + 1, 2], dtype=torch.long) for prompt in prompts]
        if max_length is not None:
            rows = [row[-max_length:] for row in rows]
        width = max(row.numel() for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, width - row.numel() :] = row
            attention_mask[index, width - row.numel() :] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeRolloutEngine:
    """Canned rollout engine: every sample gets ``width`` completion tokens plus padding."""

    def __init__(
        self, *, width: int = 4, truncate_rows: int = 0, with_logps: bool = True, truncated_rows: int = 0
    ) -> None:
        self.width = width
        self.truncate_rows = truncate_rows
        self.with_logps = with_logps
        self.truncated_rows = truncated_rows
        self.calls: list[dict[str, Any]] = []
        self.policy_updates: list[Any] = []

    def generate(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 1.0,
        *,
        greedy: bool = False,
    ) -> RolloutResult:
        self.calls.append(
            {
                "batch": prompt_ids.size(0),
                "num_generations": num_generations,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "greedy": greedy,
            }
        )
        batch = prompt_ids.size(0) * num_generations
        completion_ids = torch.zeros((batch, self.width), dtype=torch.long)
        completion_mask = torch.zeros((batch, self.width), dtype=torch.long)
        for row in range(batch):
            length = 2 if row < self.truncate_rows else self.width
            completion_ids[row, :length] = torch.arange(1, length + 1)
            completion_mask[row, :length] = 1
        completion_id_lists = [row[row != 0].tolist() for row in completion_ids]
        per_token_logps = (
            torch.full((batch, self.width), -0.5) if self.with_logps else torch.empty((batch, 0))
        )
        return RolloutResult(
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            per_token_logps=per_token_logps,
            completions_text=[f"completion {index}" for index in range(batch)],
            completion_id_lists=completion_id_lists,
            completion_truncated=torch.arange(batch) < self.truncated_rows,
        )

    def update_policy(self, model: nn.Module) -> None:
        self.policy_updates.append(model)


def make_engine_factory(engine: FakeRolloutEngine):
    """Return an ``EngineFactory`` that hands out ``engine`` and records its arguments."""

    def factory(**kwargs: Any) -> FakeRolloutEngine:
        engine.init_kwargs = kwargs
        return engine

    return factory


def build_rollout_context(rollout: GRPORolloutConfig, *, tokenizer: StubTokenizer | None = None):
    from lightning_grpo.models.grpo.rollout import RolloutContext

    return RolloutContext(
        rollout=rollout,
        model=ModelConfig(model_name_or_path="toy-model", trust_remote_code=False),
        tokenizer=tokenizer or StubTokenizer(),
        device=torch.device("cpu"),
        world_size=1,
        global_rank=0,
        local_rank=0,
    )
