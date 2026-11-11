"""Smoke-test every GRPO loss preset / advantage estimator in lightning_grpo.

Runs each algorithm end-to-end (rewards -> advantages -> objective -> metrics ->
backward) on a tiny randomly initialised Qwen2 model, so no network / checkpoints
are needed.
"""

from __future__ import annotations

import math
import sys
import traceback
import warnings

import torch

REPO = r"E:\GitHub\fork\open-r1"
for path in (REPO, REPO + r"\src"):
    if path not in sys.path:
        sys.path.insert(0, path)

from lightning_grpo.models.grpo.contracts import DistributedContext
from lightning_grpo.models.grpo.loss import (
    GRPO_ALGORITHM_PRESETS,
    create_grpo_loss_computer,
    resolve_grpo_algorithm,
)
from lightning_grpo.models.grpo.metrics import GRPOMetricsAggregator
from lightning_grpo.utils.configs.grpo import GRPORolloutConfig

torch.manual_seed(0)

VOCAB = 64
PROMPT_LEN = 6
COMP_LEN = 8
NUM_GEN = 4
NUM_PROMPTS = 2
BATCH = NUM_GEN * NUM_PROMPTS
DEVICE = "cpu"


def build_model(device: str = DEVICE):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config)
    model = model.to(device=device, dtype=torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def build_batch():
    generator = torch.Generator().manual_seed(1234)
    prompt_ids = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), generator=generator)
    completion_ids = torch.randint(0, VOCAB, (BATCH, COMP_LEN), generator=generator)
    lengths = torch.tensor([COMP_LEN, 5, COMP_LEN, 3, COMP_LEN, COMP_LEN, 2, 6])
    completion_mask = (torch.arange(COMP_LEN) < lengths[:, None]).long()
    old_logps = torch.full((BATCH, COMP_LEN), -math.log(VOCAB))
    old_logps += 0.1 * torch.randn((BATCH, COMP_LEN), generator=generator)
    sample_ids = torch.arange(NUM_PROMPTS).repeat_interleave(NUM_GEN)
    # The coordinator emits ``prompts`` prompt-major (one entry per prompt) while
    # ``completions``/``metadata`` hold one row per completion.
    return {
        "prompts": [f"prompt {i}" for i in range(NUM_PROMPTS)],
        "completions": [f"completion {i}" for i in range(BATCH)],
        "completion_id_lists": completion_ids.tolist(),
        "metadata": [{"solution": str(i)} for i in range(BATCH)],
        "prompt_ids": prompt_ids,
        "prompt_mask": torch.ones_like(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completion_truncated": torch.zeros(BATCH, dtype=torch.long),
        "sample_ids": sample_ids,
        "old_per_token_logps": old_logps,
        "baseline_completions": [f"baseline completion {i}" for i in range(NUM_PROMPTS)],
        "baseline_completion_id_lists": completion_ids[::NUM_GEN].tolist(),
        "baseline_metadata": [{"solution": str(i)} for i in range(NUM_PROMPTS)],
    }


class FakeRewardManager:
    """Deterministic two-function reward manager (accuracy + format)."""

    reward_names = ["accuracy", "format"]
    reward_weight_tensor = torch.tensor([1.0, 0.5])

    def compute_rewards(self, *, prompts, completions, completion_id_lists, metadata):
        count = len(completions)
        idx = torch.arange(count)
        accuracy = (idx % 3 == 0).float()
        fmt = (idx % 2 == 0).float() * 0.5
        per_func = torch.stack([accuracy, fmt], dim=-1)
        rewards = (per_func * self.reward_weight_tensor).nansum(dim=-1)
        return rewards, per_func


class FakeSink:
    def __init__(self) -> None:
        self.logged: dict[str, torch.Tensor] = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = value

    def log_dict(self, dictionary, **kwargs):
        self.logged.update(dictionary)


def build_computer(
    rollout: GRPORolloutConfig,
    *,
    device: str = DEVICE,
    use_liger: bool = False,
    distributed: DistributedContext | None = None,
    with_reference: bool = True,
):
    algorithm = resolve_grpo_algorithm(rollout)
    policy = build_model(device)
    reference = build_model(device) if with_reference else None
    sink = FakeSink()
    aggregator = GRPOMetricsAggregator(
        sink,
        clip_reporting=algorithm.preset.clip_reporting,
        reward_names=FakeRewardManager.reward_names,
    )
    computer = create_grpo_loss_computer(
        rollout,
        FakeRewardManager(),
        aggregator,
        distributed or DistributedContext(),
        policy_fn=lambda: policy,
        reference_policy_fn=lambda: reference,
        use_liger_kernel=use_liger,
        rollout_temperature=1.0,
        compiled=False,
    )
    return computer, policy, sink


def run_case(
    label: str,
    *,
    loss_type: str = "grpo",
    advantage_estimator: str | None = None,
    use_liger: bool = False,
    overlong_buffer_len: int = 0,
    kl_beta: float = 0.01,
    num_generations: int = NUM_GEN,
    drop_zero_advantage_groups: bool = False,
    device: str = DEVICE,
    with_reference: bool = True,
    backward: bool = True,
    distributed: DistributedContext | None = None,
):
    rollout = GRPORolloutConfig(
        loss_type=loss_type,
        max_completion_length=COMP_LEN,
        num_generations=num_generations,
        kl_beta=kl_beta,
        epsilon=0.2,
        overlong_buffer_len=overlong_buffer_len,
        overlong_penalty_factor=1.0,
        advantage_estimator=advantage_estimator,
        drop_zero_advantage_groups=drop_zero_advantage_groups,
        use_reference_model=with_reference,
    )
    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        computer, policy, sink = build_computer(
            rollout,
            device=device,
            use_liger=use_liger,
            with_reference=with_reference,
            distributed=distributed,
        )
        batch = build_batch()
        batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
        loss, metrics = computer.compute_loss(batch, num_generations=num_generations)
        if backward:
            loss.backward()
        captured = [str(item.message) for item in caught]

    if not torch.isfinite(loss):
        raise AssertionError(f"{label}: non-finite loss {loss.item()}")
    if backward:
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        if not grads:
            raise AssertionError(f"{label}: backward produced no gradients")
        if not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError(f"{label}: non-finite gradients")
        grad_norm = float(torch.sqrt(sum(g.pow(2).sum() for g in grads)))
        if grad_norm == 0.0:
            raise AssertionError(f"{label}: zero gradient norm")
    missing = {"reward", "reward_std", "advantage_mean", "kl", "entropy"} - set(metrics)
    if missing:
        raise AssertionError(f"{label}: missing metrics {sorted(missing)}")
    return {
        "loss": float(loss.item()),
        "reward": float(metrics["reward"].item()),
        "adv_mean": float(metrics["advantage_mean"].item()),
        "kl": float(metrics["kl"].item()),
        "n_metrics": len(metrics),
        "warnings": captured,
    }


def check_overlong_penalty_gather() -> dict[str, object]:
    """The gathered shaped reward must keep the overlong penalty (the diff under review)."""

    rollout = GRPORolloutConfig(
        loss_type="dapo",
        max_completion_length=COMP_LEN,
        overlong_buffer_len=4,
        overlong_penalty_factor=1.0,
        kl_beta=0.0,
        use_reference_model=False,
    )

    def two_rank_gather(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor, tensor], dim=0)

    distributed = DistributedContext(rank=0, world_size=2, all_gather_fn=two_rank_gather)
    computer, _, _ = build_computer(rollout, distributed=distributed, with_reference=False)

    batch = build_batch()
    collected = computer._compute_rewards(batch)
    local_rewards, per_func = FakeRewardManager().compute_rewards(
        prompts=batch["prompts"],
        completions=batch["completions"],
        completion_id_lists=batch["completion_id_lists"],
        metadata=batch["metadata"],
    )
    expected_local = computer._apply_overlong_penalty(batch, local_rewards)
    expected = torch.cat([expected_local, expected_local], dim=0)

    if not torch.allclose(collected.global_shaped_rewards, expected, atol=1e-6):
        raise AssertionError(
            "gathered shaped rewards do not carry the overlong penalty: "
            f"{collected.global_shaped_rewards.tolist()} vs {expected.tolist()}"
        )

    # The old (buggy) formulation re-aggregated the per-function rewards and lost the penalty.
    weights = FakeRewardManager.reward_weight_tensor
    re_aggregated = (torch.cat([per_func, per_func], 0) * weights).nansum(dim=-1)
    penalised = int((~torch.isclose(expected, re_aggregated, atol=1e-6)).sum())
    if penalised == 0:
        raise AssertionError("test data does not exercise the overlong penalty")
    if torch.allclose(collected.global_shaped_rewards, re_aggregated, atol=1e-6):
        raise AssertionError("shaped rewards lost the penalty (regression)")
    return {"penalised_rows": penalised, "max_delta": float((expected - re_aggregated).abs().max())}


def main() -> int:
    failures: list[tuple[str, str]] = []
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={DEVICE}")
    print("=" * 78)

    print("[1] loss presets (standard PyTorch path, with reference model)")
    preset_results: dict[str, dict] = {}
    for loss_type in GRPO_ALGORITHM_PRESETS:
        try:
            info = run_case(f"loss_type={loss_type}", loss_type=loss_type)
            preset_results[loss_type] = info
            print(
                f"  OK   {loss_type:<11} loss={info['loss']:+.6f} reward={info['reward']:+.3f} "
                f"adv={info['adv_mean']:+.3f} kl={info['kl']:+.5f} metrics={info['n_metrics']}"
            )
        except Exception as error:  # noqa: BLE001 - report every failure
            failures.append((f"loss_type={loss_type}", repr(error)))
            print(f"  FAIL {loss_type:<11} {error!r}")
            traceback.print_exc()

    print("=" * 78)
    print("[2] advantage estimators (loss_type=grpo unless noted)")
    from lightning_grpo.utils.loss import core_algos as algos

    for estimator in sorted(algos.ADV_ESTIMATOR_REGISTRY):
        loss_type = "gpg" if estimator == "gpg" else "grpo"
        try:
            info = run_case(f"adv={estimator}", loss_type=loss_type, advantage_estimator=estimator)
            print(f"  OK   {estimator:<32} loss={info['loss']:+.6f} adv={info['adv_mean']:+.3f}")
        except Exception as error:  # noqa: BLE001
            failures.append((f"advantage_estimator={estimator}", repr(error)))
            print(f"  FAIL {estimator:<32} {error!r}")

    print("=" * 78)
    print("[3] variant knobs")
    variants = [
        ("dapo + overlong shaping", dict(loss_type="dapo", overlong_buffer_len=4)),
        ("dapo + drop_zero_advantage_groups", dict(loss_type="dapo", drop_zero_advantage_groups=True)),
        ("dr_grpo + kl_beta=0", dict(loss_type="dr_grpo", kl_beta=0.0)),
        ("grpo + no reference model", dict(loss_type="grpo", with_reference=False)),
        ("grpo + overlong shaping (no ref)", dict(loss_type="grpo", overlong_buffer_len=4, with_reference=False)),
        ("sapo + no reference model", dict(loss_type="sapo", with_reference=False)),
    ]
    for label, kwargs in variants:
        try:
            info = run_case(label, **kwargs)
            print(f"  OK   {label:<36} loss={info['loss']:+.6f} reward={info['reward']:+.3f}")
        except Exception as error:  # noqa: BLE001
            failures.append((label, repr(error)))
            print(f"  FAIL {label:<36} {error!r}")

    print("=" * 78)
    print("[4] shaped-reward gathering keeps the overlong penalty")
    try:
        info = check_overlong_penalty_gather()
        print(f"  OK   penalised_rows={info['penalised_rows']} max_delta={info['max_delta']:.4f}")
    except Exception as error:  # noqa: BLE001
        failures.append(("overlong penalty gather", repr(error)))
        print(f"  FAIL {error!r}")

    print("=" * 78)
    print("[5] end-to-end compute_loss on a simulated 2-rank run (dapo + overlong shaping)")

    def two_rank_gather(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor, tensor], dim=0)

    for loss_type in ("grpo", "dapo", "dr_grpo", "cispo", "gspo", "sapo"):
        label = f"2-rank/{loss_type}"
        try:
            info = run_case(
                label,
                loss_type=loss_type,
                overlong_buffer_len=4,
                distributed=DistributedContext(rank=0, world_size=2, all_gather_fn=two_rank_gather),
            )
            print(f"  OK   {label:<16} loss={info['loss']:+.6f} reward={info['reward']:+.3f}")
        except Exception as error:  # noqa: BLE001
            failures.append((label, repr(error)))
            print(f"  FAIL {label:<16} {error!r}")

    if torch.cuda.is_available():
        print("=" * 78)
        print("[6] Liger fused kernel (cuda)")
        from lightning_grpo.models.grpo.liger_loss import LIGER_LOSS_TYPES

        for loss_type in LIGER_LOSS_TYPES:
            try:
                info = run_case(f"liger/{loss_type}", loss_type=loss_type, use_liger=True, device="cuda")
                print(f"  OK   liger/{loss_type:<6} loss={info['loss']:+.6f} reward={info['reward']:+.3f}")
            except Exception as error:  # noqa: BLE001
                failures.append((f"liger/{loss_type}", repr(error)))
                print(f"  FAIL liger/{loss_type:<6} {error!r}")

    print("=" * 78)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for name, error in failures:
            print(f"  - {name}: {error}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
