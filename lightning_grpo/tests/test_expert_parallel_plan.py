"""Unit tests for expert-parallel plan resolution (no distributed setup required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lightning_grpo.strategies.expert_parallel import (
    expand_ep_plan,
    get_resolved_ep_plan,
    validate_ep_topology,
)
from lightning_grpo.strategies.tensor_parallel import _drop_expert_parallel_rules

BASE_PLAN = {
    "layers.*.mlp.experts": "ep_dispatch_experts",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
}


def _model(ep_plan: dict | None = None, base_plan: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        base_model_prefix="model",
        base_model=None,
        _ep_plan=ep_plan or {},
        config=SimpleNamespace(base_model_ep_plan=base_plan),
    )


def _ep(plan: str = "auto", overrides: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(plan=plan, plan_overrides=overrides or {}, implementation="auto")


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(2, 4))
        self.down_proj = nn.Parameter(torch.empty(4, 2))


class _MoeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].mlp = nn.Module()
        self.layers[0].mlp.experts = _Experts()


def test_base_model_plan_rules_are_qualified_with_base_model_prefix():
    plan = get_resolved_ep_plan(_model(base_plan=BASE_PLAN), _ep())

    assert plan == {f"model.{key}": value for key, value in BASE_PLAN.items()}


def test_instance_plan_overrides_base_model_plan():
    model = _model(
        base_plan=BASE_PLAN,
        ep_plan={"model.layers.*.mlp.experts": "ep_dispatch_experts"},
    )

    plan = get_resolved_ep_plan(model, _ep())

    assert plan["model.layers.*.mlp.experts"] == "ep_dispatch_experts"
    assert plan["model.layers.*.mlp.experts.down_proj"] == "grouped_gemm"


def test_plan_none_disables_sharding():
    assert get_resolved_ep_plan(_model(base_plan=BASE_PLAN), _ep(plan="none")) == {}


def test_plan_overrides_are_applied():
    plan = get_resolved_ep_plan(_model(), _ep(overrides={"model.layers.*.mlp.experts": "ep_dispatch_experts"}))

    assert plan == {"model.layers.*.mlp.experts": "ep_dispatch_experts"}


def test_expand_plan_lists_parameters_before_their_module():
    model = _MoeModel()

    expanded = expand_ep_plan(model, BASE_PLAN)

    assert list(expanded) == [
        "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.experts.down_proj",
        "layers.0.mlp.experts",
    ]
    assert expanded["layers.0.mlp.experts.gate_up_proj"] == "grouped_gemm"
    assert expanded["layers.0.mlp.experts"] == "ep_dispatch_experts"


def test_topology_accepts_ep_multiple_of_tp_dividing_the_world():
    validate_ep_topology(ep_size=4, tp_size=2, dp_size=4)


@pytest.mark.parametrize(
    ("ep_size", "tp_size", "dp_size"),
    [
        (3, 2, 4),
        (4, 2, 3),
    ],
)
def test_topology_rejects_invalid_sizes(ep_size: int, tp_size: int, dp_size: int):
    with pytest.raises(ValueError):
        validate_ep_topology(ep_size=ep_size, tp_size=tp_size, dp_size=dp_size)


def test_tp_rules_under_ep_modules_are_dropped():
    tp_plan = {
        "model.layers.0.mlp.experts.gate_up_proj": "packed_colwise",
        "model.layers.0.mlp.experts": "moe_tp_experts",
        "model.layers.0.mlp.shared_expert.gate_proj": "colwise",
    }
    ep_plan = {
        "model.layers.0.mlp.experts.gate_up_proj": "grouped_gemm",
        "model.layers.0.mlp.experts": "ep_dispatch_experts",
    }

    assert _drop_expert_parallel_rules(tp_plan, ep_plan) == {
        "model.layers.0.mlp.shared_expert.gate_proj": "colwise",
    }
