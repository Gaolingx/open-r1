"""Unit tests for tensor-parallel plan resolution (no distributed setup required)."""

from __future__ import annotations

from types import SimpleNamespace

from lightning_grpo.strategies.tensor_parallel import get_resolved_tp_plan

BASE_PLAN = {
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    "layers.*.mlp.experts": "moe_tp_experts",
}


def _model(tp_plan: dict | None = None, base_plan: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        base_model_prefix="model",
        base_model=None,
        _tp_plan=tp_plan or {},
        config=SimpleNamespace(base_model_tp_plan=base_plan),
    )


def _tp(loss_parallel: bool = False, plan: str = "auto", overrides: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(plan=plan, plan_overrides=overrides or {}, loss_parallel=loss_parallel)


def test_base_model_plan_rules_are_qualified_with_base_model_prefix():
    plan = get_resolved_tp_plan(_model(base_plan=BASE_PLAN), _tp())

    assert plan == {f"model.{key}": value for key, value in BASE_PLAN.items()}


def test_qualified_rules_are_not_prefixed_twice():
    model = _model(base_plan=BASE_PLAN, tp_plan={"model.layers.*.mlp.gate_proj": "colwise"})

    plan = get_resolved_tp_plan(model, _tp())

    assert plan["model.layers.*.mlp.gate_proj"] == "colwise"
    assert not any(key.startswith("model.model.") for key in plan)


def test_base_model_instance_keeps_relative_keys():
    model = _model(base_plan=BASE_PLAN)
    model.base_model = model

    plan = get_resolved_tp_plan(model, _tp())

    assert plan == BASE_PLAN


def test_lm_head_follows_loss_parallel_when_liger_ce_is_disabled():
    model = _model(tp_plan={"lm_head": "colwise_gather_output"})

    assert get_resolved_tp_plan(model, _tp())["lm_head"] == "colwise_gather_output"
    assert get_resolved_tp_plan(model, _tp(loss_parallel=True))["lm_head"] == "colwise"


def test_lm_head_is_dropped_when_liger_fused_ce_is_enabled():
    model = _model(tp_plan={"lm_head": "colwise_gather_output"}, base_plan=BASE_PLAN)

    plan = get_resolved_tp_plan(model, _tp(), liger_fused_ce=True)

    assert "lm_head" not in plan
    assert plan["model.layers.*.mlp.experts"] == "moe_tp_experts"


def test_plan_none_disables_sharding():
    assert get_resolved_tp_plan(_model(tp_plan={"lm_head": "colwise"}), _tp(plan="none")) == {}


def test_plan_overrides_are_applied():
    plan = get_resolved_tp_plan(_model(), _tp(overrides={"model.layers.*.mlp.gate_proj": "rowwise"}))

    assert plan == {"model.layers.*.mlp.gate_proj": "rowwise"}
