"""Auxiliary-loss-free load balancing via per-expert router bias (Megatron / DeepSeek-V3 style).

Instead of adding a differentiable auxiliary loss to the router, this module
updates a *non-gradient* per-expert bias ``e_score_correction_bias`` with a
sign-based rule every optimizer step:

    b_i <- b_i + gamma * sign(mean_i(c_i) - c_i)

where ``c_i`` is the number of tokens routed to expert ``i`` during the update
interval (accumulated in ``tokens_per_expert``) and ``gamma`` is
``router_bias_update_rate``.

References:
    - Loss-Free Balancing, arXiv:2408.15664
    - DeepSeek-V3 Technical Report, arXiv:2412.19437

The bias only participates in expert *selection* (``scores + bias``); the mixing
weights are still taken from the raw sigmoid scores, so the update never
distorts the model's output distribution.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info


def iter_routers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return every submodule that owns an ``e_score_correction_bias`` buffer."""

    return [module for module in model.modules() if getattr(module, "e_score_correction_bias", None) is not None]


class RouterBiasUpdateCallback(Callback):
    """Update ``e_score_correction_bias`` once per optimizer step.

    Token counts are accumulated inside the router's forward pass whenever the
    module is in training mode, then consumed and reset here.

    Args:
        update_rate: Bias step size ``gamma``. When ``None`` (default) the value is
            read from the policy config (``router_bias_update_rate``).
        reduce_dim_names: Device-mesh dimension names whose token counts must be
            summed before applying the update. Defaults to
            ``("data_parallel", "tensor_parallel")``: data-parallel ranks see
            different tokens, while tensor-parallel ranks hold a replica of the
            router. Summing over a replicated dimension only rescales the counts,
            which leaves both the sign rule and the logged ratios unchanged.
            ``expert_parallel`` is appended automatically when the mesh defines it.
        freeze_at_end_fraction: Stop updating the bias during the last fraction of
            training, as recommended by the DeepSeek-V3 report. ``0.0`` disables it.
    """

    def __init__(
        self,
        update_rate: float | None = None,
        reduce_dim_names: Sequence[str] = ("data_parallel", "tensor_parallel"),
        freeze_at_end_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        self.update_rate = update_rate
        self.reduce_dim_names = tuple(reduce_dim_names)
        self.freeze_at_end_fraction = freeze_at_end_fraction
        self._routers: list[torch.nn.Module] = []
        self._num_experts: int | None = None

    # ---- lifecycle -------------------------------------------------------
    def setup(self, trainer, pl_module, stage) -> None:
        """Resolve routers and bias hyper-parameters from the policy config."""

        policy = getattr(pl_module, "policy", None) or getattr(pl_module, "model", None)
        if policy is None:
            rank_zero_info("[RouterBias] No `policy`/`model` attribute found; callback disabled.")
            return

        config = getattr(policy, "config", None)
        if not getattr(config, "enable_expert_bias", True):
            rank_zero_info("[RouterBias] `enable_expert_bias=False`; callback disabled.")
            return

        routers = iter_routers(policy)
        if not routers:
            rank_zero_info("[RouterBias] No router with `e_score_correction_bias` found; callback disabled.")
            return

        if any(getattr(router, "tokens_per_expert", None) is None for router in routers):
            raise RuntimeError(
                "RouterBiasUpdateCallback requires every router to own a `tokens_per_expert` buffer. "
                "Update `NekoMindMoe2TopKRouter`, or set `enable_expert_bias=False` to opt out."
            )

        # The bias must stay in float32: a 1e-3 update is below bfloat16 resolution.
        for router in routers:
            if router.e_score_correction_bias.dtype != torch.float32:
                router.e_score_correction_bias.data = router.e_score_correction_bias.data.float()
            if router.tokens_per_expert.dtype != torch.float32:
                router.tokens_per_expert.data = router.tokens_per_expert.data.float()

        if self.update_rate is None:
            self.update_rate = float(getattr(config, "router_bias_update_rate", 1.0e-3))

        self._routers = routers
        self._num_experts = routers[0].tokens_per_expert.numel()

        rank_zero_info(
            f"[RouterBias] Tracking {len(routers)} routers (num_experts={self._num_experts}, "
            f"update_rate={self.update_rate}, reduce_dims={self._resolve_mesh_dim_names(pl_module)})"
        )

    # ---- reduction -------------------------------------------------------
    def _resolve_mesh_dim_names(self, pl_module) -> tuple[str, ...]:
        """Keep only the requested device-mesh dimensions that actually exist."""

        mesh = getattr(pl_module, "device_mesh", None)
        dim_names = list(getattr(mesh, "mesh_dim_names", None) or [])
        if not dim_names:
            return ()

        requested = [name for name in self.reduce_dim_names if name in dim_names]
        if "expert_parallel" in dim_names and "expert_parallel" not in requested:
            requested.append("expert_parallel")
        return tuple(requested)

    def _reduce_groups(self, pl_module) -> list[dist.ProcessGroup]:
        """Return the process groups whose token counts must be summed."""

        if not (dist.is_available() and dist.is_initialized()):
            return []

        mesh = getattr(pl_module, "device_mesh", None)
        if mesh is None:
            return [dist.group.WORLD]

        groups = []
        for name in self._resolve_mesh_dim_names(pl_module):
            submesh = mesh[name]
            if submesh.size() > 1:
                groups.append(submesh.get_group())
        return groups

    # ---- per-optimizer-step hook ----------------------------------------
    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if not self._routers or self.update_rate is None:
            return

        device = self._routers[0].tokens_per_expert.device

        # 0) 训练末期冻结偏置（DeepSeek-V3 的推荐做法）
        if self.freeze_at_end_fraction > 0.0:
            total_steps = getattr(trainer, "estimated_stepping_batches", 0) or 0
            if total_steps > 0 and trainer.global_step / float(total_steps) >= (1.0 - self.freeze_at_end_fraction):
                for router in self._routers:
                    router.tokens_per_expert.zero_()
                return

        with torch.no_grad():
            # 1) 汇总所有层的计数 -> [num_layers, num_experts]
            counts = torch.stack([router.tokens_per_expert for router in self._routers]).to(dtype=torch.float32)
            # 2) 全局归约（DP 上每 rank 只看到部分 token）；fp32 归约避免 bf16 精度丢失
            for group in self._reduce_groups(pl_module):
                dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)

            # 3) 符号更新：等价于 e_i = c_bar - c_i
            num_experts = self._num_experts
            total = counts.sum(dim=-1, keepdim=True)  # [num_layers, 1]
            direction = torch.sign(total - counts * num_experts)  # [num_layers, num_experts]

            # 4) 写回并清零计数
            for row, router in zip(direction, self._routers):
                router.e_score_correction_bias.add_(row * self.update_rate)
                router.tokens_per_expert.zero_()

            # 5) 监控指标（归约后各 rank 一致，无需 sync_dist）
            mean = total / num_experts
            max_violation = ((counts - mean).abs().amax(dim=-1) / mean.clamp_min(1.0)).mean()
            load_cv = (counts.std(dim=-1) / mean.squeeze(-1).clamp_min(1.0)).mean()
            dead_experts = (counts == 0.0).to(dtype=torch.float32).mean()
            bias = torch.stack([router.e_score_correction_bias.detach().float() for router in self._routers])
            bias_abs_mean = bias.abs().mean()
            bias_abs_max = bias.abs().max()

        if float(total.sum()) > 0.0:
            log_kwargs = {"on_step": True, "on_epoch": False, "sync_dist": False, "prog_bar": False}
            pl_module.log("router/max_violation", max_violation.to(device), **log_kwargs)
            pl_module.log("router/load_cv", load_cv.to(device), **log_kwargs)
            pl_module.log("router/dead_experts", dead_experts.to(device), **log_kwargs)
            pl_module.log("router/bias_abs_mean", bias_abs_mean.to(device), **log_kwargs)
            pl_module.log("router/bias_abs_max", bias_abs_max.to(device), **log_kwargs)