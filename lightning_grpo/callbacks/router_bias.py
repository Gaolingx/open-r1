"""Auxiliary-loss-free load balancing via a per-expert router bias (DeepSeek-V3 style).

Instead of adding a differentiable auxiliary loss to the router, this callback
updates a *non-gradient* per-expert bias ``e_score_correction_bias`` with a
sign-based rule once per optimizer step:

    b_i <- b_i + gamma * sign(mean_i(c_i) - c_i)

where ``c_i`` is the number of tokens routed to expert ``i`` during the update
interval and ``gamma`` is ``router_bias_update_rate``.

Non-invasive by design
----------------------
``NekoMindMoe2TopKRouter`` is used **exactly as shipped by Hugging Face**:

* its ``forward`` already returns ``(router_logits, topk_weights, topk_indices)``,
  so token counts are recovered with a ``forward_hook`` -- no counter buffer and
  no counting code inside the model;
* it already computes ``scores_for_choice = scores + self.e_score_correction_bias``,
  so the shipped (never updated) zero-initialised buffer is all the model provides.

The counters live on the callback: they never enter the model's ``state_dict`` and
simply disappear when this callback is not registered. The bias itself is promoted
to float32 in memory at ``setup`` time, which replaces the ``_keep_in_fp32_modules``
declaration the HF file does not have.

References:
    - Loss-Free Balancing, arXiv:2408.15664
    - DeepSeek-V3 Technical Report, arXiv:2412.19437

The bias only participates in expert *selection* (``scores + bias``); the mixing
weights are still taken from the raw sigmoid scores, so the update never
distorts the model's output distribution.

Diagnostics
-----------
Single home of the ``router/*`` diagnostics: the callback sees both halves of the
router output, so it covers the realized (post-bias, post-grouped-top-k) load
*and* the raw logits.

* ``router/entropy`` -- per-token entropy of ``softmax(router_logits)``, averaged over
  tokens and layers. Logit-level, so independent of the bias and of the group
  restriction; bounded by ``ln(num_experts)``.
* ``router/load_imbalance_mean`` / ``load_max_violation`` / ``load_cv`` -- realized load
  over the ideal uniform load, all 0.0 when even. ``mean`` is the L1 deviation over all
  (layer, expert) pairs; ``load_max_violation`` is the per-layer worst single-expert
  deviation from its fair share, normalized by that share (0.35 = one expert was 35% off),
  then averaged over layers; ``load_cv`` is the per-layer L2 deviation (std / mean), which
  reacts to a few moderately hot experts sooner than ``mean``.
* ``router/dead_experts`` -- fraction of (layer, expert) pairs that got no token.
* ``router/bias_abs_mean`` / ``router/bias_dc`` -- bias magnitude. Only the CENTERED bias
  ``b_c = b - b.mean(-1)`` is observable: adding ``alpha`` to every expert of a layer leaves
  routing bit-for-bit unchanged (``group_scores`` shift by ``2*alpha`` for all groups, the
  mask and the raw-sigmoid weights are untouched), and nothing restores that per-layer
  constant -- the counts ignore it while ``sum_i sign(mean_i(c) - c_i) != 0`` in general --
  so raw ``|b|`` random-walks as ``~0.8 * gamma * sqrt(steps)`` even on a healthy router
  (measured: 0.0437 raw vs 0.0127 centered). ``bias_abs_mean`` is that CENTERED magnitude;
  ``bias_dc`` is the raw per-layer mean, logged only to make the drift visible.
* ``router/bias_promote_imbalance`` -- which side the centred bias is net working on. The
  sign is the direction of the router's OWN PRIOR, not of the current load (the bias has
  already flattened that): ``b_c > 0`` = the raw scores *under*-select the expert, so the
  bias props it up; ``b_c < 0`` = over-selected, held back. A *mass* split cannot express
  this (``mass(b+) - mass(|b-|) == sum(b) == E * dc``, and after centring the two masses are
  equal by construction), so the asymmetry is a HEADCOUNT: ``(P - S) / (P + S)``, ``P``/``S``
  = experts with ``b_c > 0`` / ``b_c < 0``. 0.0 = the two camps are equally populated,
  ``> 0`` = more experts propped up than held back; range ``[-1, 1]``.
* ``router/bias_promote_max`` / ``bias_suppress_max`` -- the two tail extremes, kept because
  they CAN disagree with the headcount and neither alone is conclusive (measured: 73/27 by
  count while suppression was 1.75x stronger per expert). ``bias_abs_max`` was just the larger
  of the two and was retired. The ``0.25`` (sigmoid-score spread) bound applies to them; a
  runaway ``bias_promote_max`` -- a dead expert's sign freezes positive -- together with
  ``router/dead_experts`` is the real alarm.

``load_max_violation`` is an extreme over ``num_experts`` counts, so a perfectly balanced
router already reads ``2.2 / sqrt(c)`` with ``c = tokens_per_layer_per_step * top_k /
num_experts`` (``mean``: ``0.8 / sqrt(c)``); only a larger ``c`` lowers that floor, not
more layers.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info

DEFAULT_UPDATE_RATE = 1.0e-3


def iter_routers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return every submodule that owns an ``e_score_correction_bias`` buffer."""

    return [module for module in model.modules() if getattr(module, "e_score_correction_bias", None) is not None]


class RouterBiasUpdateCallback(Callback):
    """Update ``e_score_correction_bias`` once per optimizer step.

    A ``forward_hook`` on every router accumulates the ``topk_indices`` that the
    shipped ``forward`` already returns, plus the router logits behind the entropy
    diagnostic; the accumulators are then consumed and reset in
    ``on_before_optimizer_step``. Nothing in the HF modeling code is touched, and
    this callback is the single owner of the ``router/*`` metrics.

    Args:
        enabled: Master switch. When false, ``setup`` returns immediately and the bias is
            left untouched (equivalent to plain top-k routing).
        update_rate: Bias step size ``gamma``. When ``None`` (default) the value is read
            from the policy config (``router_bias_update_rate``) as a fallback.
        reduce_dim_names: Device-mesh dimension names whose token counts must be
            summed before applying the update. Defaults to
            ``("data_parallel", "tensor_parallel")``: data-parallel ranks see
            different tokens, while tensor-parallel ranks hold a replica of the
            router. Summing over a replicated dimension only rescales the counts,
            which leaves the sign rule, the logged ratios and the entropy mean
            (sum and token count scale together) unchanged. ``expert_parallel`` is
            appended automatically when the mesh defines it.
        freeze_at_end_fraction: Stop updating the bias during the last fraction of
            training, as recommended by the DeepSeek-V3 report. ``0.0`` disables it.
    """

    def __init__(
        self,
        enabled: bool = True,
        update_rate: float | None = None,
        reduce_dim_names: Sequence[str] = ("data_parallel", "tensor_parallel"),
        freeze_at_end_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.update_rate = update_rate
        self.reduce_dim_names = tuple(reduce_dim_names)
        self.freeze_at_end_fraction = freeze_at_end_fraction
        self._routers: list[torch.nn.Module] = []
        self._counts: list[torch.Tensor] = []
        self._entropy_sums: list[torch.Tensor] = []
        self._token_counts: list[torch.Tensor] = []
        self._handles: list[Any] = []
        self._num_experts: int | None = None

    # ---- lifecycle -------------------------------------------------------
    def setup(self, trainer, pl_module, stage: str = "fit") -> None:
        """Resolve routers, promote the bias to fp32, and install counting hooks."""

        self._detach_hooks()
        self._routers = []
        self._counts = []
        self._entropy_sums = []
        self._token_counts = []
        self._num_experts = None

        if stage != "fit":
            return

        if not self.enabled:
            rank_zero_info("[RouterBias] Disabled via `optimization.router_bias_enabled`; callback inactive.")
            return

        policy = getattr(pl_module, "policy", None) or getattr(pl_module, "model", None)
        if policy is None:
            rank_zero_info("[RouterBias] No `policy`/`model` attribute found; callback disabled.")
            return

        routers = iter_routers(policy)
        if not routers:
            rank_zero_info("[RouterBias] No module exposing `e_score_correction_bias` found; callback disabled.")
            return

        if self.update_rate is None:
            # Fallback for callers that do not pass a rate explicitly.
            config = getattr(policy, "config", None)
            self.update_rate = float(getattr(config, "router_bias_update_rate", DEFAULT_UPDATE_RATE))
        if self.update_rate <= 0.0:
            rank_zero_info(f"[RouterBias] `update_rate={self.update_rate}`; callback disabled.")
            return

        num_experts = routers[0].e_score_correction_bias.numel()
        if any(router.e_score_correction_bias.numel() != num_experts for router in routers):
            raise ValueError("[RouterBias] All routers must expose the same number of experts.")

        self._routers = routers
        self._num_experts = num_experts
        for index, router in enumerate(routers):
            bias = router.e_score_correction_bias
            # Keep the bias in fp32 (a 1e-3 step is below bfloat16 resolution); swapping
            # `.data` preserves the buffer registration.
            if bias.dtype != torch.float32:
                bias.data = bias.data.to(torch.float32)
            self._counts.append(torch.zeros(num_experts, dtype=torch.float32, device=bias.device))
            # Per-layer scalars: sum and token count kept apart so the mean stays exact.
            self._entropy_sums.append(torch.zeros((), dtype=torch.float32, device=bias.device))
            self._token_counts.append(torch.zeros((), dtype=torch.float32, device=bias.device))
            self._handles.append(router.register_forward_hook(self._make_hook(index)))

        rank_zero_info(
            f"[RouterBias] Tracking {len(routers)} routers (num_experts={num_experts}, "
            f"update_rate={self.update_rate}, reduce_dims={self._resolve_mesh_dim_names(pl_module)})"
        )

    def teardown(self, trainer, pl_module, stage: str = "fit") -> None:
        """Remove the counting hooks once the stage is over."""

        self._detach_hooks()
        self._routers = []
        self._counts = []
        self._entropy_sums = []
        self._token_counts = []
        self._num_experts = None

    # ---- accumulation (forward hook; the model stays untouched) ----------
    def _make_hook(self, index: int):
        """Build the per-router hook that accumulates ``topk_indices`` and the entropy."""

        def _hook(module, args, output):
            # `TorchRolloutEngine` reuses this very policy instance under ``torch.no_grad()``,
            # so training mode alone cannot distinguish a training forward from generation.
            if not module.training or not torch.is_grad_enabled():
                return
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                return

            indices = output[2]
            if not torch.is_tensor(indices) or indices.dtype not in (torch.int32, torch.int64):
                return

            with torch.no_grad():
                flat = indices.reshape(-1)
                counts = self._counts[index]
                if counts.device != flat.device:
                    counts = counts.to(flat.device)
                    self._counts[index] = counts
                # ``index_add_`` keeps the output shape static, which keeps dynamo and CUDA
                # graphs happy (``bincount`` would produce a data-dependent length).
                counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))

                # Router-logit entropy: softmax over the raw logits, mean over tokens.
                logits = output[0]
                if torch.is_tensor(logits) and logits.ndim >= 2 and logits.shape[-1] == self._num_experts:
                    probs = logits.detach().to(dtype=torch.float32).softmax(dim=-1)
                    entropy_sum = -(probs * probs.clamp_min(1.0e-8).log()).sum()
                    self._entropy_sums[index] = self._entropy_sums[index].to(probs.device) + entropy_sum
                    self._token_counts[index] = self._token_counts[index].to(probs.device) + probs.shape[0]

        return _hook

    def _detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _zero_counts(self) -> None:
        for counts in self._counts:
            counts.zero_()
        for entropy_sum in self._entropy_sums:
            entropy_sum.zero_()
        for token_count in self._token_counts:
            token_count.zero_()

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
        if not self._routers or not self._counts or self.update_rate is None:
            return

        device = self._counts[0].device
        num_experts = self._num_experts

        # 0) Freeze the bias over the tail of training (the DeepSeek-V3 recipe).
        if self.freeze_at_end_fraction > 0.0:
            total_steps = getattr(trainer, "estimated_stepping_batches", 0) or 0
            if total_steps > 0 and trainer.global_step / float(total_steps) >= (1.0 - self.freeze_at_end_fraction):
                self._zero_counts()
                return

        with torch.no_grad():
            # 1) Stack every layer's counter -> [num_layers, num_experts], and the
            #    per-layer entropy accumulators -> [num_layers] each.
            counts = torch.stack([counter.to(dtype=torch.float32) for counter in self._counts])
            entropy_sums = torch.stack([value.to(dtype=torch.float32) for value in self._entropy_sums])
            token_counts = torch.stack([value.to(dtype=torch.float32) for value in self._token_counts])
            # 2) Sum over the ranks that saw different tokens (data parallel). The
            #    reduction runs in fp32 so no precision is lost before taking the sign.
            for group in self._reduce_groups(pl_module):
                dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(entropy_sums, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(token_counts, op=dist.ReduceOp.SUM, group=group)

            # 3) Sign update, algebraically equivalent to `e_i = c_bar - c_i`.
            total = counts.sum(dim=-1, keepdim=True)  # [num_layers, 1]
            direction = torch.sign(total - counts * num_experts)  # [num_layers, num_experts]

            # 4) Write the bias back and reset the counters. A repeated forward only
            #    rescales `counts` uniformly, which leaves the sign rule unchanged.
            for row, router in zip(direction, self._routers):
                router_bias = router.e_score_correction_bias
                if router_bias.dtype != torch.float32:
                    router_bias.data = router_bias.data.to(torch.float32)
                router_bias.add_(row * self.update_rate)
            self._zero_counts()

            # 5) Monitoring metrics. Counts are already reduced, so every rank agrees
            entropy = (entropy_sums / token_counts.clamp_min(1.0)).mean()
            ideal_load = (total / num_experts).clamp_min(1.0)  # [num_layers, 1]
            load_ratios = counts / ideal_load  # [num_layers, num_experts]
            load_imbalance_mean = (load_ratios - 1.0).abs().mean()
            load_max_violation = ((counts - ideal_load).abs().amax(dim=-1) / ideal_load.squeeze(-1)).mean()
            load_cv = (counts.std(dim=-1) / ideal_load.squeeze(-1)).mean()
            dead_experts = (counts == 0.0).to(dtype=torch.float32).mean()

            # 5b) Bias diagnostics (see module docstring): only the centered bias is observable.
            bias = torch.stack([router.e_score_correction_bias.detach().float() for router in self._routers])
            bias_dc = bias.mean()
            centered = bias - bias.mean(dim=-1, keepdim=True)
            bias_abs_mean = centered.abs().mean()
            # Signed headcount (P - S) / (P + S); all-zero bias still reads 0.0.
            signs = torch.sign(centered)
            bias_promote_imbalance = signs.mean() / signs.abs().mean().clamp_min(1.0e-8)
            bias_promote_max = centered.max()
            bias_suppress_max = centered.min()

        if float(total.sum()) > 0.0:
            log_kwargs = {"on_step": True, "on_epoch": False, "sync_dist": False, "prog_bar": False}
            pl_module.log("router/entropy", entropy.to(device), **log_kwargs)
            pl_module.log("router/load_imbalance_mean", load_imbalance_mean.to(device), **log_kwargs)
            pl_module.log("router/load_max_violation", load_max_violation.to(device), **log_kwargs)
            pl_module.log("router/load_cv", load_cv.to(device), **log_kwargs)
            pl_module.log("router/dead_experts", dead_experts.to(device), **log_kwargs)
            pl_module.log("router/bias_abs_mean", bias_abs_mean.to(device), **log_kwargs)
            pl_module.log("router/bias_dc", bias_dc.to(device), **log_kwargs)
            pl_module.log("router/bias_promote_imbalance", bias_promote_imbalance.to(device), **log_kwargs)
            pl_module.log("router/bias_promote_max", bias_promote_max.to(device), **log_kwargs)
            pl_module.log("router/bias_suppress_max", bias_suppress_max.to(device), **log_kwargs)
