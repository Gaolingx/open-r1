from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import loss_parallel

from lightning_grpo.utils.loss import core_algos as algos


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute a mask-aware mean."""

    mask = mask.to(values.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (values * mask).sum() / denom


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute token-level entropy from logits."""

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def approx_kl_divergence(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """Approximate KL divergence between policy and reference log-probabilities."""

    return torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1.0


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    # for index with shape (...):
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    # for index with shape (..., K):
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(..., K)` or `(...)`, specifying the positions to gather from the log-softmax
            output. When the last case is used, `K` log-probabilities are gathered per position (e.g. for top-K)

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)

    return per_token_logps


def tensor_parallel_loss_context(enabled: bool) -> Any:
    """Return the PyTorch loss-parallel context when vocab logits are sharded."""

    return loss_parallel() if enabled else nullcontext()


def materialize_vocab_parallel_logits(logits: torch.Tensor) -> torch.Tensor:
    """Gather DTensor vocabulary-sharded logits for metrics that require full vocab tensors."""

    if isinstance(logits, DTensor):
        return logits.redistribute(placements=[Replicate()]).to_local()
    return logits


def masked_token_stats(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> dict[str, torch.Tensor]:
    """Compute reusable masked token-level metrics for LM training."""

    logits = materialize_vocab_parallel_logits(logits)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels != ignore_index
    if not mask.any():
        zero = shift_logits.new_tensor(0.0)
        return {
            "token_accuracy": zero,
            "entropy": zero,
            "mean_logprob": zero,
            "perplexity": torch.exp(zero),
        }

    predictions = shift_logits.argmax(dim=-1)
    token_accuracy = (((predictions == shift_labels) & mask).sum().to(dtype=torch.float32) / mask.sum().to(dtype=torch.float32))
    entropy = masked_mean(entropy_from_logits(shift_logits), mask)
    per_token_logps = selective_log_softmax(shift_logits, shift_labels.masked_fill(~mask, 0))
    mean_logprob = masked_mean(per_token_logps, mask)
    perplexity = torch.exp(-mean_logprob)
    return {
        "token_accuracy": token_accuracy,
        "entropy": entropy,
        "mean_logprob": mean_logprob,
        "perplexity": perplexity,
    }


def resolve_token_stats(
    metrics: dict[str, Any],
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Return token-level stats, deriving them from logits whenever they were materialized.

    Liger's fused linear cross-entropy does not materialize logits while the model is in
    training mode, so the metrics already produced by the loss are reused in that case.
    """

    logits = getattr(metrics.get("_policy_outputs"), "logits", None)
    if logits is None:
        return metrics
    return masked_token_stats(logits, labels, ignore_index=ignore_index)


# SFT Loss
def compute_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_parallel_enabled: bool = False,
) -> torch.Tensor:
    """Compute token-level next-token loss with optional label smoothing."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)

    with tensor_parallel_loss_context(loss_parallel_enabled):
        return F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )


def compute_standard_cross_entropy_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_parallel_enabled: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute standard CE loss and collect MoE routing diagnostics."""
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    ce_loss = compute_cross_entropy_loss(
        outputs.logits,
        labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        loss_parallel_enabled=loss_parallel_enabled,
    )
    loss = ce_loss

    metrics = {"lm_loss": ce_loss.detach(), "_policy_outputs": outputs}

    return loss, metrics


def dft_loss(outputs, labels, num_items_in_batch=None):
    """
    DFT loss function, as presented in [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward
    Rectification](https://huggingface.co/papers/2508.05629)
    """
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:]
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    per_token_loss = -logprobs.exp().detach() * logprobs
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss


def compute_standard_sft_loss(
    model: torch.nn.Module,
    loss_type: str,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_parallel_enabled: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute SFT/DFT loss based on the specified loss type.
    
    Args:
        loss_type: Type of loss to compute. Options:
            - "nll": Standard cross-entropy loss
            - "dft": DFT loss from paper "On the Generalization of SFT"
        model: The policy model
        batch: Input batch containing at minimum 'input_ids'
        labels: Target labels for next-token prediction
        ignore_index: Token id to ignore in loss computation
        label_smoothing: Label smoothing factor for CE loss
        loss_parallel_enabled: Whether to use loss parallelism for tensor parallel
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    if loss_type == "nll":
        return compute_standard_cross_entropy_loss(
            model=model,
            batch=batch,
            labels=labels,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            loss_parallel_enabled=loss_parallel_enabled,
        )
    elif loss_type == "dft":
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # Compute DFT loss
        loss = dft_loss(outputs, labels)

        # Gather metrics
        metrics = {"lm_loss": loss.detach(), "_policy_outputs": outputs}

        return loss, metrics
    else:
        raise ValueError(f"Unknown SFT loss_type: {loss_type}. Expected 'nll' or 'dft'.")


# DPO Loss
def _dpo_loss(
    loss_type: str,
    beta: float,
    chosen_logratios: torch.Tensor,
    rejected_logratios: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the DPO loss given log-ratios."""

    delta_score = chosen_logratios - rejected_logratios

    if loss_type == "sigmoid":
        loss = -torch.nn.functional.logsigmoid(beta * delta_score)
    elif loss_type == "hinge":
        loss = torch.relu(1 - beta * delta_score)
    elif loss_type == "ipo":
        # IPO normalizes by completion length
        chosen_mask, rejected_mask = completion_mask.chunk(2, dim=0)
        chosen_avg = chosen_logratios / chosen_mask.sum(dim=1).clamp(min=1.0)
        rejected_avg = rejected_logratios / rejected_mask.sum(dim=1).clamp(min=1.0)
        ipo_delta = chosen_avg - rejected_avg
        loss = (ipo_delta - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"Unknown DPO loss_type: {loss_type}")

    return loss.mean()


def compute_liger_dpo_loss(
    liger_loss_computer: Any,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute DPO loss using the shared Liger loss computer."""

    if liger_loss_computer is None:
        raise RuntimeError("Liger DPO loss computer is not initialized. Call configure_model() first.")
    return liger_loss_computer.compute_loss(batch)


def compute_standard_dpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    beta: float,
    loss_type: str,
    batch: dict[str, torch.Tensor],
    nll_coeff: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute DPO loss using standard logits computation (fallback when Liger is disabled)."""

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    completion_mask = batch["completion_mask"]

    # Forward through policy model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_completion_mask = completion_mask[..., 1:].contiguous()

    # Compute per-token log-probabilities
    per_token_logps = selective_log_softmax(shift_logits, shift_labels)
    per_token_logps[shift_completion_mask == 0] = 0.0

    # Sum log-probs over sequence
    logps = per_token_logps.sum(dim=1)
    chosen_logps, rejected_logps = logps.chunk(2, dim=0)

    # Reference model forward (no gradients)
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
        ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
        ref_per_token_logps[shift_completion_mask == 0] = 0.0
        ref_logps = ref_per_token_logps.sum(dim=1)
        ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

    # Compute log-ratios
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps

    # Compute DPO loss based on loss_type
    loss = _dpo_loss(loss_type, beta, chosen_logratios, rejected_logratios, completion_mask)

    # NLL regularization on chosen completions to prevent logps collapse
    nll_loss = torch.tensor(0.0, device=loss.device)
    if nll_coeff > 0.0:
        batch_size = shift_logits.size(0) // 2
        chosen_logits = shift_logits[:batch_size]
        chosen_labels = shift_labels[:batch_size]
        chosen_completion_mask = shift_completion_mask[:batch_size]
        # Mask non-completion tokens with ignore_index
        chosen_nll_labels = chosen_labels.clone()
        chosen_nll_labels[chosen_completion_mask == 0] = -100
        vocab_size = chosen_logits.size(-1)
        nll_loss = F.cross_entropy(
            chosen_logits.reshape(-1, vocab_size),
            chosen_nll_labels.reshape(-1),
            ignore_index=-100,
        )
        loss = loss + nll_coeff * nll_loss

    # Compute rewards for logging
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    metrics_dict = {
        "chosen_logps": chosen_logps.detach(),
        "rejected_logps": rejected_logps.detach(),
        "chosen_logits_mean": shift_logits[:shift_logits.size(0) // 2].mean().detach(),
        "rejected_logits_mean": shift_logits[shift_logits.size(0) // 2:].mean().detach(),
        "nll_loss": nll_loss.detach(),
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
    }

    return loss, metrics_dict


# GRPO family
@dataclass(frozen=True)
class GRPOAlgorithmPreset:
    """Static description of one GRPO-family algorithm.

    A preset ties together the pieces that have to agree for the algorithm to be
    *correct*, instead of leaving them as freely mixable hyperparameters:

    ``policy_loss``
        Key in :data:`lightning_grpo.utils.loss.core_algos.POLICY_LOSS_REGISTRY`.
    ``advantage_estimator``
        Default key in ``core_algos.ADV_ESTIMATOR_REGISTRY``.
    ``loss_agg_mode``
        Token/sequence aggregation the paper assumes.
    ``epsilon_high``
        Upper clipping bound used when ``rollout.epsilon_high`` is left as ``None``.
    ``clip_reporting``
        How the token-level clipping diagnostics are derived
        (``"ratio"`` | ``"cispo"`` | ``"none"``).
    ``liger_supported``
        Whether ``LigerFusedLinearGRPOLoss`` implements this objective.
    """

    policy_loss: str
    advantage_estimator: str
    loss_agg_mode: str
    epsilon_high: float | None = None
    norm_adv_by_std_in_grpo: bool = True
    clip_reporting: str = "ratio"
    liger_supported: bool = True
    description: str = ""


GRPO_ALGORITHM_PRESETS: dict[str, GRPOAlgorithmPreset] = {
    "grpo": GRPOAlgorithmPreset(
        policy_loss="vanilla",
        advantage_estimator="grpo",
        loss_agg_mode="token-mean",
        description="PPO-style clipped surrogate with group-normalized GRPO advantages.",
    ),
    "dapo": GRPOAlgorithmPreset(
        policy_loss="vanilla",
        advantage_estimator="grpo",
        loss_agg_mode="token-mean",
        # "clip-higher": a looser upper than lower bound keeps low-probability tokens
        # explorable (DAPO uses 0.2 / 0.28).
        epsilon_high=0.28,
        description=(
            "DAPO (https://arxiv.org/pdf/2503.14476): token-level objective (token-mean, "
            "already the default) plus 'clip-higher' - epsilon_high (0.28) is above epsilon "
            "(0.2) so the upper clip bound is looser, which keeps low-probability tokens "
            "explorable. Use kl_beta=0.0 for the KL-free objective and "
            "drop_zero_advantage_groups=true for DAPO's dynamic sampling (groups with a "
            "degenerate reward signal are removed from the gradient instead of being "
            "resampled). DAPO's overlong reward shaping is available through "
            "overlong_buffer_len / overlong_penalty_factor."
        ),
    ),
    "dr_grpo": GRPOAlgorithmPreset(
        policy_loss="vanilla",
        advantage_estimator="grpo",
        loss_agg_mode="seq-mean-token-sum-norm",
        norm_adv_by_std_in_grpo=False,
        description=(
            "Dr.GRPO (https://arxiv.org/abs/2503.20783): drop the std scaling of the "
            "advantage (norm_adv_by_std_in_grpo=false) and divide every sequence's token "
            "sum by a constant horizon so long completions are no longer overweighted."
        ),
    ),
    "cispo": GRPOAlgorithmPreset(
        policy_loss="cispo",
        advantage_estimator="grpo",
        loss_agg_mode="token-mean",
        epsilon_high=5.0,
        clip_reporting="cispo",
        description=(
            "CISPO (https://arxiv.org/pdf/2506.13585): clip the *detached* importance "
            "weight bilaterally and multiply it by the current log-probability, so gradients "
            "only flow through log pi. epsilon_high defaults to 5.0 to preserve the "
            "historical upper-only clamp; lower it for a symmetric objective."
        ),
    ),
    "gspo": GRPOAlgorithmPreset(
        policy_loss="gspo",
        advantage_estimator="grpo",
        loss_agg_mode="seq-mean-token-mean",
        liger_supported=False,
        description=(
            "GSPO (https://arxiv.org/abs/2507.18071): clip a *sequence-level* importance "
            "ratio (geometric mean of the token ratios), so the whole completion is either "
            "accepted or rejected as a unit."
        ),
    ),
    "sapo": GRPOAlgorithmPreset(
        policy_loss="sapo",
        advantage_estimator="grpo",
        # SAPO hard-codes its own seq-mean-token-mean aggregation internally; this mode is
        # only used for the KL term.
        loss_agg_mode="token-mean",
        liger_supported=False,
        description=(
            "SAPO: smooth, temperature-controlled (tau_pos/tau_neg) surrogate that softens "
            "the hard PPO clipping boundary into a sigmoid gate."
        ),
    ),
    "geo_mean": GRPOAlgorithmPreset(
        policy_loss="geo_mean",
        advantage_estimator="grpo",
        loss_agg_mode="token-mean",
        liger_supported=False,
        description="Symmetric geometric-mean clipped surrogate (clip ratio both sides).",
    ),
    "dro": GRPOAlgorithmPreset(
        policy_loss="dro",
        advantage_estimator="grpo",
        loss_agg_mode="token-mean",
        liger_supported=False,
        description="DRO: distributionally-robust surrogate (dro_beta) over the ratio.",
    ),
    "clip_cov": GRPOAlgorithmPreset(
        policy_loss="clip_cov",
        advantage_estimator="grpo",
        loss_agg_mode="token-mean",
        liger_supported=False,
        description="Clip-Cov: mask the highest-covariance (ratio, advantage) tokens.",
    ),
    "kl_cov": GRPOAlgorithmPreset(
        policy_loss="kl_cov",
        advantage_estimator="grpo",
        loss_agg_mode="token-mean",
        liger_supported=False,
        description="KL-Cov: variant of clip_cov that masks on the measured KL.",
    ),
    "dppo_tv": GRPOAlgorithmPreset(
        policy_loss="dppo_tv",
        advantage_estimator="grpo",
        loss_agg_mode="token-mean",
        liger_supported=False,
        description="DPPO with a total-variation trust region instead of a ratio clip.",
    ),
    "dppo_kl": GRPOAlgorithmPreset(
        policy_loss="dppo_kl",
        advantage_estimator="grpo",
        loss_agg_mode="token-mean",
        liger_supported=False,
        description="DPPO with a KL-divergence trust region instead of a ratio clip.",
    ),
    "gpg": GRPOAlgorithmPreset(
        policy_loss="gpg",
        advantage_estimator="gpg",
        loss_agg_mode="token-mean",
        clip_reporting="none",
        liger_supported=False,
        description="GPG: plain advantage-weighted log-probabilities, no ratio at all.",
    ),
    "reinforce": GRPOAlgorithmPreset(
        policy_loss="reinforce",
        advantage_estimator="reinforce_plus_plus",
        loss_agg_mode="seq-mean-token-sum",
        clip_reporting="none",
        liger_supported=False,
        description=(
            "REINFORCE: raw advantage-weighted log-probabilities with no clipping. Pair it "
            "with advantage_estimator=reinforce_plus_plus (batch-whitened) or "
            "reinforce_plus_plus_baseline (group-mean centred, then whitened)."
        ),
    ),
}

#: ``bnpo`` used to be a pure alias of ``grpo`` (its Liger name). It is kept as an
#: accepted-but-deprecated spelling so existing configs keep working.
_DEPRECATED_LOSS_TYPE_ALIASES: dict[str, str] = {"bnpo": "grpo"}
_WARNED_LOSS_TYPES: set[str] = set()


@dataclass(frozen=True)
class ResolvedGRPOAlgorithm:
    """Concrete (advantage estimator, policy objective) pair for one training run."""

    loss_type: str
    preset: GRPOAlgorithmPreset
    advantage_estimator: str
    loss_agg_mode: str
    norm_adv_by_std_in_grpo: bool
    epsilon_low: float
    epsilon_high: float
    actor_config: algos.ActorConfig


def _normalize_loss_type(loss_type: str) -> str:
    """Resolve deprecated spellings and validate the preset name."""

    name = str(loss_type).lower()
    alias = _DEPRECATED_LOSS_TYPE_ALIASES.get(name)
    if alias is not None:
        if name not in _WARNED_LOSS_TYPES:
            _WARNED_LOSS_TYPES.add(name)
            warnings.warn(
                f"rollout.loss_type='{name}' is deprecated and identical to '{alias}'; "
                f"use loss_type='{alias}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return alias
    if name not in GRPO_ALGORITHM_PRESETS:
        raise ValueError(
            f"Unknown rollout.loss_type={loss_type!r}. Supported algorithms: "
            f"{sorted(GRPO_ALGORITHM_PRESETS)}."
        )
    return name


def resolve_grpo_algorithm(rollout_config: Any) -> ResolvedGRPOAlgorithm:
    """Map ``config.rollout`` onto a policy objective plus an advantage estimator.

    ``advantage_estimator`` / ``loss_agg_mode`` override the preset defaults when set,
    which is what allows combinations such as the DAPO objective with RLOO advantages.
    """

    loss_type = _normalize_loss_type(rollout_config.loss_type)
    preset = GRPO_ALGORITHM_PRESETS[loss_type]

    epsilon_low = float(rollout_config.epsilon)
    epsilon_high = rollout_config.epsilon_high
    if epsilon_high is None:
        epsilon_high = preset.epsilon_high if preset.epsilon_high is not None else epsilon_low
    epsilon_high = float(epsilon_high)

    advantage_estimator = rollout_config.advantage_estimator or preset.advantage_estimator
    if advantage_estimator not in algos.ADV_ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown rollout.advantage_estimator={advantage_estimator!r}. Supported: "
            f"{sorted(algos.ADV_ESTIMATOR_REGISTRY)}."
        )

    requested_norm = rollout_config.norm_adv_by_std_in_grpo
    norm_adv_by_std_in_grpo = preset.norm_adv_by_std_in_grpo if requested_norm is None else bool(requested_norm)
    if requested_norm is not None and bool(requested_norm) != preset.norm_adv_by_std_in_grpo:
        # Explicitness wins, but the combination is almost never what was meant.
        warnings.warn(
            f"loss_type='{loss_type}' expects norm_adv_by_std_in_grpo="
            f"{preset.norm_adv_by_std_in_grpo}; the explicitly requested "
            f"norm_adv_by_std_in_grpo={bool(requested_norm)} is used instead.",
            stacklevel=3,
        )

    loss_agg_mode = rollout_config.loss_agg_mode or preset.loss_agg_mode

    actor_config = algos.ActorConfig(
        clip_ratio=epsilon_low,
        clip_ratio_low=epsilon_low,
        clip_ratio_high=epsilon_high,
        clip_ratio_c=float(getattr(rollout_config, "clip_ratio_c", 3.0)),
        gamma=float(getattr(rollout_config, "gamma", 1.0)),
        tau_pos=float(getattr(rollout_config, "sapo_tau_pos", 1.0)),
        tau_neg=float(getattr(rollout_config, "sapo_tau_neg", 1.0)),
        policy_loss=algos.PolicyLossSubConfig(
            dro_beta=float(getattr(rollout_config, "dro_beta", 0.1)),
            clip_cov_ratio=float(getattr(rollout_config, "clip_cov_ratio", 2.0e-4)),
            clip_cov_lb=float(getattr(rollout_config, "clip_cov_lb", 1.0)),
            clip_cov_ub=float(getattr(rollout_config, "clip_cov_ub", 5.0)),
            kl_cov_ratio=float(getattr(rollout_config, "kl_cov_ratio", 2.0e-4)),
            ppo_kl_coef=float(getattr(rollout_config, "ppo_kl_coef", 1.0)),
        ),
        extra={"f_norm": 1.0, "alpha": float(getattr(rollout_config, "gpg_alpha", 1.0))},
    )
    return ResolvedGRPOAlgorithm(
        loss_type=loss_type,
        preset=preset,
        advantage_estimator=advantage_estimator,
        loss_agg_mode=loss_agg_mode,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        actor_config=actor_config,
    )


def _call_advantage_estimator(fn: Any, token_level_rewards: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Call a registered advantage estimator, forwarding only the kwargs it declares.

    The registry mixes signatures (``index`` only for group-based estimators,
    ``reward_baselines`` only for ReMax, ``config`` whenever a discount factor or
    heuristic weight is needed), so filtering by signature keeps this dispatcher
    version tolerant.
    """

    parameters = inspect.signature(fn).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        selected = kwargs
    else:
        selected = {name: value for name, value in kwargs.items() if name in parameters}
    advantages, _ = fn(token_level_rewards, **selected)
    return advantages


@dataclass
class GRPORolloutRewards:
    """Rewards of the current rollout batch, locally and gathered across ranks."""

    local: torch.Tensor
    global_rewards: torch.Tensor
    global_per_func: torch.Tensor
    weights: torch.Tensor


@dataclass
class GRPOAdvantages:
    """Advantages of the current rollout batch, locally and gathered across ranks."""

    local: torch.Tensor
    global_advantages: torch.Tensor


@dataclass
class GRPOLossContext:
    """Everything a concrete policy objective needs for one rollout batch."""

    rollout_batch: dict[str, Any]
    loss_mask: torch.Tensor
    advantages: torch.Tensor
    rewards: GRPORolloutRewards


class GRPOLossComputer(ABC):
    """Template for a GRPO-family loss: rewards -> advantages -> objective -> metrics.

    Subclasses only implement :meth:`_policy_loss`; the reward computation, the
    distributed advantage normalization and the metric contract are shared here, which is
    exactly the logic that used to be duplicated between the standard and the Liger
    kernel implementations.
    """

    #: ``loss_type`` values the subclass can actually evaluate.
    supported_loss_types: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        module: Any,
        reward_manager: Any,
        metrics_aggregator: Any,
        *,
        rollout_temperature: float,
        loss_parallel_enabled: bool = False,
    ) -> None:
        self.module = module
        self.reward_manager = reward_manager
        self.metrics_aggregator = metrics_aggregator
        self.rollout_temperature = rollout_temperature
        self.loss_parallel_enabled = loss_parallel_enabled
        self.algorithm = resolve_grpo_algorithm(module.config.rollout)
        self._uses_greedy_baseline = self.algorithm.advantage_estimator == "remax"

    # ------------------------------------------------------------------ API
    @abstractmethod
    def _policy_loss(self, context: GRPOLossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the scalar objective plus *local* per-token diagnostics."""

    def compute_loss(
        self,
        rollout_batch: dict[str, Any],
        *,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the objective and the logged metrics for one rollout batch."""

        num_generations = self.module.rollout_coordinator.resolve_num_generations(training)
        rewards = self._compute_rewards(rollout_batch)
        advantages = self._compute_advantages(rollout_batch, rewards, num_generations=num_generations)
        context = GRPOLossContext(
            rollout_batch=rollout_batch,
            loss_mask=self._resolve_loss_mask(rollout_batch),
            advantages=advantages.local,
            rewards=rewards,
        )
        loss, local_metrics = self._policy_loss(context)
        metrics = self._build_training_metrics(
            rollout_batch, rewards, advantages, local_metrics, num_generations
        )
        return loss, metrics

    # ------------------------------------------------------- shared helpers
    def _resolve_loss_mask(self, rollout_batch: dict[str, Any]) -> torch.Tensor:
        """Completion mask, further masked by the tool-call mask when present."""

        loss_mask = rollout_batch["completion_mask"]
        tool_mask = rollout_batch.get("tool_mask")
        if tool_mask is not None:
            loss_mask = loss_mask * tool_mask
        return loss_mask

    def _compute_rewards(self, rollout_batch: dict[str, Any]) -> GRPORolloutRewards:
        """Score the completions locally, then gather the per-function rewards."""

        rewards, rewards_per_func = self.reward_manager.compute_rewards(
            prompts=rollout_batch["prompts"],
            completions=rollout_batch["completions"],
            completion_id_lists=rollout_batch["completion_id_lists"],
            metadata=rollout_batch["metadata"],
        )
        rewards = self._apply_overlong_penalty(rollout_batch, rewards)
        global_per_func = self.metrics_aggregator.gather_tensor(rewards_per_func.detach())
        weights = self.reward_manager.reward_weight_tensor.to(global_per_func.device)
        global_rewards = (global_per_func * weights.unsqueeze(0)).nansum(dim=-1)
        return GRPORolloutRewards(
            local=rewards.detach(),
            global_rewards=global_rewards,
            global_per_func=global_per_func,
            weights=weights,
        )

    def _apply_overlong_penalty(self, rollout_batch: dict[str, Any], rewards: torch.Tensor) -> torch.Tensor:
        """DAPO-style overlong reward shaping (linear length penalty).

        Disabled unless ``rollout.overlong_buffer_len > 0``. Completions longer than
        ``max_completion_length - overlong_buffer_len`` are penalised linearly, up to
        ``overlong_penalty_factor``, and only when their total reward is non-positive -
        correct-but-long answers keep their reward, as in DAPO.
        """

        buffer_len = int(getattr(self.module.config.rollout, "overlong_buffer_len", 0) or 0)
        if buffer_len <= 0:
            return rewards
        factor = float(getattr(self.module.config.rollout, "overlong_penalty_factor", 1.0))
        horizon = int(self.module.config.rollout.max_completion_length)
        lengths = rollout_batch["completion_mask"].sum(dim=1).to(rewards.dtype)
        exceedance = ((lengths - (horizon - buffer_len)) / float(buffer_len)).clamp(min=0.0, max=1.0)
        penalty = -factor * exceedance
        return rewards + torch.where(rewards <= 0, penalty, torch.zeros_like(penalty))

    def _compute_advantages(
        self,
        rollout_batch: dict[str, Any],
        rewards: GRPORolloutRewards,
        *,
        num_generations: int,
    ) -> GRPOAdvantages:
        """Run the configured advantage estimator over the *global* rollout batch.

        Normalizing on the gathered batch (rather than per rank) keeps every prompt
        group complete: with ``num_generations`` completions per prompt, a rank-local
        batch would otherwise split groups across ranks and change the baseline.
        """

        sample_ids = rollout_batch["sample_ids"].detach()
        global_sample_ids = self.metrics_aggregator.gather_tensor(sample_ids)

        # Every estimator reachable from here is outcome-only: the scalar reward sits on the
        # last valid token and the resulting advantage is broadcast over the sequence. A
        # single "outcome token" per completion is therefore exact, and it keeps the
        # gathered tensor independent of the per-rank padded completion width. As a
        # consequence sequence statistics are computed over completions (as intended)
        # rather than over padded tokens.
        token_level_rewards = rewards.global_rewards.unsqueeze(-1)
        estimator_kwargs: dict[str, Any] = {
            "response_mask": torch.ones_like(token_level_rewards),
            "index": global_sample_ids.cpu().numpy(),
            "epsilon": float(self.module.config.rollout.advantage_epsilon),
            "norm_adv_by_std_in_grpo": bool(self.algorithm.norm_adv_by_std_in_grpo),
            "config": self.algorithm.actor_config,
            **self.algorithm.actor_config.extra,
        }
        if self._uses_greedy_baseline:
            estimator_kwargs["reward_baselines"] = self._remax_baselines(rollout_batch)

        estimator = algos.get_adv_estimator_fn(self.algorithm.advantage_estimator)
        global_advantages = _call_advantage_estimator(estimator, token_level_rewards, **estimator_kwargs).squeeze(-1)
        local_advantages = self._local_advantages(
            global_advantages,
            global_sample_ids,
            sample_ids,
            num_generations=num_generations,
        )
        return GRPOAdvantages(local=local_advantages, global_advantages=global_advantages)

    def _remax_baselines(self, rollout_batch: dict[str, Any]) -> torch.Tensor:
        """Score the greedy baseline completions produced by the rollout coordinator."""

        baselines = rollout_batch.get("baseline_completions")
        if not baselines:
            raise RuntimeError(
                "advantage_estimator='remax' needs a greedy baseline rollout. Set "
                "rollout.advantage_estimator='remax' so the rollout coordinator generates it."
            )
        prompts = list(rollout_batch["prompts"])
        repeats = max(1, len(baselines) // max(1, len(prompts)))
        _, baseline_per_func = self.reward_manager.compute_rewards(
            prompts=[prompt for prompt in prompts for _ in range(repeats)],
            completions=list(baselines),
            completion_id_lists=rollout_batch["baseline_completion_id_lists"],
            metadata=rollout_batch["baseline_metadata"],
        )
        weights = self.reward_manager.reward_weight_tensor.to(baseline_per_func.device)
        local_baselines = (baseline_per_func * weights.unsqueeze(0)).nansum(dim=-1)
        return self.metrics_aggregator.gather_tensor(local_baselines.detach())

    def _local_advantages(
        self,
        global_advantages: torch.Tensor,
        global_sample_ids: torch.Tensor,
        local_sample_ids: torch.Tensor,
        *,
        num_generations: int,
    ) -> torch.Tensor:
        """Recover this rank's slice of the globally normalized advantages."""

        global_size = global_advantages.numel()
        local_size = local_sample_ids.numel()
        if local_size == 0 or global_size % local_size != 0:
            raise RuntimeError(
                f"Gathered reward batch ({global_size}) is not a whole multiple of the local "
                f"rollout batch ({local_size}). Every rank must hold the same local batch size."
            )
        process_index = getattr(getattr(self.module, "trainer", None), "global_rank", 0)
        start = int(process_index) * local_size
        stop = start + local_size
        if not torch.equal(global_sample_ids[start:stop].to(local_sample_ids.device), local_sample_ids):
            raise RuntimeError(
                "Failed to recover this rank's advantages from the gathered reward tensor. "
                "Ensure every rank receives the same local rollout batch size."
            )

        local_advantages = global_advantages[start:stop].to(local_sample_ids.device)
        keep = self._zero_advantage_group_mask(global_advantages, num_generations)
        if keep is not None:
            local_advantages = local_advantages * keep[start:stop].to(local_advantages.dtype)
        return local_advantages

    def _zero_advantage_group_mask(
        self,
        global_advantages: torch.Tensor,
        num_generations: int,
    ) -> torch.Tensor | None:
        """Per-completion keep mask that drops groups with a degenerate reward signal.

        This is the "light" form of DAPO's dynamic sampling: the prompt group is not
        discarded and resampled, its completions are simply removed from the gradient -
        they carry no learning signal either way.
        """

        if not getattr(self.module.config.rollout, "drop_zero_advantage_groups", False):
            return None
        if num_generations <= 1 or global_advantages.numel() % num_generations != 0:
            return None
        grouped = global_advantages.view(-1, num_generations)
        keep = grouped.abs().amax(dim=1) > 0.0
        return keep.repeat_interleave(num_generations)

    def _clip_diagnostics(
        self,
        *,
        per_token_log_probs: torch.Tensor,
        old_per_token_log_probs: torch.Tensor,
        token_advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Token-level clipping indicators consumed by :class:`GRPOMetricsAggregator`.

        The aggregator is built around per-token tensors, whereas ``core_algos`` reports a
        scalar ``pg_clipfrac``, so the indicators are rebuilt from the importance ratio.
        """

        zeros = torch.zeros_like(per_token_log_probs)
        reporting = self.algorithm.preset.clip_reporting
        if reporting == "none":
            return {
                "is_low_clipped": zeros,
                "is_high_clipped": zeros,
                "is_region_clipped": zeros,
                "is_cispo_clipped": zeros,
            }

        log_ratio = torch.clamp(per_token_log_probs - old_per_token_log_probs, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        low_bound = 1.0 - self.algorithm.epsilon_low
        high_bound = 1.0 + self.algorithm.epsilon_high
        if reporting == "cispo":
            # CISPO clips the detached weight on both sides regardless of the advantage sign.
            return {
                "is_low_clipped": zeros,
                "is_high_clipped": zeros,
                "is_region_clipped": zeros,
                "is_cispo_clipped": ((ratio < low_bound) | (ratio > high_bound)).to(ratio.dtype),
            }
        is_low_clipped = ((ratio < low_bound) & (token_advantages < 0)).to(ratio.dtype)
        is_high_clipped = ((ratio > high_bound) & (token_advantages > 0)).to(ratio.dtype)
        return {
            "is_low_clipped": is_low_clipped,
            "is_high_clipped": is_high_clipped,
            "is_region_clipped": torch.clamp(is_low_clipped + is_high_clipped, max=1.0),
            "is_cispo_clipped": zeros,
        }

    def _build_training_metrics(
        self,
        rollout_batch: dict[str, Any],
        rewards: GRPORolloutRewards,
        advantages: GRPOAdvantages,
        local_metrics: dict[str, torch.Tensor],
        num_generations: int,
    ) -> dict[str, torch.Tensor]:
        """Gather the local diagnostics and assemble the logged metric dict."""

        aggregator = self.metrics_aggregator
        completion_lengths = rollout_batch["completion_mask"].sum(dim=1).float()
        completion_truncated = rollout_batch["completion_truncated"].to(torch.float32)
        return aggregator.build_training_metrics(
            global_rewards_per_func=rewards.global_per_func,
            reward_weights=rewards.weights,
            global_shaped_rewards=rewards.global_rewards,
            num_generations=num_generations,
            global_per_token_kl=aggregator.gather_tensor(local_metrics["per_token_kl"].detach()),
            global_loss_mask=aggregator.gather_tensor(local_metrics["loss_mask"].detach()),
            global_entropy=aggregator.gather_tensor(local_metrics["entropy"].detach()),
            global_completion_lengths=aggregator.gather_tensor(completion_lengths.detach()),
            global_completion_truncated=aggregator.gather_tensor(completion_truncated.detach()),
            global_is_low_clipped=aggregator.gather_tensor(local_metrics["is_low_clipped"].detach()),
            global_is_high_clipped=aggregator.gather_tensor(local_metrics["is_high_clipped"].detach()),
            global_is_region_clipped=aggregator.gather_tensor(local_metrics["is_region_clipped"].detach()),
            global_is_cispo_clipped=aggregator.gather_tensor(local_metrics["is_cispo_clipped"].detach()),
            global_advantages=advantages.global_advantages,
            reward_names=self.module.config.reward.reward_funcs,
        )


def _call_policy_loss(
    fn: Any,
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    config: Any,
) -> torch.Tensor:
    """Call a registered policy loss, naming the old-policy log-probs as it expects.

    ``core_algos`` mixes ``old_log_prob`` (PPO-style) with ``rollout_log_prob`` (the
    REINFORCE loss), so the argument name is resolved from the signature.
    """

    parameters = inspect.signature(fn).parameters
    candidates = {
        "old_log_prob": old_log_prob,
        "rollout_log_prob": old_log_prob,
        "log_prob": log_prob,
        "advantages": advantages,
        "response_mask": response_mask,
        "loss_agg_mode": loss_agg_mode,
        "config": config,
    }
    loss, _ = fn(**{name: value for name, value in candidates.items() if name in parameters})
    return loss


class StandardGRPOLossComputer(GRPOLossComputer):
    """GRPO-family loss computed with regular PyTorch ops (materialized logits).

    Every preset in :data:`GRPO_ALGORITHM_PRESETS` is supported: the objective itself is
    looked up in ``core_algos.POLICY_LOSS_REGISTRY``, so adding an algorithm only requires
    registering it there and adding a preset here.
    """

    supported_loss_types = frozenset(GRPO_ALGORITHM_PRESETS)

    def __init__(
        self,
        module: Any,
        reward_manager: Any,
        metrics_aggregator: Any,
        *,
        rollout_temperature: float,
        loss_parallel_enabled: bool = False,
    ) -> None:
        if loss_parallel_enabled:
            raise RuntimeError(
                "Standard GRPO loss is incompatible with Tensor Parallel loss parallelism "
                "(distributed.tensor_parallel.loss_parallel=True). Disable loss parallelism to "
                "use this path; otherwise vocab-sharded logits would be silently replicated "
                "per rank, risking OOM."
            )
        super().__init__(
            module,
            reward_manager,
            metrics_aggregator,
            rollout_temperature=rollout_temperature,
            loss_parallel_enabled=loss_parallel_enabled,
        )

    def _policy_loss(self, context: GRPOLossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate the configured policy objective on materialized logits."""

        module = self.module
        rollout_batch = context.rollout_batch
        prompt_ids = rollout_batch["prompt_ids"]
        prompt_mask = rollout_batch["prompt_mask"]
        completion_ids = rollout_batch["completion_ids"]
        completion_mask = rollout_batch["completion_mask"]
        loss_mask = context.loss_mask
        logits_to_keep = completion_ids.size(1)

        model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        policy_outputs = module.policy(
            input_ids=model_input_ids,
            attention_mask=model_attention_mask,
            use_cache=False,
        )
        completion_logits = self._completion_logits(policy_outputs.logits, logits_to_keep)
        per_token_logps = selective_log_softmax(completion_logits, completion_ids)

        old_per_token_logps = rollout_batch.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        # Without a reference model the KL term is identically zero, which is what the old
        # ``per_token_logps.detach()`` fallback expressed.
        ref_per_token_logps = per_token_logps.detach()
        if module.reference_model is not None:
            with torch.no_grad():
                ref_outputs = module.reference_model(
                    input_ids=model_input_ids,
                    attention_mask=model_attention_mask,
                    use_cache=False,
                )
                ref_completion_logits = self._completion_logits(ref_outputs.logits, logits_to_keep)
                ref_per_token_logps = selective_log_softmax(ref_completion_logits, completion_ids)

        # Broadcast the per-sequence advantage over the tokens, zeroing the padding so the
        # objective sees a clean (advantages, response_mask) pair.
        token_advantages = context.advantages.to(per_token_logps.dtype).view(-1, 1)
        token_advantages = token_advantages.expand_as(per_token_logps) * loss_mask.to(per_token_logps.dtype)

        loss = _call_policy_loss(
            algos.get_policy_loss_fn(self.algorithm.preset.policy_loss),
            old_log_prob=old_per_token_logps,
            log_prob=per_token_logps,
            advantages=token_advantages,
            response_mask=loss_mask,
            loss_agg_mode=self.algorithm.loss_agg_mode,
            config=self.algorithm.actor_config,
        )

        per_token_kl = approx_kl_divergence(per_token_logps, ref_per_token_logps)
        kl_beta = float(module.config.rollout.kl_beta)
        if kl_beta != 0.0:
            loss = loss + kl_beta * algos.agg_loss(
                loss_mat=per_token_kl,
                loss_mask=loss_mask,
                loss_agg_mode=self.algorithm.loss_agg_mode,
            )

        with torch.no_grad():
            entropy = entropy_from_logits(completion_logits)
            clip_metrics = self._clip_diagnostics(
                per_token_log_probs=per_token_logps.detach(),
                old_per_token_log_probs=old_per_token_logps.detach(),
                token_advantages=token_advantages.detach(),
            )
        local_metrics = {
            "loss_mask": loss_mask.detach(),
            "per_token_kl": per_token_kl.detach(),
            "entropy": entropy.detach(),
            **{name: value.detach() for name, value in clip_metrics.items()},
        }
        return loss, local_metrics

    def _completion_logits(self, logits: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
        """Materialize the logits and slice out the columns that predict completion tokens."""

        materialized = materialize_vocab_parallel_logits(logits)
        completion_logits = materialized[:, :-1, :][:, -logits_to_keep:, :]
        if self.rollout_temperature != 1.0:
            completion_logits = completion_logits / self.rollout_temperature
        return completion_logits


def create_grpo_loss_computer(
    module: Any,
    reward_manager: Any,
    metrics_aggregator: Any,
    *,
    use_liger_kernel: bool,
    rollout_temperature: float,
    loss_parallel_enabled: bool = False,
    compiled: bool = True,
) -> GRPOLossComputer:
    """Build the loss computer, using the Liger fused kernel only where it applies.

    ``LigerFusedLinearGRPOLoss`` implements a fixed set of objectives, so the request is
    honoured only when the selected algorithm is one of them. Otherwise the standard
    PyTorch path is used (with a warning) instead of quietly training a different objective.
    """

    algorithm = resolve_grpo_algorithm(module.config.rollout)
    if use_liger_kernel and algorithm.preset.liger_supported:
        # Imported lazily: the Liger path needs the optional ``liger_kernel`` package and
        # this module is imported by every training entry point.
        from lightning_grpo.models.grpo.liger_loss import LigerGRPOLossComputer

        return LigerGRPOLossComputer(
            module,
            reward_manager,
            metrics_aggregator,
            rollout_temperature=rollout_temperature,
            loss_parallel_enabled=loss_parallel_enabled,
            compiled=compiled,
        )
    if use_liger_kernel:
        warnings.warn(
            "liger_kernel is enabled but LigerFusedLinearGRPOLoss does not implement "
            f"loss_type='{algorithm.loss_type}'; falling back to the standard PyTorch loss path.",
            stacklevel=2,
        )
    return StandardGRPOLossComputer(
        module,
        reward_manager,
        metrics_aggregator,
        rollout_temperature=rollout_temperature,
        loss_parallel_enabled=loss_parallel_enabled,
    )
