"""Shared optimization helpers for Lightning modules."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import loss_parallel
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.optimization import get_scheduler
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.utils.configs.base import OptimizationConfig, PrecisionConfig, ModelConfig

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def resolve_torch_dtype(precision_config: PrecisionConfig) -> torch.dtype:
    """Resolve the parameter dtype from configuration."""

    return DTYPE_MAP[precision_config.model_param_dtype]


def compile_model_if_configured(model: torch.nn.Module, model_config: ModelConfig) -> torch.nn.Module:
    """Apply torch.compile after distributed wrapping is complete."""

    if model_config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, **model_config.compile_kwargs)
        rank_zero_info("Applied torch.compile to the model")
    return model


def unwrap_parallel_model(model: torch.nn.Module) -> torch.nn.Module:
    """Remove Lightning/Distributed wrappers while preserving model internals."""

    return model.module if hasattr(model, "module") else model


def get_peft_base_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying Hugging Face model from a PEFT wrapper."""

    model = unwrap_parallel_model(model)
    return model.get_base_model() if isinstance(model, PeftModel) else model


def get_transformer_backbone_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the module that owns decoder layers for PEFT/Hugging Face CausalLM wrappers."""

    model = get_peft_base_model(model)

    base_model_prefix = getattr(model, "base_model_prefix", None)
    if isinstance(base_model_prefix, str) and hasattr(model, base_model_prefix):
        candidate = getattr(model, base_model_prefix)
        if isinstance(candidate, torch.nn.Module) and hasattr(candidate, "layers"):
            return candidate
    for attribute in ("model", "transformer"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, torch.nn.Module) and hasattr(candidate, "layers"):
            return candidate
    return model


def get_lm_head_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the LM head from a possibly wrapped PEFT/CausalLM model."""

    model = get_peft_base_model(model)
    return model.lm_head


def build_optimizer(parameters: Any, optimization: OptimizationConfig) -> torch.optim.Optimizer:
    """Create the optimizer from configuration."""

    optimizer_config = optimization.optimizer
    optimizer_type = optimizer_config.type.lower()

    if optimizer_type in {"adamw8bit", "adam8bit"}:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ImportError("bitsandbytes is required for 8-bit optimizers.") from error

    if optimizer_type == "adamw8bit":
        return bnb.optim.AdamW8bit(
            parameters,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            amsgrad=optimizer_config.amsgrad,
        )

    if optimizer_type == "adam8bit":
        return bnb.optim.Adam8bit(
            parameters,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            amsgrad=optimizer_config.amsgrad,
        )

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            amsgrad=optimizer_config.amsgrad,
        )

    if optimizer_type == "adam":
        return torch.optim.Adam(
            parameters,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            amsgrad=optimizer_config.amsgrad,
        )

    if optimizer_type == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=optimizer_config.learning_rate,
            momentum=optimizer_config.momentum,
            dampening=optimizer_config.dampening,
            weight_decay=optimizer_config.weight_decay,
            nesterov=optimizer_config.nesterov,
        )

    raise ValueError(
        f"Unknown optimizer type: {optimizer_config.type}. Supported: adamw, adam, adamw8bit, adam8bit, sgd."
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    optimization: OptimizationConfig,
    estimated_stepping_batches: int,
) -> dict[str, Any]:
    """Create a per-step learning rate scheduler."""
    scheduler_config = optimization.scheduler

    if optimization.max_steps and optimization.max_steps > 0:
        total_steps = optimization.max_steps
    else:
        total_steps = max(1, estimated_stepping_batches)

    num_warmup_steps = min(max(0, scheduler_config.warmup_steps), total_steps)
    scheduler_specific_kwargs = dict(scheduler_config.scheduler_specific_kwargs)

    scheduler = get_scheduler(
        name=scheduler_config.type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs or None,
    )

    return {
        "scheduler": scheduler,
        "interval": "step",
        "frequency": 1,
        "name": "lr",
    }


def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizerBase:
    """Load and normalize the tokenizer."""

    if model_config.custom_model and not model_config.tokenizer_name_or_path:
        raise ValueError(f"custom_model requires `tokenizer_name_or_path` for tokenizer loading.")

    tokenizer_name = model_config.tokenizer_name_or_path or model_config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )

    if model_config.chat_template:
        tokenizer.chat_template = model_config.chat_template
    if model_config.eos_token:
        tokenizer.eos_token = model_config.eos_token
    if model_config.pad_token:
        tokenizer.pad_token = model_config.pad_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def count_trainable_parameters(model: PreTrainedModel) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def save_pth_weights(model: PreTrainedModel, filepath: str) -> Path | None:
    """Persist a plain PyTorch state dict next to the Lightning checkpoint."""

    path = Path(filepath)
    pth_path = path.with_suffix(".pth")
    pth_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state_dict, pth_path)
    return pth_path


def export_configured_model(
    model: PreTrainedModel,
    model_config: ModelConfig,
    base_dir: str | Path,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> dict[str, Path]:
    """Export model artifacts according to config flags using standard directory names."""

    export_root = Path(base_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    exported_paths: dict[str, Path] = {}

    if model_config.save_pth_format:
        pth_dir = export_root / "pt_checkpoint"
        pth_stem = pth_dir / "pretrain_model.ckpt"
        pth_path = save_pth_weights(model, pth_stem)
        if pth_path is not None:
            exported_paths["pth"] = pth_path

    if model_config.save_safetensors_format:
        hf_dir = export_root / "hf_checkpoint"
        export_hf_model(
            model,
            model_config,
            hf_dir,
            tokenizer=tokenizer,
            safe_serialization=True,
        )
        exported_paths["safetensors"] = hf_dir

    return exported_paths


def export_hf_model(
    model: PreTrainedModel,
    model_config: ModelConfig,
    export_dir: str | Path,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
    safe_serialization: bool = False,
) -> Path:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    save_model = model
    if isinstance(model, PeftModel):
        save_model = model
    else:
        save_model = get_peft_base_model(model)

    save_kwargs = {"safe_serialization": safe_serialization}
    if state_dict is not None:
        save_kwargs["state_dict"] = state_dict
    save_model.save_pretrained(str(export_path), **save_kwargs)

    resolved_tokenizer = tokenizer
    if resolved_tokenizer is None and model_config.tokenizer_name_or_path:
        resolved_tokenizer = load_tokenizer(model_config)
    if resolved_tokenizer is not None:
        resolved_tokenizer.save_pretrained(str(export_path))

    return export_path


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


def selective_log_softmax(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Gather token log-probabilities for target ids."""

    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


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


def compute_liger_cross_entropy_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_parallel_enabled: bool = False,
) -> torch.Tensor:
    """Compute token-level next-token loss using Liger Kernel's fused kernel."""
    from lightning_grpo.models.grpo.liger_loss import LigerCELossComputer

    loss_computer = LigerCELossComputer(model, loss_parallel_enabled=loss_parallel_enabled)
    return loss_computer.compute_loss(
        batch=batch,
        labels=labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
