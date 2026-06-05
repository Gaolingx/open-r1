"""Shared optimization helpers for Lightning modules."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.optimization import get_scheduler
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.utils.configs.base import OptimizationConfig, ModelConfig
from lightning_grpo.utils.parallel.tp_utils import gather_state_dict_for_save

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def resolve_torch_dtype(model_param_dtype: str) -> torch.dtype:
    """Resolve the parameter dtype from configuration."""

    return DTYPE_MAP[model_param_dtype]


def get_gathered_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor] | None:
    """
    Internal helper to gather sharded TP/FSDP weights.
    Must be called on all ranks if TP is enabled.
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        raw_state_dict = model.state_dict()

    tp_plan = getattr(model, "tp_plan", None)
    device_mesh = getattr(model, "_device_mesh", None)
    tp_size = getattr(model, "_tp_size", 1)

    if tp_plan and device_mesh and tp_size > 1:
        gathered_sd = gather_state_dict_for_save(
            state_dict=raw_state_dict,
            tp_plan=tp_plan,
            device_mesh=device_mesh,
            tp_size=tp_size
        )
    else:
        gathered_sd = raw_state_dict

    if dist.is_initialized():
        if dist.get_rank() == 0:
            return gathered_sd
        return None

    return gathered_sd


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


def compute_per_token_logps(
    module: Any,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute old policy log-probabilities for generated completion tokens."""

    policy = module.policy if hasattr(module, "policy") else module
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    outputs = policy(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shift_logits = outputs.logits[:, :-1, :]
    completion_logits = shift_logits[:, -logits_to_keep:, :]
    if temperature != 1.0:
        completion_logits = completion_logits / temperature
    per_token_logps = torch.log_softmax(completion_logits, dim=-1).gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
    return per_token_logps * completion_mask.to(per_token_logps.dtype)


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


PT_SUBDIR = "pt_checkpoint"
HF_SUBDIR = "hf_checkpoint"
PT_FILENAME = "pretrain_model.ckpt"


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pth_weights_direct(state_dict: Dict[str, torch.Tensor], filepath: Path) -> Path:
    """Helper to save a pre-gathered state dict."""
    ensure_dir(filepath.parent)
    cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_state_dict, filepath)
    return filepath


def export_hf_model(
    model: PreTrainedModel,
    model_config: ModelConfig,
    export_dir: Path,
    *,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    safe_serialization: bool = True,
) -> Path:
    """Helper to save a model and its configuration file to a directory."""
    ensure_dir(export_dir)
    save_model = model if isinstance(model, PeftModel) else get_peft_base_model(model)

    save_kwargs = {
        "save_directory": str(export_dir),
        "safe_serialization": safe_serialization,
        "state_dict": state_dict
    }
    save_model.save_pretrained(**save_kwargs)
    resolved_tokenizer = tokenizer or (load_tokenizer(model_config) if model_config.tokenizer_name_or_path else None)
    if resolved_tokenizer:
        resolved_tokenizer.save_pretrained(str(export_dir))

    return export_dir


def export_configured_model(
    model: PreTrainedModel,
    model_config: ModelConfig,
    base_dir: Union[str, Path],
    *,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Dict[str, Path]:
    """Export model artifacts according to config flags using standard directory names."""
    root_path = ensure_dir(base_dir)
    exported_paths: Dict[str, Path] = {}

    full_state_dict = get_gathered_state_dict(model)
    if full_state_dict is None:
        return exported_paths

    if model_config.save_pth_format:
        pth_output_dir = root_path / PT_SUBDIR
        pth_file_path = pth_output_dir / PT_FILENAME
        save_pth_weights_direct(full_state_dict, pth_file_path)
        exported_paths["pth"] = pth_output_dir
    if model_config.save_safetensors_format:
        hf_output_dir = root_path / HF_SUBDIR
        export_hf_model(
            model=model,
            model_config=model_config,
            export_dir=hf_output_dir,
            tokenizer=tokenizer,
            state_dict=full_state_dict,
            safe_serialization=True
        )
        exported_paths["safetensors"] = hf_output_dir

    if dist.is_initialized():
        dist.barrier()

    return exported_paths
