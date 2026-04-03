"""Model and tokenizer utilities for the Lightning GRPO pipeline."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
from typing import Any, Callable, Optional

import lightning as L
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from lightning.pytorch.utilities import rank_zero_info
from transformers.configuration_utils import PreTrainedConfig

from lightning_grpo.utils.config import load_json_config
from lightning_grpo.utils.configs.base import ModelConfig, PrecisionConfig
from lightning_grpo.utils.reflection import PreTrainedModelReflectionRegistry, import_causal_lm_class

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

CustomModelBuilder = Callable[[ModelConfig, PrecisionConfig], PreTrainedModel]
LOCAL_MODEL_REGISTRY = PreTrainedModelReflectionRegistry(("lightning_grpo.module",))


def resolve_torch_dtype(precision_config: PrecisionConfig) -> torch.dtype:
    """Resolve the parameter dtype from configuration."""

    return DTYPE_MAP[precision_config.parameter_dtype]


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


def _freeze_embeddings_if_needed(model: PreTrainedModel, freeze_embeddings: bool) -> None:
    """Freeze token embeddings for parameter-efficient adaptation."""

    if not freeze_embeddings:
        return

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        for parameter in input_embeddings.parameters():
            parameter.requires_grad = False

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        for parameter in output_embeddings.parameters():
            parameter.requires_grad = False


def _apply_lora_if_needed(model: PreTrainedModel, model_config: ModelConfig) -> PreTrainedModel:
    """Wrap the model with LoRA adapters when enabled."""

    if not model_config.lora.enabled:
        return model

    if model_config.lora.init_path:
        lora_path = Path(model_config.lora.init_path).expanduser()
        rank_zero_info(f"Loading LoRA adapters from {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=True)

        if model_config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        return model

    lora_config = LoraConfig(
        r=model_config.lora.r,
        lora_alpha=model_config.lora.alpha,
        lora_dropout=model_config.lora.dropout,
        bias=model_config.lora.bias,
        target_modules=model_config.lora.target_modules,
        modules_to_save=model_config.lora.modules_to_save or None,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if model_config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model


def _resolve_model_init_kwargs(model_config: ModelConfig) -> dict:
    """Resolve custom model init kwargs from JSON config and inline overrides."""

    init_kwargs: dict = {}
    if model_config.model_config_path:
        config_path = Path(model_config.model_config_path)
        raw_config = load_json_config(config_path)
        init_kwargs.update(raw_config)

    init_kwargs.update(model_config.model_init_kwargs)
    return init_kwargs


def _resolve_checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Normalize checkpoint containers to a plain state dict."""

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected checkpoint to resolve to a mapping-based state dict.")

    return state_dict


def _maybe_load_custom_weights(model: PreTrainedModel, model_config: ModelConfig) -> PreTrainedModel:
    """Load an optional local PyTorch checkpoint into a freshly built model."""

    if model_config.custom_weight_dir is None:
        return model

    weight_path = Path(model_config.custom_weight_dir).expanduser()
    if weight_path.suffix == "":
        pth_candidate = weight_path.with_suffix(".pth")
        if pth_candidate.exists():
            weight_path = pth_candidate

    if not weight_path.exists():
        raise FileNotFoundError(f"Custom checkpoint not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = _resolve_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model


def _build_reflected_model(model_config: ModelConfig, precision_config: PrecisionConfig) -> PreTrainedModel:
    """Build a local `PreTrainedModel` by matching config `model_type` via reflection."""

    init_kwargs = _resolve_model_init_kwargs(model_config)
    model_type = init_kwargs.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError(
            "Custom reflected model loading requires `model_type` in `model_config_path` or `model_init_kwargs`."
        )

    reflected_spec = LOCAL_MODEL_REGISTRY.resolve(model_type)
    model_hf_config = reflected_spec.config_class(**init_kwargs)
    model = reflected_spec.model_class(model_hf_config)
    model = model.to(dtype=resolve_torch_dtype(precision_config))

    return _maybe_load_custom_weights(model, model_config)


def _build_configured_model_class(model_config: ModelConfig, precision_config: PrecisionConfig) -> PreTrainedModel:
    """Build a local model from an explicit YAML-configured class path."""

    init_kwargs = _resolve_model_init_kwargs(model_config)
    model_class = import_causal_lm_class(model_config.model_class_path or "")

    config_class = getattr(model_class, "config_class", None)
    if not inspect.isclass(config_class) or not issubclass(config_class, PreTrainedConfig):
        raise TypeError(
            f"Configured model class must expose a PreTrainedConfig `config_class`: {model_config.model_class_path}"
        )

    model_hf_config = config_class(**init_kwargs)
    model = model_class(model_hf_config)
    model = model.to(dtype=resolve_torch_dtype(precision_config))

    return _maybe_load_custom_weights(model, model_config)


def load_causal_lm(model_config: ModelConfig, precision_config: PrecisionConfig) -> PreTrainedModel:
    """Load a decoder-only language model with optional LoRA support."""

    if model_config.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
            dtype=resolve_torch_dtype(precision_config),
        )
    elif model_config.model_class_path:
        model = _build_configured_model_class(model_config, precision_config)
    else:
        model = _build_reflected_model(model_config, precision_config)

    if hasattr(model, "config"):
        model.config.use_cache = model_config.use_cache

    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    _freeze_embeddings_if_needed(model, model_config.freeze_embeddings)
    model = _apply_lora_if_needed(model, model_config)

    if model_config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model


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


def resolve_export_model(pl_module: L.LightningModule) -> torch.nn.Module | None:
    """Return the underlying trainable model that should be exported."""

    return getattr(pl_module, "policy", None) or getattr(pl_module, "model", None)


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


def unwrap_model(model: PreTrainedModel) -> PreTrainedModel:
    return model.get_base_model() if isinstance(model, PeftModel) else model


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
        save_model = unwrap_model(model)

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


@staticmethod
def format_metric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu().item())
    return float(value)


def collect_moe_metrics(outputs: Any) -> dict[str, torch.Tensor]:
    """Extract aggregate MoE routing diagnostics from model outputs."""

    metrics: dict[str, torch.Tensor] = {}

    def _get_output_value(outputs: Any, key: str) -> Any:
        if isinstance(outputs, dict):
            return outputs.get(key)
        return getattr(outputs, key, None)

    if outputs is None:
        return metrics

    router_logits = _get_output_value(outputs, "router_logits")
    aux_loss = _get_output_value(outputs, "aux_loss")
    z_loss = _get_output_value(outputs, "z_loss")

    if aux_loss is not None:
        metrics["aux_loss"] = aux_loss.detach().to(dtype=torch.float32)
    if z_loss is not None:
        metrics["z_loss"] = z_loss.detach().to(dtype=torch.float32)

    if router_logits is None:
        return metrics

    if torch.is_tensor(router_logits):
        router_logits = (router_logits,)
    elif not isinstance(router_logits, (tuple, list)):
        return metrics

    valid_router_logits = [layer_logits for layer_logits in router_logits if torch.is_tensor(layer_logits)]
    if not valid_router_logits:
        return metrics

    layer_entropies: list[torch.Tensor] = []
    layer_load_std: list[torch.Tensor] = []
    layer_top1_occupancies: list[torch.Tensor] = []
    layer_dead_expert_fractions: list[torch.Tensor] = []

    for layer_logits in valid_router_logits:
        probs = layer_logits.detach().to(dtype=torch.float32)
        if probs.ndim == 0 or probs.shape[-1] == 0:
            continue

        probs = probs.reshape(-1, probs.shape[-1])
        if probs.numel() == 0:
            continue

        row_sums = probs.sum(dim=-1)
        is_probability_distribution = torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1.0e-4,
            rtol=1.0e-4,
        ) and torch.all(probs >= 0)
        if not is_probability_distribution:
            probs = torch.softmax(probs, dim=-1)

        mean_probs = probs.mean(dim=0)
        entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=-1).mean()
        load_std = mean_probs.std(unbiased=False)
        top1_experts = probs.argmax(dim=-1)
        num_experts = probs.shape[-1]
        top1_counts = torch.bincount(top1_experts, minlength=num_experts).to(dtype=torch.float32)
        top1_occupancy = (top1_counts / max(top1_experts.numel(), 1)).max()
        dead_expert_fraction = (top1_counts == 0).to(dtype=torch.float32).mean()

        layer_entropies.append(entropy)
        layer_load_std.append(load_std)
        layer_top1_occupancies.append(top1_occupancy)
        layer_dead_expert_fractions.append(dead_expert_fraction)

    if layer_entropies:
        metrics["router_entropy"] = torch.stack(layer_entropies).mean()
        metrics["expert_load_std"] = torch.stack(layer_load_std).mean()
        metrics["top1_expert_occupancy"] = torch.stack(layer_top1_occupancies).mean()
        metrics["dead_expert_fraction"] = torch.stack(layer_dead_expert_fractions).mean()

    return metrics


def log_moe_metrics(
    module: L.LightningModule,
    outputs_or_metrics: Any,
    stage: str,
    *,
    on_step: bool,
    on_epoch: bool = True,
    sync_dist: bool = True,
) -> None:
    """Log shared MoE diagnostics from raw outputs or a precomputed metric dict."""

    metrics = collect_moe_metrics(outputs_or_metrics)

    if isinstance(outputs_or_metrics, dict):
        get_metric = outputs_or_metrics.get
        metrics.update({
            "aux_loss": get_metric("aux_loss", metrics.get("aux_loss")),
            "z_loss": get_metric("z_loss", metrics.get("z_loss")),
            "router_entropy": get_metric("router_entropy", metrics.get("router_entropy")),
            "expert_load_std": get_metric("expert_load_std", metrics.get("expert_load_std")),
            "top1_expert_occupancy": get_metric("top1_expert_occupancy", metrics.get("top1_expert_occupancy")),
            "dead_expert_fraction": get_metric("dead_expert_fraction", metrics.get("dead_expert_fraction")),
        })
        metrics = {key: value for key, value in metrics.items() if value is not None}

    if not metrics:
        return

    log_kwargs = {"prog_bar": False, "on_step": on_step, "on_epoch": on_epoch, "sync_dist": sync_dist}

    for key, value in metrics.items():
        formatted_value = format_metric_value(value)
        if formatted_value is not None:
            module.log(f"{stage}/moe_{key}", formatted_value, **log_kwargs)
