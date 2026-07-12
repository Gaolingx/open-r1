from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from lightning_grpo.models.common import resolve_torch_dtype
from lightning_grpo.utils.configs.base import ModelConfig, PrecisionConfig


def build_configured_model_class(
    model_config: ModelConfig,
    precision_config: PrecisionConfig,
) -> PreTrainedModel:
    """Build a model using AutoConfig.from_pretrained with custom overrides."""
    base_config = AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )

    for key, value in model_config.model_init_kwargs.items():
        setattr(base_config, key, value)

    model = AutoModelForCausalLM.from_config(
        base_config,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        dtype=resolve_torch_dtype(precision_config.model_param_dtype),
    ).cpu()

    _maybe_load_state_dict(model, model_config.model_name_or_path)
    return model


def _maybe_load_state_dict(model: PreTrainedModel, path: str) -> None:
    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir():
        return

    safetensors_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if safetensors_files:
        _load_safetensors_shards(model, safetensors_files)
        return

    bin_files = sorted(checkpoint_dir.glob("pytorch_model*.bin"))
    if bin_files:
        _load_torch_bin_shards(model, bin_files)
        return

    return


def _load_shards(
    model: PreTrainedModel,
    files: List[Path],
    load_fn,
) -> None:
    for f in files:
        shard = load_fn(f)
        model.load_state_dict(shard)
        del shard  # Release the current shard memory in time


def _load_safetensors_shards(model: PreTrainedModel, files: List[Path]) -> None:
    from safetensors.torch import load_file

    _load_shards(
        model,
        files,
        load_fn=lambda f: load_file(str(f), device="cpu"),
    )


def _load_torch_bin_shards(model: PreTrainedModel, files: List[Path]) -> None:
    _load_shards(
        model,
        files,
        load_fn=lambda f: torch.load(str(f), map_location="cpu"),
    )
