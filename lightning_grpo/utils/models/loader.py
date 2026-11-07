import gc
import json
import os

import torch

from functools import partial
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.utils import check_torch_load_is_safe
from safetensors.torch import load_file as safe_load_file

from lightning_grpo.models.common import resolve_torch_dtype
from lightning_grpo.utils.configs.base import ModelConfig, PrecisionConfig

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"


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

    load_sharded_checkpoint(model, model_config.model_name_or_path)
    return model


def _find_tied_parameter_groups(model) -> list[list[str]]:
    """Group the names of parameters that share the same tensor.

    Port of `accelerate.utils.find_tied_parameters`, which is what
    `transformers.modeling_utils._find_missing_and_unexpected_keys` uses to reconcile the checkpoint keys with the
    model keys.
    """
    all_named_parameters = {name: param for name, param in model.named_parameters(remove_duplicate=False)}
    unique_named_parameters = {name: param for name, param in model.named_parameters(remove_duplicate=True)}

    # Parameters with duplicate names are exactly the tied ones.
    tied_param_names = set(all_named_parameters) - set(unique_named_parameters)

    tied_param_groups = {}
    for tied_param_name in tied_param_names:
        tied_param = all_named_parameters[tied_param_name]
        for param_name, param in unique_named_parameters.items():
            if param is tied_param:
                tied_param_groups.setdefault(param_name, set()).add(tied_param_name)

    return [sorted([weight, *tied]) for weight, tied in tied_param_groups.items()]


def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    """
    This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional*, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `True`):
            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not safe_index_present:
        filenames = (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME)
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = safe_index_present and (prefer_safe or not index_present)
    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loaded_keys = set(index["weight_map"].keys())

    if load_safe:
        loader = safe_load_file
    else:
        check_torch_load_is_safe()
        loader = partial(torch.load, map_location="cpu", weights_only=True)

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    # Reconcile keys *after* loading, like `transformers.modeling_utils._find_missing_and_unexpected_keys`.
    # Checking `weight_map` alone breaks on tied weights: e.g. Qwen3 saves only one copy of
    # `lm_head.weight`/`embed_tokens.weight`, so the other is legitimately absent from the index.
    model_keys = set(model.state_dict().keys())
    missing_keys = sorted(model_keys - loaded_keys)
    unexpected_keys = sorted(loaded_keys - model_keys)

    # Non-persistent buffers are absent from `state_dict()` but may still be stored in the checkpoint.
    model_buffers = {name for name, _ in model.named_buffers()}
    unexpected_keys = [key for key in unexpected_keys if key not in model_buffers]

    # A missing tied key is fine if another member of its group was loaded; only fully missing groups are errors.
    for group in _find_tied_parameter_groups(model):
        group = set(group)
        missing_in_group = [key for key in missing_keys if key in group]
        if 0 < len(missing_in_group) < len(group):
            missing_keys = [key for key in missing_keys if key not in missing_in_group]

    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nUnexpected key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    # Return the same thing as PyTorch load_state_dict function.
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
