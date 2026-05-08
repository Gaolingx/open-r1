"""Reflection helpers for resolving local Hugging Face model classes."""

from __future__ import annotations

import importlib
import inspect

from transformers import PreTrainedModel


def import_causal_lm_class(class_path: str) -> type[PreTrainedModel]:
    """Import a custom `*ForCausalLM` class from a fully-qualified dotted path."""

    module_path, _, class_name = class_path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(
            "`model.model_class_path` must be a fully-qualified dotted path, for example "
            "`lightning_grpo.module.minimind.modeling_minimind_moe.MiniMindMoeForCausalLM`."
        )

    module = importlib.import_module(module_path)
    class_object = getattr(module, class_name, None)
    if not inspect.isclass(class_object) or not issubclass(class_object, PreTrainedModel):
        raise TypeError(f"Configured model class must be a PreTrainedModel subclass: {class_path}")
    if not class_object.__name__.endswith("ForCausalLM"):
        raise TypeError(f"Configured model class must end with `ForCausalLM`: {class_path}")

    return class_object
