"""Reflection helpers for resolving local Hugging Face model classes."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from types import ModuleType

from transformers import PreTrainedModel
from transformers.configuration_utils import PreTrainedConfig


@dataclass(frozen=True)
class ReflectedModelSpec:
    """Resolved model/config pair discovered from local modules."""

    model_class: type[PreTrainedModel]
    config_class: type[PreTrainedConfig]


class PreTrainedModelReflectionRegistry:
    """Discover local `PreTrainedModel` subclasses keyed by config `model_type`."""

    def __init__(self, package_names: tuple[str, ...]):
        if not package_names:
            raise ValueError("At least one package name is required for model reflection.")
        self._package_names = package_names
        self._registry: dict[str, ReflectedModelSpec] | None = None

    def resolve(self, model_type: str) -> ReflectedModelSpec:
        """Resolve a local model class from a `PreTrainedConfig.model_type` string."""

        normalized_model_type = (model_type or "").strip()
        if not normalized_model_type:
            raise ValueError("`model_type` must be a non-empty string.")

        registry = self._ensure_registry()
        spec = registry.get(normalized_model_type)
        if spec is None:
            available = ", ".join(sorted(registry)) or "<none>"
            raise ValueError(
                f"Unable to find a local PreTrainedModel class for model_type '{normalized_model_type}'. "
                f"Available model types: {available}."
            )
        return spec

    def _ensure_registry(self) -> dict[str, ReflectedModelSpec]:
        if self._registry is None:
            self._registry = self._build_registry()
        return self._registry

    def _build_registry(self) -> dict[str, ReflectedModelSpec]:
        registry: dict[str, ReflectedModelSpec] = {}
        for package_name in self._package_names:
            package = importlib.import_module(package_name)
            for module in self._iter_modules(package):
                for _, candidate in inspect.getmembers(module, inspect.isclass):
                    if not issubclass(candidate, PreTrainedModel) or candidate is PreTrainedModel:
                        continue
                    if inspect.isabstract(candidate):
                        continue
                    if not candidate.__name__.endswith("ForCausalLM"):
                        continue

                    config_class = getattr(candidate, "config_class", None)
                    if not inspect.isclass(config_class) or not issubclass(config_class, PreTrainedConfig):
                        continue

                    model_type = getattr(config_class, "model_type", None)
                    if not isinstance(model_type, str) or not model_type:
                        continue

                    existing = registry.get(model_type)
                    if existing is not None and existing.model_class is not candidate:
                        raise ValueError(
                            f"Duplicate reflected model_type '{model_type}' for "
                            f"{existing.model_class.__module__}.{existing.model_class.__name__} and "
                            f"{candidate.__module__}.{candidate.__name__}."
                        )

                    registry[model_type] = ReflectedModelSpec(
                        model_class=candidate,
                        config_class=config_class,
                    )

        return registry

    def _iter_modules(self, package: ModuleType) -> list[ModuleType]:
        modules = [package]
        package_path = getattr(package, "__path__", None)
        if package_path is None:
            return modules

        prefix = f"{package.__name__}."
        for module_info in pkgutil.walk_packages(package_path, prefix):
            modules.append(importlib.import_module(module_info.name))
        return modules


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
