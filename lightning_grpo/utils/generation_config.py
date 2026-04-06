"""Helpers for managing rollout generation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import GenerationConfig

from lightning_grpo.utils.config import load_json_config


@dataclass
class GenerationConfigManager:
    """Thin wrapper around ``transformers.GenerationConfig`` for rollout and log-prob usage."""

    config: GenerationConfig

    @classmethod
    def from_json(cls, config_path: str | Path | None = None, **overrides: Any) -> "GenerationConfigManager":
        """Create a manager from a JSON file with optional keyword overrides."""

        if not config_path:
            return cls.from_kwargs(**overrides)

        config_data = dict(load_json_config(config_path))
        config_data.pop("transformers_version", None)
        config_data.update(overrides)
        return cls(GenerationConfig(**config_data))

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "GenerationConfigManager":
        """Create a manager directly from generation config keyword arguments."""

        default_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
        }
        default_kwargs.update(kwargs)
        return cls(GenerationConfig(**default_kwargs))

    @property
    def temperature(self) -> float:
        """Return a safe temperature for log-prob computation."""

        if self.config.temperature is None:
            return 1.0
        return float(self.config.temperature)

    def to_generation_config(self, **overrides: Any) -> GenerationConfig:
        """Return a concrete ``GenerationConfig`` instance for model.generate calls."""

        config_dict = {key: value for key, value in self.config.to_dict().items() if value is not None}
        config_dict.update(overrides)
        return GenerationConfig(**config_dict)

    def to_sampling_params(self, *, exclude_keys: set[str] | frozenset[str] | None = None) -> dict[str, Any]:
        """Return non-null config values as a plain dict."""

        excluded = set(exclude_keys or ())
        return {
            key: value
            for key, value in self.config.to_dict().items()
            if key not in excluded and value is not None
        }


def load_generation_config(config_path: str | Path | None = None, **kwargs: Any) -> GenerationConfigManager:
    """Load a generation config manager from JSON or keyword arguments."""

    if config_path:
        return GenerationConfigManager.from_json(config_path, **kwargs)
    return GenerationConfigManager.from_kwargs(**kwargs)
