"""Configuration loading helpers for the Lightning GRPO pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lightning_grpo.configs.base import ExperimentConfig
from lightning_grpo.configs.grpo import GRPOConfig
from lightning_grpo.configs.sft import SFTConfig

CONFIG_REGISTRY = {
    "sft": SFTConfig,
    "grpo": GRPOConfig,
}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a typed experiment configuration from YAML."""

    payload = load_config(path)

    task = payload.get("task", "sft")
    config_cls = CONFIG_REGISTRY.get(task)
    if config_cls is None:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {sorted(CONFIG_REGISTRY)}.")

    return config_cls._from_mapping(payload)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a raw experiment configuration mapping from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Save a typed experiment configuration to YAML."""

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)

    return config_path
