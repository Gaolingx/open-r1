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

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}

    task = payload.get("task", "sft")
    config_cls = CONFIG_REGISTRY.get(task)
    if config_cls is None:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {sorted(CONFIG_REGISTRY)}.")

    return config_cls._from_mapping(payload)
