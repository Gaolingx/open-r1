"""Configuration loading helpers for the Lightning GRPO pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
import json

from lightning_grpo.configs.base import ExperimentConfig
from lightning_grpo.configs.grpo import GRPOConfig
from lightning_grpo.configs.sft import SFTConfig

CONFIG_REGISTRY = {
    "sft": SFTConfig,
    "grpo": GRPOConfig,
}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a typed experiment configuration from YAML."""

    payload = load_yaml_config(path)

    task = payload.get("task", "sft")
    config_cls = CONFIG_REGISTRY.get(task)
    if config_cls is None:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {sorted(CONFIG_REGISTRY)}.")

    return config_cls._from_mapping(payload)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a raw experiment configuration mapping from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml_config(config: Dict[str, Any], path: str | Path) -> None:
    """Save a typed experiment configuration to YAML."""

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def load_json_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg


def save_json_config(config: Dict[str, Any], path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
