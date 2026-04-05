from __future__ import annotations

from pathlib import Path

from lightning_grpo.utils.config import load_yaml_config
from lightning_grpo.utils.configs.base import TrainingBaseConfig
from lightning_grpo.utils.configs.grpo import GRPOConfig
from lightning_grpo.utils.configs.pretrain import PretrainConfig
from lightning_grpo.utils.configs.sft import SFTConfig

CONFIG_REGISTRY = {
    "sft": SFTConfig,
    "grpo": GRPOConfig,
    "pretrain": PretrainConfig,
}


def load_experiment_config(path: str | Path) -> TrainingBaseConfig:
    """Load a typed experiment configuration from YAML."""

    payload = load_yaml_config(path)
    task = payload.get("task", "sft")
    config_cls = CONFIG_REGISTRY.get(task)
    if config_cls is None:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {sorted(CONFIG_REGISTRY)}.")

    return config_cls.from_yaml(path)
