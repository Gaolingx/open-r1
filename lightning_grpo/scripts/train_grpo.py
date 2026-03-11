"""CLI entrypoint for Lightning-based GRPO training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lightning as L

from lightning_grpo.configs.grpo import GRPOConfig
from lightning_grpo.data.grpo import GRPODataModule
from lightning_grpo.models.grpo_module import GRPOLightningModule
from lightning_grpo.utils.config import load_experiment_config
from lightning_grpo.utils.trainer import build_trainer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train a GRPO model with PyTorch Lightning.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Load config, construct trainer objects, and launch training."""

    args = parse_args()
    config = load_experiment_config(args.config)
    if not isinstance(config, GRPOConfig):
        raise TypeError("Expected a GRPO config for train_grpo.py")

    L.seed_everything(config.seed, workers=True)
    data_module = GRPODataModule(
        data_config=config.data,
        model_config=config.model,
        optimization_config=config.optimization,
        rollout_config=config.rollout,
        system_prompt=config.system_prompt,
    )
    module = GRPOLightningModule(config)
    trainer = build_trainer(
        config,
        enable_validation=bool(config.data.val_split),
    )
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
