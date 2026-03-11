"""CLI entrypoint for Lightning-based supervised fine-tuning."""

from __future__ import annotations

import argparse

import lightning as L

from lightning_grpo.configs.sft import SFTConfig
from lightning_grpo.data.sft import SFTDataModule
from lightning_grpo.models.sft_module import SFTLightningModule
from lightning_grpo.utils.config import load_experiment_config
from lightning_grpo.utils.trainer import build_trainer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train an SFT model with PyTorch Lightning.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit GPU ids to use, for example --gpus 0 1.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, construct trainer objects, and launch training."""

    args = parse_args()
    config = load_experiment_config(args.config)
    if not isinstance(config, SFTConfig):
        raise TypeError("Expected an SFT config for train_sft.py")

    trainer_devices = args.gpus
    trainer_accelerator = "gpu" if trainer_devices is not None else None

    L.seed_everything(config.seed, workers=True)
    data_module = SFTDataModule(
        data_config=config.data,
        model_config=config.model,
        optimization_config=config.optimization,
        system_prompt=config.system_prompt,
        completion_only_loss=config.completion_only_loss,
    )
    module = SFTLightningModule(config)
    trainer = build_trainer(
        config,
        devices=trainer_devices,
        accelerator=trainer_accelerator,
    )
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
