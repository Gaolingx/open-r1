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

from lightning_grpo.utils.configs.grpo import GRPOConfig
from lightning_grpo.data.grpo_datamodule import GRPODataModule
from lightning_grpo.models.grpo_module import GRPOLightningModule
from lightning_grpo.utils.config import load_experiment_config
from lightning_grpo.utils.trainer import build_trainer, find_resume_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train a GRPO model with PyTorch Lightning.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["16", "32", "bf16", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--lora_init_path",
        type=str,
        default=None,
        help="Optional LoRA initialization path. When provided, LoRA is enabled and initialized from this path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, construct trainer objects, and launch training."""

    args = parse_args()
    config = load_experiment_config(args.config)
    if not isinstance(config, GRPOConfig):
        raise TypeError("Expected a GRPO config for train_grpo.py")

    if args.seed is not None:
        config.seed = args.seed
    if args.precision is not None:
        config.precision.trainer_precision = args.precision
    if args.gpus is not None:
        config.distributed.devices = args.gpus
    if args.lora_init_path is not None:
        config.model.lora.enabled = True
        setattr(config.model.lora, "init_path", args.lora_init_path)
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    L.seed_everything(config.seed, workers=True)
    data_module = GRPODataModule(
        data_config=config.data,
        model_config=config.model,
        optimization_config=config.optimization,
        rollout_config=config.rollout,
        system_prompt=config.system_prompt,
    )
    module = GRPOLightningModule(config)
    trainer = build_trainer(config)
    ckpt_path = find_resume_checkpoint(args.resume_from_checkpoint, config.checkpoint.dirpath)
    trainer.fit(module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
