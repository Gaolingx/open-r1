"""CLI entrypoint for Lightning-based supervised fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lightning as L

from lightning_grpo.configs.sft import SFTConfig
from lightning_grpo.data.sft import SFTDataModule
from lightning_grpo.models.sft_module import SFTLightningModule
from lightning_grpo.utils.config import load_experiment_config
from lightning_grpo.utils.trainer import build_trainer


def find_resume_checkpoint(resume_arg: str, default_ckpt_dir: str) -> Optional[str]:
    """Resolve a checkpoint path for resuming training."""

    if not resume_arg:
        return None

    if resume_arg.lower() == "last":
        last_ckpt = Path(default_ckpt_dir) / "last.ckpt"
        if last_ckpt.exists():
            print(f"Resuming from latest checkpoint: {last_ckpt}")
            return str(last_ckpt)
        return None

    p = Path(resume_arg)
    if p.is_file() and p.suffix == ".ckpt":
        print(f"Resuming from checkpoint file: {p}")
        return str(p)
    if p.is_dir():
        candidates = sorted(
            [x for x in p.rglob("*.ckpt") if x.name != "last.ckpt"],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            print(f"Resuming from checkpoint in dir: {candidates[0]}")
            return str(candidates[0])

        last_ckpt = p / "last.ckpt"
        if last_ckpt.exists():
            print(f"Resuming from last checkpoint in dir: {last_ckpt}")
            return str(last_ckpt)
    return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train an SFT model with PyTorch Lightning.")
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
    return parser.parse_args()


def main() -> None:
    """Load config, construct trainer objects, and launch training."""

    args = parse_args()
    config = load_experiment_config(args.config)
    if not isinstance(config, SFTConfig):
        raise TypeError("Expected an SFT config for train_sft.py")

    if args.seed is not None:
        config.seed = args.seed
    if args.precision is not None:
        config.precision = args.precision
    if args.gpus is not None:
        config.distributed.devices = args.gpus

    L.seed_everything(config.seed, workers=True)
    data_module = SFTDataModule(
        data_config=config.data,
        model_config=config.model,
        optimization_config=config.optimization,
        system_prompt=config.system_prompt,
    )
    module = SFTLightningModule(config)
    trainer = build_trainer(config)
    ckpt_path = find_resume_checkpoint(args.resume_from_checkpoint, config.checkpoint.dirpath)
    trainer.fit(module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
