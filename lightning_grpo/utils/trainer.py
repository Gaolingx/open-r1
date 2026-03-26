"""Trainer construction helpers for multi-GPU Lightning execution."""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from lightning_grpo.callbacks import build_callbacks
from lightning_grpo.utils.configs.base import ExperimentConfig
from lightning_grpo.strategies import trainer_strategy_kwargs


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


def build_loggers(config: ExperimentConfig) -> list[Any]:
    """Build logger instances from experiment configuration."""

    loggers: list[Any] = []
    if config.logging.enable_csv:
        loggers.append(CSVLogger(save_dir=config.output_dir, name="csv_logs"))
    if config.logging.enable_wandb:
        loggers.append(
            WandbLogger(
                project=config.logging.project,
                name=config.logging.run_name,
                save_dir=config.output_dir,
            )
        )
    return loggers


def build_eval_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    """Build Lightning validation scheduling kwargs from rollout configuration."""

    rollout_config = getattr(config, "rollout", None)
    eval_strategy = getattr(rollout_config, "eval_strategy", "no")

    if eval_strategy == "no":
        return {
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
        }
    if eval_strategy == "steps":
        return {
            "check_val_every_n_epoch": None,
            "val_check_interval": max(1, config.logging.log_every_n_steps),
        }
    return {"check_val_every_n_epoch": 1}


def build_trainer(config: ExperimentConfig) -> L.Trainer:
    """Create a Lightning trainer with DDP or FSDP support."""

    strategy_kwargs = trainer_strategy_kwargs(
        config.distributed,
        config.precision,
    )
    eval_strategy_kwargs = build_eval_kwargs(config)

    return L.Trainer(
        default_root_dir=config.output_dir,
        max_epochs=config.optimization.max_epochs,
        max_steps=config.optimization.max_steps,
        accumulate_grad_batches=config.optimization.gradient_accumulation_steps,
        gradient_clip_val=config.optimization.gradient_clip_val,
        log_every_n_steps=config.logging.log_every_n_steps,
        callbacks=build_callbacks(config),
        logger=build_loggers(config),
        benchmark=True,
        deterministic=False,
        **strategy_kwargs,
        **eval_strategy_kwargs,
    )
