"""Trainer construction helpers for multi-GPU Lightning execution."""

from __future__ import annotations

from typing import Any

import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from lightning_grpo.callbacks import build_callbacks
from lightning_grpo.configs.base import ExperimentConfig
from lightning_grpo.strategies import trainer_strategy_kwargs


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


def build_trainer(config: ExperimentConfig) -> L.Trainer:
    """Create a Lightning trainer with DDP or FSDP support."""

    strategy_kwargs = trainer_strategy_kwargs(config.distributed)
    return L.Trainer(
        default_root_dir=config.output_dir,
        precision=config.precision.trainer_precision,
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
    )
