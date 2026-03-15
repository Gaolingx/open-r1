"""Reusable Lightning callbacks for training control and observability."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info

from lightning_grpo.configs.base import CheckpointConfig, EarlyStoppingConfig, ExperimentConfig, LoggingConfig
from lightning_grpo.utils.modeling import load_tokenizer


class TrainingStateCallback(Callback):
    """Track coarse training lifecycle events and global progress."""

    def __init__(self) -> None:
        self.train_start_time: float | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Mark the beginning of the fit loop."""

        self.train_start_time = time.perf_counter()
        if pl_module.logger is not None:
            pl_module.logger.log_metrics({"system/global_rank": float(trainer.global_rank)}, step=trainer.global_step)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log wall-clock duration after training completes."""

        if self.train_start_time is None:
            return
        elapsed = time.perf_counter() - self.train_start_time
        if pl_module.logger is not None and trainer.is_global_zero:
            pl_module.logger.log_metrics({"system/train_time_seconds": elapsed}, step=trainer.global_step)


class EfficiencyMonitorCallback(Callback):
    """Log simple throughput and token-efficiency metrics."""

    def __init__(self, log_every_n_steps: int = 10) -> None:
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.step_start_time: float | None = None

    def on_train_batch_start(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            batch: dict[str, Any],
            batch_idx: int,
    ) -> None:
        """Capture batch start time for throughput estimation."""

        self.step_start_time = time.perf_counter()

    def on_train_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: dict[str, Any],
            batch_idx: int,
    ) -> None:
        """Log tokens per second and sequence statistics periodically."""

        if self.step_start_time is None:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        elapsed = max(time.perf_counter() - self.step_start_time, 1.0e-6)
        token_count = 0
        sequence_count = 0
        if isinstance(batch, dict) and "attention_mask" in batch:
            token_count = int(batch["attention_mask"].sum().item())
            sequence_count = int(batch["attention_mask"].shape[0])
        elif isinstance(batch, dict) and "input_ids" in batch:
            token_count = int(torch.numel(batch["input_ids"]))
            sequence_count = int(batch["input_ids"].shape[0])

        pl_module.log("perf/tokens_per_second", token_count / elapsed, on_step=True, sync_dist=True)
        pl_module.log("perf/sequences_per_second", sequence_count / elapsed, on_step=True, sync_dist=True)
        pl_module.log("perf/batch_time_seconds", elapsed, on_step=True, sync_dist=True)


class PeriodicSampleGenerationCallback(Callback):
    """Generate text samples during training for qualitative inspection."""

    def __init__(self, logging_config: LoggingConfig, model_name_or_path: str) -> None:
        self.logging_config = logging_config
        self.tokenizer = load_tokenizer(
            type("ModelConfigProxy", (), {
                "model_name_or_path": model_name_or_path,
                "tokenizer_name_or_path": None,
                "model_revision": "main",
                "trust_remote_code": False,
                "chat_template": None,
                "eos_token": None,
                "pad_token": None,
            })()
        )

    @rank_zero_only
    def on_train_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: dict[str, Any],
            batch_idx: int,
    ) -> None:
        """Generate periodic sample completions and save them to disk."""

        every_n_steps = self.logging_config.sample_every_n_steps
        if every_n_steps <= 0 or not self.logging_config.sample_prompts:
            return
        if trainer.global_step == 0 or trainer.global_step % every_n_steps != 0:
            return

        model = getattr(pl_module, "policy", None) or getattr(pl_module, "model", None)
        if model is None:
            return

        prompts = self.logging_config.sample_prompts
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        tokenized = {key: value.to(pl_module.device) for key, value in tokenized.items()}
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        output_dir = Path(trainer.default_root_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_path = output_dir / f"step-{trainer.global_step:08d}.json"
        with sample_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [{"prompt": prompt, "generation": generation} for prompt, generation in zip(prompts, texts)],
                handle,
                ensure_ascii=False,
                indent=2,
            )


class NaNLossCallback(Callback):
    """Immediately stop training when a NaN or Inf loss is detected."""

    def on_train_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
    ) -> None:
        """Stop training when the batch loss becomes non-finite."""

        loss = None
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]

        if loss is not None and not torch.isfinite(loss):
            rank_zero_info(
                f"\n[NaNLossCallback] Non-finite loss detected at "
                f"step {trainer.global_step} (batch_idx={batch_idx}): {loss.item():.6f}. "
                f"Stopping training."
            )
            trainer.should_stop = True


class ConfigSnapshotCallback(Callback):
    """Persist the resolved experiment config next to checkpoints."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @rank_zero_only
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write the experiment configuration once at the start of training."""

        output_dir = Path(trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "resolved_config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(self.config.to_dict(), handle, ensure_ascii=False, indent=2)


def build_checkpoint_callback(checkpoint_config: CheckpointConfig) -> ModelCheckpoint:
    """Create the default checkpoint callback."""

    return ModelCheckpoint(
        dirpath=checkpoint_config.dirpath,
        monitor=checkpoint_config.monitor,
        mode=checkpoint_config.mode,
        save_top_k=checkpoint_config.save_top_k,
        save_last=checkpoint_config.save_last,
        every_n_train_steps=checkpoint_config.every_n_train_steps,
        filename="{epoch:02d}-{step:08d}-{" + checkpoint_config.monitor.replace("/", "_") + ":.4f}",
        auto_insert_metric_name=False,
    )


def build_early_stopping_callback(
        early_stopping_config: EarlyStoppingConfig,
        checkpoint_config: CheckpointConfig,
) -> EarlyStopping | None:
    """Create an early stopping callback when the feature is enabled."""

    if not early_stopping_config.enabled:
        return None

    return EarlyStopping(
        monitor=early_stopping_config.monitor or checkpoint_config.monitor,
        mode=early_stopping_config.mode or checkpoint_config.mode,
        patience=max(0, early_stopping_config.patience),
        min_delta=early_stopping_config.min_delta,
        check_finite=early_stopping_config.check_finite,
        stopping_threshold=early_stopping_config.stopping_threshold,
        divergence_threshold=early_stopping_config.divergence_threshold,
        verbose=early_stopping_config.verbose,
    )


def build_callbacks(config: ExperimentConfig) -> list[Callback]:
    """Build the callback stack for Lightning training."""

    callbacks: list[Callback] = [
        build_checkpoint_callback(config.checkpoint),
        LearningRateMonitor(logging_interval="step"),
        TrainingStateCallback(),
        EfficiencyMonitorCallback(log_every_n_steps=config.logging.log_every_n_steps),
        NaNLossCallback(),
        ConfigSnapshotCallback(config),
    ]
    early_stopping_callback = build_early_stopping_callback(config.early_stopping, config.checkpoint)
    if early_stopping_callback is not None:
        callbacks.append(early_stopping_callback)
    if config.logging.sample_every_n_steps > 0 and config.logging.sample_prompts:
        callbacks.append(
            PeriodicSampleGenerationCallback(
                logging_config=config.logging,
                model_name_or_path=config.model.model_name_or_path,
            )
        )
    return callbacks
