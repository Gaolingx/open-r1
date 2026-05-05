"""Reusable Lightning callbacks for training control and observability."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info
from transformers.optimization import get_scheduler

from lightning_grpo.models.rollout_engine import PolicyRolloutEngine
from lightning_grpo.utils.configs.base import LoggingConfig, ModelConfig, TrainingBaseConfig
from lightning_grpo.utils.modeling import save_pth_weights, load_tokenizer, resolve_export_model
from lightning_grpo.utils.config import save_json_config


class CheckpointCallback(ModelCheckpoint):
    """ModelCheckpoint with optional torch export delegated to LightningModule."""

    def __init__(self, *args: Any, save_pt_format: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_pt_format = save_pt_format

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if not self.save_pt_format or not trainer.is_global_zero:
            return

        model = resolve_export_model(trainer.lightning_module)
        if model is None:
            return

        save_path = Path(filepath).parent / "pt_checkpoint"
        save_file = save_path / "pretrain_model"
        save_pth_weights(model, save_file)


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


class GlobalSampleCountCallback(Callback):
    """Log the total number of global samples seen during training."""

    def __init__(self, log_every_n_steps: int = 10) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.global_samples_seen: int = 0
        self.global_tokens_seen: int = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "global_samples_seen": int(self.global_samples_seen),
            "global_tokens_seen": int(self.global_tokens_seen),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.global_samples_seen = int(state_dict.get("global_samples_seen", 0))
        self.global_tokens_seen = int(state_dict.get("global_tokens_seen", 0))

    @staticmethod
    def _extract_batch_size(batch: Any) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.size(0)
        if isinstance(batch, dict):
            if "attention_mask" in batch:
                return batch["attention_mask"].size(0)
            if "input_ids" in batch:
                return batch["input_ids"].size(0)
            for val in batch.values():
                if isinstance(val, torch.Tensor):
                    return val.size(0)
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            if isinstance(batch[0], torch.Tensor):
                return batch[0].size(0)
        return 1

    @staticmethod
    def _extract_token_count(batch: Any) -> int:
        if isinstance(batch, torch.Tensor):
            return int(batch.numel())
        if isinstance(batch, dict):
            if "attention_mask" in batch:
                return int(batch["attention_mask"].sum().item())
            if "input_ids" in batch:
                return int(torch.numel(batch["input_ids"]))
            for val in batch.values():
                if isinstance(val, torch.Tensor):
                    return int(torch.numel(val))
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            if isinstance(batch[0], torch.Tensor):
                return int(torch.numel(batch[0]))
        return 0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        local_batch_size = self._extract_batch_size(batch)
        local_token_count = self._extract_token_count(batch)

        self.global_samples_seen += local_batch_size * trainer.world_size
        self.global_tokens_seen += local_token_count * trainer.world_size

        step = int(trainer.global_step)
        if step > 0 and step % self.log_every_n_steps == 0:
            pl_module.log("sample/global_samples", float(self.global_samples_seen), on_step=True, sync_dist=False)
            pl_module.log("sample/global_tokens", float(self.global_tokens_seen), on_step=True, sync_dist=False)


class GradParamNormCallback(Callback):
    """Log global parameter/gradient L2 norms as training metrics."""

    def __init__(self, log_every_n_steps: int = 1) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))

    @staticmethod
    def _compute_global_norm(pl_module: L.LightningModule, *, use_grad: bool) -> torch.Tensor:
        reference = None
        total = None

        for param in pl_module.parameters():
            if not param.requires_grad:
                continue

            tensor = param.grad if use_grad else param.detach()
            if tensor is None:
                continue

            reference = tensor
            part = tensor.detach().float().pow(2).sum()
            total = part if total is None else total + part

        if total is None:
            if reference is not None:
                return torch.tensor(0.0, device=reference.device)
            return torch.tensor(0.0, device=pl_module.device)

        return total.sqrt()

    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        grad_norm = self._compute_global_norm(pl_module, use_grad=True).detach().cpu()

        pl_module.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

    def on_before_zero_grad(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        grad_norm = self._compute_global_norm(pl_module, use_grad=True).detach().cpu()

        pl_module.log("train/grad_norm_clip", grad_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)


class LRandSchedulerOverrideCallback(Callback):
    """Override optimizer LR and optionally reset scheduler state after ckpt resume."""

    def __init__(self, config: TrainingBaseConfig) -> None:
        super().__init__()
        self.config = config
        self.applied = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.applied:
            return

        optimization = self.config.optimization
        resume_override = optimization.resume_override
        reset_lr = resume_override.override_lr_on_resume
        reset_scheduler = resume_override.reset_scheduler_on_resume

        if not reset_lr and not reset_scheduler:
            return

        if not trainer.optimizers:
            return

        optimizer = trainer.optimizers[0]
        target_lr = float(optimization.optimizer.learning_rate)

        if reset_lr:
            for group in optimizer.param_groups:
                group["lr"] = target_lr
                group["initial_lr"] = target_lr
            rank_zero_info(f"[LRandSchedulerOverrideCallback] Reset optimizer LR to {target_lr}.")

        if reset_scheduler:
            scheduler_config = optimization.scheduler
            scheduler_type = str(scheduler_config.type)

            if optimization.max_steps and optimization.max_steps > 0:
                total_steps = optimization.max_steps
            else:
                total_steps = max(1, trainer.estimated_stepping_batches)

            num_warmup_steps = min(max(0, scheduler_config.warmup_steps), total_steps)
            scheduler_specific_kwargs = dict(scheduler_config.scheduler_specific_kwargs)

            new_scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                scheduler_specific_kwargs=scheduler_specific_kwargs or None,
            )

            if trainer.lr_scheduler_configs:
                trainer.lr_scheduler_configs[0].scheduler = new_scheduler
            rank_zero_info(
                "[LRandSchedulerOverrideCallback] Reset scheduler state "
                f"(type={scheduler_type}, warmup={num_warmup_steps}, total_steps={total_steps})."
            )

        self.applied = True


class PeriodicSampleGenerationCallback(Callback):
    """Generate text samples during training for qualitative inspection."""

    def __init__(self, logging_config: LoggingConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.logging_config = logging_config
        self.tokenizer = load_tokenizer(model_config)
        self.rollout_engine: PolicyRolloutEngine | None = None
        self.last_sample_step: int = -1

    def _ensure_rollout_engine(self, pl_module: L.LightningModule) -> PolicyRolloutEngine | None:
        """Create or refresh the in-process rollout engine for the current policy."""

        model = resolve_export_model(pl_module)
        if model is None:
            return None

        model_config = getattr(model, "config", None)
        if self.rollout_engine is None:
            self.rollout_engine = PolicyRolloutEngine(
                policy_model=model,
                tokenizer=self.tokenizer,
                generation_config_path=self.logging_config.sample_generation_config_path,
                model_config=model_config,
            )
        else:
            self.rollout_engine.update_policy(model)
        return self.rollout_engine

    def _write_csv_samples(self, logger: CSVLogger, rows: list[dict[str, Any]]) -> None:
        """Append generated samples to a CSV file under the CSV logger directory."""

        csv_path = Path(logger.log_dir) / "sample_generations.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "prompt_index", "prompt", "completion"])
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _write_wandb_samples(self, logger: WandbLogger, rows: list[dict[str, Any]], step: int) -> None:
        """Log generated samples as a W&B table when W&B is enabled."""

        try:
            import wandb
        except ImportError:
            rank_zero_info("[PeriodicSampleGenerationCallback] wandb is not installed; skipping sample table logging.")
            return

        table = wandb.Table(columns=["step", "prompt_index", "prompt", "completion"])
        for row in rows:
            table.add_data(row["step"], row["prompt_index"], row["prompt"], row["completion"])
        logger.experiment.log({"samples/generations": table}, step=step)

    def _log_samples(self, trainer: L.Trainer, rows: list[dict[str, Any]], step: int) -> None:
        """Print samples and persist them to supported loggers."""

        loggers = trainer.loggers if isinstance(trainer.loggers, list) else [trainer.logger]
        for logger in [item for item in loggers if item is not None]:
            if isinstance(logger, CSVLogger):
                self._write_csv_samples(logger, rows)
            elif isinstance(logger, WandbLogger):
                self._write_wandb_samples(logger, rows, step)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Periodically switch to inference mode and generate configured samples."""

        step = int(trainer.global_step)
        every_n_steps = max(1, int(self.logging_config.sample_every_n_steps))
        if step <= 0 or step == self.last_sample_step or step % every_n_steps != 0:
            return

        rollout_engine = self._ensure_rollout_engine(pl_module)
        if rollout_engine is None:
            rank_zero_info("[PeriodicSampleGenerationCallback] No exportable model found; skipping sample generation.")
            return

        prompts = list(self.logging_config.sample_prompts)
        if not prompts:
            return

        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt")
        device = pl_module.device
        prompt_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        model = resolve_export_model(pl_module)
        was_training = bool(model.training) if model is not None else bool(pl_module.training)
        if model is not None:
            model.eval()
        else:
            pl_module.eval()

        try:
            with torch.inference_mode():
                rollout = rollout_engine.rollout(
                    prompt_ids=prompt_ids,
                    attention_mask=attention_mask,
                    num_generations=1,
                )
        finally:
            if was_training:
                if model is not None:
                    model.train()
                else:
                    pl_module.train()

        rows = [
            {
                "step": step,
                "prompt_index": index,
                "prompt": prompt,
                "completion": completion,
            }
            for index, (prompt, completion) in enumerate(zip(prompts, rollout.completions_text))
        ]
        self._log_samples(trainer, rows, step)
        self.last_sample_step = step


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

    def __init__(self, config: TrainingBaseConfig) -> None:
        self.config = config

    @rank_zero_only
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write the experiment configuration once at the start of training."""

        output_dir = Path(trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "resolved_config.json"
        save_json_config(self.config.to_dict(), config_path)


def build_callbacks(config: TrainingBaseConfig) -> list[Callback]:
    """Build the callback stack for Lightning training."""

    ckpt_callback = CheckpointCallback(
        dirpath=config.checkpoint.dirpath,
        filename="model-{epoch:02d}-{step:06d}-{train/loss:.4f}",
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        save_top_k=config.checkpoint.save_top_k,
        save_last=config.checkpoint.save_last,
        every_n_train_steps=config.checkpoint.every_n_train_steps,
        save_pt_format=config.checkpoint.save_pt_format,
    )

    callbacks: list[Callback] = [
        ckpt_callback,
        LRandSchedulerOverrideCallback(config),
        EfficiencyMonitorCallback(log_every_n_steps=config.logging.log_every_n_steps),
        GlobalSampleCountCallback(log_every_n_steps=config.logging.log_every_n_steps),
        GradParamNormCallback(log_every_n_steps=config.logging.log_every_n_steps),
        NaNLossCallback(),
        ConfigSnapshotCallback(config),
        RichProgressBar(),
    ]
    if config.logging.enable_csv or config.logging.enable_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if config.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=config.early_stopping.monitor or config.checkpoint.monitor,
                mode=config.early_stopping.mode or config.checkpoint.mode,
                patience=max(0, config.early_stopping.patience),
                min_delta=config.early_stopping.min_delta,
                check_finite=config.early_stopping.check_finite,
                stopping_threshold=config.early_stopping.stopping_threshold,
                divergence_threshold=config.early_stopping.divergence_threshold,
                verbose=config.early_stopping.verbose,
            )
        )
    if config.logging.sample_every_n_steps > 0 and config.logging.sample_prompts:
        callbacks.append(PeriodicSampleGenerationCallback(logging_config=config.logging, model_config=config.model))
    return callbacks
