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
from transformers import GenerationConfig
from transformers.optimization import get_scheduler

from lightning_grpo.utils.configs.base import CheckpointConfig, EarlyStoppingConfig, ExperimentConfig, LoggingConfig, ModelConfig
from lightning_grpo.utils.modeling import export_configured_model, load_tokenizer, resolve_export_model
from lightning_grpo.utils.config import load_json_config, save_json_config


class CheckpointCallback(ModelCheckpoint):
    """ModelCheckpoint with optional torch export delegated to LightningModule."""

    def __init__(
        self,
        *args: Any,
        save_pth_format: bool = True,
        save_safetensors_format: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_pth_format = save_pth_format
        self.save_safetensors_format = save_safetensors_format

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if not trainer.is_global_zero:
            return

        model = resolve_export_model(trainer.lightning_module)
        if model is None:
            return

        tokenizer = getattr(trainer.lightning_module, "tokenizer", None)

        model_config = trainer.lightning_module.config.model
        original_save_pth_format = model_config.save_pth_format
        original_save_safetensors_format = model_config.save_safetensors_format

        model_config.save_pth_format = self.save_pth_format
        model_config.save_safetensors_format = self.save_safetensors_format

        try:
            export_configured_model(
                model,
                model_config,
                Path(filepath).parent,
                tokenizer=tokenizer,
            )
        finally:
            model_config.save_pth_format = original_save_pth_format
            model_config.save_safetensors_format = original_save_safetensors_format


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

    def __init__(self, config: ExperimentConfig) -> None:
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
        self.logging_config = logging_config
        self.generation_config = self._load_generation_config(logging_config.sample_generation_config_path)
        self.tokenizer = load_tokenizer(model_config)

    @staticmethod
    def _load_generation_config(config_path: str | None) -> GenerationConfig:
        """Load text generation settings from a JSON config file."""

        if not config_path:
            return GenerationConfig(max_new_tokens=128)

        return GenerationConfig(load_json_config(config_path))

    @staticmethod
    def _extract_generations(decoded_sequences: list[str], prompts: list[str]) -> list[str]:
        """Strip prompt prefixes from decoded full sequences when possible."""

        generations: list[str] = []
        for prompt, text in zip(prompts, decoded_sequences):
            if text.startswith(prompt):
                generations.append(text[len(prompt):].lstrip())
            else:
                generations.append(text)
        return generations

    @staticmethod
    def _resolve_logger_collection(trainer: L.Trainer) -> list[Any]:
        """Return all attached logger instances as a flat list."""

        if trainer.loggers:
            return list(trainer.loggers)
        if trainer.logger is None:
            return []
        return [trainer.logger]

    def _log_samples_to_loggers(
        self,
        trainer: L.Trainer,
        rows: list[dict[str, Any]],
        sample_path: Path,
    ) -> None:
        """Push sample generations to configured experiment loggers."""

        loggers = self._resolve_logger_collection(trainer)
        if not loggers:
            return

        preview_lines = [
            f"[{index}] prompt: {row['prompt']}\n[{index}] generation: {row['generation']}"
            for index, row in enumerate(rows)
        ]
        preview_text = "\n\n".join(preview_lines)
        base_metrics = {
            "samples/count": float(len(rows)),
            "samples/path": str(sample_path),
        }

        for logger in loggers:
            logger.log_metrics(base_metrics, step=trainer.global_step)

            experiment = getattr(logger, "experiment", None)
            if isinstance(logger, WandbLogger) and experiment is not None:
                try:
                    import wandb

                    experiment.log(
                        {
                            "samples/table": wandb.Table(
                                columns=["step", "prompt", "generation", "path"],
                                data=[
                                    [trainer.global_step, row["prompt"], row["generation"], str(sample_path)]
                                    for row in rows
                                ],
                            ),
                            "samples/text": preview_text,
                        },
                        step=trainer.global_step,
                    )
                except Exception as exc:  # pragma: no cover - logger-specific best effort path
                    rank_zero_info(f"[PeriodicSampleGenerationCallback] Failed to log samples to Weights & Biases: {exc}")
            elif isinstance(logger, CSVLogger):
                self._append_samples_csv(logger.log_dir, trainer.global_step, rows, sample_path)
            elif experiment is not None:
                add_text = getattr(experiment, "add_text", None)
                if callable(add_text):
                    add_text("samples/text", preview_text, global_step=trainer.global_step)

    @staticmethod
    def _append_samples_csv(log_dir: str, step: int, rows: list[dict[str, Any]], sample_path: Path) -> None:
        """Persist sample rows under the CSV logger directory for easy inspection."""

        csv_path = Path(log_dir) / "sample_generations.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "prompt", "generation", "path"])
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "step": step,
                        "prompt": row["prompt"],
                        "generation": row["generation"],
                        "path": str(sample_path),
                    }
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

        was_training = model.training
        prompts = self.logging_config.sample_prompts
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        tokenized = {key: value.to(pl_module.device) for key, value in tokenized.items()}
        if hasattr(model, "eval"):
            model.eval()
        with torch.inference_mode():
            generated = model.generate(
                **tokenized,
                generation_config=self.generation_config,
            )
        if was_training and hasattr(model, "train"):
            model.train()

        decoded_sequences = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        texts = self._extract_generations(decoded_sequences, prompts)
        output_dir = Path(trainer.default_root_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_path = output_dir / f"step-{trainer.global_step:08d}.json"
        payload = {
            "step": trainer.global_step,
            "timestamp": time.time(),
            "samples": [
                {
                    "prompt": prompt,
                    "generation": generation,
                    "full_text": full_text,
                }
                for prompt, generation, full_text in zip(prompts, texts, decoded_sequences)
            ],
        }
        save_json_config(payload, sample_path)
        self._log_samples_to_loggers(trainer, payload["samples"], sample_path)


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
        save_json_config(self.config.to_dict(), config_path)


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

    ckpt_callback = CheckpointCallback(
        dirpath=config.checkpoint.dirpath,
        filename="model-{epoch:02d}-{step:06d}-{train/loss:.4f}",
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        save_top_k=config.checkpoint.save_top_k,
        save_last=config.checkpoint.save_last,
        every_n_train_steps=config.checkpoint.every_n_train_steps,
        save_pth_format=config.checkpoint.save_pth_format,
        save_safetensors_format=config.checkpoint.save_safetensors_format,
    )

    callbacks: list[Callback] = [
        ckpt_callback,
        LearningRateMonitor(logging_interval="step"),
        LRandSchedulerOverrideCallback(config),
        EfficiencyMonitorCallback(log_every_n_steps=config.logging.log_every_n_steps),
        GradParamNormCallback(log_every_n_steps=config.logging.log_every_n_steps),
        NaNLossCallback(),
        ConfigSnapshotCallback(config),
        RichProgressBar(),
    ]
    early_stopping_callback = build_early_stopping_callback(config.early_stopping, config.checkpoint)
    if early_stopping_callback is not None:
        callbacks.append(early_stopping_callback)
    if config.logging.sample_every_n_steps > 0 and config.logging.sample_prompts:
        callbacks.append(
            PeriodicSampleGenerationCallback(
                logging_config=config.logging,
                model_config=config.model,
            )
        )
    return callbacks
