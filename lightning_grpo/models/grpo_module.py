"""Lightning module for GRPO-style online RL fine-tuning."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.models.common import build_optimizer, build_scheduler, resolve_generation_config
from lightning_grpo.models.grpo import (
    GRPOLossComputer,
    GRPOMetricsAggregator,
    GRPORewardManager,
    GRPORolloutCoordinator,
)
from lightning_grpo.utils.configs.grpo import GRPOConfig
from lightning_grpo.utils.modeling import count_trainable_parameters, export_configured_model, load_causal_lm, load_tokenizer


class GRPOLightningModule(L.LightningModule):
    """Lightning-native GRPO implementation aligned with the core TRL training flow."""

    def __init__(self, config: GRPOConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = load_causal_lm(config.model, config.precision)
        self.reference_model = load_causal_lm(config.model, config.precision) if config.rollout.use_reference_model else None
        if self.reference_model is not None:
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()

        self.tokenizer = load_tokenizer(config.model)
        self.rollout_coordinator = GRPORolloutCoordinator(config, self.policy, self.tokenizer)
        self.rollout_engine = self.rollout_coordinator.rollout_engine
        self.reward_model_engine = self.rollout_coordinator.reward_model_engine
        self.rollout_generation_config = resolve_generation_config(config.rollout.generation_config_path, self.model.config)
        self.reward_manager = GRPORewardManager(config, self.tokenizer, self.rollout_engine, self.reward_model_engine, self.device)
        self.register_buffer("reward_weights", self.reward_manager.reward_weight_tensor.clone(), persistent=False)
        self.metrics_aggregator = GRPOMetricsAggregator(self)
        self.loss_computer = GRPOLossComputer(
            self,
            self.reward_manager,
            self.metrics_aggregator,
            rollout_temperature=self.rollout_generation_config.temperature,
        )
        self.save_hyperparameters(config.to_dict())

        trainable, total = count_trainable_parameters(self.policy)
        self.trainable_parameter_count = trainable
        self.total_parameter_count = total

    def on_fit_start(self) -> None:
        """Log static parameter counts once training starts."""

        if self.logger is None or not self.trainer.is_global_zero:
            return

        self.logger.log_metrics(
            {
                "model/trainable_parameters": float(self.trainable_parameter_count),
                "model/total_parameters": float(self.total_parameter_count),
            },
            step=self.global_step,
        )

    def forward(self, **batch: torch.Tensor) -> Any:
        """Forward prompts and completions through the policy model."""

        return self.policy(**batch)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one online rollout and optimization step."""

        debug_every = self.config.rollout.debug.every_n_steps
        if self.config.rollout.debug.enabled and debug_every > 0 and self.global_step % debug_every == 0:
            self.rollout_coordinator.emit_debug_samples(self.trainer, self.device)

        rollout_batch = self.rollout_coordinator.generate(batch, training=True)
        loss, metrics = self.loss_computer.compute_loss(rollout_batch, training=True)
        self.metrics_aggregator.log_metrics("train", loss, metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Evaluate the current policy with a rollout batch."""

        rollout_batch = self.rollout_coordinator.generate(batch, training=False)
        loss, metrics = self.loss_computer.compute_loss(rollout_batch, training=False)
        self.metrics_aggregator.log_metrics("val", loss, metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and scheduler for Lightning."""

        optimizer = build_optimizer(self.policy.parameters(), self.config.optimization)
        scheduler = build_scheduler(optimizer, self.config.optimization, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_batch_end(self, outputs: Any, batch: dict[str, Any], batch_idx: int) -> None:
        """Sync rollout backend after optimizer updates."""

        self.rollout_coordinator.sync_policy(self.policy)

    def on_train_end(self) -> None:
        """Export a Hugging Face-compatible model directory after training."""

        if not self.trainer.is_global_zero:
            return

        export_dir = self.config.output_dir + "/hf_final"
        exported_paths = export_configured_model(
            self.policy,
            self.config.model,
            export_dir,
            tokenizer=self.tokenizer,
            generation_config=self.rollout_generation_config.to_generation_config(),
        )
        if exported_paths:
            rank_zero_info(f"Exported model artifacts to {export_dir}: {sorted(exported_paths)}")
