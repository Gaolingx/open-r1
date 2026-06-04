"""Lightning module for Group Relative Policy Optimization (GRPO)."""

from __future__ import annotations

import copy
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.models.common import (
    build_optimizer,
    build_scheduler,
    compile_model_if_configured,
    count_trainable_parameters,
    export_configured_model,
    load_tokenizer,
)
from lightning_grpo.models.grpo.rollout import LocalGenerateRolloutCoordinator
from lightning_grpo.models.grpo.loss import StandardGRPOLossComputer
from lightning_grpo.models.grpo.liger_loss import LigerGRPOLossComputer
from lightning_grpo.models.grpo.metrics import GRPOMetricsAggregator
from lightning_grpo.models.grpo.reward import GRPORewardManager
from lightning_grpo.models.grpo.tool_call import GRPOToolCallMixin
from lightning_grpo.strategies.fsdp2 import configure_fully_shard
from lightning_grpo.utils.configs.grpo import GRPOConfig
from lightning_grpo.utils.modeling import load_causal_lm


class GRPOLightningModule(GRPOToolCallMixin, L.LightningModule):
    """Lightning module for local-rollout GRPO, reasoning RL, and agentic RL."""

    def __init__(self, config: GRPOConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = load_causal_lm(config.model, config.precision)
        self.save_hyperparameters(config.to_dict())

        trainable, total = count_trainable_parameters(self.policy)
        self.trainable_parameter_count = trainable
        self.total_parameter_count = total
        self.tokenizer = load_tokenizer(config.model)
        self.reference_model = self._build_reference_model()
        self.rollout_coordinator = LocalGenerateRolloutCoordinator(self)
        self.metrics_aggregator = GRPOMetricsAggregator(self)
        self.reward_manager = GRPORewardManager(config, self.tokenizer, device=self.device)
        self._liger_loss_computer: LigerGRPOLossComputer | None = None
        self._standard_loss_computer: StandardGRPOLossComputer | None = None
        self._init_tool_executor()

    def _build_reference_model(self) -> torch.nn.Module | None:
        """Create the frozen reference model used for KL regularization."""

        if not self.config.rollout.use_reference_model:
            return None
        reference_model = load_causal_lm(self.config.ref_model, self.config.precision) if self.config.ref_model else copy.deepcopy(self.policy)
        reference_model.requires_grad_(False)
        reference_model.eval()
        return reference_model

    def forward(self, **batch: torch.Tensor) -> Any:
        """Forward tokens through the wrapped policy model."""

        return self.policy(**batch)

    def configure_model(self) -> None:
        """Apply tensor parallelism, FSDP2, compile, then initialize GRPO loss."""

        configure_fully_shard(self.policy, self.config.distributed, self.config.precision, self.device_mesh)
        self.policy = compile_model_if_configured(self.policy, self.config.model)
        self.rollout_coordinator.update_policy()
        if self.reference_model is not None:
            configure_fully_shard(self.reference_model, self.config.distributed, self.config.precision, self.device_mesh)

        if self.config.liger_kernel.enabled:
            self._liger_loss_computer = LigerGRPOLossComputer(
                self,
                self.reward_manager,
                self.metrics_aggregator,
                rollout_temperature=self.config.rollout.temperature,
            )
        else:
            self._standard_loss_computer = StandardGRPOLossComputer(
                self,
                self.reward_manager,
                self.metrics_aggregator,
                rollout_temperature=self.config.rollout.temperature,
            )

    def _shared_step(self, batch: dict[str, list[Any]], stage: str) -> torch.Tensor:
        """Generate rollouts, compute GRPO loss, and log metrics."""

        rollout_batch = self.rollout_coordinator.rollout(batch, training=stage == "train")

        # Run tool calling loop if enabled and executor is initialized
        if self.tool_executor is not None:
            rollout_batch = self._run_tool_calling(rollout_batch)

        self.reward_manager.device = self.device
        if self.config.liger_kernel.enabled:
            if self._liger_loss_computer is None:
                raise RuntimeError("GRPO loss computer is not initialized. Call configure_model() first.")
            loss, metrics = self._liger_loss_computer.compute_loss(rollout_batch, training=stage == "train")
        else:
            if self._standard_loss_computer is None:
                raise RuntimeError("Standard GRPO loss computer is not initialized. Call configure_model() first.")
            loss, metrics = self._standard_loss_computer.compute_loss(rollout_batch, training=stage == "train")
        self.metrics_aggregator.log_metrics(stage, loss, metrics, on_step=stage == "train", on_epoch=True)

        if self.config.rollout.debug_samples and self.trainer.is_global_zero:
            every = max(1, self.config.rollout.debug_every_n_steps)
            if self.global_step % every == 0:
                rank_zero_info(f"[GRPO DEBUG] prompt={rollout_batch['prompts'][0]!r}")
                rank_zero_info(f"[GRPO DEBUG] completion={rollout_batch['completions'][0]!r}")
        return loss

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

    def training_step(self, batch: dict[str, list[Any]], batch_idx: int) -> torch.Tensor:
        """Compute one on-policy GRPO training step."""

        return self._shared_step(batch, "train")

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Refresh external rollout-engine weights after optimizer updates."""

        if self.config.rollout.engine == "sglang":
            self.rollout_coordinator.update_policy()

    def validation_step(self, batch: dict[str, list[Any]], batch_idx: int) -> torch.Tensor:
        """Run validation rollouts with one completion per prompt."""

        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and scheduler for Lightning."""

        optimizer = build_optimizer(self.parameters(), self.config.optimization)
        scheduler = build_scheduler(optimizer, self.config.optimization, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_end(self) -> None:
        """Export a Hugging Face-compatible model directory after training."""

        # Clean up tool executor resources
        if self.tool_executor is not None and hasattr(self.tool_executor, "shutdown"):
            self.tool_executor.shutdown()

        if not self.trainer.is_global_zero:
            return
        export_dir = self.config.output_dir + "/hf_final"
        exported_paths = export_configured_model(self.policy, self.config.model, export_dir, tokenizer=self.tokenizer)
        if exported_paths:
            rank_zero_info(f"Exported model artifacts to {export_dir}: {sorted(exported_paths)}")
