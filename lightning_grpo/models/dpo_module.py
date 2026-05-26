"""Lightning module for Direct Preference Optimization (DPO) training.

Implements the DPO algorithm using LigerFusedLinearDPOLoss for memory-efficient
fused computation that avoids materializing full vocabulary logits.

Reference:
    - DPO paper: https://huggingface.co/papers/2305.18290
    - Liger Kernel: https://github.com/linkedin/Liger-Kernel
"""

from __future__ import annotations

import copy
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.utils.configs.dpo import DPOConfig
from lightning_grpo.models.common import (
    compile_model_if_configured,
    count_trainable_parameters,
    build_optimizer,
    build_scheduler,
    export_configured_model,
    load_tokenizer,
)
from lightning_grpo.models.grpo.loss import compute_liger_dpo_loss, compute_standard_dpo_loss
from lightning_grpo.models.grpo.liger_loss import LigerDPOLossComputer
from lightning_grpo.strategies.fsdp2 import configure_fully_shard
from lightning_grpo.strategies.tensor_parallel import configure_tensor_parallel
from lightning_grpo.utils.modeling import load_causal_lm


class DPOLightningModule(L.LightningModule):
    """Lightning module for Direct Preference Optimization.

    Uses LigerFusedLinearDPOLoss to fuse the LM head projection with the DPO loss
    computation, avoiding materialization of full vocabulary logits and reducing
    peak VRAM usage significantly for large vocabulary models.
    """

    def __init__(self, config: DPOConfig) -> None:
        super().__init__()
        self.config = config
        self.model = load_causal_lm(config.model, config.precision)
        self.save_hyperparameters(config.to_dict())

        trainable, total = count_trainable_parameters(self.model)
        self.trainable_parameter_count = trainable
        self.total_parameter_count = total
        self.tokenizer = load_tokenizer(config.model) if config.model.tokenizer_name_or_path else None

        # Reference model: frozen copy of the policy model
        self.ref_model = self._build_ref_model()

        # DPO hyperparameters
        self.beta = config.beta
        self.loss_type = config.loss_type
        self.label_smoothing = config.label_smoothing

        # Liger fused DPO loss computer (initialized lazily in configure_model)
        self._liger_loss_computer = None

    def _build_ref_model(self) -> torch.nn.Module:
        """Create a frozen reference model for DPO log-probability computation."""

        if self.config.ref_model_name_or_path:
            ref_model = load_causal_lm(
                self.config.model.__class__(
                    model_name_or_path=self.config.ref_model_name_or_path,
                    **{k: v for k, v in self.config.model.__dict__.items() if k != "model_name_or_path"},
                ),
                self.config.precision,
            )
        else:
            ref_model = copy.deepcopy(self.model)

        # Freeze all reference model parameters
        ref_model.requires_grad_(False)
        ref_model.eval()
        return ref_model

    def forward(self, **batch: torch.Tensor) -> Any:
        """Forward tokens through the wrapped language model."""

        return self.model(**batch)

    def configure_model(self) -> None:
        """Apply tensor parallelism, then composable FSDP2 after Lightning creates the device mesh."""

        configure_tensor_parallel(self.model, self.config.distributed, self.device_mesh)
        configure_fully_shard(self.model, self.config.distributed, self.config.precision, self.device_mesh)
        self.model = compile_model_if_configured(self.model, self.config.model)

        # Also shard the reference model
        configure_tensor_parallel(self.ref_model, self.config.distributed, self.device_mesh)
        configure_fully_shard(self.ref_model, self.config.distributed, self.config.precision, self.device_mesh)

        # Initialize Liger fused DPO loss if enabled
        if self.config.liger_kernel.enabled:
            self._liger_loss_computer = LigerDPOLossComputer(
                self.model,
                self.ref_model,
                beta=self.beta,
                loss_type=self.loss_type,
                label_smoothing=self.label_smoothing,
                loss_parallel_enabled=self.config.distributed.tensor_parallel.loss_parallel,
            )

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Run one optimization/evaluation step and log metrics."""

        use_liger = self.config.liger_kernel.enabled

        if use_liger:
            loss, metrics = compute_liger_dpo_loss(batch)
        else:
            loss, metrics = compute_standard_dpo_loss(batch)

        # Log metrics
        on_step = stage == "train"
        prog_bar = stage in {"train", "val"}

        self.log(f"{stage}/loss", loss, prog_bar=prog_bar, on_step=on_step, on_epoch=True, sync_dist=True)

        with torch.no_grad():
            chosen_rewards = metrics["chosen_rewards"]
            rejected_rewards = metrics["rejected_rewards"]

            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()

            self.log(f"{stage}/rewards/chosen", chosen_rewards.mean(), on_step=on_step, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/rewards/rejected", rejected_rewards.mean(), on_step=on_step, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/rewards/accuracy", reward_accuracy, prog_bar=prog_bar, on_step=on_step, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/rewards/margin", reward_margin, on_step=on_step, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/logps/chosen", metrics["chosen_logps"].mean(), on_step=on_step, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/logps/rejected", metrics["rejected_logps"].mean(), on_step=on_step, on_epoch=True, sync_dist=True)

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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute the DPO preference loss."""

        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run validation with the same DPO objective."""

        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and scheduler for Lightning."""

        optimizer = build_optimizer(self.parameters(), self.config.optimization)
        scheduler = build_scheduler(optimizer, self.config.optimization, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_end(self) -> None:
        """Export a Hugging Face-compatible model directory after training."""

        if not self.trainer.is_global_zero:
            return

        export_dir = self.config.output_dir + "/hf_final"
        exported_paths = export_configured_model(
            self.model,
            self.config.model,
            export_dir,
            tokenizer=self.tokenizer,
        )
        if exported_paths:
            rank_zero_info(f"Exported model artifacts to {export_dir}: {sorted(exported_paths)}")
