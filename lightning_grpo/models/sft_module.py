"""Lightning module for supervised fine-tuning."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch

from lightning_grpo.configs.sft import SFTConfig
from lightning_grpo.models.common import build_optimizer, build_scheduler
from lightning_grpo.utils.modeling import count_trainable_parameters, load_causal_lm


class SFTLightningModule(L.LightningModule):
    """A reusable Lightning module for decoder-only SFT."""

    def __init__(self, config: SFTConfig) -> None:
        super().__init__()
        self.config = config
        self.model = load_causal_lm(config.model, config.precision)
        self.save_hyperparameters(config.to_dict())

        trainable, total = count_trainable_parameters(self.model)
        self.trainable_parameter_count = trainable
        self.total_parameter_count = total

    def forward(self, **batch: torch.Tensor) -> Any:
        """Forward tokens through the wrapped language model."""

        return self.model(**batch)

    def on_fit_start(self) -> None:
        """Log static parameter counts once training starts."""

        self.log("model/trainable_parameters", float(self.trainable_parameter_count), rank_zero_only=True)
        self.log("model/total_parameters", float(self.total_parameter_count), rank_zero_only=True)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute the next-token prediction loss."""

        outputs = self(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run validation with the same language modeling objective."""

        outputs = self(**batch)
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and scheduler for Lightning."""

        optimizer = build_optimizer(self.parameters(), self.config.optimization)
        scheduler = build_scheduler(optimizer, self.config.optimization, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
