"""Lightning module for causal language model pretraining."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.models.common import build_optimizer, build_scheduler, masked_token_stats
from lightning_grpo.utils.configs.pretrain import PretrainConfig
from lightning_grpo.utils.modeling import count_trainable_parameters, describe_model_source, load_causal_lm


class PretrainLightningModule(L.LightningModule):
    """Lightning module for decoder-only causal LM pretraining."""

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__()
        self.config = config
        self.model = load_causal_lm(config.model, config.precision)
        self.save_hyperparameters(config.to_dict())
        rank_zero_info(f"Loaded pretraining model from {describe_model_source(config.model)}")

        trainable, total = count_trainable_parameters(self.model)
        self.trainable_parameter_count = trainable
        self.total_parameter_count = total

    def forward(self, **batch: torch.Tensor) -> Any:
        """Forward tokens through the wrapped language model."""

        return self.model(**batch)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Run one pretraining optimization/evaluation step and log metrics."""

        labels = batch["labels"]
        outputs = self(**{**batch, "use_cache": False})
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        stats = masked_token_stats(outputs.logits, labels)
        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/token_accuracy", stats["token_accuracy"], prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/entropy", stats["entropy"], prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/mean_logprob", stats["mean_logprob"], prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/perplexity", stats["perplexity"], prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
            self.log(f"{stage}/aux_loss", outputs.aux_loss, prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute next-token prediction loss for pretraining."""

        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run validation with the same causal LM objective."""

        return self._shared_step(batch, "val")

    def on_fit_start(self) -> None:
        """Log parameter counts once training starts."""

        if self.logger is None or not self.trainer.is_global_zero:
            return

        self.logger.log_metrics(
            {
                "model/trainable_parameters": float(self.trainable_parameter_count),
                "model/total_parameters": float(self.total_parameter_count),
            },
            step=self.global_step,
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and scheduler for Lightning."""

        optimizer = build_optimizer(self.parameters(), self.config.optimization)
        scheduler = build_scheduler(optimizer, self.config.optimization, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}