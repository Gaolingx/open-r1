"""Lightning module for supervised fine-tuning."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from lightning_grpo.configs.sft import SFTConfig
from lightning_grpo.models.common import build_optimizer, build_scheduler, entropy_from_logits
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

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute token-level next-token loss with optional label smoothing."""

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)

        return F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )

    @staticmethod
    def _compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute next-token accuracy over non-masked labels."""

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100

        if not mask.any():
            return shift_logits.new_tensor(0.0)

        predictions = shift_logits.argmax(dim=-1)
        correct = ((predictions == shift_labels) & mask).sum()
        total = mask.sum()
        return correct.to(dtype=torch.float32) / total.to(dtype=torch.float32)

    @staticmethod
    def _compute_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean entropy on valid next-token positions."""

        token_entropy = entropy_from_logits(logits[..., :-1, :].contiguous())
        mask = labels[..., 1:].contiguous() != -100

        if not mask.any():
            return token_entropy.new_tensor(0.0)

        mask = mask.to(dtype=token_entropy.dtype)
        return (token_entropy * mask).sum() / mask.sum()

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Run one SFT optimization/evaluation step and log metrics."""

        labels = batch["labels"]
        model_inputs = {**batch, "use_cache": False}
        outputs = self(**model_inputs)
        loss = self._compute_loss(outputs.logits, labels)
        accuracy = self._compute_token_accuracy(outputs.logits, labels)
        entropy = self._compute_entropy(outputs.logits, labels)

        on_step = stage == "train"
        prog_bar = stage in {"train", "val"}
        self.log(f"{stage}/loss", loss, prog_bar=prog_bar, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/token_accuracy", accuracy, prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/entropy", entropy, prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)

        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
            self.log(f"{stage}/aux_loss", outputs.aux_loss, prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)

        return loss

    def on_fit_start(self) -> None:
        """Log static parameter counts once training starts."""

        self.log("model/trainable_parameters", float(self.trainable_parameter_count), rank_zero_only=True)
        self.log("model/total_parameters", float(self.total_parameter_count), rank_zero_only=True)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute the next-token prediction loss."""

        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run validation with the same language modeling objective."""

        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        """Create optimizer and scheduler for Lightning."""

        optimizer = build_optimizer(self.parameters(), self.config.optimization)
        scheduler = build_scheduler(optimizer, self.config.optimization, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
