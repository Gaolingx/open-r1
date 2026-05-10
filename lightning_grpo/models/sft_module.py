"""Lightning module for supervised fine-tuning."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.utils.configs.sft import SFTConfig
from lightning_grpo.models.common import build_optimizer, build_scheduler, masked_token_stats, compute_cross_entropy_loss
from lightning_grpo.strategies.fsdp2 import configure_fully_shard
from lightning_grpo.strategies.tensor_parallel import configure_tensor_parallel
from lightning_grpo.utils.modeling import count_trainable_parameters, export_configured_model, load_causal_lm, load_tokenizer, log_moe_metrics


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
        self.tokenizer = load_tokenizer(config.model) if config.model.tokenizer_name_or_path else None

    def forward(self, **batch: torch.Tensor) -> Any:
        """Forward tokens through the wrapped language model."""

        return self.model(**batch)

    def configure_model(self) -> None:
        """Apply tensor parallelism, then composable FSDP2 after Lightning creates the device mesh."""

        configure_tensor_parallel(self.model, self.config.distributed, getattr(self, "device_mesh", None))
        configure_fully_shard(self.model, self.config.distributed, self.config.precision, getattr(self, "device_mesh", None))

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Run one SFT optimization/evaluation step and log metrics."""

        labels = batch["labels"]
        if self.config.distributed.tensor_parallel.loss_parallel:
            outputs = self(**{key: value for key, value in batch.items() if key != "labels"}, use_cache=False)
            loss = compute_cross_entropy_loss(
                outputs.logits,
                labels,
                ignore_index=self.config.data.ignore_index,
                label_smoothing=self.config.label_smoothing,
                loss_parallel_enabled=True,
            )
        else:
            outputs = self(**{**batch, "use_cache": False})
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = compute_cross_entropy_loss(
                    outputs.logits,
                    labels,
                    ignore_index=self.config.data.ignore_index,
                    label_smoothing=self.config.label_smoothing,
                )

        with torch.no_grad():
            stats = masked_token_stats(outputs.logits, labels, ignore_index=self.config.data.ignore_index)

        on_step = stage == "train"
        prog_bar = stage in {"train", "val"}
        self.log(f"{stage}/loss", loss, prog_bar=prog_bar, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/token_accuracy", stats["token_accuracy"], prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/entropy", stats["entropy"], prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/mean_logprob", stats["mean_logprob"], prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/perplexity", stats["perplexity"], prog_bar=False, on_step=on_step, on_epoch=True, sync_dist=True)
        log_moe_metrics(self, outputs, stage, on_step=on_step)

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
