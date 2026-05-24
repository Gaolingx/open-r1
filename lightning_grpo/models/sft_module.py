"""Lightning module for supervised fine-tuning."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.utils.configs.sft import SFTConfig
from lightning_grpo.models.common import build_optimizer, build_scheduler, masked_token_stats, compute_cross_entropy_loss, compute_liger_cross_entropy_loss
from lightning_grpo.strategies.fsdp2 import configure_fully_shard
from lightning_grpo.strategies.tensor_parallel import configure_tensor_parallel
from lightning_grpo.utils.modeling import compile_model_if_configured, count_trainable_parameters, export_configured_model, load_causal_lm, load_tokenizer, log_moe_metrics


class SFTLightningModule(L.LightningModule):
    """Lightning module for decoder-only causal LM SFT."""

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

        configure_tensor_parallel(self.model, self.config.distributed, self.device_mesh)
        configure_fully_shard(self.model, self.config.distributed, self.config.precision, self.device_mesh)
        self.model = compile_model_if_configured(self.model, self.config.model)

    def _model_forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        exclude_labels: bool = True,
        output_hidden_states: bool = False,
    ) -> Any:
        """Unified forward pass through the model.

        Args:
            batch: Input batch dictionary.
            exclude_labels: If True, strip 'labels' from inputs before forwarding.
            output_hidden_states: If True, request hidden states from the model.
        """
        inputs = {k: v for k, v in batch.items() if k != "labels"} if exclude_labels else batch
        return self.model(**inputs, use_cache=False, output_hidden_states=output_hidden_states)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Run one optimization/evaluation step and log metrics."""

        labels = batch["labels"]
        use_liger = self.config.use_liger_kernel

        if use_liger:
            # Liger path: forward without computing logits, use fused linear + CE kernel
            outputs = self._model_forward(batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            loss = compute_liger_cross_entropy_loss(
                model=self.model,
                hidden_states=hidden_states,
                labels=labels,
                ignore_index=self.config.data.ignore_index,
                label_smoothing=self.config.label_smoothing,
                loss_parallel_enabled=self.config.distributed.tensor_parallel.loss_parallel,
            )
        elif self.config.distributed.tensor_parallel.loss_parallel:
            outputs = self._model_forward(batch)
            loss = compute_cross_entropy_loss(
                outputs.logits,
                labels,
                ignore_index=self.config.data.ignore_index,
                label_smoothing=self.config.label_smoothing,
                loss_parallel_enabled=True,
            )
        else:
            outputs = self._model_forward(batch, exclude_labels=False)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = compute_cross_entropy_loss(
                    outputs.logits,
                    labels,
                    ignore_index=self.config.data.ignore_index,
                    label_smoothing=self.config.label_smoothing,
                )

        on_step = stage == "train"
        prog_bar = stage in {"train", "val"}
        self.log(f"{stage}/loss", loss, prog_bar=prog_bar, on_step=on_step, on_epoch=True, sync_dist=True)

        if not use_liger:
            # Full metrics only available when logits are materialized
            with torch.no_grad():
                stats = masked_token_stats(outputs.logits, labels, ignore_index=self.config.data.ignore_index)
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
