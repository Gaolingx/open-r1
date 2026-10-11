"""Lightning module for Group Relative Policy Optimization (GRPO)."""

from __future__ import annotations

import copy
from typing import Any

import lightning as L
import torch
import torch.distributed as dist

from lightning_grpo.models.common import (
    build_optimizer,
    build_scheduler,
    compile_model_if_configured,
    count_trainable_parameters,
    export_configured_model,
    load_tokenizer,
)
from lightning_grpo.models.grpo.contracts import DistributedContext
from lightning_grpo.models.grpo.rollout import LocalGenerateRolloutCoordinator, RolloutContext
from lightning_grpo.models.grpo.loss import GRPOLossComputer, create_grpo_loss_computer, resolve_grpo_algorithm
from lightning_grpo.models.grpo.metrics import GRPOMetricsAggregator
from lightning_grpo.models.grpo.reward import GRPORewardManager
from lightning_grpo.models.grpo.tool_call import GRPOToolCallMixin
from lightning_grpo.strategies.fsdp2 import configure_fully_shard
from lightning_grpo.strategies.tensor_parallel import configure_tensor_parallel
from lightning_grpo.utils.configs.grpo import GRPOConfig
from lightning_grpo.utils.modeling import load_causal_lm
from lightning_grpo.utils.liger_kernel.warpper import apply_liger_kernel


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
        self.setup_tool_calling()

        self.metrics_aggregator: GRPOMetricsAggregator | None = None
        self.rollout_coordinator: LocalGenerateRolloutCoordinator | None = None
        self.reward_manager: GRPORewardManager | None = None
        self.loss_computer: GRPOLossComputer | None = None

    def _build_reference_model(self) -> torch.nn.Module | None:
        """Create the frozen reference model used for KL regularization."""

        if not self.config.rollout.use_reference_model:
            return None
        reference_model = load_causal_lm(self.config.ref_model, self.config.precision) if self.config.ref_model else copy.deepcopy(self.policy)
        reference_model.requires_grad_(False)
        reference_model.eval()
        return reference_model

    def _build_distributed_context(self) -> DistributedContext:
        """Snapshot rank/world size and the all-gather service for the GRPO collaborators."""

        trainer = self.trainer
        if dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = trainer.global_rank, trainer.world_size
        return DistributedContext(rank=rank, world_size=world_size, all_gather_fn=self.all_gather)

    def forward(self, **batch: torch.Tensor) -> Any:
        """Forward tokens through the wrapped policy model."""

        return self.policy(**batch)

    def configure_model(self) -> None:
        """Apply tensor parallelism, FSDP2, compile, then initialize the GRPO collaborators."""

        # Apply liger kernel
        if self.config.liger_kernel.patch_model_enabled():
            apply_liger_kernel(model=self.policy, kernel_config=self.config.liger_kernel.kernel_config)

        configure_tensor_parallel(self.policy, self.config.distributed, self.device_mesh)
        configure_fully_shard(self.policy, self.config.distributed, self.config.precision, self.device_mesh)
        self.policy = compile_model_if_configured(self.policy, self.config.model)

        def policy_provider() -> torch.nn.Module:
            """Return the policy as it is bound right now, not as it was when loaded."""

            return self.policy

        def reference_policy_provider() -> torch.nn.Module | None:
            """Return the reference model, or ``None`` when KL regularization is off."""

            return self.reference_model

        distributed = self._build_distributed_context()
        self.rollout_coordinator = LocalGenerateRolloutCoordinator(
            RolloutContext(
                rollout=self.config.rollout,
                model=self.config.model,
                tokenizer=self.tokenizer,
                device=self.device,
                world_size=distributed.world_size,
                global_rank=distributed.rank,
                local_rank=self.local_rank,
            ),
            policy_provider,
        )
        self.rollout_coordinator.update_policy()
        if self.reference_model is not None:
            configure_tensor_parallel(self.reference_model, self.config.distributed, self.device_mesh)
            configure_fully_shard(self.reference_model, self.config.distributed, self.config.precision, self.device_mesh)

        self.reward_manager = GRPORewardManager(self.config.reward, self.tokenizer, device=self.device)
        self.metrics_aggregator = GRPOMetricsAggregator(
            self,
            clip_reporting=resolve_grpo_algorithm(self.config.rollout).preset.clip_reporting,
            reward_names=self.reward_manager.reward_names,
        )

        # ``create_grpo_loss_computer`` honours the Liger request only when the kernel
        # implements the selected loss; otherwise it warns and uses the standard path.
        self.loss_computer = create_grpo_loss_computer(
            self.config.rollout,
            self.reward_manager,
            self.metrics_aggregator,
            distributed,
            policy_fn=policy_provider,
            reference_policy_fn=reference_policy_provider,
            use_liger_kernel=self.config.liger_kernel.liger_grpo_enabled(),
            rollout_temperature=self.config.rollout.temperature,
            loss_parallel_enabled=self.config.distributed.tensor_parallel.loss_parallel,
            compiled=self.config.liger_kernel.compiled,
        )

    def _shared_step(self, batch: dict[str, list[Any]], stage: str) -> torch.Tensor:
        """Generate rollouts, compute GRPO loss, and log metrics."""

        training = stage == "train"
        rollout_batch = self.rollout_coordinator.rollout(batch, training=training)

        # Run tool calling loop if enabled and executor is initialized
        if self.tool_executor is not None:
            rollout_batch = self._run_tool_calling(rollout_batch)

        if self.loss_computer is None or self.metrics_aggregator is None:
            raise RuntimeError("GRPO loss computer is not initialized. Call configure_model() first.")
        loss, metrics = self.loss_computer.compute_loss(
            rollout_batch,
            num_generations=self.rollout_coordinator.resolve_num_generations(training),
        )
        self.metrics_aggregator.log_metrics(stage, loss, metrics, on_step=training, on_epoch=True)

        if self.config.rollout.debug_samples and self.trainer.is_global_zero:
            every = max(1, self.config.rollout.debug_every_n_steps)
            if self.global_step % every == 0:
                self.print(f"[GRPO DEBUG] prompt={rollout_batch['prompts'][0]!r}")
                self.print(f"[GRPO DEBUG] completion={rollout_batch['completions'][0]!r}")
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

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        """Refresh the rollout-engine weights right after every policy update."""

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure)
        if self.config.rollout.engine == "vllm":
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
        self.shutdown_tool_calling()
        # Release the rollout engine (colocated vLLM weights / server communicator) before exporting.
        if self.rollout_coordinator is not None:
            self.rollout_coordinator.shutdown()

        export_dir = self.config.output_dir + "/hf_final"
        exported_paths = export_configured_model(self.policy, self.config.model, export_dir, tokenizer=self.tokenizer)
        if exported_paths:
            self.print(f"Exported model artifacts to: {', '.join(sorted(str(p) for p in exported_paths.values()))}")
