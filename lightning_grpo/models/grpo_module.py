"""Lightning module for GRPO-style online RL fine-tuning."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_info

from lightning_grpo.models.common import build_optimizer, build_scheduler
from lightning_grpo.models.grpo import (
    GRPOLossComputer,
    GRPOMetricsAggregator,
    GRPORewardManager,
    GRPORolloutCoordinator,
    ToolCallExecutor,
)
from lightning_grpo.strategies.fsdp2 import configure_fully_shard
from lightning_grpo.strategies.tensor_parallel import configure_tensor_parallel
from lightning_grpo.utils.configs.grpo import GRPOConfig
from lightning_grpo.utils.modeling import compile_model_if_configured, count_trainable_parameters, export_configured_model, load_causal_lm, load_tokenizer


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
        self.rollout_generation_config = self.rollout_engine.generation_config
        self.reward_manager = GRPORewardManager(config, self.tokenizer, self.rollout_engine, self.reward_model_engine, self.device)
        self.register_buffer("reward_weights", self.reward_manager.reward_weight_tensor.clone(), persistent=False)
        self.metrics_aggregator = GRPOMetricsAggregator(self)

        # Select loss computer: Liger Kernel fused loss or standard loss
        use_liger = config.use_liger_kernel or config.rollout.liger_kernel.enabled
        if use_liger:
            from lightning_grpo.models.grpo.liger_loss import LigerGRPOLossComputer
            self.loss_computer = LigerGRPOLossComputer(
                self,
                self.reward_manager,
                self.metrics_aggregator,
                rollout_temperature=self.rollout_generation_config.temperature,
            )
            rank_zero_info("Using Liger Kernel fused GRPO loss for memory-efficient training")
        else:
            self.loss_computer = GRPOLossComputer(
                self,
                self.reward_manager,
                self.metrics_aggregator,
                rollout_temperature=self.rollout_generation_config.temperature,
            )

        # Initialize tool calling executor if configured
        self.tool_executor: ToolCallExecutor | None = None
        if config.rollout.tool_calling.enabled:
            from lightning_grpo.models.grpo.tool_calling import load_tools_from_names
            tools = load_tools_from_names(config.rollout.tool_calling.tools)
            self.tool_executor = ToolCallExecutor(
                tools=tools,
                tokenizer=self.tokenizer,
                max_iterations=config.rollout.tool_calling.max_iterations,
                max_completion_length=config.rollout.max_completion_length,
                chat_template=config.rollout.tool_calling.chat_template,
                chat_template_kwargs=config.rollout.tool_calling.chat_template_kwargs,
            )
            rank_zero_info(
                "Tool calling enabled with %d tools, max %d iterations",
                len(tools), config.rollout.tool_calling.max_iterations,
            )

        self.save_hyperparameters(config.to_dict())

        trainable, total = count_trainable_parameters(self.policy)
        self.trainable_parameter_count = trainable
        self.total_parameter_count = total

    def on_fit_start(self) -> None:
        """Log static parameter counts once training starts."""

        self.reward_manager.device = self.device
        if self.reward_model_engine is not None and hasattr(self.reward_model_engine, "to"):
            self.reward_model_engine.to(self.device)

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

    def configure_model(self) -> None:
        """Apply tensor parallelism, then composable FSDP2 to the trainable policy model."""

        configure_tensor_parallel(self.policy, self.config.distributed, self.device_mesh)
        configure_fully_shard(self.policy, self.config.distributed, self.config.precision, self.device_mesh)
        self.policy = compile_model_if_configured(self.policy, self.config.model)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one online rollout and optimization step."""

        debug_every = self.config.rollout.debug.every_n_steps
        if self.config.rollout.debug.enabled and debug_every > 0 and self.global_step % debug_every == 0:
            self.rollout_coordinator.emit_debug_samples(self.trainer, self.device)

        rollout_batch = self.rollout_coordinator.generate(batch, training=True)

        # Run tool calling loop if enabled and completions contain tool calls
        if self.tool_executor is not None:
            rollout_batch = self._run_tool_calling(rollout_batch)

        loss, metrics = self.loss_computer.compute_loss(rollout_batch, training=True)
        self.metrics_aggregator.log_metrics("train", loss, metrics, on_step=True, on_epoch=True)
        return loss

    def _run_tool_calling(self, rollout_batch: dict[str, Any]) -> dict[str, Any]:
        """Execute multi-turn tool calling loop on rollout results.

        Detects tool calls in completions, executes them, regenerates, and
        produces a tool_mask that excludes tool-injected tokens from the loss.
        """
        completions = rollout_batch.get("completions", [])
        if not completions:
            return rollout_batch

        # Check if any completion has tool calls
        has_tool_calls = any(
            isinstance(c, list) and c and isinstance(c[-1], dict) and c[-1].get("tool_calls")
            for c in completions
        )
        if not has_tool_calls:
            return rollout_batch

        # Build generate function for regeneration after tool execution
        def _generate_fn(conversations):
            """Regenerate completions for conversations with tool results."""
            engine = self.rollout_engine
            # Use vLLM chat generation if available
            if hasattr(engine, "generate_chat"):
                _, new_ids, new_logprobs = engine.generate_chat(
                    conversations,
                    num_generations=1,
                    tools=self.tool_executor.tool_schemas,
                    chat_template=self.config.rollout.tool_calling.chat_template,
                    chat_template_kwargs=self.config.rollout.tool_calling.chat_template_kwargs,
                )
                return new_ids, new_logprobs
            # Fallback: tokenize and generate via policy
            return [], None

        # Run the tool loop
        prompt_ids_list = rollout_batch.get("prompt_id_lists", [
            ids[mask.bool()].tolist()
            for ids, mask in zip(rollout_batch["prompt_ids"], rollout_batch["prompt_mask"])
        ])
        completion_ids_list = rollout_batch.get("completion_id_lists", [
            ids.tolist() for ids in rollout_batch["completion_ids"]
        ])
        logprobs_list = rollout_batch.get("old_per_token_logps_list", None)

        # Get prompts as conversations
        prompts = rollout_batch.get("prompts_structured", rollout_batch.get("prompts", []))

        tool_result = self.tool_executor.run_tool_loop(
            prompts=prompts if isinstance(prompts[0], list) else [[{"role": "user", "content": p}] for p in prompts],
            prompt_ids=prompt_ids_list,
            completion_ids=completion_ids_list,
            completions=completions,
            logprobs=logprobs_list,
            generate_fn=_generate_fn,
        )

        # Update rollout_batch with tool calling results
        if tool_result["tool_mask"]:
            # Convert tool_mask to tensor matching completion_ids shape
            max_len = rollout_batch["completion_ids"].shape[1]
            device = rollout_batch["completion_ids"].device
            tool_mask_tensor = torch.zeros(
                len(tool_result["tool_mask"]), max_len, dtype=torch.long, device=device
            )
            for i, mask in enumerate(tool_result["tool_mask"]):
                length = min(len(mask), max_len)
                tool_mask_tensor[i, :length] = torch.tensor(mask[:length], dtype=torch.long)
            rollout_batch["tool_mask"] = tool_mask_tensor

        # Log tool calling metrics
        if self.trainer.is_global_zero:
            self.log("train/tool_call_count", float(tool_result["tool_call_count"]), on_step=True)
            self.log("train/tool_failure_count", float(tool_result["tool_failure_count"]), on_step=True)

        return rollout_batch

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

        # Clean up tool executor resources
        if self.tool_executor is not None:
            self.tool_executor.shutdown()

        # Clean up vLLM engine if applicable
        if hasattr(self.rollout_engine, "shutdown"):
            self.rollout_engine.shutdown()

        if not self.trainer.is_global_zero:
            return

        export_dir = self.config.output_dir + "/hf_final"
        exported_paths = export_configured_model(
            self.policy,
            self.config.model,
            export_dir,
            tokenizer=self.tokenizer,
        )
        if exported_paths:
            rank_zero_info(f"Exported model artifacts to {export_dir}: {sorted(exported_paths)}")
