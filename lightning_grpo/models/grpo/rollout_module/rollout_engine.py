"""Pluggable rollout backends for Lightning GRPO."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import PreTrainedTokenizerBase

from lightning_grpo.models.grpo.rollout_module.utils import RolloutEngine, RolloutResult, truncate_completions
from lightning_grpo.utils.configs.grpo import VLLMConfig


# ===== PyTorch Rollout Engine =====
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def generate(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8, top_p: float = 1.0) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        model = getattr(model, '_orig_mod', model)

        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_router_logits=False,
            ).clone()

        prompt_len = prompt_ids.size(1)
        repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        repeated_prompt_mask = attention_mask.repeat_interleave(num_generations, dim=0)
        completion_ids = output_ids[:, prompt_len:]
        completion_ids, completion_mask, completion_truncated, completion_id_lists = truncate_completions(
            completion_ids,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )
        completions_text = self.tokenizer.batch_decode(completion_id_lists, skip_special_tokens=True)
        output_ids = torch.cat([repeated_prompt_ids, completion_ids], dim=1)
        return RolloutResult(
            output_ids=output_ids,
            prompt_ids=repeated_prompt_ids,
            prompt_mask=repeated_prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            per_token_logps=torch.empty((completion_ids.size(0), 0), dtype=torch.float32, device=completion_ids.device),
            completions_text=completions_text,
            completion_id_lists=completion_id_lists,
            completion_truncated=completion_truncated,
        )

    def update_policy(self, model: torch.nn.Module):
        self.policy_model = model


# ===== vLLM Rollout Engine =====
class VLLMRolloutEngineWrapper(RolloutEngine):
    def __init__(
        self,
        *,
        vllm_config: VLLMConfig,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        sampling_config_path: Optional[str] = None,
        max_completion_length: int = 2048,
        world_size: int = 1,
        local_rank: int = 0,
        global_rank: int = 0,
    ) -> None:
        from lightning_grpo.models.grpo.rollout_module.vllm_rollout_engine import VLLMRolloutEngine

        self.engine = VLLMRolloutEngine(
            vllm_config=vllm_config,
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            sampling_config_path=sampling_config_path,
            max_completion_length=max_completion_length,
            world_size=world_size,
            local_rank=local_rank,
            global_rank=global_rank,
        )
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 1.0,
    ) -> RolloutResult:
        old_max_new_tokens = self.engine.generation_config.max_new_tokens
        old_temperature = self.engine.generation_config.temperature
        old_top_p = self.engine.generation_config.top_p

        self.engine.generation_config.max_new_tokens = max_new_tokens
        self.engine.generation_config.temperature = temperature
        self.engine.generation_config.top_p = top_p

        try:
            return self.engine.rollout(
                prompt_ids=prompt_ids,
                attention_mask=attention_mask,
                num_generations=num_generations,
            )
        finally:
            self.engine.generation_config.max_new_tokens = old_max_new_tokens
            self.engine.generation_config.temperature = old_temperature
            self.engine.generation_config.top_p = old_top_p

    def update_policy(self, model: torch.nn.Module):
        return self.engine.update_policy(model)

    def generate_chat(self, *args, **kwargs):
        return self.engine.generate_chat(*args, **kwargs)

    def shutdown(self) -> None:
        if hasattr(self.engine, "shutdown"):
            self.engine.shutdown()


def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer: PreTrainedTokenizerBase = None,
    device: str = "cuda",
    autocast_ctx = None,
    vllm_config: Optional[VLLMConfig] = None,
    model_name_or_path: Optional[str] = None,
    sampling_config_path: Optional[str] = None,
    max_completion_length: int = 2048,
    world_size: int = 1,
    local_rank: int = 0,
    global_rank: int = 0,
    **kwargs: Any,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    if engine_type == "vllm":
        if vllm_config is None:
            raise ValueError("rollout.vllm config must be set when using the vllm rollout engine")
        if not model_name_or_path:
            raise ValueError("model_name_or_path must be provided for the vllm rollout engine")
        return VLLMRolloutEngineWrapper(
            vllm_config=vllm_config,
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            sampling_config_path=sampling_config_path,
            max_completion_length=max_completion_length,
            world_size=world_size,
            local_rank=local_rank,
            global_rank=global_rank,
        )
    raise ValueError(f"Not Support Rollout Engine Type: {engine_type}")
