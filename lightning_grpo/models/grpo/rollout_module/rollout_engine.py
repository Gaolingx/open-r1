"""Pluggable rollout backends for Lightning GRPO."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import PreTrainedTokenizerBase

from lightning_grpo.models.grpo.rollout_module.utils import (
    RolloutEngine,
    RolloutResult,
    pad_float_sequences,
    pad_sequences,
    truncate_completions,
    load_generation_config,
    looks_like_fsdp_enabled,
    sampled_token_logprobs,
)
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
        policy_model: torch.nn.Module,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        sampling_config_path: Optional[str] = None,
        max_completion_length: int = 2048,
        world_size: int = 1,
        local_rank: int = 0,
        global_rank: int = 0,
    ) -> None:
        from lightning_grpo.models.grpo.rollout_module.vllm_generation import VLLMGeneration

        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.world_size = world_size
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.generation_config = load_generation_config(sampling_config_path, SimpleNamespace(to_dict=lambda: {}))
        self.generation_config.max_new_tokens = self.generation_config.max_new_tokens or max_completion_length
        self._policy_model = policy_model
        self._model_name_or_path = model_name_or_path
        model_for_vllm = self._prepare_model_for_vllm(policy_model)
        self.engine = VLLMGeneration(
            model=model_for_vllm,
            accelerator=_LightningVLLMAcceleratorAdapter(self.device, world_size, local_rank, global_rank),
            is_fsdp_enabled=looks_like_fsdp_enabled(policy_model),
            processing_class=tokenizer,
            mode=vllm_config.mode,
            structured_outputs_regex=vllm_config.structured_outputs_regex,
            server_base_url=vllm_config.server_base_url,
            server_host=vllm_config.server_host,
            server_port=vllm_config.server_port,
            server_timeout=vllm_config.server_timeout,
            group_port=vllm_config.group_port,
            tensor_parallel_size=vllm_config.tensor_parallel_size,
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
            max_model_length=vllm_config.max_model_length,
            max_num_seqs=vllm_config.max_num_seqs,
            enable_sleep_mode=vllm_config.enable_sleep_mode,
            model_impl=vllm_config.model_impl,
            repetition_penalty=vllm_config.repetition_penalty,
            temperature=self.generation_config.temperature or 1.0,
            top_p=self.generation_config.top_p or 1.0,
            top_k=self.generation_config.top_k if self.generation_config.top_k is not None else 0,
            max_completion_length=self.generation_config.max_new_tokens,
            logprobs=vllm_config.logprobs,
            generation_kwargs=vllm_config.generation_kwargs,
        )

    def generate(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 1.0,
    ) -> RolloutResult:
        old_max_completion_length = self.engine.max_completion_length
        old_temperature = self.engine.temperature
        old_top_p = self.engine.top_p
        self.engine.max_completion_length = max_new_tokens
        self.engine.temperature = temperature
        self.engine.top_p = top_p
        try:
            prompt_id_lists = [ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, attention_mask, strict=True)]
            # VLLMGeneration.generate expects prompts to be expanded to one row per requested completion.
            # In server mode it de-duplicates with prompts[::num_generations]; in colocate mode it samples n=1.
            repeated_prompts = [ids for ids in prompt_id_lists for _ in range(num_generations)]
            generated_prompt_ids, completion_id_lists, logprobs, logprob_token_ids = self.engine.generate(
                prompts=repeated_prompts,
                images=None,
                num_generations=num_generations,
            )
            per_token_logps_list = sampled_token_logprobs(logprobs, completion_id_lists, logprob_token_ids)
            return self._build_rollout_result(generated_prompt_ids, completion_id_lists, per_token_logps_list, prompt_ids.device)
        finally:
            self.engine.max_completion_length = old_max_completion_length
            self.engine.temperature = old_temperature
            self.engine.top_p = old_top_p

    def update_policy(self, model: torch.nn.Module):
        self._policy_model = model
        self.engine.model = self._prepare_model_for_vllm(model)
        return self.engine.sync_weights()        

    def shutdown(self) -> None:
        llm = getattr(self.engine, "llm", None)
        if llm is not None:
            del self.engine.llm

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        return getattr(unwrapped, "_orig_mod", unwrapped)

    def _prepare_model_for_vllm(self, model: torch.nn.Module) -> torch.nn.Module:
        unwrapped = self._unwrap_model(model)
        # VLLMGeneration initializes colocated vLLM with model.name_or_path. Lightning-loaded or wrapped
        # models may not preserve that attribute, so force the configured path used by rollout creation.
        setattr(unwrapped, "name_or_path", self._model_name_or_path)
        return unwrapped

    def _build_rollout_result(
        self,
        prompt_ids_list: list[list[int]],
        completion_ids_list: list[list[int]],
        logprobs_list: list[list[float]] | None,
        device: torch.device,
    ) -> RolloutResult:
        repeated_prompt_ids = pad_sequences(prompt_ids_list, self.tokenizer.pad_token_id, device)
        repeated_prompt_mask = (repeated_prompt_ids != self.tokenizer.pad_token_id).long()
        completion_ids_tensor = pad_sequences(completion_ids_list, self.tokenizer.pad_token_id, device)
        completion_ids, completion_mask, completion_truncated, completion_id_lists = truncate_completions(
            completion_ids_tensor,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )
        if logprobs_list is None:
            per_token_logps = torch.empty((completion_ids.size(0), 0), dtype=torch.float32, device=device)
        else:
            aligned_logprobs = [row[:len(ids)] for row, ids in zip(logprobs_list, completion_id_lists, strict=True)]
            per_token_logps = pad_float_sequences(aligned_logprobs, completion_ids.size(1), device)
        completions_text = self.tokenizer.batch_decode(completion_id_lists, skip_special_tokens=True)
        output_ids = torch.cat([repeated_prompt_ids, completion_ids], dim=1)
        return RolloutResult(
            output_ids=output_ids,
            prompt_ids=repeated_prompt_ids,
            prompt_mask=repeated_prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            per_token_logps=per_token_logps,
            completions_text=completions_text,
            completion_id_lists=completion_id_lists,
            completion_truncated=completion_truncated,
        )


class _LightningVLLMAcceleratorAdapter:
    """Minimal Accelerate-compatible adapter used by TRL's vLLM generation helper."""

    def __init__(self, device: torch.device, world_size: int, local_rank: int, global_rank: int) -> None:
        self.device = device
        self.num_processes = world_size
        self.local_process_index = local_rank
        self.process_index = global_rank
        self.is_main_process = global_rank == 0
        self.state = SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=SimpleNamespace(fsdp_version=2))

    def wait_for_everyone(self) -> None:
        if dist.is_initialized():
            dist.barrier()


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
            policy_model=policy_model,
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            device=device,
            sampling_config_path=sampling_config_path,
            max_completion_length=max_completion_length,
            world_size=world_size,
            local_rank=local_rank,
            global_rank=global_rank,
        )
    raise ValueError(f"Not Support Rollout Engine Type: {engine_type}")
