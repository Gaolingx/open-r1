"""vLLM-based rollout engine for Lightning GRPO.

Supports two deployment modes:
- **server**: Connects to an external vLLM server via HTTP. The training process
  sends generation requests and weight updates over the network. Suitable for
  multi-node setups where inference runs on dedicated GPU nodes.
- **colocate**: Runs vLLM in-process on the same GPUs as training. Uses sleep mode
  to time-share GPU memory between training and inference. Lower latency but
  requires careful memory management.

Weight synchronization supports FSDP2 (composable), FSDP1, and plain DDP models.
Inspired by TRL's VLLMGeneration class but adapted for Lightning's distributed
training primitives.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase

from lightning_grpo.models.common import materialize_vocab_parallel_logits
from lightning_grpo.models.rollout_engine import (
    RolloutEngine,
    RolloutResult,
    _load_generation_config,
    _resolve_eos_token_ids,
    compute_per_token_logps,
    pad_float_sequences,
    pad_sequences,
    truncate_completions,
)
from lightning_grpo.utils.configs.grpo import VLLMConfig
from lightning_grpo.utils.modeling import DTYPE_MAP

logger = logging.getLogger(__name__)


def _empty_cache() -> None:
    """Empty GPU cache for the available device backend."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif hasattr(torch, "mlu") and torch.mlu.is_available():
        torch.mlu.empty_cache()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def _is_vllm_available() -> bool:
    """Check if vLLM is installed."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_logprobs(all_outputs: list) -> tuple[list[list[float]] | None, list[list[int]] | None]:
    """Extract per-token logprobs from vLLM RequestOutput objects.

    Returns:
        Tuple of (logprobs_list, token_ids_list) where each inner list is per-sequence.
        Returns (None, None) if logprobs are not available.
    """
    all_logprobs: list[list[float]] = []
    all_token_ids: list[list[int]] = []

    for outputs in all_outputs:
        for output in outputs.outputs:
            if output.logprobs is None:
                return None, None
            seq_logprobs: list[float] = []
            seq_token_ids: list[int] = []
            for lp in output.logprobs:
                # Get the sampled token's logprob (rank 1)
                sorted_items = sorted(lp.items(), key=lambda x: x[1].rank)
                token_id, logprob_info = sorted_items[0]
                lp_val = logprob_info.logprob
                if math.isnan(lp_val):
                    lp_val = 0.0
                seq_logprobs.append(lp_val)
                seq_token_ids.append(token_id)
            all_logprobs.append(seq_logprobs)
            all_token_ids.append(seq_token_ids)

    return all_logprobs, all_token_ids


class VLLMRolloutEngine(RolloutEngine):
    """vLLM-based rollout engine supporting server and colocate modes.

    In **server** mode, generation requests are sent to an external vLLM server.
    Weight updates are pushed via a NCCL communicator or HTTP endpoint.

    In **colocate** mode, vLLM runs in-process sharing GPUs with training.
    Sleep mode allows time-sharing memory between training and inference.

    Args:
        vllm_config: VLLMConfig dataclass with all vLLM parameters.
        model_name_or_path: HuggingFace model ID or local path.
        tokenizer: Pre-loaded tokenizer instance.
        sampling_config_path: Path to generation config JSON.
        world_size: Total number of training processes.
        local_rank: This process's local rank.
        global_rank: This process's global rank.
    """

    def __init__(
        self,
        *,
        vllm_config: VLLMConfig,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        sampling_config_path: Optional[str] = None,
        world_size: int = 1,
        local_rank: int = 0,
        global_rank: int = 0,
    ) -> None:
        self.config = vllm_config
        self.mode = vllm_config.mode
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.world_size = world_size
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.is_main_process = (global_rank == 0)

        # Load generation config
        self.generation_config = _load_generation_config(
            sampling_config_path, GenerationConfig()
        )

        # Override temperature from vllm config if set
        self._temperature = vllm_config.repetition_penalty  # stored for sampling params

        # Initialize vLLM backend
        self._vllm_client = None  # For server mode
        self._llm = None  # For colocate mode
        self._tp_group = None

        self._init_vllm()

    def _init_vllm(self) -> None:
        """Initialize vLLM in the configured mode."""
        if not _is_vllm_available():
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )

        if self.mode == "server":
            self._init_server_mode()
        elif self.mode == "colocate":
            self._init_colocate_mode()
        else:
            raise ValueError(f"vLLM mode must be 'server' or 'colocate', got '{self.mode}'")

        # Synchronize all processes after vLLM initialization
        if dist.is_initialized():
            dist.barrier()

    def _init_server_mode(self) -> None:
        """Initialize server mode with HTTP client."""
        if self.is_main_process:
            from lightning_grpo.models.vllm_client import VLLMClient

            base_url = self.config.server_base_url
            if base_url is None:
                base_url = f"http://{self.config.server_host}:{self.config.server_port}"

            self._vllm_client = VLLMClient(
                base_url=base_url,
                group_port=self.config.group_port,
                connection_timeout=self.config.server_timeout,
            )
            self._vllm_client.init_communicator(device=torch.cuda.current_device())

    def _init_colocate_mode(self) -> None:
        """Initialize colocate mode with in-process vLLM."""
        import vllm
        from vllm import LLM

        tp_size = self.config.tensor_parallel_size

        # Validate TP size divides world size
        if self.world_size % tp_size != 0:
            raise ValueError(
                f"tensor_parallel_size ({tp_size}) must divide world_size "
                f"({self.world_size}) evenly."
            )

        # Create TP subgroups if needed
        if tp_size > 1 and dist.is_initialized():
            self._tp_group, _ = dist.new_subgroups_by_enumeration(
                [
                    list(range(i * tp_size, (i + 1) * tp_size))
                    for i in range(self.world_size // tp_size)
                ]
            )

        # Set environment variables required by vLLM
        os.environ.setdefault("RANK", str(self.global_rank))
        os.environ.setdefault("LOCAL_RANK", str(self.local_rank))
        os.environ.setdefault("WORLD_SIZE", str(self.world_size))
        # Ensure MASTER_ADDR and MASTER_PORT are set
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        # Build LLM instance
        llm_kwargs: dict[str, Any] = {
            "model": self.model_name_or_path,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "distributed_executor_backend": "external_launcher",
            "seed": self.global_rank // tp_size,
            "max_num_batched_tokens": 4096,
        }

        if self.config.max_model_length is not None:
            llm_kwargs["max_model_len"] = self.config.max_model_length
        if self.config.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = self.config.max_num_seqs
        if self.config.model_impl != "auto":
            llm_kwargs["model_impl"] = self.config.model_impl

        self._llm = LLM(**llm_kwargs)

        if self.config.enable_sleep_mode:
            self._llm.sleep(level=2)

    def _build_sampling_params(self, num_generations: int = 1):
        """Build vLLM SamplingParams from config."""
        from vllm import SamplingParams

        config = self.config
        gen_config = self.generation_config

        params: dict[str, Any] = {
            "n": num_generations if self.mode == "server" else 1,
            "temperature": gen_config.temperature if gen_config.temperature else 1.0,
            "top_p": gen_config.top_p if gen_config.top_p else 1.0,
            "top_k": gen_config.top_k if gen_config.top_k else -1,
            "max_tokens": gen_config.max_new_tokens if gen_config.max_new_tokens else 2048,
            "repetition_penalty": config.repetition_penalty,
            "logprobs": config.logprobs,
        }

        # Apply any extra generation kwargs
        if config.generation_kwargs:
            params.update(config.generation_kwargs)

        # Handle structured outputs
        if config.structured_outputs_regex:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                params["structured_outputs"] = StructuredOutputsParams(
                    regex=config.structured_outputs_regex
                )
            except ImportError:
                from vllm.sampling_params import GuidedDecodingParams
                params["guided_decoding"] = GuidedDecodingParams(
                    regex=config.structured_outputs_regex
                )

        return SamplingParams(**params)

    def rollout(
        self,
        *,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
    ) -> RolloutResult:
        """Generate completions using vLLM.

        In server mode: main process gathers all prompts, generates, and broadcasts.
        In colocate mode: each process generates its own batch (with TP coordination).
        """
        if self.mode == "server":
            return self._rollout_server(prompt_ids, attention_mask, num_generations)
        else:
            return self._rollout_colocate(prompt_ids, attention_mask, num_generations)

    def _rollout_server(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
    ) -> RolloutResult:
        """Server mode: main process generates, broadcasts results."""
        device = prompt_ids.device

        # Prepare prompt token lists (remove padding)
        input_ids_list = [
            ids[mask.bool()].tolist()
            for ids, mask in zip(prompt_ids, attention_mask, strict=True)
        ]

        # Gather prompts from all ranks to main process
        if dist.is_initialized() and self.world_size > 1:
            all_prompts_gathered = [None] * self.world_size
            dist.all_gather_object(all_prompts_gathered, input_ids_list)
            all_input_ids = [ids for rank_ids in all_prompts_gathered for ids in rank_ids]
        else:
            all_input_ids = input_ids_list

        # Main process generates
        if self.is_main_process:
            # Deduplicate: take unique prompts and generate num_generations each
            unique_prompts = all_input_ids[::num_generations] if num_generations > 1 else all_input_ids

            sampling_params = self._build_sampling_params(num_generations)

            # Use vLLM client to generate
            output = self._vllm_client.generate(
                prompts=unique_prompts,
                sampling_params=sampling_params,
            )

            all_completion_ids = output["completion_ids"]
            all_logprobs = output.get("logprobs")
            # Duplicate prompt_ids for num_generations
            all_prompt_ids_out = [ids for ids in all_input_ids for _ in range(num_generations)] if num_generations > 1 else all_input_ids

            payload = (all_prompt_ids_out, all_completion_ids, all_logprobs)
        else:
            payload = None

        # Broadcast results
        obj_list = [payload]
        if dist.is_initialized():
            dist.broadcast_object_list(obj_list, src=0)
        all_prompt_ids_out, all_completion_ids, all_logprobs = obj_list[0]

        # Slice this rank's portion
        local_batch = len(input_ids_list) * num_generations
        start = self.global_rank * local_batch
        end = start + local_batch

        local_prompt_ids = all_prompt_ids_out[start:end]
        local_completion_ids = all_completion_ids[start:end]
        local_logprobs = all_logprobs[start:end] if all_logprobs else None

        return self._build_rollout_result(local_prompt_ids, local_completion_ids, local_logprobs, device)

    def _rollout_colocate(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
    ) -> RolloutResult:
        """Colocate mode: each rank generates with local vLLM instance."""
        device = prompt_ids.device
        tp_size = self.config.tensor_parallel_size

        # Wake up vLLM if in sleep mode
        if self.config.enable_sleep_mode:
            _empty_cache()
            self._llm.wake_up(tags=["weights"])
            try:
                self._llm.collective_rpc("reload_weights")
            except (NotImplementedError, AttributeError):
                pass

        # Prepare prompts - repeat for num_generations
        input_ids_list = [
            ids[mask.bool()].tolist()
            for ids, mask in zip(prompt_ids, attention_mask, strict=True)
        ]
        # In colocate mode, each prompt is already repeated num_generations times by the dataloader
        prompts = input_ids_list

        # For TP > 1, gather prompts within TP group
        if tp_size > 1 and self._tp_group is not None:
            orig_size = len(prompts)
            gathered = [None] * tp_size
            dist.all_gather_object(gathered, prompts, group=self._tp_group)
            all_prompts = [p for sublist in gathered for p in sublist]
        else:
            orig_size = len(prompts)
            all_prompts = prompts

        # Generate with vLLM
        sampling_params = self._build_sampling_params(num_generations=1)

        if self.config.enable_sleep_mode:
            self._llm.wake_up(tags=["kv_cache"])

        all_outputs = self._llm.generate(
            prompt_token_ids=all_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Extract results
        all_prompt_ids_out = [output.prompt_token_ids for output in all_outputs]
        all_completion_ids_out = [
            list(output.outputs[0].token_ids) for output in all_outputs
        ]
        all_logprobs_out, _ = _extract_logprobs(all_outputs)

        # Slice for this rank within TP group
        if tp_size > 1 and self._tp_group is not None:
            local_rank_in_group = dist.get_rank(group=self._tp_group)
            tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
            local_prompt_ids = all_prompt_ids_out[tp_slice]
            local_completion_ids = all_completion_ids_out[tp_slice]
            local_logprobs = all_logprobs_out[tp_slice] if all_logprobs_out else None
        else:
            local_prompt_ids = all_prompt_ids_out
            local_completion_ids = all_completion_ids_out
            local_logprobs = all_logprobs_out

        if self.config.enable_sleep_mode:
            self._llm.sleep(level=2)

        return self._build_rollout_result(local_prompt_ids, local_completion_ids, local_logprobs, device)

    def _build_rollout_result(
        self,
        prompt_ids_list: list[list[int]],
        completion_ids_list: list[list[int]],
        logprobs_list: Optional[list[list[float]]],
        device: torch.device,
    ) -> RolloutResult:
        """Convert raw lists into a structured RolloutResult with proper padding."""
        # Pad prompt IDs
        repeated_prompt_ids = pad_sequences(
            prompt_ids_list, pad_value=self.tokenizer.pad_token_id, device=device
        )
        repeated_prompt_mask = (repeated_prompt_ids != self.tokenizer.pad_token_id).long()

        # Pad completion IDs and truncate at EOS
        completion_ids_tensor = pad_sequences(
            completion_ids_list, pad_value=self.tokenizer.pad_token_id, device=device
        )
        completion_ids, completion_mask, completion_truncated, completion_id_lists = truncate_completions(
            completion_ids_tensor,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )

        # Align logprobs with truncated completions
        if logprobs_list is not None:
            aligned_logprobs = [
                lp[:len(ids)] for lp, ids in zip(logprobs_list, completion_id_lists)
            ]
            per_token_logps = pad_float_sequences(
                aligned_logprobs, completion_ids.shape[1], device=device
            )
        else:
            # If no logprobs from vLLM, we'll need to compute them via forward pass
            per_token_logps = torch.zeros(
                completion_ids.shape, dtype=torch.float32, device=device
            )

        completions_text = self.tokenizer.batch_decode(
            completion_id_lists, skip_special_tokens=True
        )
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

    def update_policy(self, model: torch.nn.Module) -> None:
        """Synchronize training model weights to vLLM.

        Handles FSDP2 (composable), FSDP1, and plain DDP/single-GPU models.
        For colocate mode, weights are loaded directly into the vLLM model.
        For server mode, weights are pushed via the NCCL communicator.
        """
        # Wake up vLLM weights before sync
        if self.mode == "colocate" and self.config.enable_sleep_mode:
            _empty_cache()
            self._llm.wake_up(tags=["weights"])

        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

        # Determine if we're using FSDP2 (composable) by checking for DTensor params
        is_fsdp2 = self._is_fsdp2_model(unwrapped)

        if is_fsdp2:
            self._sync_fsdp2_weights(unwrapped)
        else:
            self._sync_standard_weights(unwrapped)

        # Reset KV cache after weight update
        if self.mode == "server" and self.is_main_process and self._vllm_client is not None:
            self._vllm_client.reset_prefix_cache()
        elif self.mode == "colocate" and self._llm is not None:
            self._llm.reset_prefix_cache()

        # Barrier to ensure all ranks are synchronized
        if dist.is_initialized():
            dist.barrier()

    def _is_fsdp2_model(self, model: torch.nn.Module) -> bool:
        """Check if model uses FSDP2 (composable) by looking for DTensor parameters."""
        try:
            from torch.distributed._tensor import DTensor
            for param in model.parameters():
                if isinstance(param, DTensor):
                    return True
        except ImportError:
            pass
        return False

    def _sync_fsdp2_weights(self, model: torch.nn.Module) -> None:
        """Sync weights from an FSDP2 (composable) model to vLLM."""
        # For FSDP2, state_dict() returns full tensors via DTensor.full_tensor()
        for name, param in model.state_dict().items():
            # Clean up parameter name for vLLM compatibility
            name = self._fix_param_name(name)

            # Materialize DTensor to regular tensor
            if hasattr(param, "full_tensor"):
                param = param.full_tensor()
            if param.is_cpu:
                param = param.to(torch.device("cuda"))

            self._push_weight(name, param)

    def _sync_standard_weights(self, model: torch.nn.Module) -> None:
        """Sync weights from a standard (non-FSDP2) model to vLLM.

        For FSDP1 models, uses get_model_state_dict with full_state_dict option.
        For plain models, iterates parameters directly.
        """
        try:
            # Try FSDP-aware state dict extraction
            state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        except (TypeError, RuntimeError):
            # Fallback for non-FSDP models
            state_dict = {k: v.detach() for k, v in model.state_dict().items()}

        for name, param in state_dict.items():
            name = self._fix_param_name(name)
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            self._push_weight(name, param)

    def _push_weight(self, name: str, param: torch.Tensor) -> None:
        """Push a single weight tensor to vLLM."""
        if self.mode == "server":
            if self.is_main_process and self._vllm_client is not None:
                self._vllm_client.update_named_param(name, param.data)
        elif self.mode == "colocate":
            if self._llm is not None:
                llm_model = self._llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param.data)])

    @staticmethod
    def _fix_param_name(name: str) -> str:
        """Fix parameter names for vLLM compatibility.

        Removes common wrapper prefixes added by FSDP, DDP, or checkpoint wrappers.
        """
        prefixes_to_remove = [
            "_checkpoint_wrapped_module.",
            "_fsdp_wrapped_module.",
            "module.",
        ]
        for prefix in prefixes_to_remove:
            name = name.replace(prefix, "")
        return name

    def generate_chat(
        self,
        conversations: list[list[dict[str, str]]],
        num_generations: int = 1,
        tools: Optional[list[dict]] = None,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
    ) -> tuple[list[list[int]], list[list[int]], list[list[float]] | None]:
        """Generate completions from chat conversations (for tool calling).

        This method handles conversational inputs directly, useful for multi-turn
        tool calling where the conversation grows with each iteration.

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs) as lists.
        """
        chat_template_kwargs = chat_template_kwargs or {}

        if self.mode == "server" and self.is_main_process and self._vllm_client is not None:
            sampling_params = {
                "n": num_generations,
                "temperature": self.generation_config.temperature or 1.0,
                "top_p": self.generation_config.top_p or 1.0,
                "top_k": self.generation_config.top_k or -1,
                "max_tokens": self.generation_config.max_new_tokens or 2048,
                "repetition_penalty": self.config.repetition_penalty,
                "logprobs": self.config.logprobs,
            }
            if self.config.generation_kwargs:
                sampling_params.update(self.config.generation_kwargs)

            output = self._vllm_client.chat(
                messages=conversations,
                tools=tools,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                **sampling_params,
            )
            return output["prompt_ids"], output["completion_ids"], output.get("logprobs")

        elif self.mode == "colocate" and self._llm is not None:
            sampling_params = self._build_sampling_params(num_generations)

            if self.config.enable_sleep_mode:
                self._llm.wake_up(tags=["weights", "kv_cache"])

            all_outputs = self._llm.chat(
                conversations,
                sampling_params=sampling_params,
                use_tqdm=False,
                tools=tools,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
            )

            prompt_ids = [output.prompt_token_ids for output in all_outputs]
            completion_ids = [
                list(output.outputs[0].token_ids) for output in all_outputs
            ]
            logprobs_list, _ = _extract_logprobs(all_outputs)

            if self.config.enable_sleep_mode:
                self._llm.sleep(level=2)

            return prompt_ids, completion_ids, logprobs_list

        else:
            return [], [], None

    def score(self, samples: list[dict[str, Any]]) -> list[float]:
        """Not supported for vLLM engine."""
        raise NotImplementedError("VLLMRolloutEngine does not support reward scoring.")

    def shutdown(self) -> None:
        """Clean up vLLM resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._vllm_client is not None:
            self._vllm_client = None
        _empty_cache()
