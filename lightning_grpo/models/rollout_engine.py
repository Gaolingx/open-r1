"""Pluggable rollout backends for Lightning GRPO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any, Optional, Sequence

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase

from lightning_grpo.models.common import materialize_vocab_parallel_logits
from lightning_grpo.utils.configs.grpo import RewardModelConfig
from lightning_grpo.utils.modeling import DTYPE_MAP


logger = logging.getLogger(__name__)


def _load_generation_config(sampling_config_path: Optional[str], fallback: GenerationConfig) -> GenerationConfig:
    """Load rollout sampling config, falling back to the model generation config."""

    if sampling_config_path:
        return GenerationConfig.from_pretrained(pretrained_model_name=sampling_config_path)
    return GenerationConfig.from_dict(fallback.to_dict())


def _resolve_eos_token_ids(eos_token_id: Optional[int | Sequence[int]]) -> list[int]:
    """Normalize single or multiple EOS token ids into a list."""

    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    return [int(token_id) for token_id in eos_token_id]

def compute_per_token_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    n_keep: int,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float,
) -> torch.Tensor:
    """Compute per-token log-probabilities for the sampled completion suffix."""

    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    temperature = 1.0 if temperature is None else float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive when computing log-probabilities, got {temperature}.")

    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    outputs = unwrapped(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = materialize_vocab_parallel_logits(outputs.logits)[:, :-1, :]
    logits = logits[:, -n_keep:, :] / temperature
    target_ids = input_ids[:, -n_keep:]
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


@dataclass
class RolloutResult:
    """Structured rollout outputs shared by all rollout backends."""

    output_ids: torch.Tensor
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    per_token_logps: torch.Tensor
    completions_text: list[str]
    completion_id_lists: list[list[int]]
    completion_truncated: torch.Tensor


class RolloutEngine(ABC):
    """Abstract rollout backend."""

    tokenizer: PreTrainedTokenizerBase

    @abstractmethod
    def rollout(
        self,
        *,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
    ) -> RolloutResult:
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, model: torch.nn.Module) -> None:
        raise NotImplementedError

    def score(self, samples: list[dict[str, Any]]) -> list[float]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support reward-model scoring.")


class PolicyRolloutEngine(RolloutEngine):
    """In-process rollout using the current policy model."""

    _GENERATION_CONFIG_OVERRIDE_KEYS = frozenset({"num_return_sequences"})

    def __init__(
        self,
        policy_model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        sampling_config_path: Optional[str],
        generation_batch_size: int = 0,
        output_router_logits: bool = False,
    ) -> None:
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.generation_config = _load_generation_config(sampling_config_path, policy_model.generation_config)
        self.generation_batch_size = max(0, int(generation_batch_size))
        self.output_router_logits = output_router_logits

    def rollout(
        self,
        *,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
    ) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        original_padding_side = self.tokenizer.padding_side
        if original_padding_side != "left":
            self.tokenizer.padding_side = "left"

        try:
            with torch.no_grad():
                generated = self._generate_in_chunks(
                    model=model,
                    prompt_ids=prompt_ids,
                    attention_mask=attention_mask,
                    num_generations=num_generations,
                )
        finally:
            self.tokenizer.padding_side = original_padding_side

        repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        repeated_prompt_mask = attention_mask.repeat_interleave(num_generations, dim=0)
        prompt_length = repeated_prompt_ids.shape[1]
        completion_ids = generated[:, prompt_length:]
        completion_ids, completion_mask, completion_truncated, completion_id_lists = truncate_completions(
            completion_ids,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )
        completions_text = self.tokenizer.batch_decode(completion_id_lists, skip_special_tokens=False)
        model_input_ids = torch.cat([repeated_prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([repeated_prompt_mask, completion_mask], dim=1)
        per_token_logps = compute_per_token_logps(
            self.policy_model,
            model_input_ids,
            completion_ids.shape[1],
            attention_mask=model_attention_mask,
            temperature=self.generation_config.temperature,
        )
        return RolloutResult(
            output_ids=model_input_ids,
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
        self.policy_model = model

    def _build_generation_config(self, num_generations: int) -> GenerationConfig:
        """Build the per-call generation config for local policy rollouts."""

        generation_config = {
            key: value
            for key, value in self.generation_config.to_dict().items()
            if key not in self._GENERATION_CONFIG_OVERRIDE_KEYS and value is not None
        }
        generation_config["num_return_sequences"] = num_generations
        generation_config.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generation_config.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        return GenerationConfig(**generation_config)

    def _generate_in_chunks(
        self,
        *,
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
    ) -> torch.Tensor:
        """Generate completions in prompt chunks to cap rollout memory usage."""

        chunk_size = self.generation_batch_size or prompt_ids.size(0)
        if chunk_size <= 0:
            chunk_size = prompt_ids.size(0)

        generated_chunks: list[torch.Tensor] = []
        for start in range(0, prompt_ids.size(0), chunk_size):
            end = min(start + chunk_size, prompt_ids.size(0))
            generated_chunk = model.generate(
                input_ids=prompt_ids[start:end],
                attention_mask=attention_mask[start:end],
                generation_config=self._build_generation_config(num_generations),
                output_router_logits=self.output_router_logits,
            )
            generated_chunks.append(generated_chunk)

        if not generated_chunks:
            return prompt_ids.new_empty((0, prompt_ids.size(1)), dtype=prompt_ids.dtype)
        return torch.cat(generated_chunks, dim=0)


class RewardModelRolloutEngine(RolloutEngine):
    """Local reward-model inference backend for GRPO reward scoring."""

    def __init__(self, reward_model_config: RewardModelConfig) -> None:
        if not reward_model_config.model_name_or_path:
            raise ValueError("reward_model.model_name_or_path must be set when reward_model is enabled.")

        self.config = reward_model_config
        tokenizer_name = reward_model_config.tokenizer_name_or_path or reward_model_config.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=reward_model_config.model_revision,
            trust_remote_code=reward_model_config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "revision": reward_model_config.model_revision,
            "trust_remote_code": reward_model_config.trust_remote_code,
            "torch_dtype": DTYPE_MAP[reward_model_config.dtype],
        }
        if reward_model_config.attn_implementation:
            model_kwargs["attn_implementation"] = reward_model_config.attn_implementation

        self.model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_config.model_name_or_path,
            **model_kwargs,
        )
        self.model.eval()

    def to(self, device: torch.device | str) -> "RewardModelRolloutEngine":
        """Move the local reward model to the active Lightning device."""

        self.model.to(device)
        return self

    def rollout(
        self,
        *,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
    ) -> RolloutResult:
        raise NotImplementedError("RewardModelRolloutEngine is only used for reward scoring, not text generation.")

    def update_policy(self, model: torch.nn.Module) -> None:
        return None

    @torch.no_grad()
    def score(self, samples: list[dict[str, Any]]) -> list[float]:
        if not samples:
            return []

        device = next(self.model.parameters()).device
        outputs: list[float] = []
        batch_size = max(1, int(self.config.batch_size))
        score_field = self.config.score_field

        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            texts = [sample["text"] for sample in chunk]
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            model_outputs = self.model(**tokenized)

            if hasattr(model_outputs, score_field):
                scores = getattr(model_outputs, score_field)
            elif hasattr(model_outputs, "logits"):
                scores = model_outputs.logits
            else:
                raise AttributeError(
                    f"Reward model output has neither '{score_field}' nor 'logits'."
                )

            if scores.ndim > 1:
                if scores.shape[-1] == 1:
                    scores = scores.squeeze(-1)
                else:
                    scores = scores[..., 0]

            if self.config.normalize:
                scores = torch.tanh(scores)
            scores = scores * self.config.scale + self.config.bias
            outputs.extend(float(score) for score in scores.detach().cpu())

        return outputs


def truncate_completions(
    completion_ids: torch.Tensor,
    pad_token_id: int,
    eos_token_id: Optional[int | Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
    """Mask tokens after EOS and extract valid completion token lists."""

    eos_token_ids = _resolve_eos_token_ids(eos_token_id)
    if not eos_token_ids:
        completion_mask = torch.ones_like(completion_ids, dtype=torch.long)
        completion_id_lists = [row.tolist() for row in completion_ids]
        completion_truncated = torch.zeros(completion_ids.size(0), dtype=torch.bool, device=completion_ids.device)
        return completion_ids, completion_mask, completion_truncated, completion_id_lists

    eos_tensor = torch.tensor(eos_token_ids, dtype=completion_ids.dtype, device=completion_ids.device)
    is_eos = torch.isin(completion_ids, eos_tensor)
    eos_idx = torch.full((completion_ids.size(0),), completion_ids.size(1), dtype=torch.long, device=completion_ids.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    token_positions = torch.arange(completion_ids.size(1), device=completion_ids.device).expand(completion_ids.size(0), -1)
    completion_mask = (token_positions <= eos_idx.unsqueeze(1)).long()
    completion_ids = completion_ids.masked_fill(completion_mask == 0, pad_token_id)
    completion_id_lists = [ids[mask.bool()].tolist() for ids, mask in zip(completion_ids, completion_mask, strict=True)]
    completion_truncated = torch.tensor(
        [len(ids) == 0 or ids[-1] not in eos_token_ids for ids in completion_id_lists],
        device=completion_ids.device,
        dtype=torch.bool,
    )
    return completion_ids, completion_mask, completion_truncated, completion_id_lists


def pad_sequences(sequences: list[list[int]], pad_value: int, device: torch.device) -> torch.Tensor:
    """Pad integer token sequences into a dense tensor."""

    max_len = max((len(seq) for seq in sequences), default=0)
    if max_len == 0:
        return torch.empty((len(sequences), 0), dtype=torch.long, device=device)
    return torch.tensor([seq + [pad_value] * (max_len - len(seq)) for seq in sequences], dtype=torch.long, device=device)


def pad_float_sequences(sequences: list[list[float]], max_len: int, device: torch.device) -> torch.Tensor:
    """Pad float sequences into a dense tensor."""

    if max_len == 0:
        return torch.empty((len(sequences), 0), dtype=torch.float32, device=device)
    padded = [seq[:max_len] + [0.0] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.float32, device=device)


def create_rollout_engine(
    *,
    engine_type: str,
    policy_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sampling_config_path: Optional[str] = None,
    generation_batch_size: int = 0,
    reward_model_config: Optional[RewardModelConfig] = None,
    vllm_config: Optional[Any] = None,
    model_name_or_path: Optional[str] = None,
    world_size: int = 1,
    local_rank: int = 0,
    global_rank: int = 0,
    **kwargs: Any,
) -> RolloutEngine:
    """Build the configured rollout engine."""

    if engine_type == "policy":
        return PolicyRolloutEngine(
            policy_model=policy_model,
            tokenizer=tokenizer,
            sampling_config_path=sampling_config_path,
            generation_batch_size=generation_batch_size,
        )
    if engine_type == "vllm":
        from lightning_grpo.models.vllm_rollout_engine import VLLMRolloutEngine

        if vllm_config is None:
            raise ValueError("rollout.engine.vllm config must be set when using the vllm rollout engine")
        if not model_name_or_path:
            raise ValueError("model_name_or_path must be provided for the vllm rollout engine")
        return VLLMRolloutEngine(
            vllm_config=vllm_config,
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            sampling_config_path=sampling_config_path,
            world_size=world_size,
            local_rank=local_rank,
            global_rank=global_rank,
        )
    if engine_type == "reward_model":
        if reward_model_config is None:
            raise ValueError("reward_model rollout requires `reward_model_config`.")
        return RewardModelRolloutEngine(reward_model_config)
    raise ValueError(f"Unsupported rollout engine type: {engine_type}")
