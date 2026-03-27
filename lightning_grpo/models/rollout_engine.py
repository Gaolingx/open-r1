"""Pluggable rollout backends for Lightning GRPO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def compute_per_token_logps(
        model: torch.nn.Module,
        input_ids: Tensor,
        n_keep: int,
        *,
        attention_mask: Optional[Tensor] = None,
        temperature: float = 1.0,
) -> Tensor:
    """Compute per-token log-probabilities for the sampled completion suffix."""

    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    outputs = unwrapped(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    logits = logits[:, -n_keep:, :] / temperature
    target_ids = input_ids[:, -n_keep:]
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


@dataclass
class RolloutResult:
    """Structured rollout outputs shared by all rollout backends."""

    output_ids: Tensor
    prompt_ids: Tensor
    prompt_mask: Tensor
    completion_ids: Tensor
    completion_mask: Tensor
    per_token_logps: Tensor
    completions_text: list[str]
    completion_id_lists: list[list[int]]
    completion_truncated: Tensor


class RolloutEngine(ABC):
    """Abstract rollout backend."""

    tokenizer: PreTrainedTokenizerBase

    @abstractmethod
    def rollout(
            self,
            *,
            prompt_ids: Tensor,
            attention_mask: Tensor,
            num_generations: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
    ) -> RolloutResult:
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, model: torch.nn.Module) -> None:
        raise NotImplementedError


class PolicyRolloutEngine(RolloutEngine):
    """In-process rollout using the current policy model."""

    def __init__(
            self,
            policy_model: torch.nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            temperature: float,
            generation_batch_size: int = 0,
    ) -> None:
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.generation_batch_size = max(0, int(generation_batch_size))

    def rollout(
            self,
            *,
            prompt_ids: Tensor,
            attention_mask: Tensor,
            num_generations: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
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
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
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
        completions_text = self.tokenizer.batch_decode(completion_id_lists, skip_special_tokens=True)
        model_input_ids = torch.cat([repeated_prompt_ids, completion_ids], dim=1)
        model_attention_mask = torch.cat([repeated_prompt_mask, completion_mask], dim=1)
        per_token_logps = compute_per_token_logps(
            self.policy_model,
            model_input_ids,
            completion_ids.shape[1],
            attention_mask=model_attention_mask,
            temperature=self.temperature,
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

    def _generate_in_chunks(
            self,
            *,
            model: torch.nn.Module,
            prompt_ids: Tensor,
            attention_mask: Tensor,
            num_generations: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
    ) -> Tensor:
        """Generate completions in prompt chunks to cap rollout memory usage."""

        chunk_size = self.generation_batch_size or prompt_ids.size(0)
        if chunk_size <= 0:
            chunk_size = prompt_ids.size(0)

        generated_chunks: list[Tensor] = []
        for start in range(0, prompt_ids.size(0), chunk_size):
            end = min(start + chunk_size, prompt_ids.size(0))
            generated_chunk = model.generate(
                input_ids=prompt_ids[start:end],
                attention_mask=attention_mask[start:end],
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            generated_chunks.append(generated_chunk)

        if not generated_chunks:
            return prompt_ids.new_empty((0, prompt_ids.size(1)), dtype=prompt_ids.dtype)
        return torch.cat(generated_chunks, dim=0)


class SGLangRolloutEngine(RolloutEngine):
    """HTTP rollout backend backed by an SGLang server."""

    def __init__(
            self,
            *,
            base_url: str,
            model_path: str,
            shared_ckpt_path: str,
            timeout: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests

    def rollout(
            self,
            *,
            prompt_ids: Tensor,
            attention_mask: Tensor,
            num_generations: int,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
    ) -> RolloutResult:
        input_ids_list = [ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, attention_mask, strict=True)]
        repeated_prompt_ids_list = [ids for ids in input_ids_list for _ in range(num_generations)]
        payload = {
            "input_ids": repeated_prompt_ids_list,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else [],
            },
            "return_logprob": True,
        }
        response = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        results = response.json()
        if not isinstance(results, list):
            results = [results]

        completions, logprobs = [], []
        for result in results:
            meta = result.get("meta_info", {})
            completion = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            parsed_logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and item:
                    parsed_logprobs.append(float(item[0]))
                elif isinstance(item, (int, float)):
                    parsed_logprobs.append(float(item))
            completions.append(completion)
            logprobs.append(parsed_logprobs)

        repeated_prompt_ids = pad_sequences(repeated_prompt_ids_list, pad_value=self.tokenizer.pad_token_id, device=prompt_ids.device)
        repeated_prompt_mask = (repeated_prompt_ids != self.tokenizer.pad_token_id).long()
        completion_ids = pad_sequences(completions, pad_value=self.tokenizer.pad_token_id, device=prompt_ids.device)
        completion_ids, completion_mask, completion_truncated, completion_id_lists = truncate_completions(
            completion_ids,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )
        per_token_logps = pad_float_sequences(logprobs, completion_ids.shape[1], device=prompt_ids.device)
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

    def update_policy(self, model: torch.nn.Module) -> None:
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        target_path = Path(self.shared_ckpt_path).resolve()
        target_path.mkdir(parents=True, exist_ok=True)
        state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(str(target_path), state_dict=state_dict, safe_serialization=False)
        self.tokenizer.save_pretrained(str(target_path))
        response = self.http.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": str(target_path)},
            timeout=self.timeout,
        )
        response.raise_for_status()


def truncate_completions(
        completion_ids: Tensor,
        pad_token_id: int,
        eos_token_id: Optional[int],
) -> tuple[Tensor, Tensor, Tensor, list[list[int]]]:
    """Mask tokens after EOS and extract valid completion token lists."""

    if eos_token_id is None:
        completion_mask = torch.ones_like(completion_ids, dtype=torch.long)
        completion_id_lists = [row.tolist() for row in completion_ids]
        completion_truncated = torch.zeros(completion_ids.size(0), dtype=torch.bool, device=completion_ids.device)
        return completion_ids, completion_mask, completion_truncated, completion_id_lists

    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((completion_ids.size(0),), completion_ids.size(1), dtype=torch.long, device=completion_ids.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    token_positions = torch.arange(completion_ids.size(1), device=completion_ids.device).expand(completion_ids.size(0), -1)
    completion_mask = (token_positions <= eos_idx.unsqueeze(1)).long()
    completion_ids = completion_ids.masked_fill(completion_mask == 0, pad_token_id)
    completion_id_lists = [ids[mask.bool()].tolist() for ids, mask in zip(completion_ids, completion_mask, strict=True)]
    completion_truncated = torch.tensor(
        [len(ids) == 0 or ids[-1] != eos_token_id for ids in completion_id_lists],
        device=completion_ids.device,
        dtype=torch.bool,
    )
    return completion_ids, completion_mask, completion_truncated, completion_id_lists


def pad_sequences(sequences: list[list[int]], pad_value: int, device: torch.device) -> Tensor:
    """Pad integer token sequences into a dense tensor."""

    max_len = max((len(seq) for seq in sequences), default=0)
    if max_len == 0:
        return torch.empty((len(sequences), 0), dtype=torch.long, device=device)
    return torch.tensor([seq + [pad_value] * (max_len - len(seq)) for seq in sequences], dtype=torch.long, device=device)


def pad_float_sequences(sequences: list[list[float]], max_len: int, device: torch.device) -> Tensor:
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
        temperature: float,
        generation_batch_size: int = 0,
        sglang_base_url: Optional[str] = None,
        sglang_model_path: Optional[str] = None,
        sglang_shared_path: Optional[str] = None,
        request_timeout: int = 120,
) -> RolloutEngine:
    """Build the configured rollout engine."""

    if engine_type == "policy":
        return PolicyRolloutEngine(
            policy_model=policy_model,
            tokenizer=tokenizer,
            temperature=temperature,
            generation_batch_size=generation_batch_size,
        )
    if engine_type == "sglang":
        if not sglang_model_path:
            raise ValueError("rollout.engine.sglang_model_path must be set when using the sglang rollout engine")
        return SGLangRolloutEngine(
            base_url=sglang_base_url or "http://localhost:8996",
            model_path=sglang_model_path,
            shared_ckpt_path=sglang_shared_path or "./sglang_ckpt_grpo",
            timeout=request_timeout,
        )
    raise ValueError(f"Unsupported rollout engine type: {engine_type}")
