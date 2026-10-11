from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
import socket
from typing import List, Optional, Sequence

import torch
from torch import Tensor
from transformers import GenerationConfig


logger = logging.getLogger(__name__)


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_free_port() -> int:
    candidates = (29500, 23456, 12355, 12345)
    for p in candidates:
        if _is_port_free(p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def ensure_master_addr_port(addr: str | None = None, port: int | None = None) -> None:
    """
    Ensure `MASTER_ADDR`/`MASTER_PORT` are set safely.

    - Respects existing environment variables.
    - Defaults `MASTER_ADDR` to localhost if unset.
    - Chooses a free TCP port if `MASTER_PORT` is unset to avoid collisions.
    - If `MASTER_PORT` is set to `"0"` or `"auto"`, it is resolved to a free port.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR") or addr or "localhost"

    env_port = os.environ.get("MASTER_PORT", "").strip().lower()
    if port is None and env_port not in {"", "0", "auto"}:
        try:
            port = int(env_port)
        except ValueError:
            pass

    os.environ["MASTER_PORT"] = str(_find_free_port() if port in (None, 0) else port)


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
    tokenizer = None

    @abstractmethod
    def generate(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8, top_p: float = 1.0, *, greedy: bool = False) -> RolloutResult:
        """Sample ``num_generations`` completions per prompt.

        ``greedy=True`` switches to deterministic decoding, which is what the ReMax reward
        baseline needs.
        """
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass

    def shutdown(self) -> None:
        """Release engine-owned resources. Engines without long-lived resources need not override this."""


def load_generation_config(sampling_config_path: Optional[str], fallback: GenerationConfig) -> GenerationConfig:
    """Load rollout sampling config, falling back to the model generation config."""

    if sampling_config_path:
        return GenerationConfig.from_pretrained(pretrained_model_name=sampling_config_path)
    return GenerationConfig.from_dict(fallback.to_dict())


def detect_fsdp_version(model: torch.nn.Module) -> Optional[int]:
    """Return the FSDP version sharding ``model``, or ``None`` when it is not sharded at all."""

    if model is None:
        return None
    for module in model.modules():
        name = module.__class__.__name__
        if name == "FullyShardedDataParallel":
            return 1
        if name == "FSDPModule":
            return 2
    return 2 if any(hasattr(param, "full_tensor") for param in model.state_dict().values()) else None


def sampled_token_logprobs(
    logprobs: list | None,
    completion_ids: list[list[int]],
    logprob_token_ids: list | None,
) -> list[list[float]] | None:
    if logprobs is None:
        return None
    if logprob_token_ids is None:
        raise ValueError("vLLM logprobs were returned without logprob_token_ids; cannot align sampled token logprobs.")
    sampled_logprobs: list[list[float]] = []
    for sequence_ids, sequence_logprobs, sequence_logprob_token_ids in zip(completion_ids, logprobs, logprob_token_ids, strict=True):
        sequence_sampled_logprobs: list[float] = []
        for token_id, position_logprobs, position_token_ids in zip(sequence_ids, sequence_logprobs, sequence_logprob_token_ids, strict=True):
            try:
                token_index = position_token_ids.index(token_id)
            except ValueError as exc:
                raise ValueError(f"Sampled token id {token_id} is missing from vLLM logprob_token_ids {position_token_ids}.") from exc
            value = position_logprobs[token_index]
            sequence_sampled_logprobs.append(float(value) if value is not None else 0.0)
        sampled_logprobs.append(sequence_sampled_logprobs)
    return sampled_logprobs


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


def _resolve_eos_token_ids(eos_token_id: Optional[int | Sequence[int]]) -> list[int]:
    """Normalize single or multiple EOS token ids into a list."""

    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    return [int(token_id) for token_id in eos_token_id]

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
