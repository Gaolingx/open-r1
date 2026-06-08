"""Lightweight vLLM HTTP client for server-mode rollout.

This client communicates with a running vLLM server (launched via `vllm serve`
or `trl vllm-serve`) to perform generation and weight updates. It uses NCCL
for efficient weight transfer when available, falling back to HTTP.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def _is_vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


class VLLMClient:
    """HTTP + NCCL client for communicating with a vLLM inference server.

    Supports:
    - Text generation via /generate endpoint
    - Chat generation via /chat endpoint
    - Weight updates via NCCL communicator (fast GPU-to-GPU transfer)
    - Prefix cache reset

    Args:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000").
        group_port: Port for the NCCL weight update group.
        connection_timeout: Max seconds to wait for server to be ready.
    """

    def __init__(
        self,
        base_url: str,
        group_port: int = 51216,
        connection_timeout: float = 240.0,
    ) -> None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.base_url = base_url.rstrip("/")
        self.group_port = group_port
        self.connection_timeout = connection_timeout

        # Set up HTTP session with retries
        self._session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # NCCL communicator for weight updates
        self._communicator = None

        # Wait for server to be ready
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        """Wait for the vLLM server to be ready."""
        start = time.time()
        while time.time() - start < self.connection_timeout:
            try:
                resp = self._session.get(f"{self.base_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    logger.info("vLLM server is ready at %s", self.base_url)
                    return
            except Exception:
                pass
            time.sleep(2.0)
        raise ConnectionError(
            f"vLLM server at {self.base_url} did not become ready within "
            f"{self.connection_timeout}s"
        )

    def init_communicator(self, device: int | torch.device) -> None:
        """Initialize NCCL communicator for weight updates.

        This creates a process group between the training process and the vLLM
        server for efficient GPU-to-GPU weight transfer.
        """
        if not _is_vllm_available():
            logger.warning("vLLM not available, weight updates will use HTTP fallback")
            return

        try:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup

            # Get server's weight update config
            resp = self._session.get(f"{self.base_url}/get_weight_update_group")
            if resp.status_code != 200:
                logger.warning("Server does not support NCCL weight updates, using HTTP")
                return

            server_info = resp.json()
            # Create stateless process group for weight transfer
            pg = StatelessProcessGroup.create(
                host=server_info.get("host", "0.0.0.0"),
                port=self.group_port,
                rank=0,  # Training process is rank 0
                world_size=2,  # Training + vLLM server
            )
            self._communicator = PyNcclCommunicator(
                pg, device=device if isinstance(device, torch.device) else torch.device(f"cuda:{device}")
            )
            logger.info("NCCL communicator initialized for weight updates")
        except Exception as e:
            logger.warning("Failed to init NCCL communicator: %s. Using HTTP fallback.", e)
            self._communicator = None

    def generate(
        self,
        prompts: list[list[int]] | list[str],
        sampling_params: Any = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate completions from token ID prompts or text prompts.

        Args:
            prompts: List of token ID lists or text strings.
            sampling_params: vLLM SamplingParams or dict of params.
            **kwargs: Additional parameters passed to the server.

        Returns:
            Dict with 'prompt_ids', 'completion_ids', 'logprobs' keys.
        """
        # Build request payload
        payload: dict[str, Any] = {}

        if prompts and isinstance(prompts[0], list):
            payload["prompt_token_ids"] = prompts
        else:
            payload["prompts"] = prompts

        if isinstance(sampling_params, dict):
            payload["sampling_params"] = sampling_params
        elif sampling_params is not None:
            # Convert SamplingParams object to dict
            payload["sampling_params"] = {
                "n": getattr(sampling_params, "n", 1),
                "temperature": getattr(sampling_params, "temperature", 1.0),
                "top_p": getattr(sampling_params, "top_p", 1.0),
                "top_k": getattr(sampling_params, "top_k", -1),
                "max_tokens": getattr(sampling_params, "max_tokens", 2048),
                "repetition_penalty": getattr(sampling_params, "repetition_penalty", 1.0),
                "logprobs": getattr(sampling_params, "logprobs", 0),
            }

        payload.update(kwargs)

        resp = self._session.post(
            f"{self.base_url}/generate", json=payload, timeout=300.0
        )
        resp.raise_for_status()
        return resp.json()

    def chat(
        self,
        messages: list[list[dict[str, str]]],
        tools: Optional[list[dict]] = None,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate completions from chat conversations.

        Args:
            messages: List of conversations (each is a list of message dicts).
            tools: Tool schemas for function calling.
            chat_template: Custom chat template.
            chat_template_kwargs: Additional chat template parameters.
            **kwargs: Sampling parameters.

        Returns:
            Dict with 'prompt_ids', 'completion_ids', 'logprobs' keys.
        """
        payload: dict[str, Any] = {
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if chat_template:
            payload["chat_template"] = chat_template
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        payload.update(kwargs)

        resp = self._session.post(
            f"{self.base_url}/chat", json=payload, timeout=300.0
        )
        resp.raise_for_status()
        return resp.json()

    def update_named_param(self, name: str, param: torch.Tensor) -> None:
        """Update a single named parameter in the vLLM model.

        Uses NCCL communicator if available for fast GPU transfer,
        otherwise falls back to HTTP.
        """
        if self._communicator is not None:
            # Fast path: NCCL GPU-to-GPU transfer
            self._communicator.send(param.contiguous())
            # Notify server about the parameter name
            self._session.post(
                f"{self.base_url}/update_weight",
                json={"name": name, "shape": list(param.shape), "dtype": str(param.dtype)},
                timeout=30.0,
            )
        else:
            # Slow path: HTTP transfer (CPU serialization)
            import io
            buffer = io.BytesIO()
            torch.save(param.cpu(), buffer)
            self._session.post(
                f"{self.base_url}/update_weight_http",
                data=buffer.getvalue(),
                headers={"X-Param-Name": name},
                timeout=60.0,
            )

    def reset_prefix_cache(self) -> None:
        """Reset the vLLM prefix cache after weight updates."""
        try:
            self._session.post(f"{self.base_url}/reset_prefix_cache", timeout=10.0)
        except Exception as e:
            logger.warning("Failed to reset prefix cache: %s", e)
