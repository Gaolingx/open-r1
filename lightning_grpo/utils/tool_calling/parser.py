"""Shared tool-calling primitives for data preprocessing and online rollout."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import threading
from collections.abc import Callable
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _start_async_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start an event loop on a daemon thread for async tool execution."""
    loop = asyncio.new_event_loop()

    def _run(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run, args=(loop,), daemon=True)
    thread.start()
    return loop, thread


def _shutdown_async_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    """Gracefully shut down the async event loop."""
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


def load_tools_from_names(tool_names: list[str]) -> list[Callable]:
    """Load tool callables from dotted module paths."""
    tools: list[Callable] = []
    for name in tool_names:
        if "." not in name:
            raise ValueError(f"Tool '{name}' not found. Use a dotted import path like 'module.function'.")

        module_path, func_name = name.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        if not callable(func):
            raise ValueError(f"Tool '{name}' is not callable.")
        tools.append(func)
    return tools


class ToolSchemaBuilder:
    """Build OpenAI-compatible JSON tool schemas from Python callables."""

    @staticmethod
    def build_tool_schemas(tools: list[Callable]) -> list[dict[str, Any]]:
        """Build JSON tool schemas from function signatures and docstrings."""
        schemas = []
        for tool in tools:
            sig = inspect.signature(tool)
            params = {}
            for param_name, param in sig.parameters.items():
                param_info: dict[str, Any] = {"type": "string"}
                if param.annotation != inspect.Parameter.empty:
                    type_map = {
                        str: "string",
                        int: "integer",
                        float: "number",
                        bool: "boolean",
                        list: "array",
                        dict: "object",
                    }
                    param_info["type"] = type_map.get(param.annotation, "string")
                params[param_name] = param_info

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "description": (tool.__doc__ or "").strip().split("\n")[0],
                        "parameters": {
                            "type": "object",
                            "properties": params,
                            "required": list(params.keys()),
                        },
                    },
                }
            )
        return schemas


class ToolCallParser:
    """Parse and normalize tool calls shared by SFT preprocessing and GRPO rollout."""

    @staticmethod
    def normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
        """Normalize a parsed tool-call payload to OpenAI-compatible shape."""
        if isinstance(tool_call, str):
            try:
                tool_call = json.loads(tool_call)
            except json.JSONDecodeError:
                return None
        if not isinstance(tool_call, dict):
            return None

        if "function" in tool_call:
            function = tool_call.get("function") or {}
            if isinstance(function, str):
                try:
                    function = json.loads(function)
                except json.JSONDecodeError:
                    return None
            if not isinstance(function, dict):
                return None
            name = function.get("name") or tool_call.get("name")
            arguments = function.get("arguments", tool_call.get("arguments", {}))
        else:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

        if not name:
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass

        return {
            "id": str(tool_call.get("id", "")),
            "type": tool_call.get("type", "function"),
            "function": {"name": str(name), "arguments": arguments},
        }

    def parse_assistant_message(self, text: str) -> dict[str, Any]:
        """Parse generated assistant text and recover tool calls when present."""
        content = text or ""
        tool_calls: list[dict[str, Any]] = []

        matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL))
        for match in matches:
            parsed = self.normalize_tool_call(match.group(1).strip())
            if parsed is not None:
                tool_calls.append(parsed)
        if matches:
            content = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", content, flags=re.DOTALL).strip()

        if not tool_calls:
            stripped = content.strip()
            if stripped:
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    raw_calls = payload.get("tool_calls")
                    if raw_calls is None and ("name" in payload or "function" in payload):
                        raw_calls = [payload]
                    if isinstance(raw_calls, dict):
                        raw_calls = [raw_calls]
                    if isinstance(raw_calls, list):
                        for raw_call in raw_calls:
                            parsed = self.normalize_tool_call(raw_call)
                            if parsed is not None:
                                tool_calls.append(parsed)
                        if tool_calls:
                            content = str(payload.get("content", ""))

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def extract_tool_calls(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Return structured tool calls, parsing message content lazily if needed."""
        calls = message.get("tool_calls") or []
        if calls:
            normalized = [call for call in (self.normalize_tool_call(call) for call in calls) if call is not None]
            message["tool_calls"] = normalized
            return normalized

        parsed = self.parse_assistant_message(str(message.get("content", "")))
        parsed_calls = parsed.get("tool_calls") or []
        if parsed_calls:
            message["content"] = parsed.get("content", "")
            message["tool_calls"] = parsed_calls
        return parsed_calls


class ToolCallRunner(ToolCallParser):
    """Execute synchronous and asynchronous tool calls."""

    def __init__(self, tools: list[Callable]) -> None:
        self._sync_tools: dict[str, Callable] = {}
        self._async_tools: dict[str, Callable] = {}
        for tool in tools:
            name = tool.__name__
            if asyncio.iscoroutinefunction(tool):
                self._async_tools[name] = tool
            else:
                self._sync_tools[name] = tool

        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None
        if self._async_tools:
            self._async_loop, self._async_thread = _start_async_loop()

        self.tool_schemas = ToolSchemaBuilder.build_tool_schemas(tools)

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> tuple[str, str]:
        """Execute a single tool call and return (name, result_str)."""
        if tool_call.get("type") != "function":
            name = tool_call.get("function", {}).get("name", "unknown")
            return name, json.dumps({"error": f"Unsupported tool call type: {tool_call.get('type')}"})

        function = tool_call["function"]
        name = function["name"]
        arguments = function.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return name, json.dumps({"error": f"Invalid JSON arguments: {arguments}"})

        try:
            if name in self._sync_tools:
                result = self._sync_tools[name](**arguments)
            elif name in self._async_tools:
                if self._async_loop is None:
                    return name, json.dumps({"error": "Async loop not initialized"})
                future = asyncio.run_coroutine_threadsafe(self._async_tools[name](**arguments), self._async_loop)
                result = future.result(timeout=60.0)
            else:
                return name, json.dumps({"error": f"Tool '{name}' not found"})
        except Exception as e:
            logger.warning("Tool '%s' execution failed: %s", name, e)
            return name, json.dumps({"error": str(e)})

        return name, str(result)

    def execute_tool_calls_batch(self, tool_calls_batch: list[list[dict[str, Any]]]) -> list[list[dict[str, str]]]:
        """Execute tool calls for a batch of completions."""
        results: list[list[dict[str, str]]] = []
        for tool_calls in tool_calls_batch:
            messages = []
            async_coros = []
            sync_results = []

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name", "unknown")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        sync_results.append((name, json.dumps({"error": "Invalid JSON"})))
                        continue

                if name in self._async_tools and self._async_loop is not None:
                    async_coros.append((name, self._async_tools[name](**arguments)))
                elif name in self._sync_tools:
                    try:
                        result = self._sync_tools[name](**arguments)
                        sync_results.append((name, str(result)))
                    except Exception as e:
                        sync_results.append((name, json.dumps({"error": str(e)})))
                else:
                    sync_results.append((name, json.dumps({"error": f"Tool '{name}' not found"})))

            if async_coros and self._async_loop is not None:

                async def _gather(coros):
                    tasks = [coro for _, coro in coros]
                    return await asyncio.gather(*tasks, return_exceptions=True)

                future = asyncio.run_coroutine_threadsafe(_gather(async_coros), self._async_loop)
                async_results = future.result(timeout=120.0)
                for (name, _), result in zip(async_coros, async_results):
                    if isinstance(result, Exception):
                        sync_results.append((name, json.dumps({"error": str(result)})))
                    else:
                        sync_results.append((name, str(result)))

            for name, content in sync_results:
                messages.append({"role": "tool", "name": name, "content": content})
            results.append(messages)

        return results

    def shutdown(self) -> None:
        """Clean up async resources."""
        if self._async_loop is not None and self._async_thread is not None:
            _shutdown_async_loop(self._async_loop, self._async_thread)
            self._async_loop = None
            self._async_thread = None
