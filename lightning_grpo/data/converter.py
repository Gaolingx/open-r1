"""Dataset row converters for SFT chat-style training data."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal, Optional, TypedDict

from dacite import Config, from_dict

DatasetFormat = Literal["auto", "openai", "sharegpt", "alpaca"]
SFTSample = dict[str, Any]
ConverterFn = Callable[[Mapping[str, Any]], SFTSample]


class AlpacaSample(TypedDict, total=False):
    """Alpaca-style instruction tuning row."""

    system: str
    instruction: str
    input: str
    output: str


class SharegptMessage(TypedDict, total=False):
    """ShareGPT/LLaMA-Factory conversation message."""

    from_: str
    value: Any


class SharegptSample(TypedDict, total=False):
    """ShareGPT/LLaMA-Factory conversation row."""

    conversations: list[SharegptMessage]
    tools: Any


class OpenaiMessage(TypedDict, total=False):
    """OpenAI-compatible chat message."""

    role: str
    content: Any
    name: str
    tool_call_id: str
    tool_calls: Any
    tools: Any


class OpenaiSample(TypedDict, total=False):
    """OpenAI-compatible chat row."""

    messages: list[OpenaiMessage]
    tools: Any


@dataclass
class ToolFunctionSchema:
    """Nested function schema used by chat-template tool definitions."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSchema:
    """OpenAI-compatible tool definition with a nested JSON schema."""

    type: str = "function"
    function: ToolFunctionSchema = field(default_factory=ToolFunctionSchema)


@dataclass
class ToolCallFunctionSchema:
    """Nested function call emitted by assistant messages."""

    name: str = ""
    arguments: Any = field(default_factory=dict)


@dataclass
class ToolCallSchema:
    """OpenAI-compatible assistant tool-call payload."""

    id: str = ""
    type: str = "function"
    function: Optional[ToolCallFunctionSchema] = None
    name: str = ""
    arguments: Any = field(default_factory=dict)


@dataclass
class MessageSchema:
    """Canonical chat message schema used by SFT preprocessing."""

    role: str = ""
    content: Any = ""
    reasoning_content: str = ""
    name: str = ""
    tool_call_id: str = ""
    tools: Optional[list[ToolSchema]] = None
    tool_calls: Optional[list[ToolCallSchema]] = None


DACITE_JSON_CONFIG = Config(cast=[str], strict=False)

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "instruction": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "function_call": "assistant",
    "function": "tool",
    "observation": "tool",
    "tool": "tool",
}


class UnknownDatasetFormatError(ValueError):
    """Raised when an unsupported dataset format is requested."""


def json_loads_if_needed(value: Any) -> Any:
    """Parse JSON-looking strings while preserving ordinary strings."""

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{\"":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def drop_empty_schema_values(value: Any) -> Any:
    """Remove dacite/default placeholders that should not be sent to chat templates."""

    if isinstance(value, dict):
        return {
            key: drop_empty_schema_values(item)
            for key, item in value.items()
            if item is not None and item != "" and item != {} and item != []
        }
    if isinstance(value, list):
        return [drop_empty_schema_values(item) for item in value]
    return value


def _maybe_text_content(value: Any) -> Any:
    """Turn common single text blocks into plain strings for tokenizer templates."""

    value = json_loads_if_needed(value)
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        text_parts = []
        for item in value:
            item_type = item.get("type")
            if item_type not in {None, "text"}:
                return [dict(part) for part in value]
            if "text" in item:
                text_parts.append(str(item["text"]))
            elif "value" in item:
                text_parts.append(str(item["value"]))
            else:
                return [dict(part) for part in value]
        return "".join(text_parts)
    return value


def normalize_role(role: Any) -> str:
    """Normalize known role aliases to OpenAI chat roles."""

    return ROLE_ALIASES.get(str(role or "").lower(), str(role or ""))


def normalize_tool_definition(tool: Any) -> Any:
    """Normalize one nested tool schema dict into a stable OpenAI-compatible structure."""

    tool = json_loads_if_needed(tool)
    if not isinstance(tool, Mapping):
        return tool
    if "function" not in tool and "name" in tool:
        tool = {"type": "function", "function": dict(tool)}
    normalized = from_dict(data_class=ToolSchema, data=dict(tool), config=DACITE_JSON_CONFIG)
    return drop_empty_schema_values(asdict(normalized))


def normalize_tool_call(tool_call: Any) -> Any:
    """Normalize one assistant tool-call payload into OpenAI-compatible structure."""

    tool_call = json_loads_if_needed(tool_call)
    if not isinstance(tool_call, Mapping):
        return tool_call

    normalized_call = dict(tool_call)
    if "function" in normalized_call:
        normalized_call["function"] = json_loads_if_needed(normalized_call["function"])
        if isinstance(normalized_call["function"], Mapping) and "arguments" in normalized_call["function"]:
            function = dict(normalized_call["function"])
            function["arguments"] = json_loads_if_needed(function["arguments"])
            normalized_call["function"] = function
    elif "name" in normalized_call or "arguments" in normalized_call:
        normalized_call["function"] = {
            "name": normalized_call.get("name", ""),
            "arguments": json_loads_if_needed(normalized_call.get("arguments", {})),
        }

    normalized = from_dict(data_class=ToolCallSchema, data=normalized_call, config=DACITE_JSON_CONFIG)
    return drop_empty_schema_values(asdict(normalized))


def normalize_tools(tools: Any) -> Any:
    """Normalize row-level tool definitions."""

    tools = json_loads_if_needed(tools)
    if tools in (None, "", [], {}):
        return None
    if isinstance(tools, list):
        return [normalize_tool_definition(tool) for tool in tools]
    return tools


def _normalize_openai_message(message: Any) -> dict[str, Any]:
    """Normalize one OpenAI/ShareGPT-like message into the canonical message schema."""

    message = json_loads_if_needed(message)
    if not isinstance(message, Mapping):
        message = {"role": "user", "content": message}

    raw = dict(message)
    if "from" in raw and "role" not in raw:
        raw["role"] = raw["from"]
    if "value" in raw and "content" not in raw:
        raw["content"] = raw["value"]

    role = normalize_role(raw.get("role"))
    content = _maybe_text_content(raw.get("content", ""))
    normalized: dict[str, Any] = {"role": role, "content": content}

    for key in ("reasoning_content", "name", "tool_call_id"):
        if raw.get(key) not in (None, ""):
            normalized[key] = raw[key]

    if raw.get("tools") not in (None, "", [], {}):
        normalized["tools"] = normalize_tools(raw["tools"])

    if raw.get("tool_calls") not in (None, "", [], {}):
        tool_calls = json_loads_if_needed(raw["tool_calls"])
        normalized["tool_calls"] = (
            [normalize_tool_call(tool_call) for tool_call in tool_calls]
            if isinstance(tool_calls, list)
            else tool_calls
        )

    schema = from_dict(data_class=MessageSchema, data=normalized, config=DACITE_JSON_CONFIG)
    known = drop_empty_schema_values(asdict(schema))
    extras = {
        key: json_loads_if_needed(value)
        for key, value in raw.items()
        if key not in MessageSchema.__dataclass_fields__ and key not in {"from", "value"}
    }
    return {**extras, **known}


def _attach_tools(messages: list[dict[str, Any]], tools: Any) -> list[dict[str, Any]]:
    """Attach row-level tools to the first system message for tokenizer chat templates."""

    tools = normalize_tools(tools)
    if tools is None:
        return messages
    normalized = [dict(message) for message in messages]
    for message in normalized:
        if message.get("role") == "system":
            message.setdefault("tools", tools)
            return normalized
    return [{"role": "system", "content": "", "tools": tools}, *normalized]


def openai_converter(raw_sample: Mapping[str, Any]) -> SFTSample:
    """Convert an OpenAI-compatible sample to canonical SFT messages."""

    messages = json_loads_if_needed(raw_sample.get("messages", []))
    if not isinstance(messages, list):
        messages = [messages]
    normalized_messages = [_normalize_openai_message(message) for message in messages]
    tools = normalize_tools(raw_sample.get("tools"))
    return {"messages": _attach_tools(normalized_messages, tools), "tools": tools}


def _sharegpt_tool_call_message(value: Any) -> dict[str, Any] | None:
    """Convert a ShareGPT function_call payload into an assistant tool-call message."""

    tool_calls = json_loads_if_needed(value)
    if tool_calls in (None, ""):
        return None
    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [normalize_tool_call(tool_call) for tool_call in tool_calls],
    }


def sharegpt_converter(raw_sample: Mapping[str, Any]) -> SFTSample:
    """Convert ShareGPT/LLaMA-Factory rows to canonical SFT messages."""

    raw_messages = json_loads_if_needed(raw_sample.get("conversations", []))
    if not isinstance(raw_messages, list):
        raw_messages = [raw_messages]

    messages: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        raw_message = json_loads_if_needed(raw_message)
        if not isinstance(raw_message, Mapping):
            messages.append({"role": "user", "content": raw_message})
            continue

        role = normalize_role(raw_message.get("from", raw_message.get("role")))
        value = raw_message.get("value", raw_message.get("content", ""))
        if str(raw_message.get("from", raw_message.get("role", ""))).lower() == "function_call":
            tool_call_message = _sharegpt_tool_call_message(value)
            if tool_call_message is not None:
                messages.append(tool_call_message)
            continue

        message = dict(raw_message)
        message["role"] = role
        message["content"] = value
        messages.append(_normalize_openai_message(message))

    tools = normalize_tools(raw_sample.get("tools"))
    return {"messages": _attach_tools(messages, tools), "tools": tools}


def alpaca_converter(raw_sample: Mapping[str, Any]) -> SFTSample:
    """Convert Alpaca instruction rows to canonical SFT messages."""

    messages: list[dict[str, Any]] = []
    system = raw_sample.get("system")
    if system not in (None, ""):
        messages.append({"role": "system", "content": _maybe_text_content(system)})

    instruction = str(raw_sample.get("instruction", "") or "")
    input_text = str(raw_sample.get("input", "") or "")
    prompt = instruction if not input_text else f"{instruction}\n{input_text}" if instruction else input_text
    if prompt:
        messages.append({"role": "user", "content": prompt})

    output = raw_sample.get("output", raw_sample.get("response"))
    if output not in (None, ""):
        messages.append({"role": "assistant", "content": _maybe_text_content(output)})

    tools = normalize_tools(raw_sample.get("tools"))
    return {"messages": _attach_tools(messages, tools), "tools": tools}


def _fallback_prompt_response_converter(raw_sample: Mapping[str, Any]) -> SFTSample:
    """Convert generic prompt/response rows to canonical SFT messages."""

    messages: list[dict[str, Any]] = []
    system = raw_sample.get("system")
    if system not in (None, ""):
        messages.append({"role": "system", "content": _maybe_text_content(system)})
    prompt = raw_sample.get("prompt", raw_sample.get("question", raw_sample.get("problem")))
    if prompt is not None:
        messages.append({"role": "user", "content": _maybe_text_content(prompt)})
    response = raw_sample.get("response", raw_sample.get("answer", raw_sample.get("solution")))
    if response is not None:
        messages.append({"role": "assistant", "content": _maybe_text_content(response)})
    tools = normalize_tools(raw_sample.get("tools"))
    return {"messages": _attach_tools(messages, tools), "tools": tools}


DATA_CONVERTERS: dict[str, ConverterFn] = {
    "openai": openai_converter,
    "sharegpt": sharegpt_converter,
    "alpaca": alpaca_converter,
}


def detect_dataset_format(raw_sample: Mapping[str, Any]) -> DatasetFormat:
    """Detect the best converter for one dataset row."""

    if raw_sample.get("messages") is not None:
        return "openai"
    if raw_sample.get("conversations") is not None:
        return "sharegpt"
    if raw_sample.get("instruction") is not None or raw_sample.get("output") is not None:
        return "alpaca"
    return "auto"


def get_data_converter(format_name: DatasetFormat | str) -> ConverterFn:
    """Return a converter by format name."""

    if format_name == "auto":
        return convert_auto_sample
    try:
        return DATA_CONVERTERS[str(format_name)]
    except KeyError as exc:
        raise UnknownDatasetFormatError(f"Unsupported dataset_format: {format_name}") from exc


def convert_auto_sample(raw_sample: Mapping[str, Any]) -> SFTSample:
    """Automatically convert OpenAI, ShareGPT, Alpaca, or prompt/response rows."""

    detected = detect_dataset_format(raw_sample)
    if detected == "auto":
        return _fallback_prompt_response_converter(raw_sample)
    return get_data_converter(detected)(raw_sample)


def convert_sft_sample(raw_sample: Mapping[str, Any], dataset_format: DatasetFormat | str = "auto") -> SFTSample:
    """Convert one raw dataset row into canonical SFT preprocessing input."""

    converter = get_data_converter(dataset_format)
    return converter(raw_sample)
