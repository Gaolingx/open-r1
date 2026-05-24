"""Stream-clean JSONL chat datasets to make conversation message schemas uniform.

This script preserves row order and writes exactly one output JSON object per input line.
It normalizes sparse per-message fields under `messages` or `conversations` so Hugging Face
`datasets` can infer a stable Arrow schema.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

MESSAGE_KEYS = (
    "role",
    "content",
    "reasoning_content",
    "tools",
    "tool_calls",
    "name",
    "tool_call_id",
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize sparse message fields in a JSONL chat dataset without changing row order."
    )
    parser.add_argument("input", type=Path, help="Input .jsonl file path.")
    parser.add_argument("output", type=Path, help="Output .jsonl file path.")
    parser.add_argument(
        "--source-column",
        choices=("auto", "messages", "conversations"),
        default="auto",
        help="Column containing chat messages. With auto, messages is preferred, then conversations.",
    )
    parser.add_argument(
        "--target-column",
        choices=("same", "messages", "conversations"),
        default="conversations",
        help="Column to write normalized messages to. Default matches the requested examples: conversations.",
    )
    parser.add_argument(
        "--drop-source-column",
        action="store_true",
        help="When target column differs from source column, remove the original source column.",
    )
    parser.add_argument(
        "--empty-sparse-as-empty-string",
        action="store_true",
        help="Use empty strings instead of null for missing sparse fields.",
    )
    parser.add_argument(
        "--ensure-all-message-keys",
        action="store_true",
        help="Add every known message key to every message. By default only role/content plus observed sparse keys are added.",
    )
    parser.add_argument(
        "--keep-extra-message-keys",
        action="store_true",
        help="Keep message keys outside the normalized key set.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON without spaces.",
    )
    return parser.parse_args()


def loads_json_if_needed(value: Any) -> Any:
    """Parse JSON-like strings while preserving ordinary text."""

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{\"":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def dumps_sparse_value(value: Any) -> Any:
    """Store nested sparse fields as stable JSON strings for Arrow compatibility."""

    value = loads_json_if_needed(value)
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def normalize_role(value: Any) -> str:
    role = str(value or "")
    return ROLE_ALIASES.get(role.lower(), role)


def message_from_value(raw_message: Any) -> dict[str, Any]:
    """Convert OpenAI or ShareGPT-like message payloads into one normalized dict."""

    raw_message = loads_json_if_needed(raw_message)
    if not isinstance(raw_message, Mapping):
        return {"role": "user", "content": "" if raw_message is None else str(raw_message)}

    message = dict(raw_message)
    if "from" in message and "role" not in message:
        message["role"] = message["from"]
    if "value" in message and "content" not in message:
        message["content"] = message["value"]

    role = normalize_role(message.get("role"))
    content = loads_json_if_needed(message.get("content", ""))
    if content is None:
        content = ""
    elif isinstance(content, (dict, list)):
        content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    else:
        content = str(content)

    normalized: dict[str, Any] = {
        "role": role,
        "content": content,
    }

    if message.get("reasoning_content") not in (None, ""):
        normalized["reasoning_content"] = str(message["reasoning_content"])
    if message.get("tools") not in (None, "", [], {}):
        normalized["tools"] = dumps_sparse_value(message["tools"])
    if message.get("tool_calls") not in (None, "", [], {}):
        normalized["tool_calls"] = dumps_sparse_value(message["tool_calls"])
    if message.get("name") not in (None, ""):
        normalized["name"] = str(message["name"])
    if message.get("tool_call_id") not in (None, ""):
        normalized["tool_call_id"] = str(message["tool_call_id"])

    return normalized


def resolve_source_column(row: Mapping[str, Any], source_column: str) -> str | None:
    if source_column != "auto":
        return source_column if source_column in row else None
    if "messages" in row:
        return "messages"
    if "conversations" in row:
        return "conversations"
    return None


def normalize_messages(
    raw_messages: Any,
    *,
    empty_sparse_value: Any,
    ensure_all_message_keys: bool,
    keep_extra_message_keys: bool,
) -> list[dict[str, Any]]:
    raw_messages = loads_json_if_needed(raw_messages)
    if not isinstance(raw_messages, list):
        raw_messages = [] if raw_messages in (None, "") else [raw_messages]

    converted = [message_from_value(message) for message in raw_messages]
    observed_sparse_keys = {
        key
        for message in converted
        for key, value in message.items()
        if key not in {"role", "content"} and value not in (None, "")
    }
    keys_to_emit = list(MESSAGE_KEYS) if ensure_all_message_keys else ["role", "content", *[k for k in MESSAGE_KEYS if k in observed_sparse_keys]]

    normalized_messages: list[dict[str, Any]] = []
    for original, message in zip(raw_messages, converted, strict=True):
        normalized = {key: message.get(key, empty_sparse_value if key not in {"role", "content"} else "") for key in keys_to_emit}
        normalized["role"] = message.get("role", "")
        normalized["content"] = message.get("content", "")
        if keep_extra_message_keys and isinstance(original, Mapping):
            for key, value in original.items():
                if key not in normalized and key not in {"from", "value"}:
                    normalized[key] = loads_json_if_needed(value)
        normalized_messages.append(normalized)
    return normalized_messages


def clean_stream(input_file: TextIO, output_file: TextIO, args: argparse.Namespace) -> tuple[int, int]:
    total = 0
    missing_message_rows = 0
    empty_sparse_value = "" if args.empty_sparse_as_empty_string else None
    json_kwargs = {"ensure_ascii": False}
    if args.compact:
        json_kwargs["separators"] = (",", ":")

    for line_number, line in enumerate(input_file, start=1):
        if not line.strip():
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} must be a JSON object, got {type(row).__name__}.")

        source_column = resolve_source_column(row, args.source_column)
        if source_column is None:
            missing_message_rows += 1
            output_file.write(json.dumps(row, **json_kwargs) + "\n")
            continue

        target_column = source_column if args.target_column == "same" else args.target_column
        cleaned_row = dict(row)
        cleaned_row[target_column] = normalize_messages(
            row[source_column],
            empty_sparse_value=empty_sparse_value,
            ensure_all_message_keys=args.ensure_all_message_keys,
            keep_extra_message_keys=args.keep_extra_message_keys,
        )
        if args.drop_source_column and target_column != source_column:
            cleaned_row.pop(source_column, None)

        output_file.write(json.dumps(cleaned_row, **json_kwargs) + "\n")

    return total, missing_message_rows


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must be different to avoid corrupting the dataset.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as input_file, args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        total, missing_message_rows = clean_stream(input_file, output_file, args)

    print(f"Wrote {total} rows to {args.output}")
    if missing_message_rows:
        print(f"Warning: {missing_message_rows} rows had no messages/conversations column and were copied unchanged.")


if __name__ == "__main__":
    main()
