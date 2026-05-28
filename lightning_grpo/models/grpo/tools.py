"""Tool-call parsing and local mock tool execution for agentic GRPO rollouts."""

from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse ``<tool_call>{...}</tool_call>`` blocks from model text."""

    calls: list[dict[str, Any]] = []
    for match in re.findall(r"<tool_call>(.*?)</tool_call>", str(text), re.DOTALL):
        try:
            payload = json.loads(match.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            calls.append(payload)
    return calls


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval_math(expression: str) -> float:
    """Evaluate a small arithmetic expression without exposing Python builtins."""

    normalized = expression.replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    normalized = normalized.replace("（", "(").replace("）", ")")
    tree = ast.parse(normalized, mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
            return _ALLOWED_BIN_OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
            return _ALLOWED_UNARY_OPS[type(node.op)](visit(node.operand))
        raise ValueError("unsupported expression")

    return visit(tree)


WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("USD", "EUR"): 0.92}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592}


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Execute built-in mock tools for local agentic RL rollouts."""

    try:
        if name == "calculate_math":
            return {"result": str(_safe_eval_math(str(args.get("expression", "0"))))}
        if name == "unit_converter":
            key = f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}"
            return {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(key, 1), 4)}
        if name == "get_current_weather":
            weather = WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))
            return {"city": args.get("location"), "temperature": weather[0], "humidity": "65%", "condition": weather[1]}
        if name == "get_current_time":
            timezone = args.get("timezone", "Asia/Shanghai")
            return {"datetime": TIME_DATA.get(timezone, TIME_DATA["Asia/Shanghai"]), "timezone": timezone}
        if name == "get_exchange_rate":
            pair = (args.get("from_currency"), args.get("to_currency"))
            return {"from": pair[0], "to": pair[1], "rate": EXCHANGE_DATA.get(pair, 1.0)}
        if name == "translate_text":
            key = (args.get("text"), args.get("target_language"))
            return {"translated_text": TRANSLATE_DATA.get(key, args.get("text", ""))}
    except Exception:
        return None
    return None
