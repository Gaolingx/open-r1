"""Reward management helpers for GRPO training."""

from __future__ import annotations

from typing import Any

import torch

import ast
import json
import operator
import re


def repetition_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    """Return a bounded n-gram repetition penalty for generated text."""

    tokens = re.findall(r"\w+|[^\w\s]", str(text).lower())
    grams = [tuple(tokens[index: index + n]) for index in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


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


def validate_gt_in_text(text: str, gt_list: Any) -> set[Any]:
    """Return ground-truth values that are present as text or exact numeric matches."""

    if gt_list is None:
        return set()
    if not isinstance(gt_list, list):
        gt_list = [gt_list]
    text = str(text)
    text_num = text.replace(",", "")
    nums = [float(value) for value in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])", text_num)]
    verified = set()
    for gt in gt_list:
        raw = str(gt).strip()
        if not raw:
            continue
        numeric = raw.replace(",", "")
        if raw.lower() in text.lower():
            verified.add(gt)
        elif re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric):
            expected = float(numeric)
            if any(abs(expected - number) < 1.0e-6 for number in nums):
                verified.add(gt)
    return verified

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

CHECK_ARGS = {
    "calculate_math": lambda args: bool(args.get("expression")),
    "unit_converter": lambda args: args.get("value") is not None and args.get("from_unit") and args.get("to_unit"),
    "get_current_weather": lambda args: bool(args.get("location")),
    "get_current_time": lambda args: True,
    "get_exchange_rate": lambda args: bool(args.get("from_currency")) and bool(args.get("to_currency")),
    "translate_text": lambda args: bool(args.get("text")) and bool(args.get("target_language")),
}


class GRPORewardManager:
    """Build and evaluate reward functions for GRPO rollouts."""

    def __init__(self, config: Any, tokenizer: Any, rollout_engine: Any = None, reward_model_engine: Any = None, device: torch.device | None = None) -> None:
        self.config = config
        self.reward_config = config.reward.active
        self.tokenizer = tokenizer
        self.rollout_engine = rollout_engine
        self.reward_model_engine = reward_model_engine
        self.device = device

    def _format_reward(self, response: str, final_answer: str) -> float:
        """Reward length, thinking-tag structure, and discourage repetition."""

        cfg = self.config.reward
        reward = 0.5 if cfg.min_response_chars <= len(response.strip()) <= cfg.max_response_chars else -0.5
        if "</think>" in response:
            think, answer = response.split("</think>", 1)
            reward += 1.0 if cfg.min_think_chars <= len(think.strip()) <= cfg.max_think_chars else -0.5
            reward += 0.25 if response.count("</think>") == 1 else -0.25
            final_answer = answer.strip()
        reward -= repetition_penalty(final_answer, n=cfg.repetition_ngram, cap=cfg.repetition_cap)
        return float(max(min(reward, 3.0), -3.0))

    def _gt_reward(self, final_answer: str, gt: Any) -> float:
        """Reward exact text/numeric ground-truth matches."""

        gt_list = gt if isinstance(gt, list) else ([] if gt in (None, "") else [gt])
        if not gt_list:
            return 0.0
        verified = validate_gt_in_text(final_answer, gt_list)
        return float(2.5 * len(verified) / max(len(gt_list), 1))

    def _tool_reward(self, turn_outputs: list[str], final_answer: str, gt: Any, tools: Any, unfinished: bool) -> float:
        """Reward valid tool calls and final answers grounded in tool results."""

        valid_names = {tool["function"]["name"] for tool in tools or [] if isinstance(tool, dict) and "function" in tool}
        tool_calls: list[dict[str, Any]] = []
        reward = 0.0
        for turn in turn_outputs:
            tool_calls.extend(parse_tool_calls(turn))
            reward -= 0.5 * abs(turn.count("<tool_call>") - turn.count("</tool_call>"))
        if not tool_calls:
            return reward

        valid_call_count = 0
        for tool_call in tool_calls:
            name = tool_call.get("name", "")
            raw_args = tool_call.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            check = CHECK_ARGS.get(name)
            valid_call_count += int(bool(name in valid_names and check and check(raw_args)))

        gt_list = gt if isinstance(gt, list) else ([] if gt in (None, "") else [gt])
        tool_gap = abs(valid_call_count - len(gt_list)) + max(0, len(tool_calls) - valid_call_count)
        reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap
        if gt_list:
            reward += self._gt_reward(final_answer, gt_list)
        if unfinished:
            reward -= 0.5
        return float(max(min(reward, 3.0), -3.0))

    def _reward_model_reward(self, prompt: str, final_answer: str) -> float:
        """Score a completion with an optional external reward model engine."""

        if self.reward_model_engine is None:
            return 0.0
        pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
        matches = re.findall(pattern, prompt, re.DOTALL)
        messages = [{"role": role, "content": content.strip()} for role, content in matches]
        return float(self.reward_model_engine.get_score(messages, final_answer))

    def compute_rewards(
        self,
        *,
        prompts: list[str],
        completions: list[str],
        completion_id_lists: list[list[int]],
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scalar rewards and per-function reward columns."""

        reward_names = list(self.reward_config.reward_funcs)
        rewards_per_func = torch.zeros((len(completions), len(reward_names)), device=self.device)
        num_generations = max(1, len(completions) // max(len(prompts), 1))

        for index, response in enumerate(completions):
            sample_index = index // num_generations
            meta = metadata[index] if index < len(metadata) else {}
            prompt = prompts[sample_index] if sample_index < len(prompts) else ""
            turn_outputs = meta.get("turn_outputs") or [response]
            answer = turn_outputs[-1].split("</think>", 1)[-1].strip() if turn_outputs else response.strip()
            if "</tool_call>" in answer:
                answer = answer.split("</tool_call>")[-1].strip()

            values = {
                "format": self._format_reward(response, answer),
                "gt": self._gt_reward(answer, meta.get("gt")),
                "tool": self._tool_reward(turn_outputs, answer, meta.get("gt"), meta.get("tools"), bool(meta.get("unfinished"))),
                "reward_model": self._reward_model_reward(prompt, answer),
            }
            for reward_index, reward_name in enumerate(reward_names):
                rewards_per_func[index, reward_index] = values.get(reward_name, 0.0)

        weights = torch.tensor(self.reward_config.weights, device=self.device, dtype=rewards_per_func.dtype)
        if weights.numel() != len(reward_names):
            weights = torch.ones(len(reward_names), device=self.device, dtype=rewards_per_func.dtype)
        rewards = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=-1)
        return rewards, rewards_per_func
