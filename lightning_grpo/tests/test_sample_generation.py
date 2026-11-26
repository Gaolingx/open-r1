"""Periodic sample prompts must follow the chat format the model was trained on."""

import lightning_grpo.callbacks.core as callbacks_core
from lightning_grpo.callbacks.core import PeriodicSampleGenerationCallback
from lightning_grpo.utils.configs.base import LoggingConfig, ModelConfig


class _RecordingTokenizer:
    padding_side = "right"

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.rendered = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.rendered.append((messages, tokenize, add_generation_prompt))
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


def _build_callback(monkeypatch, tokenizer):
    monkeypatch.setattr(callbacks_core, "load_tokenizer", lambda _: tokenizer)
    return PeriodicSampleGenerationCallback(LoggingConfig(), ModelConfig())


def test_prompts_are_wrapped_in_the_chat_template(monkeypatch):
    tokenizer = _RecordingTokenizer(chat_template="<|begin|>")
    callback = _build_callback(monkeypatch, tokenizer)

    texts, add_special_tokens = callback._render_prompts(["法国的首都在哪里？"])

    assert texts == ["<user>法国的首都在哪里？<assistant>"]
    assert add_special_tokens is False
    assert tokenizer.rendered == [([{"role": "user", "content": "法国的首都在哪里？"}], False, True)]


def test_prompts_stay_raw_without_a_chat_template(monkeypatch):
    tokenizer = _RecordingTokenizer(chat_template=None)
    callback = _build_callback(monkeypatch, tokenizer)

    texts, add_special_tokens = callback._render_prompts(["hello", "hi"])

    assert texts == ["hello", "hi"]
    assert add_special_tokens is True
    assert tokenizer.rendered == []
