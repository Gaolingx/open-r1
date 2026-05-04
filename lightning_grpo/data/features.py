from __future__ import annotations

from datasets import Features, Sequence, Value

TOKENIZED_PRETRAIN_FEATURES = Features({
    "input_ids": Sequence(Value("int64")),
    "attention_mask": Sequence(Value("int64")),
    "labels": Sequence(Value("int64")),
})

TOKENIZED_SFT_FEATURES = Features({
    "input_ids": Sequence(Value("int64")),
    "attention_mask": Sequence(Value("int64")),
    "labels": Sequence(Value("int64")),
})

GRPO_PROMPT_FEATURES = Features({
    "prompt_text": Value("string"),
    "metadata": Value("string"),
    "sample_id": Value("int64"),
})

LOCAL_CHAT_DATASET_FEATURES = Features({
    "conversations": [{
        "role": Value("string"),
        "content": Value("string"),
        "reasoning_content": Value("string"),
        "tools": Value("string"),
        "tool_calls": Value("string"),
    }],
})
