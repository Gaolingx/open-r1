from types import SimpleNamespace

import torch

from lightning_grpo.models.grpo.rollout_module.rollout_engine import VLLMRolloutEngineWrapper
from lightning_grpo.models.grpo.tool_call import GRPOToolCallMixin
from lightning_grpo.tests.support import ToyCausalLM
from lightning_grpo.tests.test_rollout_module import make_coordinator
from open_r1.rewards import format_reward


def test_vllm_packing_preserves_eos_but_does_not_invent_it_for_short_rows():
    coordinator, _, tokenizer = make_coordinator()
    tokenizer.eos_token_id = tokenizer.pad_token_id = 2
    tokenizer.batch_decode = lambda rows, **kw: list(map(str, rows))
    wrapper = object.__new__(VLLMRolloutEngineWrapper)
    wrapper.tokenizer = tokenizer
    result = wrapper._build_rollout_result([[3, 2], [4], []], [[-1., -2.], [-3.], []], torch.device('cpu'))
    packed = coordinator._pack_rollout(['p'], result.completion_ids, result.completion_mask, result.completions_text,
                                      [{}, {}, {}], 3, result.per_token_logps, engine_completion_truncated=result.completion_truncated)
    assert packed['completion_id_lists'] == [[3, 2], [4], []]
    assert packed['completion_truncated'].tolist() == [0, 1, 1]


def test_tool_round_masks_and_final_answer(monkeypatch):
    assistant = {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {'name': 'lookup', 'arguments': {}}}]}
    formatted = '<think>\nreason\n</think>\n<answer>\n42\n</answer>'
    answer = {'role': 'assistant', 'content': formatted}
    def parse(tokenizer, ids):
        return assistant.copy() if ids == [1, 2] else answer.copy()
    monkeypatch.setattr('lightning_grpo.models.grpo.tool_call.parse_response', parse)
    host = GRPOToolCallMixin()
    host.device = torch.device('cpu')
    host.policy = ToyCausalLM()
    host.policy.config = SimpleNamespace(max_position_embeddings=32)
    host.tokenizer = host._tokenizer = SimpleNamespace(pad_token_id=2, eos_token_id=2, decode=lambda ids, **kw: str(ids))
    host.config = SimpleNamespace(rollout=SimpleNamespace(engine='torch', max_total_length=16, max_completion_length=8, temperature=1., top_p=1.))
    host.max_tool_calling_iterations = 1
    host._sync_tools, host._async_tools = {'lookup': lambda: '42'}, {}
    host._get_tool_suffix_ids = lambda messages: [4, 2]

    def generate(prompt_ids, attention_mask, **kw):
        assert prompt_ids.tolist() == [[3, 1, 2, 4, 2]]
        assert attention_mask.tolist() == [[1, 1, 1, 1, 1]]
        return SimpleNamespace(completion_id_lists=[[5, 2]], per_token_logps=torch.empty(0))

    host.rollout_coordinator = SimpleNamespace(rollout_engine=SimpleNamespace(generate=generate))
    packed = host._run_tool_calling({'prompt_ids': torch.tensor([[3], [3]]), 'prompt_mask': torch.ones(2, 1),
                                     'prompts': ['p', 'q'], 'completion_id_lists': [[1, 2], [5, 2]]})
    assert packed['completion_id_lists'] == [[1, 2, 4, 2, 5, 2], [5, 2]]
    assert packed['completion_mask'].tolist() == [[1]*6, [1, 1, 0, 0, 0, 0]]
    assert packed['tool_mask'].tolist() == [[1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0]]
    assert format_reward(packed['completions']) == [1., 1.]
    assert format_reward([[assistant, {'role': 'tool', 'content': formatted}]]) == [0.]
