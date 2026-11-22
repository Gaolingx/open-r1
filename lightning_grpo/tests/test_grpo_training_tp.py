from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from lightning_grpo.models.grpo.reward import GRPORewardManager
from lightning_grpo.models.grpo.rollout_module.rollout_engine import VLLMRolloutEngineWrapper
from lightning_grpo.models.grpo.rollout_module.vllm.vllm_generation import VLLMGeneration
from lightning_grpo.tests.support import StubTokenizer
from lightning_grpo.utils.configs.grpo import GRPORewardConfig


def test_tp_peer_packs_leader_trajectory(monkeypatch):
    wrapper = object.__new__(VLLMRolloutEngineWrapper)
    wrapper.engine = SimpleNamespace(max_completion_length=4, temperature=1., top_p=1.,
                                     generate=lambda **kwargs: ([], [[3]], [[[-2.]]], [[[3]]], None))
    wrapper.training_tp_group = object()
    wrapper._profiler = None
    wrapper.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9, batch_decode=lambda rows, **kw: list(map(str, rows)))

    def broadcast(payload, *, group, group_src, device):
        assert group is wrapper.training_tp_group and group_src == 0
        assert payload == [([[3]], [[-2.]])]
        payload[0] = ([[5, 9]], [[-1., -0.5]])

    monkeypatch.setattr(dist, 'broadcast_object_list', broadcast)
    prompt = torch.tensor([[2]])
    result = wrapper.generate(prompt, torch.ones_like(prompt), 1, 4)
    assert result.completion_id_lists == [[5, 9]]
    assert result.completion_mask.tolist() == [[1, 1]]
    assert result.per_token_logps.tolist() == [[-1., -0.5]]


def _tp_worker(rank, rendezvous):
    dist.init_process_group('gloo', rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=45))
    try:
        mesh = DeviceMesh('cpu', [0, 1])
        group = mesh.get_group()
        ids = [1, 9] if rank == 0 else [3, 4, 9]
        sampled = [[[-0.5 - rank] for _ in ids]]
        engine = SimpleNamespace(max_completion_length=4, temperature=1., top_p=1.,
                                 generate=lambda **kwargs: ([], [ids], sampled, [[[i] for i in ids]], None))
        wrapper = object.__new__(VLLMRolloutEngineWrapper)
        wrapper.engine, wrapper.training_tp_group, wrapper._profiler = engine, group, None
        wrapper.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9, batch_decode=lambda rows, **kw: list(map(str, rows)))
        prompt = torch.tensor([[2]])
        result = wrapper.generate(prompt, torch.ones_like(prompt), 1, 4)
        assert result.completion_ids.tolist() == [[1, 9]]
        assert result.completion_mask.tolist() == [[1, 1]]
        assert result.per_token_logps.tolist() == [[-0.5, -0.5]]
        manager = GRPORewardManager(GRPORewardConfig(reward_funcs=['format']), StubTokenizer(), torch.device('cpu'), training_tp_group=group)
        def score(**kwargs):
            assert rank == 0
            return [3.]
        manager.reward_funcs = [score]
        totals, scores = manager.compute_rewards(['p'], result.completions_text, result.completion_id_lists, [{}])
        assert totals.tolist() == [3.] and scores.tolist() == [[3.]]

        model = torch.nn.Module()
        dense = torch.arange(16.).reshape(8, 2)
        model.packed = torch.nn.Parameter(distribute_tensor(dense, mesh, [_StridedShard(0, split_factor=2)]))
        model.bias = torch.nn.Parameter(torch.ones(2))
        backend = object.__new__(VLLMGeneration)
        backend.accelerator = SimpleNamespace(device=torch.device('cpu'))
        weights = dict(backend._iter_fsdp2_params(model))
        torch.testing.assert_close(weights['packed'], dense)
        torch.testing.assert_close(weights['bias'], model.bias)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason='requires Gloo')
def test_training_tp_shares_samples_scores_and_full_weights(tmp_path):
    try:
        dist.ProcessGroupGloo.create_device(hostname='127.0.0.1')
    except RuntimeError as exc:
        if 'unsupported gloo device' not in str(exc):
            raise
        pytest.skip(str(exc))
    torch.multiprocessing.spawn(_tp_worker, args=((tmp_path / 'store').as_uri(),), nprocs=2)
