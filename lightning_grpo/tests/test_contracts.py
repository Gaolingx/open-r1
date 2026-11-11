"""Unit tests for the GRPO collaborator contracts (distributed gather semantics)."""

from __future__ import annotations

import pytest
import torch

from lightning_grpo.models.grpo.contracts import DistributedContext, uses_greedy_baseline


class TestGather:
    def test_single_rank_is_a_passthrough(self) -> None:
        context = DistributedContext()
        tensor = torch.arange(6.0).reshape(3, 2)

        assert context.is_distributed is False
        assert context.gather(tensor) is tensor

    def test_multi_rank_concatenates_in_rank_order(self) -> None:
        context = DistributedContext(rank=1, world_size=2, all_gather_fn=lambda tensor: torch.cat([tensor, tensor]))
        local = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        gathered = context.gather(local)

        assert gathered.shape == (4, 2)
        assert torch.equal(gathered[:2], local)
        assert torch.equal(gathered[2:], local)

    def test_padding_to_the_largest_shard_is_applied_before_gathering(self) -> None:
        local = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        peer = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        payloads: list[torch.Size] = []

        def all_gather(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() == 1 and tensor.numel() == 2:
                # Shape descriptor: report both ranks' local shapes.
                return torch.cat([torch.tensor(local.shape), torch.tensor(peer.shape)])
            payloads.append(tensor.shape)
            return torch.cat([tensor, peer])

        context = DistributedContext(rank=0, world_size=2, all_gather_fn=all_gather)
        gathered = context.gather(local)

        assert payloads == [torch.Size([3, 2])]
        assert gathered.shape == (6, 2)
        assert torch.equal(gathered[:2], local)
        # The short rank is right-padded with zeros up to the largest local batch size.
        assert torch.equal(gathered[2], torch.zeros(2))
        assert torch.equal(gathered[3:], peer)

    def test_scalar_gather_is_flattened(self) -> None:
        context = DistributedContext(
            rank=0, world_size=2, all_gather_fn=lambda tensor: torch.stack([tensor, tensor])
        )

        gathered = context.gather(torch.tensor(3.0))

        assert gathered.shape == (2,)
        assert torch.equal(gathered, torch.tensor([3.0, 3.0]))

    def test_missing_all_gather_fn_raises(self) -> None:
        context = DistributedContext(rank=0, world_size=2)

        with pytest.raises(RuntimeError, match="all_gather_fn"):
            context.gather(torch.zeros(2))


class TestUsesGreedyBaseline:
    def test_remax_needs_the_baseline(self) -> None:
        assert uses_greedy_baseline("remax") is True

    @pytest.mark.parametrize("estimator", ["grpo", "rloo", "reinforce_plus_plus", "grpo_passk"])
    def test_other_estimators_do_not(self, estimator: str) -> None:
        assert uses_greedy_baseline(estimator) is False
