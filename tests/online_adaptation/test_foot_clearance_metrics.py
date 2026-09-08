import pytest
import torch

from rl_training.foot_metrics import relative_foot_clearance


def test_relative_foot_clearance_uses_lowest_foot_as_local_ground():
    foot_pos_w = torch.tensor(
        [[[0.2, 0.1, 0.02], [0.2, -0.1, 0.07], [-0.2, 0.1, 0.04], [-0.2, -0.1, 0.02]]]
    )

    assert torch.allclose(
        relative_foot_clearance(foot_pos_w),
        torch.tensor([[0.0, 0.05, 0.02, 0.0]]),
    )


def test_relative_foot_clearance_rejects_ambiguous_foot_shape():
    with pytest.raises(ValueError, match="num_envs, 4, 3"):
        relative_foot_clearance(torch.zeros(2, 3, 3))
