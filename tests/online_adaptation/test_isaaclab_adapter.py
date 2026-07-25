import pytest
import torch

from rl_training.online_adaptation.isaaclab_adapter import compose_proprio_frame

pytestmark = pytest.mark.isaaclab


def test_compose_proprio_frame_has_fixed_order_and_49_dimensions():
    terms = [
        torch.full((2, 3), 1.0),
        torch.full((2, 3), 2.0),
        torch.full((2, 12), 3.0),
        torch.full((2, 12), 4.0),
        torch.full((2, 4), 5.0),
        torch.full((2, 12), 6.0),
        torch.full((2, 3), 7.0),
    ]
    frame = compose_proprio_frame(*terms)
    assert frame.shape == (2, 49)
    assert torch.equal(frame[:, :3], terms[0])
    assert torch.equal(frame[:, -3:], terms[-1])
