import pytest
import torch

from rl_training.command_slew import slew_toward


def test_slew_toward_moves_each_axis_in_both_directions_without_overshoot():
    current = torch.tensor([[0.0, 0.4, -0.1], [0.49, -0.49, 0.0]])
    original = current.clone()
    target = torch.tensor([[0.5, 0.1, -0.5], [0.5, -0.5, 0.0]])

    result = slew_toward(current, target, max_delta=(0.1, 0.2, 0.3))

    assert torch.allclose(
        result,
        torch.tensor([[0.1, 0.2, -0.4], [0.5, -0.5, 0.0]]),
    )
    assert torch.equal(current, original)


@pytest.mark.parametrize(
    "max_delta, message",
    [
        ((0.1, 0.2), "three components"),
        ((0.1, 0.0, 0.3), "positive"),
        ((0.1, float("inf"), 0.3), "finite"),
    ],
)
def test_slew_toward_rejects_invalid_limits(max_delta, message):
    with pytest.raises(ValueError, match=message):
        slew_toward(torch.zeros(1, 3), torch.ones(1, 3), max_delta=max_delta)


def test_slew_toward_rejects_mismatched_command_shapes():
    with pytest.raises(ValueError, match="same shape"):
        slew_toward(torch.zeros(1, 3), torch.ones(2, 3), max_delta=(0.1, 0.1, 0.1))
