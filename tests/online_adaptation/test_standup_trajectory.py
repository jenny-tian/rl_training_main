import pytest
import torch

from rl_training.standup_trajectory import (
    FOLDED_JOINT_POS,
    STANDING_JOINT_POS,
    cubic_hermite_position,
    physical_targets_to_policy_action,
    standup_targets,
)


def test_cubic_hermite_has_exact_position_and_zero_velocity_endpoints():
    start = torch.tensor([[0.0, -1.0]])
    end = torch.tensor([[1.0, 2.0]])
    result, velocity = cubic_hermite_position(start, torch.zeros_like(start), end, torch.zeros_like(end), 1.5, 1.5)
    assert torch.equal(result, end)
    assert torch.equal(velocity, torch.zeros_like(end))


def test_standup_targets_reaches_reference_folded_and_standing_poses():
    start = FOLDED_JOINT_POS.clone()
    phase_one, phase_one_vel = standup_targets(start, torch.zeros_like(start), elapsed=1.5, phase_duration=1.5, phase=1)
    phase_two, phase_two_vel = standup_targets(start, torch.zeros_like(start), elapsed=1.5, phase_duration=1.5, phase=2)
    assert torch.allclose(phase_one, FOLDED_JOINT_POS, atol=1e-6)
    assert torch.allclose(phase_two, STANDING_JOINT_POS, atol=1e-6)
    assert torch.allclose(phase_one_vel, torch.zeros_like(start), atol=1e-6)
    assert torch.allclose(phase_two_vel, torch.zeros_like(start), atol=1e-6)


def test_standup_phase_does_not_overshoot_targets():
    start = torch.zeros(1, 12)
    folded, _ = standup_targets(start, torch.zeros_like(start), elapsed=0.75, phase_duration=1.5, phase=1)
    assert torch.all((folded[:, 1::3] <= 0.0) & (folded[:, 1::3] >= FOLDED_JOINT_POS[:, 1::3]))
    assert torch.all((folded[:, 2::3] >= 0.0) & (folded[:, 2::3] <= FOLDED_JOINT_POS[:, 2::3]))


def test_physical_target_round_trip_uses_policy_action_convention():
    action = physical_targets_to_policy_action(STANDING_JOINT_POS)
    assert torch.allclose(action, torch.tensor([[0.0, -0.0141407, -0.1156335] * 4]), atol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"elapsed": -0.1, "phase_duration": 1.5, "phase": 1},
        {"elapsed": 0.0, "phase_duration": 0.0, "phase": 1},
        {"elapsed": 0.0, "phase_duration": 1.5, "phase": 3},
    ],
)
def test_standup_targets_rejects_invalid_phase_inputs(kwargs):
    with pytest.raises(ValueError):
        standup_targets(FOLDED_JOINT_POS, torch.zeros_like(FOLDED_JOINT_POS), **kwargs)
