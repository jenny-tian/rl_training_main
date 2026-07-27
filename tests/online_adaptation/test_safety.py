import torch

from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState


def state(tilt=0.0, height=0.35):
    return SafetyState(
        projected_gravity_xy=torch.tensor([[tilt, 0.0]]),
        base_height=torch.tensor([[height]]),
        joint_pos=torch.zeros(1, 12),
        joint_vel=torch.zeros(1, 12),
        joint_lower=torch.full((1, 12), -1.0),
        joint_upper=torch.full((1, 12), 1.0),
        default_joint_pos=torch.zeros(1, 12),
        action_scale=torch.full((1, 12), 0.25),
        nonfoot_contact=torch.zeros(1, 1, dtype=torch.bool),
        control_timed_out=torch.zeros(1, 1, dtype=torch.bool),
    )


def test_rate_and_joint_target_projection():
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2), residual_rate_limit=0.05))
    decision = filt.project(torch.zeros(1, 12), torch.ones(1, 12), state())
    assert torch.allclose(decision.residual, torch.full((1, 12), 0.05))
    assert decision.projected.item()


def test_unsafe_tilt_pauses_and_requires_stable_window_to_resume():
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2), stable_steps_to_resume=3))
    unsafe = filt.project(torch.ones(1, 12), torch.zeros(1, 12), state(tilt=0.8))
    assert unsafe.paused.item()
    assert torch.equal(unsafe.action, torch.zeros(1, 12))
    assert torch.equal(unsafe.residual, torch.zeros(1, 12))
    for _ in range(2):
        assert filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state()).paused.item()
    assert not filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state()).paused.item()
