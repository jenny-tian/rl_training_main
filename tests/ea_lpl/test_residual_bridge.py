import pytest
import torch

from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState

LIMITS = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4))


def nominal_state(n=2, contact=False):
    zeros = torch.zeros(n, 12)
    return SafetyState(projected_gravity_xy=torch.zeros(n, 2), base_height=torch.full((n, 1), 0.35),
                       joint_pos=zeros.clone(), joint_vel=zeros.clone(), joint_lower=torch.full((n, 12), -10.0),
                       joint_upper=torch.full((n, 12), 10.0), default_joint_pos=zeros.clone(),
                       action_scale=torch.full((n, 12), 0.25), nonfoot_contact=torch.full((n, 1), contact),
                       control_timed_out=torch.zeros(n, 1, dtype=torch.bool))


class FixedAdapter:
    def __init__(self, state):
        self.state = state

    def safety_state(self, control_timed_out):
        return self.state


def test_residual_ramps_at_the_rate_limit_then_holds():
    bridge = ResidualSafetyBridge(FixedAdapter(nominal_state()), SafetyFilter(LIMITS))
    request = torch.full((2, 12), 0.1)
    timed_out = torch.zeros(2, 1, dtype=torch.bool)
    executed = [bridge.step(torch.zeros(2, 12), request, timed_out).executed_action[0, 0].item() for _ in range(7)]
    assert executed == pytest.approx([0.02, 0.04, 0.06, 0.08, 0.10, 0.10, 0.10])


def test_contact_guard_zeroes_the_residual_and_keeps_the_request():
    bridge = ResidualSafetyBridge(FixedAdapter(nominal_state(contact=True)), SafetyFilter(LIMITS))
    trace = bridge.step(torch.zeros(2, 12), torch.full((2, 12), 0.1), torch.zeros(2, 1, dtype=torch.bool))
    assert torch.equal(trace.executed_action, torch.zeros(2, 12))
    assert torch.equal(trace.residual_request, torch.full((2, 12), 0.1))
    assert bool(trace.decision.projected.all())


def test_nonfinite_request_is_rejected():
    bridge = ResidualSafetyBridge(FixedAdapter(nominal_state()), SafetyFilter(LIMITS))
    request = torch.zeros(2, 12)
    request[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        bridge.step(torch.zeros(2, 12), request, torch.zeros(2, 1, dtype=torch.bool))
