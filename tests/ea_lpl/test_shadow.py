import torch

from rl_training.ea_lpl.basis import basis
from rl_training.ea_lpl.shadow import (finite_difference_jacobian, parameterization_jacobian, relative_deviation,
                                       request_from_codes, sensitivity)
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState

LIMITS = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4))


def nominal_state(n, contact=False):
    zeros = torch.zeros(n, 12)
    return SafetyState(projected_gravity_xy=torch.zeros(n, 2), base_height=torch.full((n, 1), 0.35),
                       joint_pos=zeros.clone(), joint_vel=zeros.clone(), joint_lower=torch.full((n, 12), -10.0),
                       joint_upper=torch.full((n, 12), 10.0), default_joint_pos=zeros.clone(),
                       action_scale=torch.full((n, 12), 0.25), nonfoot_contact=torch.full((n, 1), contact),
                       control_timed_out=torch.zeros(n, 1, dtype=torch.bool))


def snapshot_with(n, previous):
    live = SafetyFilter(LIMITS)
    live.temporal_snapshot(torch.zeros(n, 12))
    live._previous_residual = previous.clone()
    return live.temporal_snapshot(torch.zeros(n, 12))


def test_shadow_calls_leave_the_live_filter_unchanged():
    live = SafetyFilter(LIMITS)
    live.project(torch.zeros(2, 12), torch.full((2, 12), 0.05), nominal_state(2))
    before = live.temporal_snapshot(torch.zeros(2, 12))
    v = basis("sag").expand(2, 12, 2)
    finite_difference_jacobian(LIMITS, before, [torch.zeros(2, 12)] * 3, [nominal_state(2)] * 3, v,
                               torch.zeros(2, 2), 0.6)
    after = live.temporal_snapshot(torch.zeros(2, 12))
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_small_delta_at_the_hold_point_equals_the_parameterization():
    v = basis("sag").expand(2, 12, 2)
    u_ref = torch.tensor([[1.0, -0.5], [0.0, 0.0]])
    snapshot = snapshot_with(2, request_from_codes(v, u_ref))
    j = finite_difference_jacobian(LIMITS, snapshot, [torch.zeros(2, 12)], [nominal_state(2)], v, u_ref, 0.05)
    assert torch.allclose(j, parameterization_jacobian(v, u_ref, 0.05), atol=1e-5)


def test_rate_limit_shrinks_the_exploration_scale_response():
    v = basis("sag").expand(1, 12, 2)
    snapshot = snapshot_with(1, torch.zeros(1, 12))
    m = sensitivity(finite_difference_jacobian(LIMITS, snapshot, [torch.zeros(1, 12)], [nominal_state(1)], v,
                                               torch.zeros(1, 2), 0.6))
    m0 = sensitivity(parameterization_jacobian(v, torch.zeros(1, 2), 0.6))
    assert float(relative_deviation(m, m0)[0]) > 0.3


def test_contact_guard_gives_zero_sensitivity():
    v = basis("roll").expand(1, 12, 2)
    snapshot = snapshot_with(1, torch.zeros(1, 12))
    j = finite_difference_jacobian(LIMITS, snapshot, [torch.zeros(1, 12)], [nominal_state(1, contact=True)], v,
                                   torch.zeros(1, 2), 0.05)
    assert torch.count_nonzero(j) == 0
