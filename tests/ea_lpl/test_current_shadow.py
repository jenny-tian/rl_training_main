import pytest
import torch

from rl_training.ea_lpl.basis import basis
from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter
from rl_training.ea_lpl.current_shadow import finite_difference_jacobian, rollout_mean_action
from rl_training.ea_lpl.shadow import parameterization_jacobian
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState


def fixture():
    zeros = torch.zeros(2, 12)
    state = SafetyState(torch.zeros(2, 2), torch.full((2, 1), .35), zeros, zeros,
        torch.full_like(zeros, -1.), torch.ones_like(zeros), zeros, torch.full_like(zeros, .25),
        torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool))
    limits = SafetyLimits(residual_limits=torch.tensor([.24, .20, .20] * 4))
    return zeros, state, limits


def test_current_shadow_uses_repaired_memory_across_projection_and_recovery():
    zero, state, limits = fixture()
    live = CurrentModelSafetyFilter(limits)
    snapshot = live.temporal_snapshot(zero)
    high = zero.clone(); high[:, 1] = 4.02
    expected = (live.project(high, zero, state).action + live.project(zero, zero, state).action) / 2
    assert torch.equal(rollout_mean_action(limits, snapshot, [high, zero], [state, state], zero), expected)
    assert torch.count_nonzero(snapshot['filter_previous_residual_action']) == 0


def test_current_shadow_matches_parameterization_at_small_unguarded_perturbation():
    zero, state, limits = fixture()
    live = CurrentModelSafetyFilter(limits)
    snapshot = live.temporal_snapshot(zero)
    v = basis('roll').expand(2, 12, 2)
    u = torch.zeros(2, 2)
    result = finite_difference_jacobian(limits, snapshot, [zero], [state], v, u, .05)
    assert torch.allclose(result, parameterization_jacobian(v, u, .05), atol=1e-6)
    assert all(torch.equal(t, live.temporal_snapshot(zero)[key]) for key, t in snapshot.items())


def test_current_shadow_rejects_legacy_snapshot():
    zero, state, limits = fixture()
    with pytest.raises(ValueError, match='not a current'):
        rollout_mean_action(limits, SafetyFilter(limits).temporal_snapshot(zero), [zero], [state], zero)
