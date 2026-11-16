from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter, projected_base_action
from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState


def state(n=2):
    default = torch.tensor([[0., -.65, 1.3] * 4] * n)
    return SafetyState(torch.zeros(n, 2), torch.full((n, 1), .35), default.clone(), torch.zeros(n, 12),
        default - 2., default + 2., default, torch.tensor([[.125, .25, .25] * 4]),
        torch.zeros(n, 1, dtype=torch.bool), torch.zeros(n, 1, dtype=torch.bool))


def limits():
    return SafetyLimits(residual_limits=torch.tensor([.24, .20, .20] * 4))


def test_legacy_reproduces_zero_request_memory_leak_current_filter_removes_it():
    s = state()
    old, new = SafetyFilter(limits()), CurrentModelSafetyFilter(limits())
    base, zero = torch.zeros(2, 12), torch.zeros(2, 12)
    base[0, 1] = 8.02  # Physical target is 0.025 rad past the .02 rad margin.
    first_old, first_new = old.project(base, zero, s), new.project(base, zero, s)
    assert torch.equal(first_old.action, first_new.action)  # Physical safety unchanged.
    old_after, new_after = old.project(zero, zero, s), new.project(zero, zero, s)
    assert old_after.action[0, 1].item() == pytest.approx(-.08, abs=2e-6)
    assert torch.equal(new_after.action, zero)
    assert torch.equal(new._previous_residual, zero)


@pytest.mark.parametrize('large', [-40., 40.])
def test_large_base_corrections_cannot_escape_residual_amplitude_envelope(large):
    s, filt = state(), CurrentModelSafetyFilter(limits())
    base = torch.full((2, 12), large)
    zero = torch.zeros_like(base)
    decision = filt.project(base, zero, s)
    assert torch.equal(decision.action, projected_base_action(base, s))
    assert torch.equal(filt.execution_components()['learned_effect'], zero)
    assert torch.equal(filt.project(zero, zero, s).action, zero)


def test_mixed_zero_and_nonzero_codes_keep_independent_slew_memory():
    s, filt = state(), CurrentModelSafetyFilter(limits())
    bridge = ResidualSafetyBridge(SimpleNamespace(safety_state=lambda _: s), filt,
                                  action_clip=lambda a: a.clamp(-100, 100))
    base, request = torch.zeros(2, 12), torch.zeros(2, 12)
    base[0, 1] = 8.02
    request[1] = .1
    first = bridge.step(base, request, s.control_timed_out)
    second = bridge.step(torch.zeros_like(base), request, s.control_timed_out)
    assert torch.equal(second.executed_action[0], torch.zeros(12))
    assert torch.allclose(first.executed_action[1], torch.full((12,), .02))
    assert torch.allclose(second.executed_action[1], torch.full((12,), .04))
    filt.reset_temporal_state(torch.tensor([0]))
    third = bridge.step(torch.zeros_like(base), request, s.control_timed_out)
    assert torch.allclose(third.executed_action[1], torch.full((12,), .06))


def test_legitimate_nonzero_to_zero_request_retains_slew_limit():
    s, filt = state(), CurrentModelSafetyFilter(limits())
    zero = torch.zeros(2, 12)
    for _ in range(5): filt.project(zero, torch.full_like(zero, .1), s)
    after = filt.project(zero, zero, s)
    assert torch.allclose(after.action, torch.full_like(zero, .08))


@pytest.mark.parametrize('guard', ['contact', 'tilt', 'height', 'velocity', 'timeout'])
def test_existing_guards_still_remove_learned_effect_and_enforce_joint_limits(guard):
    s, filt = state(), CurrentModelSafetyFilter(limits())
    s = {'contact': replace(s, nonfoot_contact=torch.ones(2, 1, dtype=torch.bool)),
         'tilt': replace(s, projected_gravity_xy=torch.tensor([[.8, 0.]] * 2)),
         'height': replace(s, base_height=torch.full((2, 1), .1)),
         'velocity': replace(s, joint_vel=torch.full((2, 12), 35.)),
         'timeout': replace(s, control_timed_out=torch.ones(2, 1, dtype=torch.bool))}[guard]
    base = torch.full((2, 12), 40.)
    decision = filt.project(base, torch.full_like(base, .1), s)
    assert torch.allclose(decision.action, projected_base_action(base, s), atol=2e-6, rtol=0)
    assert torch.count_nonzero(filt._previous_residual) == 0


def test_projected_nonzero_effect_is_distinct_from_base_correction():
    s, filt = state(), CurrentModelSafetyFilter(limits())
    base, request = torch.zeros(2, 12), torch.zeros(2, 12)
    base[:, 1] = 7.93
    request[:, 1] = -.1
    decision = filt.project(base, request, s)
    components = filt.execution_components()
    assert components['base_correction'][0, 1].item() == pytest.approx(-.01, abs=2e-6)
    assert components['learned_effect'][0, 1].item() == pytest.approx(-.01, abs=2e-6)
    assert torch.allclose(decision.action - base,
                          components['base_correction'] + components['learned_effect'], atol=1e-6)


def test_shadow_copy_uses_current_semantics_and_does_not_mutate_live_memory():
    s, live = state(), CurrentModelSafetyFilter(limits())
    base = torch.zeros(2, 12); base[0, 1] = 8.02
    live.project(base, torch.zeros_like(base), s)
    snapshot = live.temporal_snapshot(base)
    shadow = CurrentModelSafetyFilter.from_snapshot(limits(), snapshot)
    out = shadow.project(torch.zeros_like(base), torch.full_like(base, .1), s)
    assert torch.allclose(out.action, torch.full_like(base, .02))
    assert all(torch.equal(v, live.temporal_snapshot(base)[k]) for k, v in snapshot.items())
    with pytest.raises(ValueError, match='not a current'):
        CurrentModelSafetyFilter.from_snapshot(limits(), SafetyFilter(limits()).temporal_snapshot(base))


def test_repeated_joint_boundary_crossings_preserve_all_zero_codes():
    torch.manual_seed(20261116)
    s, filt = state(32), CurrentModelSafetyFilter(limits())
    for _ in range(100):
        base = torch.randn(32, 12) * 15.
        decision = filt.project(base, torch.zeros_like(base), s)
        safe_base = projected_base_action(base, s)
        assert torch.equal(decision.action, safe_base)
        assert torch.count_nonzero(filt._previous_residual) == 0


def test_random_mixed_residuals_keep_physical_bounds_and_memory_envelope():
    torch.manual_seed(20261117)
    s, filt = state(32), CurrentModelSafetyFilter(limits())
    for step in range(100):
        base = (torch.randn(32, 12) * 15.).clamp(-100, 100)
        request = torch.randn_like(base) * .4
        request[:16] = 0
        if step % 11 == 0:
            filt.reset_temporal_state(torch.tensor([1, 18]))
        decision = filt.project(base, request, s)
        target = s.default_joint_pos + s.action_scale * decision.action
        assert (target >= s.joint_lower + .02 - 2e-6).all()
        assert (target <= s.joint_upper - .02 + 2e-6).all()
        assert (filt._previous_residual.abs() <= limits().residual_limits).all()
        assert torch.count_nonzero(filt._previous_residual[:16]) == 0
        assert torch.equal(decision.action[:16], projected_base_action(base, s)[:16])
