from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/tools'))
from reference_online_interface import (DEFAULT, SCALE, EpisodeLedger, clip_action, observation,
                                        safety_diagnostics, wrap_manager_compute)
from reference_online_targets import protocol, rows, choose_targets
from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState


def make_bridge():
    default = torch.tensor([DEFAULT] * 2)
    state = SafetyState(torch.zeros(2, 2), torch.full((2, 1), .35), default.clone(), torch.zeros(2, 12),
        default - 2., default + 2., default, torch.tensor([SCALE]),
        torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool))
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.tensor([.24, .20, .20] * 4)))
    return ResidualSafetyBridge(NS(safety_state=lambda _: state), filt, action_clip=clip_action), state


@pytest.mark.parametrize('keyword', [False, True])
def test_native_manager_keyword_callback_contract_order_and_return(keyword):
    events = []
    reward = torch.tensor([1., 2.])
    class Manager:
        def compute(self, *, dt):
            events.append(('native', dt))
            return reward
    manager = Manager()
    wrap_manager_compute(manager, lambda step_dt: events.append(('observer', step_dt)))
    result = manager.compute(dt=.02) if keyword else manager.compute(.02)
    assert result is reward
    assert events == [('native', .02), ('observer', .02)]


def test_reward_capture_observes_terminal_state_before_reset():
    events = []
    terminal = torch.tensor([True, False])
    class Manager:
        def compute(self, *, dt):
            return torch.zeros(2)
    manager = Manager()
    wrap_manager_compute(manager, lambda step_dt: events.append(terminal.clone()))
    manager.compute(dt=.02)
    terminal[:] = False  # The native step performs its reset after reward compute.
    assert torch.equal(events[0], torch.tensor([True, False]))


def test_zero_preserves_current_feasible_action_outside_legacy_clip():
    bridge, state = make_bridge()
    action = torch.zeros(2, 12)
    action[:, 2] = -3.  # Legacy knee lower clip -2.57 is not this policy's clip.
    trace = bridge.step(action, torch.zeros_like(action), state.control_timed_out)
    assert torch.equal(trace.executed_action, action)
    projected, changed = safety_diagnostics(trace)
    assert not projected.any() and not changed.any()


def test_zero_physical_projection_is_explicit_and_not_called_base_equivalence():
    bridge, state = make_bridge()
    action = torch.full((2, 12), 40.)
    trace = bridge.step(action, torch.zeros_like(action), state.control_timed_out)
    projected, changed = safety_diagnostics(trace)
    assert projected.all() and changed.all()
    assert trace.decision.projected.all()


def test_asynchronous_filter_reset_leaves_other_history_and_restart_ramp():
    bridge, state = make_bridge()
    action = torch.zeros(2, 12)
    request = torch.full_like(action, .1)
    for _ in range(5): bridge.step(action, request, state.control_timed_out)
    bridge.safety_filter.reset_temporal_state(torch.tensor([0]))
    trace = bridge.step(action, request, state.control_timed_out)
    assert torch.allclose(trace.executed_action[0], torch.full((12,), .02))
    assert torch.allclose(trace.executed_action[1], torch.full((12,), .1))


def test_observation_uses_executed_action_and_native_clip_then_scale():
    perm = [2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10]
    data = NS(root_ang_vel_b=torch.full((2, 3), 200.), projected_gravity_b=torch.tensor([[0., 0., -1.]] * 2),
        joint_pos=torch.arange(24.).reshape(2, 12), default_joint_pos=torch.zeros(2, 12),
        joint_vel=torch.full((2, 12), 200.), default_joint_vel=torch.zeros(2, 12))
    prev = torch.arange(24.).reshape(2, 12) / 10
    result = observation(NS(data=data), perm, torch.tensor([[.4, 0., 0.]] * 2), prev)
    assert result.shape == (2, 45)
    assert (result[:, :3] == 25.).all() and (result[:, 21:33] == 5.).all()
    assert torch.equal(result[:, 9:21], data.joint_pos[:, perm])
    assert torch.equal(result[:, 33:], prev)


def test_multiepisode_reset_keeps_other_keys_and_budget_rejects_before_mutation():
    a, b = EpisodeLedger([8, 9], 12, 4), EpisodeLedger([8, 9], 12, 4)
    assert np.array_equal(a.reset([0, 1]), b.reset([0, 1]))
    untouched = a.keys[1]
    one, two = a.reset([0]), a.reset([0])
    assert a.keys[1] == untouched and a.keys[0][-1] == 2
    assert not np.array_equal(one, two)
    assert np.array_equal(one, b.reset([0]))
    before = a.keys.copy()
    with pytest.raises(RuntimeError): a.reset([1])
    assert a.keys == before and a.started == 4
    a.count_steps([True, False]); a.count_steps([True, True])
    assert a.steps == 3


def test_phase_budgets_and_sealed_confirmation_starts():
    assert len(rows('baseline')) * 2 == 384
    grid = rows('grid', 'stairs_06')
    assert len(grid) == 600 and len({r['code'] for r in grid}) == 75
    confirm = rows('confirmation', 'stairs_06', dict(code='best', basis='sag', u=[1., 0.]))
    assert len(confirm) * 2 == 64
    assert grid[0]['namespace'] != confirm[0]['namespace']
    assert protocol()['budgets']['total'] == 384 + 1200 + 64


@pytest.mark.parametrize('count,eligible', [(6, False), (7, True), (19, True), (20, False)])
def test_selection_does_not_relax_registered_success_window(count, eligible):
    records = [dict(condition=c, outcome='success' if i < count else 'timeout', progress_fraction=.6)
               for c in protocol()['conditions'] for i in range(32)]
    result = choose_targets({'B': dict(records=records)})
    assert ('stairs_06' in result['selected']) == eligible
    assert 'stairs_05' not in result['selected']
    assert result['selected'] == (['stairs_06', 'boxes'] if eligible else [])
