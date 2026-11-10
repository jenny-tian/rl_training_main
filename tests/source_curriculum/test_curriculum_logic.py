import math
from types import SimpleNamespace

import pytest
import torch

from rl_training.source_curriculum.logic import (
    OMNI_FAMILY_ID, FamilyStats, apply_tile_commands, column_family_ids, curriculum_moves,
    omni_levels, omni_truncation_bounds, outside_bounds, resolve_outcomes, safety_unsafe)

LIMITS = SimpleNamespace(max_projected_gravity_xy=0.65, min_base_height=0.20, max_joint_velocity=30.0,
                         nonfoot_contact_grace_steps=10)
B = lambda *v: torch.tensor(v, dtype=torch.bool)


def test_failure_beats_completion_beats_timeout():
    c, f, t = resolve_outcomes(B(1, 1, 0, 0), B(1, 0, 0, 0), B(1, 1, 1, 0))
    assert c.tolist() == [False, True, False, False]
    assert f.tolist() == [True, False, False, False]
    assert t.tolist() == [False, False, True, False]


def test_moves():
    up, down = curriculum_moves(B(1, 1, 0, 0, 0), B(1, 0, 0, 0, 0), B(0, 0, 1, 1, 0),
                                torch.tensor([9.3, 9.3, 1.0, 6.0, 0.0]), torch.full((5,), 10.0))
    assert up.tolist() == [False, True, False, False, False]
    assert down.tolist() == [True, False, True, False, False]


def test_initial_reset_has_no_moves():
    z = torch.zeros(6, dtype=torch.bool)
    up, down = curriculum_moves(z, z, z, torch.zeros(6), torch.zeros(6))
    assert not up.any() and not down.any()


def test_omni_levels_and_bounds():
    g = torch.Generator().manual_seed(0)
    levels = omni_levels(10000, num_rows=10, device="cpu", generator=g)
    assert levels.min() == 1 and levels.max() == 8
    assert omni_truncation_bounds() == (-59.0, 59.0, -11.0, 19.0)
    xy = torch.tensor([[0.0, 0.0], [59.5, 0.0], [0.0, -11.5], [math.nan, 0.0]])
    assert outside_bounds(xy, omni_truncation_bounds()).tolist() == [False, True, True, True]


def test_family_ids():
    ids = column_family_ids()
    assert ids.shape == (20,) and ids[9:13].eq(OMNI_FAMILY_ID).all() and ids[13] == 0


def test_tile_commands():
    n = 6
    vel = torch.full((n, 3), 7.0); target = torch.full((n,), 2.0)
    is_h = torch.zeros(n, dtype=torch.bool); is_s = torch.ones(n, dtype=torch.bool)
    ids = torch.tensor([1, 3, 4]); heading = torch.tensor([0.1, -3.0, 3.0])
    apply_tile_commands(vel, target, is_h, is_s, ids, heading=heading, lin_vel_x=(0.25, 0.5),
                        stiffness=0.5, ang_range=(-0.8, 0.8), generator=torch.Generator().manual_seed(1))
    assert ((vel[ids, 0] >= 0.25) & (vel[ids, 0] <= 0.5)).all() and vel[ids, 1].eq(0).all()
    assert vel[ids, 2].tolist() == pytest.approx([-0.05, 0.8, -0.8])
    assert is_h[ids].all() and not is_s[ids].any() and target[ids].eq(0).all()
    assert vel[0].tolist() == [7.0, 7.0, 7.0] and is_s[0]  # untouched


def test_safety_formula_and_nonfinite():
    n = 5
    steps = torch.zeros(n, dtype=torch.long)
    gxy = torch.tensor([[0.0, 0.0], [0.6, 0.3], [0.0, 0.0], [0.0, 0.0], [math.nan, 0.0]])
    h = torch.tensor([[0.3], [0.3], [0.19], [0.3], [0.3]])
    jv = torch.zeros(n, 12); jv[3, 5] = 31.0
    contact = torch.zeros(n, 1, dtype=torch.bool); timeout = torch.zeros(n, 1, dtype=torch.bool)
    unsafe, steps = safety_unsafe(gxy, h, jv, contact, timeout, steps, LIMITS)
    assert unsafe.tolist() == [False, True, True, True, True]
    contact[0] = True
    for k in range(11):
        unsafe, steps = safety_unsafe(torch.zeros(n, 2), torch.full((n, 1), 0.3), torch.zeros(n, 12),
                                      contact, timeout, steps, LIMITS)
    assert unsafe[0] and steps[0] == 11 and not unsafe[1]


def test_family_stats():
    stats = FamilyStats("cpu", decay=0.5)
    fam = torch.tensor([0, 0, 2, OMNI_FAMILY_ID])
    stats.update(fam, B(1, 0, 0, 0), B(0, 1, 0, 0), B(0, 0, 0, 1))
    report = stats.report(torch.tensor([9, 3, 4, 5]), fam, top_row=9)
    assert report["flat/completion_ema"] == pytest.approx(0.5)
    assert report["flat/top_row_fraction"] == pytest.approx(0.5)
    assert report["slope/completion_ema"] == 0.0  # no outcome yet
    assert report["omni_flat/timeout_ema"] == 1.0


def test_safety_parity_with_source_episode_lifecycle():
    # The full SourceEpisodeLifecycle is tensor-only, so it is driven directly (no Isaac Sim).
    from rl_training.online_adaptation.safety import SafetyLimits
    from rl_training.online_adaptation.types import SafetyState
    from rl_training.source_scan.training_lifecycle import SourceEpisodeLifecycle

    limits = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4))
    n = 64
    g = torch.Generator().manual_seed(20261109)
    lifecycle = SourceEpisodeLifecycle(n, "cpu", limits)
    steps = torch.zeros(n, dtype=torch.long)
    lower, upper = torch.full((n, 12), -1.0), torch.full((n, 12), 1.0)
    saw_unsafe = saw_safe = saw_grace = False
    for k in range(20):
        gxy = (torch.rand(n, 2, generator=g) - 0.5) * 1.1
        height = 0.17 + 0.2 * torch.rand(n, 1, generator=g)
        joint_vel = (torch.rand(n, 12, generator=g) - 0.5) * 62.0 * (torch.rand(n, 1, generator=g) < 0.2)
        contact = torch.rand(n, 1, generator=g) < 0.85
        timed_out = torch.rand(n, 1, generator=g) < 0.03
        state = SafetyState(projected_gravity_xy=gxy, base_height=height, joint_pos=torch.zeros(n, 12),
                            joint_vel=joint_vel, joint_lower=lower, joint_upper=upper,
                            default_joint_pos=torch.zeros(n, 12), action_scale=torch.full((1, 12), 0.25),
                            nonfoot_contact=contact, control_timed_out=timed_out)
        lifecycle.update(state, torch.zeros(n), torch.full((n,), k + 1, dtype=torch.long))
        unsafe, steps = safety_unsafe(gxy, height, joint_vel, contact, timed_out, steps, limits)
        assert torch.equal(unsafe, lifecycle.unsafe), k
        assert torch.equal(steps, lifecycle.contact_steps), k
        saw_unsafe |= bool(unsafe.any()); saw_safe |= bool((~unsafe).any())
        saw_grace |= bool((steps > limits.nonfoot_contact_grace_steps).any())
    assert saw_unsafe and saw_safe and saw_grace
