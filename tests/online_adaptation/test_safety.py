import pytest
import torch

from rl_training.lite3_stability_contract import (
    LITE3_MAX_PROJECTED_GRAVITY_XY,
    LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY,
)
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.isaaclab_adapter import select_joint_order
from rl_training.online_adaptation.types import SafetyState


def state(tilt=0.0, height=0.35, nonfoot_contact=False, control_timed_out=False):
    return SafetyState(
        projected_gravity_xy=torch.tensor([[tilt, 0.0]]),
        base_height=torch.tensor([[height]]),
        joint_pos=torch.zeros(1, 12),
        joint_vel=torch.zeros(1, 12),
        joint_lower=torch.full((1, 12), -1.0),
        joint_upper=torch.full((1, 12), 1.0),
        default_joint_pos=torch.zeros(1, 12),
        action_scale=torch.full((1, 12), 0.25),
        nonfoot_contact=torch.tensor([[nonfoot_contact]], dtype=torch.bool),
        control_timed_out=torch.tensor([[control_timed_out]], dtype=torch.bool),
    )


def test_select_joint_order_maps_robot_storage_to_policy_action_order():
    robot_order = torch.tensor([[10, 20, 30, 40, 50, 60]])
    policy_joint_ids = [0, 2, 4, 1, 3, 5]

    ordered = select_joint_order(robot_order, policy_joint_ids)

    assert torch.equal(ordered, torch.tensor([[10, 30, 50, 20, 40, 60]]))


def test_select_joint_order_preserves_trailing_joint_limit_pair():
    robot_order = torch.tensor(
        [[[10, 11], [20, 21], [30, 31], [40, 41], [50, 51], [60, 61]]]
    )
    policy_joint_ids = [0, 2, 4, 1, 3, 5]

    ordered = select_joint_order(robot_order, policy_joint_ids)

    assert torch.equal(
        ordered,
        torch.tensor(
            [[[10, 11], [30, 31], [50, 51], [20, 21], [40, 41], [60, 61]]]
        ),
    )


def test_rate_and_joint_target_projection():
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2), residual_rate_limit=0.05))
    decision = filt.project(torch.zeros(1, 12), torch.ones(1, 12), state())
    assert torch.allclose(decision.residual, torch.full((1, 12), 0.05))
    assert decision.projected.item()


@pytest.mark.parametrize("paused", [False, True])
def test_zero_adapter_uses_identical_feasible_action_with_or_without_force_zero(paused):
    from dataclasses import replace

    limits = SafetyLimits(residual_limits=torch.full((12,), 0.2))
    inputs = replace(state(control_timed_out=paused),
                     default_joint_pos=torch.tensor([[0.1, 0.5, -1.1] * 4]),
                     joint_lower=torch.full((1, 12), -3.0),
                     joint_upper=torch.full((1, 12), 3.0))
    base = torch.tensor([[0.40760514, -0.49224943, -0.48111793] * 4])
    ordinary = SafetyFilter(limits).project(base, torch.zeros_like(base), inputs)
    forced = SafetyFilter(limits).project(base, torch.zeros_like(base), inputs, force_zero_residual=True)
    assert torch.equal(ordinary.action, base)
    assert torch.equal(ordinary.action, forced.action)
    assert not ordinary.residual.any() and not forced.residual.any()


def test_force_zero_keeps_real_sub_tolerance_joint_limit_correction():
    from dataclasses import replace

    base = torch.full((1, 12), 0.4)
    inputs = replace(state(), joint_upper=torch.full((1, 12), 0.12 - 1e-7))
    decision = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2))).project(
        base, torch.zeros_like(base), inputs, force_zero_residual=True)
    assert torch.all(decision.action < base)
    assert torch.all(inputs.default_joint_pos + inputs.action_scale * decision.action <= inputs.joint_upper - 0.02)


def test_risk_gate_preserves_full_residual_at_low_risk():
    filt = SafetyFilter(
        SafetyLimits(
            residual_limits=torch.full((12,), 0.2),
            residual_rate_limit=1.0,
            risk_gate_enabled=True,
        )
    )
    decision = filt.project(torch.zeros(1, 12), torch.full((1, 12), 0.1), state())
    torch.testing.assert_close(decision.risk_score, torch.zeros(1, 1))
    torch.testing.assert_close(decision.risk_scale, torch.ones(1, 1))
    torch.testing.assert_close(decision.residual, torch.full((1, 12), 0.1))


def test_risk_gate_scales_residual_inside_hard_safety_margin():
    filt = SafetyFilter(
        SafetyLimits(
            residual_limits=torch.full((12,), 0.2),
            residual_rate_limit=1.0,
            risk_gate_enabled=True,
        )
    )
    decision = filt.project(
        torch.zeros(1, 12), torch.full((1, 12), 0.2), state(tilt=0.50)
    )
    # Tilt is halfway through [0.35, 0.65], so the default envelope is 0.625.
    torch.testing.assert_close(decision.risk_score, torch.full((1, 1), 0.5))
    torch.testing.assert_close(decision.risk_scale, torch.full((1, 1), 0.625))
    torch.testing.assert_close(decision.residual, torch.full((1, 12), 0.125))
    assert not decision.paused.item()


def test_risk_gate_keeps_base_action_before_hard_pause():
    filt = SafetyFilter(
        SafetyLimits(
            residual_limits=torch.full((12,), 0.2),
            residual_rate_limit=1.0,
            risk_gate_enabled=True,
        )
    )
    decision = filt.project(
        torch.full((1, 12), 0.5), torch.full((1, 12), 0.2), state(tilt=0.54)
    )
    assert not decision.paused.item()
    assert torch.all(decision.residual > 0)
    assert torch.all(decision.residual < 0.2)
    assert torch.all(decision.action > 0.5)


def test_risk_gate_parameters_validate_direction_and_bounds():
    with pytest.raises(ValueError, match="risk gate tilt end"):
        SafetyLimits(
            residual_limits=torch.ones(12),
            risk_gate_tilt_start=0.7,
            risk_gate_tilt_end=0.6,
        )
    with pytest.raises(ValueError, match="risk_gate_min_scale"):
        SafetyLimits(residual_limits=torch.ones(12), risk_gate_min_scale=1.1)


def test_force_zero_residual_bypasses_learned_rate_limit_memory():
    filt = SafetyFilter(
        SafetyLimits(residual_limits=torch.full((12,), 0.2), residual_rate_limit=0.05)
    )
    base_action = torch.zeros(1, 12)
    learned_residual = torch.full((1, 12), 0.2)
    for _ in range(4):
        filt.project(base_action, learned_residual, state())

    rate_limited = filt.project(base_action, torch.zeros_like(learned_residual), state())
    forced = filt.project(
        base_action,
        torch.zeros_like(learned_residual),
        state(),
        force_zero_residual=True,
    )

    assert torch.allclose(rate_limited.residual, torch.full((1, 12), 0.15))
    assert torch.equal(forced.residual, torch.zeros_like(learned_residual))
    assert torch.equal(forced.action, base_action)


def test_force_zero_residual_removes_joint_round_trip_noise_exactly():
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2)))
    base_action = torch.full((1, 12), 0.12345)
    noisy_state = state()
    noisy_state = SafetyState(
        **{
            **noisy_state.__dict__,
            "default_joint_pos": torch.full((1, 12), 0.1),
            "action_scale": torch.full((1, 12), 0.3),
        }
    )

    forced = filt.project(
        base_action,
        torch.zeros_like(base_action),
        noisy_state,
        force_zero_residual=True,
    )

    assert torch.equal(forced.residual, torch.zeros_like(base_action))
    assert torch.equal(forced.action, base_action)


def test_float32_joint_round_trip_noise_is_not_reported_as_projection():
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2)))
    base_action = torch.full((1, 12), 0.12345)
    noisy_state = state()
    noisy_state = SafetyState(
        **{
            **noisy_state.__dict__,
            "default_joint_pos": torch.full((1, 12), 0.1),
            "action_scale": torch.full((1, 12), 0.3),
        }
    )

    decision = filt.project(base_action, torch.full_like(base_action, 0.01), noisy_state)

    assert not decision.projected.item()
    torch.testing.assert_close(decision.residual, torch.full_like(base_action, 0.01))


def test_force_zero_residual_preserves_only_joint_limit_correction_memory():
    filt = SafetyFilter(
        SafetyLimits(residual_limits=torch.full((12,), 0.2), residual_rate_limit=0.05)
    )
    learned_residual = torch.full((1, 12), 0.2)
    for _ in range(4):
        filt.project(torch.zeros(1, 12), learned_residual, state())

    base_action = torch.zeros(1, 12)
    base_action[:, 0] = 4.0
    forced = filt.project(
        base_action,
        learned_residual,
        state(),
        force_zero_residual=True,
    )

    expected_residual = torch.zeros_like(learned_residual)
    expected_residual[:, 0] = -0.08
    assert torch.allclose(forced.residual, expected_residual, atol=1e-6)
    assert torch.allclose(forced.action, base_action + expected_residual, atol=1e-6)

    after = filt.project(
        torch.zeros_like(base_action),
        torch.zeros_like(learned_residual),
        state(),
    )
    expected_after = torch.zeros_like(learned_residual)
    expected_after[:, 0] = -0.03
    assert torch.allclose(after.residual, expected_after, atol=1e-6)


def test_unsafe_tilt_pauses_and_requires_stable_window_to_resume():
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2), stable_steps_to_resume=3))
    unsafe = filt.project(torch.ones(1, 12), torch.zeros(1, 12), state(tilt=0.8))
    assert unsafe.paused.item()
    assert torch.equal(unsafe.action, torch.ones(1, 12))
    assert torch.equal(unsafe.residual, torch.zeros(1, 12))
    for _ in range(2):
        assert filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state()).paused.item()
    assert not filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state()).paused.item()


def test_control_timeout_pauses_only_the_current_step():
    filt = SafetyFilter(
        SafetyLimits(residual_limits=torch.full((12,), 0.2), stable_steps_to_resume=3)
    )
    base_action = torch.ones(1, 12)

    timed_out = filt.project(
        base_action,
        torch.full((1, 12), 0.1),
        state(control_timed_out=True),
    )
    recovered = filt.project(base_action, torch.full((1, 12), 0.1), state())

    assert timed_out.paused.item()
    assert "control_timeout" in timed_out.reasons
    assert torch.equal(timed_out.action, base_action)
    assert not recovered.paused.item()
    assert recovered.reasons == ()


def test_control_timeout_does_not_clear_a_physical_safety_pause():
    filt = SafetyFilter(
        SafetyLimits(residual_limits=torch.full((12,), 0.2), stable_steps_to_resume=3)
    )
    base_action = torch.zeros(1, 12)

    assert filt.project(base_action, base_action, state(tilt=0.8)).paused.item()
    assert filt.project(
        base_action,
        base_action,
        state(control_timed_out=True),
    ).paused.item()
    for _ in range(2):
        assert filt.project(base_action, base_action, state()).paused.item()
    assert not filt.project(base_action, base_action, state()).paused.item()


def test_transient_nonfoot_contact_uses_a_short_grace_window():
    filt = SafetyFilter(
        SafetyLimits(residual_limits=torch.full((12,), 0.2), nonfoot_contact_grace_steps=2)
    )
    assert not filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state(nonfoot_contact=True)).paused.item()
    assert not filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state(nonfoot_contact=True)).paused.item()
    decision = filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state(nonfoot_contact=True))
    assert decision.paused.item()
    assert "nonfoot_contact" in decision.reasons


def test_transient_nonfoot_contact_disables_residual_before_pause():
    filt = SafetyFilter(
        SafetyLimits(residual_limits=torch.full((12,), 0.2), nonfoot_contact_grace_steps=2)
    )
    decision = filt.project(torch.ones(1, 12), torch.full((1, 12), 0.1), state(nonfoot_contact=True))
    assert not decision.paused.item()
    assert torch.equal(decision.action, torch.ones(1, 12))
    assert torch.equal(decision.residual, torch.zeros(1, 12))


def test_high_tilt_margin_disables_residual_before_pause():
    filt = SafetyFilter(
        SafetyLimits(residual_limits=torch.full((12,), 0.2), max_projected_gravity_xy=0.65)
    )
    decision = filt.project(torch.ones(1, 12), torch.full((1, 12), 0.1), state(tilt=0.56))
    assert not decision.paused.item()
    assert torch.equal(decision.action, torch.ones(1, 12))
    assert torch.equal(decision.residual, torch.zeros(1, 12))


def test_safety_defaults_share_lite3_training_contract():
    limits = SafetyLimits(residual_limits=torch.ones(12))

    assert limits.max_projected_gravity_xy == LITE3_MAX_PROJECTED_GRAVITY_XY
    assert (
        limits.residual_guard_projected_gravity_xy
        == LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY
    )


def test_reset_temporal_state_clears_safety_history():
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2)))
    filt.project(torch.zeros(1, 12), torch.ones(1, 12), state())
    filt.reset_temporal_state()
    assert filt._previous_residual is None
    assert filt._paused is None
    assert filt._stable_steps is None
    assert filt._contact_steps is None


def test_batched_reason_transfer_preserves_order_and_union_across_environments():
    from dataclasses import fields
    safe = state()
    values = {f.name: torch.cat([getattr(safe, f.name)] * 5, dim=0) for f in fields(safe)}
    values['projected_gravity_xy'][0, 0] = .9
    values['base_height'][1, 0] = .1
    values['joint_vel'][2, 0] = 31.
    values['nonfoot_contact'][3, 0] = True
    values['control_timed_out'][4, 0] = True
    filt = SafetyFilter(SafetyLimits(torch.full((12,), .2), nonfoot_contact_grace_steps=0))
    result = filt.project(torch.zeros(5, 12),torch.ones(5, 12),SafetyState(**values))
    assert result.reasons == ('tilt','height','joint_velocity','nonfoot_contact','control_timeout')
    assert result.paused.all()
    assert torch.count_nonzero(result.residual) == 0
