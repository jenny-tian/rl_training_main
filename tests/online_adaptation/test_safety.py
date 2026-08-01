import torch

from rl_training.lite3_stability_contract import (
    LITE3_MAX_PROJECTED_GRAVITY_XY,
    LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY,
)
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.isaaclab_adapter import select_joint_order
from rl_training.online_adaptation.types import SafetyState


def state(tilt=0.0, height=0.35, nonfoot_contact=False):
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
        control_timed_out=torch.zeros(1, 1, dtype=torch.bool),
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
