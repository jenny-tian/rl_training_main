from dataclasses import replace

import torch

from rl_training.online_adaptation.metrics import RollingMetrics
from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
from rl_training.online_adaptation.types import RewardSignals


def signals(error=0.0, tilt=0.0, batch=1, foot_height=0.022, foot_contact=1.0):
    return RewardSignals(
        base_lin_vel_xy=torch.tensor([[1.0 - error, 0.0]]).expand(batch, -1).clone(),
        command_lin_vel_xy=torch.tensor([[1.0, 0.0]]).expand(batch, -1).clone(),
        base_ang_vel_z=torch.zeros(batch, 1),
        command_ang_vel_z=torch.zeros(batch, 1),
        projected_gravity_xy=torch.tensor([[tilt, 0.0]]).expand(batch, -1).clone(),
        base_lin_vel_z=torch.zeros(batch, 1),
        foot_slip=torch.zeros(batch, 1),
        foot_impact=torch.zeros(batch, 1),
        joint_power=torch.zeros(batch, 1),
        nonfoot_contact=torch.zeros(batch, 1),
        foot_height=torch.full((batch, 4), foot_height),
        foot_contact=torch.full((batch, 4), foot_contact),
    )


def test_reward_decreases_with_tracking_error_and_tilt():
    residual = torch.zeros(1, 12)
    good = compute_online_reward(signals(), residual, residual, RewardWeights())
    bad = compute_online_reward(signals(error=0.5, tilt=0.5), residual, residual, RewardWeights())
    assert good.item() > bad.item()


def test_reward_increases_with_normalized_forward_progress():
    residual = torch.zeros(1, 12)
    stalled = compute_online_reward(
        signals(), residual, residual, RewardWeights(), forward_progress=torch.zeros(1, 1)
    )
    moving = compute_online_reward(
        signals(), residual, residual, RewardWeights(), forward_progress=torch.ones(1, 1)
    )

    assert moving.item() - stalled.item() == 0.5


def test_stalled_progress_is_penalized_even_when_robot_remains_stable():
    residual = torch.zeros(1, 12)
    weights = RewardWeights(stall_penalty_weight=2.0)
    moving = compute_online_reward(
        signals(), residual, residual, weights,
        progress_stalled_mask=torch.zeros(1, 1),
    )
    stalled = compute_online_reward(
        signals(), residual, residual, weights,
        progress_stalled_mask=torch.ones(1, 1),
    )

    assert moving.item() - stalled.item() == 2.0


def test_post_step_progress_gets_an_additional_reward():
    residual = torch.zeros(1, 12)
    weights = RewardWeights(post_step_progress_weight=2.0)
    before = compute_online_reward(
        signals(), residual, residual, weights,
        forward_progress=torch.ones(1, 1),
        post_step_mask=torch.zeros(1, 1),
    )
    after = compute_online_reward(
        signals(), residual, residual, weights,
        forward_progress=torch.ones(1, 1),
        post_step_mask=torch.ones(1, 1),
    )

    assert after.item() > before.item()


def test_completion_receives_a_one_step_bonus():
    residual = torch.zeros(1, 12)
    incomplete = compute_online_reward(
        signals(), residual, residual, RewardWeights(),
        completion_mask=torch.zeros(1, 1),
    )
    complete = compute_online_reward(
        signals(), residual, residual, RewardWeights(),
        completion_mask=torch.ones(1, 1),
    )

    assert complete.item() > incomplete.item()


def test_reward_clamps_and_sanitizes_forward_progress():
    residual = torch.zeros(2, 12)
    values = compute_online_reward(
        signals(batch=2),
        residual,
        residual,
        RewardWeights(),
        forward_progress=torch.tensor([[-1.0e6], [float("nan")]]),
    )

    assert torch.isfinite(values).all()
    assert values[1].item() - values[0].item() == 0.5


def test_post_step_stability_penalty_only_applies_after_step_edge():
    residual = torch.zeros(1, 12)
    weights = RewardWeights(post_step_stability_weight=3.0)
    before = compute_online_reward(
        signals(tilt=0.2), residual, residual, weights,
        post_step_mask=torch.zeros(1, 1),
    )
    after = compute_online_reward(
        signals(tilt=0.2), residual, residual, weights,
        post_step_mask=torch.ones(1, 1),
    )

    assert after.item() < before.item()


def test_post_step_stability_penalty_is_stronger_for_tilt_and_vertical_motion():
    residual = torch.zeros(1, 12)
    weights = RewardWeights(post_step_stability_weight=3.0)
    stable = compute_online_reward(
        signals(), residual, residual, weights,
        post_step_mask=torch.ones(1, 1),
    )
    unstable_signals = signals(tilt=0.5)
    unstable_signals = type(unstable_signals)(
        **{**unstable_signals.__dict__, "base_lin_vel_z": torch.ones(1, 1)}
    )
    unstable = compute_online_reward(
        unstable_signals, residual, residual, weights,
        post_step_mask=torch.ones(1, 1),
    )

    assert stable.item() > unstable.item()


def test_pre_step_clearance_reward_targets_noncontact_swing_foot_height():
    residual = torch.zeros(1, 12)
    weights = RewardWeights(
        pre_step_clearance_weight=1.0,
        pre_step_clearance_target=0.06,
    )
    low = compute_online_reward(
        signals(foot_height=0.03, foot_contact=0.0),
        residual,
        residual,
        weights,
        pre_step_clearance_mask=torch.ones(1, 1),
    )
    high = compute_online_reward(
        signals(foot_height=0.06, foot_contact=0.0),
        residual,
        residual,
        weights,
        pre_step_clearance_mask=torch.ones(1, 1),
    )
    inactive = compute_online_reward(
        signals(foot_height=0.03, foot_contact=0.0),
        residual,
        residual,
        weights,
        pre_step_clearance_mask=torch.zeros(1, 1),
    )

    assert high.item() > low.item() > inactive.item()


def test_pre_step_clearance_reward_ignores_stance_feet():
    residual = torch.zeros(1, 12)
    weights = RewardWeights(pre_step_clearance_weight=1.0)
    stance = compute_online_reward(
        signals(foot_height=0.06, foot_contact=1.0),
        residual,
        residual,
        weights,
        pre_step_clearance_mask=torch.ones(1, 1),
    )
    inactive = compute_online_reward(
        signals(foot_height=0.06, foot_contact=1.0),
        residual,
        residual,
        weights,
        pre_step_clearance_mask=torch.zeros(1, 1),
    )

    torch.testing.assert_close(stance, inactive)


def test_pre_step_height_and_forward_placement_must_use_the_same_foot():
    residual = torch.zeros(1, 12)
    weights = RewardWeights(
        pre_step_clearance_weight=1.0,
        pre_step_foot_forward_weight=1.0,
        pre_step_clearance_target=0.06,
        pre_step_foot_forward_target=0.04,
    )
    split = replace(
        signals(foot_contact=0.0),
        foot_height=torch.tensor([[0.06, 0.0, 0.0, 0.0]]),
        foot_edge_offset=torch.tensor([[0.0, 0.04, 0.0, 0.0]]),
    )
    coupled = replace(
        split,
        foot_edge_offset=torch.tensor([[0.04, 0.0, 0.0, 0.0]]),
    )
    split_reward = compute_online_reward(
        split, residual, residual, weights,
        pre_step_clearance_mask=torch.ones(1, 1),
    )
    coupled_reward = compute_online_reward(
        coupled, residual, residual, weights,
        pre_step_clearance_mask=torch.ones(1, 1),
    )

    assert coupled_reward.item() > split_reward.item()


def test_pre_step_forward_score_is_dense_before_crossing_the_edge():
    residual = torch.zeros(1, 12)
    weights = RewardWeights(
        pre_step_clearance_weight=1.0,
        pre_step_foot_forward_weight=1.0,
        pre_step_clearance_target=0.06,
        pre_step_foot_forward_target=0.04,
    )
    far = replace(
        signals(foot_height=0.06, foot_contact=0.0),
        foot_edge_offset=torch.full((1, 4), -0.04),
    )
    near = replace(far, foot_edge_offset=torch.full((1, 4), -0.01))
    crossed = replace(far, foot_edge_offset=torch.full((1, 4), 0.04))

    far_reward = compute_online_reward(
        far,
        residual,
        residual,
        weights,
        pre_step_clearance_mask=torch.ones(1, 1),
    )
    near_reward = compute_online_reward(
        near,
        residual,
        residual,
        weights,
        pre_step_clearance_mask=torch.ones(1, 1),
    )
    crossed_reward = compute_online_reward(
        crossed,
        residual,
        residual,
        weights,
        pre_step_clearance_mask=torch.ones(1, 1),
    )

    assert crossed_reward.item() > near_reward.item() > far_reward.item()


def test_metrics_reports_first_and_second_half():
    metrics = RollingMetrics(window=3)
    for index in range(6):
        metrics.update(
            step=index,
            in_heldout=True,
            velocity_error=1.0 - 0.1 * index,
            tilt_rms=0.5 - 0.05 * index,
            nonfoot_contact=0,
        )
    summary = metrics.heldout_summary()
    assert summary["second_half_velocity_error"] < summary["first_half_velocity_error"]
