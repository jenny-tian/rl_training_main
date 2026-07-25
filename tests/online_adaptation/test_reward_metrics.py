import torch

from rl_training.online_adaptation.metrics import RollingMetrics
from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
from rl_training.online_adaptation.types import RewardSignals


def signals(error=0.0, tilt=0.0):
    return RewardSignals(
        base_lin_vel_xy=torch.tensor([[1.0 - error, 0.0]]),
        command_lin_vel_xy=torch.tensor([[1.0, 0.0]]),
        base_ang_vel_z=torch.zeros(1, 1),
        command_ang_vel_z=torch.zeros(1, 1),
        projected_gravity_xy=torch.tensor([[tilt, 0.0]]),
        base_lin_vel_z=torch.zeros(1, 1),
        foot_slip=torch.zeros(1, 1),
        foot_impact=torch.zeros(1, 1),
        joint_power=torch.zeros(1, 1),
        nonfoot_contact=torch.zeros(1, 1),
    )


def test_reward_decreases_with_tracking_error_and_tilt():
    residual = torch.zeros(1, 12)
    good = compute_online_reward(signals(), residual, residual, RewardWeights())
    bad = compute_online_reward(signals(error=0.5, tilt=0.5), residual, residual, RewardWeights())
    assert good.item() > bad.item()


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
