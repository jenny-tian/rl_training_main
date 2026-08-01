from pathlib import Path

import torch

from rl_training.lite3_stability_contract import (
    LITE3_MAX_PROJECTED_GRAVITY_XY,
    LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY,
    projected_gravity_xy_is_unsafe,
    tilt_guard_band_penalty,
)


ROOT = Path(__file__).resolve().parents[2]


def test_diagonal_tilt_uses_norm_not_individual_components():
    gravity = torch.tensor([[0.47, 0.47, -0.75]])

    assert gravity[:, :2].abs().amax().item() < 0.7
    assert projected_gravity_xy_is_unsafe(gravity).item()


def test_tilt_boundary_is_strictly_greater_than_unchanged_limit():
    gravity = torch.tensor([[0.65, 0.0, -1.0], [0.6501, 0.0, -1.0]])

    assert LITE3_MAX_PROJECTED_GRAVITY_XY == 0.65
    assert projected_gravity_xy_is_unsafe(gravity).tolist() == [False, True]


def test_guard_penalty_is_zero_then_monotonic_and_bounded():
    gravity = torch.tensor(
        [[0.55, 0.0, -1.0], [0.60, 0.0, -1.0], [0.65, 0.0, -1.0]]
    )

    penalty = tilt_guard_band_penalty(gravity)

    assert LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY == 0.55
    assert torch.allclose(penalty, torch.tensor([0.0, 0.25, 1.0]), atol=1.0e-5)


def test_lite3_config_binds_training_to_shared_stability_contract():
    source = (
        ROOT
        / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config"
        / "quadruped/deeprobotics_lite3/rough_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert 'self.terminations.bad_orientation_2.params["max_xy_norm"]' in source
    assert "LITE3_MAX_PROJECTED_GRAVITY_XY" in source
    assert 'self.rewards.tilt_guard_band.weight = -20.0' in source
    assert 'self.rewards.tilt_guard_band.params["guard_xy_norm"]' in source
    assert "LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY" in source
    assert 'self.rewards.tilt_guard_band.params["max_xy_norm"]' in source


def test_training_termination_delegates_to_xy_norm_contract_when_configured():
    source = (
        ROOT
        / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/events.py"
    ).read_text(encoding="utf-8")

    assert "projected_gravity_xy_is_unsafe(" in source
    assert "asset.data.projected_gravity_b, max_xy_norm=max_xy_norm" in source
