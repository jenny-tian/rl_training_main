import math
from types import SimpleNamespace

import pytest
import torch

from rl_training.ea_lpl.cfg_checks import check_heading_hold, heading_yaw_rate_command


def command_cfg(**overrides):
    ranges = SimpleNamespace(lin_vel_x=(0.3, 0.3), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.8, 0.8), heading=(0.0, 0.0))
    values = dict(heading_command=True, heading_control_stiffness=0.5, rel_heading_envs=1.0, rel_standing_envs=0.0,
                  rel_zero_vel_envs=0.0, rel_only_lin_y_envs=0.0, rel_only_lin_x_envs=0.0, rel_only_ang_z_envs=0.0,
                  resampling_time_range=(1.0e9, 1.0e9), ranges=ranges)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_heading_hold_accepts_registered_settings():
    check_heading_hold(command_cfg(), enabled=True)


def test_heading_hold_rejects_zero_yaw_range_and_pure_forward_slot():
    cfg = command_cfg()
    cfg.ranges.ang_vel_z = (0.0, 0.0)
    with pytest.raises(ValueError, match="ang_vel_z"):
        check_heading_hold(cfg, enabled=True)
    with pytest.raises(ValueError, match="rel_only_lin_x_envs"):
        check_heading_hold(command_cfg(rel_only_lin_x_envs=1.0), enabled=True)


def test_disabled_heading_matches_source_baseline():
    cfg = command_cfg(heading_command=False, rel_heading_envs=0.0, rel_only_lin_x_envs=1.0)
    cfg.ranges.ang_vel_z = (0.0, 0.0)
    check_heading_hold(cfg, enabled=False)


def test_yaw_rate_command_formula():
    heading = torch.tensor([0.0, 0.2, -0.2, 3.0, -3.0])
    command = heading_yaw_rate_command(heading)
    assert torch.allclose(command[:3], torch.tensor([0.0, -0.1, 0.1]))
    assert math.isclose(float(command[3]), -0.8, rel_tol=1e-6)
    assert math.isclose(float(command[4]), 0.8, rel_tol=1e-6)
