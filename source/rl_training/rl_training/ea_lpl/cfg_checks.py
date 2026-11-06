"""Configuration checks that need no simulator import."""
from __future__ import annotations

import torch


def check_heading_hold(command_cfg, *, enabled, stiffness=0.5, max_yaw_rate=0.8):
    """Reject settings under which Isaac Lab would clip or disable the heading loop."""
    common = dict(rel_standing_envs=0.0, rel_zero_vel_envs=0.0, rel_only_lin_y_envs=0.0, rel_only_ang_z_envs=0.0)
    if enabled:
        expected = dict(common, heading_command=True, heading_control_stiffness=stiffness,
                        rel_heading_envs=1.0, rel_only_lin_x_envs=0.0)
        yaw_range = (-max_yaw_rate, max_yaw_rate)
    else:
        expected = dict(common, heading_command=False, rel_heading_envs=0.0, rel_only_lin_x_envs=1.0)
        yaw_range = (0.0, 0.0)
    for name, value in expected.items():
        actual = getattr(command_cfg, name)
        if actual != value:
            raise ValueError(f"heading setting {name}={actual!r}, expected {value!r}")
    ranges = command_cfg.ranges
    for name, value in dict(lin_vel_x=(0.3, 0.3), lin_vel_y=(0.0, 0.0), ang_vel_z=yaw_range,
                            heading=(0.0, 0.0)).items():
        actual = tuple(getattr(ranges, name))
        if actual != tuple(value):
            raise ValueError(f"command range {name}={actual!r}, expected {tuple(value)!r}")
    if tuple(command_cfg.resampling_time_range) != (1.0e9, 1.0e9):
        raise ValueError("commands must not resample inside an episode")


def heading_yaw_rate_command(heading, *, stiffness=0.5, max_rate=0.8):
    """Isaac Lab heading loop with target heading 0: clip(k * wrap(0 - heading))."""
    error = torch.atan2(torch.sin(-heading), torch.cos(-heading))
    return (stiffness * error).clamp(-max_rate, max_rate)
