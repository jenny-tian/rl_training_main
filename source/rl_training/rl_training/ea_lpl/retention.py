"""Torch-only streaming metrics for the original Flat command distribution."""
from __future__ import annotations

import torch


COMMAND_CATEGORIES = ("standing", "only_x", "only_y", "only_ang", "mixed")


def categorize_commands(command: torch.Tensor) -> torch.Tensor:
    """Categorize actual [vx, vy, wz] values; even tiny nonzero values are active."""
    if command.ndim != 2 or command.shape[1] != 3:
        raise ValueError("commands must have shape [num_envs, 3]")
    if not command.is_floating_point() or not bool(torch.isfinite(command).all()):
        raise ValueError("commands must be finite floating-point values")
    active = command != 0
    count = active.sum(-1)
    category = torch.full_like(count, 4)
    category[count == 0] = 0
    for axis in range(3):
        category[(count == 1) & active[:, axis]] = axis + 1
    return category


def _mask(value: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    if value.shape != (n,) or value.device != device or not bool(((value == 0) | (value == 1)).all()):
        raise ValueError("episode flags must be binary [num_envs] tensors on the command device")
    return value.bool()


class RetentionMetrics:
    """Accumulate per-component XY MSE and yaw MSE, excluding post-reset states.

    A fall is the union of both fall terms among all ended episodes, including
    timeouts. Empty denominators and nonfinite data raise instead of reporting
    a successful zero-error or zero-fall evaluation.
    """

    def __init__(self, *, device="cpu"):
        self.counts = torch.zeros(6, dtype=torch.int64, device=device)
        self.squared_error = torch.zeros((6, 2), dtype=torch.float64, device=device)
        self.total_env_steps = self.ended_episodes = self.fall_episodes = 0

    def update(self, command, velocity_xy_body, yaw_rate_body, done, illegal_contact, bad_orientation):
        category = categorize_commands(command)
        n = len(command)
        if command.device != self.counts.device:
            raise ValueError("command device differs from metric accumulator")
        if velocity_xy_body.shape != (n, 2) or yaw_rate_body.shape != (n,):
            raise ValueError("body velocity shapes must be [num_envs, 2] and [num_envs]")
        if velocity_xy_body.device != command.device or yaw_rate_body.device != command.device:
            raise ValueError("body velocities must be on the command device")
        done, contact, orientation = [_mask(v, n, command.device) for v in (done, illegal_contact, bad_orientation)]
        falls = contact | orientation
        if bool((falls & ~done).any()):
            raise ValueError("fall term fired without an ended episode")
        valid = ~done
        xy = velocity_xy_body[valid].double() - command[valid, :2].double()
        yaw = yaw_rate_body[valid].double() - command[valid, 2].double()
        errors = torch.stack((xy.square().sum(-1), yaw.square()), dim=-1)
        if not bool(torch.isfinite(errors).all()):
            raise FloatingPointError("nonfinite tracking errors")
        groups = category[valid] + 1
        self.counts[0] += valid.sum()
        self.squared_error[0] += errors.sum(0)
        self.counts.index_add_(0, groups, torch.ones_like(groups))
        self.squared_error.index_add_(0, groups, errors)
        self.total_env_steps += n
        self.ended_episodes += int(done.sum())
        self.fall_episodes += int((falls & done).sum())

    def summary(self):
        if bool((self.counts == 0).any()) or self.ended_episodes == 0:
            raise ValueError("retention metric denominator is zero (overall, category, or ended episodes)")
        if not bool(torch.isfinite(self.squared_error).all()):
            raise FloatingPointError("nonfinite accumulated tracking errors")
        rms = (self.squared_error / self.counts[:, None] / self.squared_error.new_tensor([2, 1])).sqrt()

        def values(i):
            return dict(tracking_samples=int(self.counts[i]), rms_xy_error=float(rms[i, 0]),
                        rms_yaw_error=float(rms[i, 1]))

        return dict(**values(0), total_env_steps=self.total_env_steps,
                    excluded_reset_samples=self.total_env_steps - int(self.counts[0]),
                    ended_episodes=self.ended_episodes, fall_episodes=self.fall_episodes,
                    falls_per_episode=self.fall_episodes / self.ended_episodes,
                    categories={name: values(i + 1) for i, name in enumerate(COMMAND_CATEGORIES)})
