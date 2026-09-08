"""Torch reference sampling for motion-prior reward terms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .schema import MotionTrajectory


def prior_weight(mode: str, step: int | float, decay_steps: int | float, weight: float) -> float:
    """Return the configured scalar prior weight for an episode/training step."""
    if mode == "on":
        return float(weight)
    if mode == "off":
        return 0.0
    if mode == "decay":
        if decay_steps <= 0:
            raise ValueError("decay_steps must be positive for decay mode")
        return float(weight) * max(0.0, 1.0 - float(step) / float(decay_steps))
    raise ValueError("prior mode must be one of: on, decay, off")


@dataclass(frozen=True)
class MotionPriorSample:
    joint_pos: torch.Tensor
    foot_pos_b: torch.Tensor
    contact: torch.Tensor
    command: torch.Tensor | None
    joint_vel: torch.Tensor | None
    base_lin_vel_b: torch.Tensor | None
    base_ang_vel_b: torch.Tensor | None
    projected_gravity_b: torch.Tensor | None
    metadata: Mapping[str, Any]


class MotionPriorManager:
    """Load one validated reference and interpolate it on a Torch device."""

    def __init__(self, path: str | Path, device: str | torch.device = "cpu") -> None:
        trajectory = MotionTrajectory.load(path)
        self.path = Path(path).expanduser().resolve()
        self.device = torch.device(device)
        self._time = torch.as_tensor(trajectory.time, dtype=torch.float32, device=self.device)
        self._joint_pos = torch.as_tensor(trajectory.joint_pos, dtype=torch.float32, device=self.device)
        self._foot_pos_b = torch.as_tensor(trajectory.foot_pos_b, dtype=torch.float32, device=self.device)
        self._contact = torch.as_tensor(trajectory.contact, dtype=torch.float32, device=self.device)
        self._command = (
            torch.as_tensor(trajectory.command, dtype=torch.float32, device=self.device)
            if trajectory.command is not None
            else None
        )
        self._joint_vel = self._optional_tensor(trajectory.joint_vel)
        self._base_lin_vel_b = self._optional_tensor(trajectory.base_lin_vel_b)
        self._base_ang_vel_b = self._optional_tensor(trajectory.base_ang_vel_b)
        self._projected_gravity_b = self._optional_tensor(trajectory.projected_gravity_b)
        self.metadata: Mapping[str, Any] = dict(trajectory.metadata)
        self.duration = float(trajectory.time[-1] - trajectory.time[0])

    def _optional_tensor(self, value) -> torch.Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _interpolate(self, values: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
        query = torch.as_tensor(time_s, dtype=self._time.dtype, device=self.device)
        if query.ndim == 0:
            query = query.unsqueeze(0)
        if not torch.isfinite(query).all():
            raise ValueError("reference query times must be finite")
        query = torch.remainder(query - self._time[0], self.duration) + self._time[0]
        upper = torch.searchsorted(self._time, query, right=True).clamp(1, self._time.numel() - 1)
        lower = upper - 1
        alpha = ((query - self._time[lower]) / (self._time[upper] - self._time[lower])).reshape(-1, *([1] * (values.ndim - 1)))
        return values[lower] * (1.0 - alpha) + values[upper] * alpha

    def sample(self, time_s: torch.Tensor | float) -> MotionPriorSample:
        query = torch.as_tensor(time_s, dtype=torch.float32, device=self.device)
        return MotionPriorSample(
            joint_pos=self._interpolate(self._joint_pos, query),
            foot_pos_b=self._interpolate(self._foot_pos_b, query),
            contact=self._interpolate(self._contact, query),
            command=self._interpolate(self._command, query) if self._command is not None else None,
            joint_vel=self._interpolate(self._joint_vel, query) if self._joint_vel is not None else None,
            base_lin_vel_b=(
                self._interpolate(self._base_lin_vel_b, query) if self._base_lin_vel_b is not None else None
            ),
            base_ang_vel_b=(
                self._interpolate(self._base_ang_vel_b, query) if self._base_ang_vel_b is not None else None
            ),
            projected_gravity_b=(
                self._interpolate(self._projected_gravity_b, query)
                if self._projected_gravity_b is not None
                else None
            ),
            metadata=self.metadata,
        )
