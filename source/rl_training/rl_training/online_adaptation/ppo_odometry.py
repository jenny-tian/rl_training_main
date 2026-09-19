"""Planar dead reckoning from measured body velocity, without world pose."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class PlanarProprioOdometry:
    """Integrate planar body velocity in a frame fixed at the episode start.

    Heading starts at zero, matching the task's fixed spawn orientation. Body
    angular velocity z is only an approximation to heading rate under tilt;
    velocity errors and finite-step turning errors accumulate. These estimates
    are not world-pose truth and use no terrain or position measurements.

    Returned context is [sin(heading), cos(heading) - 1, relative_y / scale].
    Only its last component is clamped to [-2, 2]; diagnostic displacement is
    retained without clipping. All public tensor results own their storage.
    """

    def __init__(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        lateral_scale: float = 1.65,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("dtype must be a supported real floating-point torch dtype")
        self.batch_size = batch_size
        self.dtype = dtype
        self.lateral_scale = _positive_real(lateral_scale, "lateral_scale")
        self._heading = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        self._relative_xy = torch.zeros((batch_size, 2), device=device, dtype=dtype)
        self.device = self._heading.device
        self._lateral_scale = self._heading.new_tensor(self.lateral_scale)
        if not bool((torch.isfinite(self._lateral_scale) & (self._lateral_scale > 0.0)).item()):
            raise ValueError("lateral_scale must be finite and positive in the odometry dtype")

    def _measurement(self, value: Tensor, width: int, name: str) -> Tensor:
        if not isinstance(value, Tensor) or value.shape != (self.batch_size, width):
            raise ValueError(f"{name} must have shape [{self.batch_size}, {width}]")
        if value.device != self.device or not value.is_floating_point():
            raise ValueError(f"{name} must be a real floating-point tensor on {self.device}")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain finite values")
        result = value.detach().to(dtype=self.dtype)
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError(f"{name} must be finite in the odometry dtype")
        return result

    def _reset_mask(self, mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        if not isinstance(mask, Tensor) or mask.shape not in ((self.batch_size,), (self.batch_size, 1)):
            raise ValueError(f"reset mask must have shape [{self.batch_size}] or [{self.batch_size}, 1]")
        if mask.device != self.device or mask.is_complex():
            raise ValueError(f"reset mask must be a real tensor on {self.device}")
        if not bool((torch.isfinite(mask) & ((mask == 0) | (mask == 1))).all().item()):
            raise ValueError("reset mask must contain finite binary values")
        return mask.detach().reshape(self.batch_size, 1).bool()

    def _context(self, heading: Tensor, relative_xy: Tensor) -> Tensor:
        return torch.cat(
            (heading.sin(), heading.cos() - 1.0, (relative_xy[:, 1:2] / self._lateral_scale).clamp(-2.0, 2.0)),
            dim=-1,
        )

    @torch.no_grad()
    def observe(
        self,
        body_lin_vel_xy: Tensor,
        body_ang_vel_z: Tensor,
        *,
        step_dt: float,
        reset_mask: Tensor | None = None,
    ) -> Tensor:
        """Integrate one measured interval with midpoint heading rotation.

        Optional reset_mask clears marked environments AFTER this interval,
        returning context for their new episode. To retain terminal context,
        call observe on pre-reset measurements without a mask, then reset(done).
        Invalid input or non-finite integration leaves every environment intact.
        """
        velocity = self._measurement(body_lin_vel_xy, 2, "body_lin_vel_xy")
        angular_velocity = self._measurement(body_ang_vel_z, 1, "body_ang_vel_z")
        dt = self._heading.new_tensor(_positive_real(step_dt, "step_dt"))
        if not bool((torch.isfinite(dt) & (dt > 0.0)).item()):
            raise ValueError("step_dt must be finite and positive in the odometry dtype")
        mask = self._reset_mask(reset_mask)

        heading_delta = angular_velocity * dt
        midpoint = self._heading + 0.5 * heading_delta
        vx, vy = velocity[:, :1], velocity[:, 1:2]
        cos_midpoint, sin_midpoint = midpoint.cos(), midpoint.sin()
        displacement = dt * torch.cat(
            (cos_midpoint * vx - sin_midpoint * vy, sin_midpoint * vx + cos_midpoint * vy), dim=-1
        )
        relative_xy = self._relative_xy + displacement
        unwrapped_heading = self._heading + heading_delta
        heading = torch.atan2(unwrapped_heading.sin(), unwrapped_heading.cos())
        if not bool(torch.isfinite(heading).all().item()) or not bool(torch.isfinite(relative_xy).all().item()):
            raise ValueError("odometry integration produced non-finite state")
        if mask is not None:
            heading = torch.where(mask, torch.zeros_like(heading), heading)
            relative_xy = torch.where(mask, torch.zeros_like(relative_xy), relative_xy)
        context = self._context(heading, relative_xy)
        self._heading, self._relative_xy = heading, relative_xy
        return context

    @torch.no_grad()
    def context(self) -> Tensor:
        """Return a detached context without updating the odometry."""
        return self._context(self._heading, self._relative_xy)

    @torch.no_grad()
    def reset(self, mask: Tensor | None = None) -> None:
        """Clear all environments, or the selected finite binary mask rows."""
        selected = self._reset_mask(mask)
        heading, relative_xy = torch.zeros_like(self._heading), torch.zeros_like(self._relative_xy)
        if selected is not None:
            heading = torch.where(selected, heading, self._heading)
            relative_xy = torch.where(selected, relative_xy, self._relative_xy)
        self._heading, self._relative_xy = heading, relative_xy

    def snapshot(self) -> dict[str, Tensor]:
        """Return independent heading [N, 1] and unclipped displacement [N, 2]."""
        return {"heading": self._heading.detach().clone(), "relative_xy": self._relative_xy.detach().clone()}


__all__ = ["PlanarProprioOdometry"]
