from __future__ import annotations

import torch


LITE3_MAX_PROJECTED_GRAVITY_XY = 0.65
LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY = 0.55


def sustained_contact_confirmation(
    contact: torch.Tensor,
    previous_count: torch.Tensor,
    required_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confirm contact only after it persists for whole control steps."""
    if required_steps < 1:
        raise ValueError("required_steps must be positive")
    contact = contact.bool()
    if contact.shape != previous_count.shape:
        raise ValueError("contact and previous_count must have matching shapes")
    next_count = torch.where(contact, previous_count + 1, torch.zeros_like(previous_count))
    return next_count, next_count >= required_steps


def projected_gravity_xy_is_unsafe(
    projected_gravity: torch.Tensor,
    *,
    max_xy_norm: float = LITE3_MAX_PROJECTED_GRAVITY_XY,
) -> torch.Tensor:
    return (projected_gravity[..., 2] > 0) | (
        projected_gravity[..., :2].norm(dim=-1) > max_xy_norm
    )


def tilt_guard_band_penalty(
    projected_gravity: torch.Tensor,
    *,
    guard_xy_norm: float = LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY,
    max_xy_norm: float = LITE3_MAX_PROJECTED_GRAVITY_XY,
) -> torch.Tensor:
    if not 0.0 <= guard_xy_norm < max_xy_norm:
        raise ValueError("tilt guard must satisfy 0 <= guard < maximum")
    normalized = (projected_gravity[..., :2].norm(dim=-1) - guard_xy_norm) / (
        max_xy_norm - guard_xy_norm
    )
    return normalized.clamp(0.0, 1.0).square()
