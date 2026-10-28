"""Per-environment Source terrain bounds checks that run without Isaac Sim."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence


def source_tile_bounds_terminated(
    body_positions_w: torch.Tensor,
    body_linear_velocities_w: torch.Tensor,
    terrain_origins: torch.Tensor,
    terrain_levels: torch.Tensor,
    terrain_types: torch.Tensor,
    *,
    size: Sequence[float],
    flat_length_m: float,
    collision_radius_m: float,
    numerical_margin_m: float,
    control_period_s: float,
) -> torch.Tensor:
    """Return one fail-closed tile-bound termination flag per environment.

    ``terrain_origins[terrain_levels, terrain_types]`` is the robot reset origin.
    The track occupies local x ``[-flat_length/2, size_x-flat_length/2]`` and
    local y ``[-size_y/2, size_y/2]`` around that origin. Every rigid-body
    frame is tested with a spherical collision envelope, then a one-control-
    period linear position prediction is checked as an early warning. The
    prediction is not a continuous-time proof when acceleration is unbounded.
    Invalid indices and non-finite state terminate only their corresponding
    environment.
    """

    if len(size) != 2:
        raise ValueError("size must contain track length and width")
    size_x, size_y = (float(size[0]), float(size[1]))
    scalars = {
        "size_x": size_x,
        "size_y": size_y,
        "flat_length_m": float(flat_length_m),
        "collision_radius_m": float(collision_radius_m),
        "numerical_margin_m": float(numerical_margin_m),
        "control_period_s": float(control_period_s),
    }
    if any(not math.isfinite(value) for value in scalars.values()):
        raise ValueError("tile geometry, margins, and control period must be finite")
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError("tile size must be positive")
    if not 0.0 < scalars["flat_length_m"] <= size_x:
        raise ValueError("flat_length_m must be in (0, size_x]")
    if scalars["collision_radius_m"] <= 0.0:
        raise ValueError("collision_radius_m must be positive")
    if scalars["numerical_margin_m"] < 0.0:
        raise ValueError("numerical_margin_m must be nonnegative")
    if scalars["control_period_s"] <= 0.0:
        raise ValueError("control_period_s must be positive")

    margin = scalars["collision_radius_m"] + scalars["numerical_margin_m"]
    if margin >= min(size_x / 2.0, size_y / 2.0):
        raise ValueError("collision margin leaves no valid interior tile region")

    tensors = (
        body_positions_w,
        body_linear_velocities_w,
        terrain_origins,
        terrain_levels,
        terrain_types,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("body state and terrain allocation inputs must be torch tensors")
    if body_positions_w.ndim != 3 or body_positions_w.shape[-1] < 2:
        raise ValueError("body_positions_w must have shape [env, body, >=2]")
    if body_positions_w.shape[1] == 0:
        raise ValueError("at least one rigid-body position is required")
    if body_linear_velocities_w.shape != body_positions_w.shape:
        raise ValueError("body velocity shape must match body position shape")
    if terrain_origins.ndim != 3 or terrain_origins.shape[-1] < 2:
        raise ValueError("terrain_origins must have shape [row, column, >=2]")
    env_count = body_positions_w.shape[0]
    if terrain_levels.shape != (env_count,) or terrain_types.shape != (env_count,):
        raise ValueError("terrain row/column assignments must contain one index per environment")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if terrain_levels.dtype not in integer_dtypes or terrain_types.dtype not in integer_dtypes:
        raise ValueError("terrain row/column assignments must use integer tensors")

    device = body_positions_w.device
    levels = terrain_levels.to(device=device, dtype=torch.long)
    types = terrain_types.to(device=device, dtype=torch.long)
    origins = terrain_origins.to(device=device, dtype=body_positions_w.dtype)
    row_count, column_count = origins.shape[:2]
    valid_indices = (
        (levels >= 0)
        & (levels < row_count)
        & (types >= 0)
        & (types < column_count)
    )
    safe_levels = levels.clamp(0, row_count - 1)
    safe_types = types.clamp(0, column_count - 1)
    reset_origins = origins[safe_levels, safe_types, :2]

    lower = reset_origins.new_tensor((-scalars["flat_length_m"] / 2.0, -size_y / 2.0))
    upper = reset_origins.new_tensor((size_x - scalars["flat_length_m"] / 2.0, size_y / 2.0))
    safe_lower = reset_origins + lower + margin
    safe_upper = reset_origins + upper - margin

    positions = body_positions_w[..., :2]
    velocities = body_linear_velocities_w[..., :2]
    next_positions = positions + velocities * scalars["control_period_s"]
    lower_per_body = safe_lower.unsqueeze(1)
    upper_per_body = safe_upper.unsqueeze(1)

    outside_now = (
        (positions < lower_per_body) | (positions > upper_per_body)
    ).any(dim=(1, 2))
    outside_prediction = (
        (next_positions < lower_per_body) | (next_positions > upper_per_body)
    ).any(dim=(1, 2))
    invalid_state = (
        ~torch.isfinite(positions).all(dim=(1, 2))
        | ~torch.isfinite(velocities).all(dim=(1, 2))
        | ~torch.isfinite(reset_origins).all(dim=1)
    )
    return ~valid_indices | invalid_state | outside_now | outside_prediction


__all__ = ["source_tile_bounds_terminated"]
