"""Plane configuration and comparable metrics for frozen flat retention probes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


FLAT_RETENTION_SCHEMA = "plane_fixed_command_steady_velocity_v1"


def configure_flat_retention_plane(env_cfg: Any) -> dict[str, object]:
    """Keep the task's robot and safety rules while replacing its terrain.

    Isaac Lab's TerrainImporter handles ``plane`` without a generator and its
    RayCaster explicitly supports the resulting PhysX infinite plane. Keep
    the existing pre-reset capture environment and material configuration.
    """
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.terminations.time_out = None
    env_cfg.terminations.terrain_out_of_bounds = None
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.terrain_levels = None
    return {"terrain_type": "plane", "terrain_generator": None}


def summarize_flat_retention(
    rows: Sequence[Mapping[str, Any]],
    *,
    initial_x: float,
    step_dt: float,
    max_steps: int,
    fell: bool,
    safety_violation: bool,
    reset_count: int,
    warmup_seconds: float = 5.0,
) -> dict[str, object]:
    """Measure the first physical trial, excluding the initial five seconds.

    Velocity MAE uses body-frame command errors. ``steady_velocity_xy_mae``
    averages absolute errors over samples and both coordinates; per-axis MAE
    and mean vector L2 error are also reported. Progress uses world x and the
    position at the warmup boundary, including its first following interval.
    Comparison eligibility is a data/safety prerequisite, not a retention pass.
    """
    if not math.isfinite(initial_x) or not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("initial_x and positive step_dt must be finite")
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0.0:
        raise ValueError("warmup_seconds must be finite and non-negative")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    trajectory = []
    # Terminal capture supplies the position before the first reset. Never
    # combine its displacement with positions from a subsequently reset trial.
    for index, row in enumerate(rows):
        if row["step"] != index + 1:
            raise ValueError("flat retention rows must form a contiguous trial")
        time_s = float(row["step"]) * step_dt
        x = float(row["base_x"])
        velocity, command = row["base_lin_vel_xy"], row["command_lin_vel_xy"]
        if len(velocity) != 2 or len(command) != 2:
            raise ValueError("flat retention velocity and command must each have two coordinates")
        dx, dy = float(velocity[0]) - float(command[0]), float(velocity[1]) - float(command[1])
        if not all(math.isfinite(value) for value in (time_s, x, dx, dy)):
            raise ValueError("flat retention samples contain non-finite values")
        trajectory.append((time_s, x, abs(dx), abs(dy)))
        if row.get("done", False):
            break

    steady = [row for row in trajectory if row[0] > warmup_seconds]
    boundary_x = None
    previous_time, previous_x = 0.0, float(initial_x)
    for time_s, x, _, _ in trajectory:
        if time_s >= warmup_seconds:
            fraction = (warmup_seconds - previous_time) / (time_s - previous_time)
            boundary_x = previous_x + fraction * (x - previous_x)
            break
        previous_time, previous_x = time_s, x

    sample_count = len(trajectory)
    steady_count = len(steady)
    available = steady_count > 0
    full_budget = sample_count == max_steps and len(rows) == max_steps
    final_x = trajectory[-1][1] if trajectory else float(initial_x)
    x_mae = sum(row[2] for row in steady) / steady_count if available else None
    y_mae = sum(row[3] for row in steady) / steady_count if available else None
    return {
        "schema": FLAT_RETENTION_SCHEMA,
        "warmup_seconds": warmup_seconds,
        "step_dt_seconds": step_dt,
        "sample_count": sample_count,
        "expected_sample_count": max_steps,
        "steady_sample_count": steady_count,
        "steady_duration_seconds": trajectory[-1][0] - warmup_seconds if available else 0.0,
        "steady_data_available": available,
        "velocity_frame": "body",
        "velocity_units": "m/s",
        "steady_velocity_x_mae": x_mae,
        "steady_velocity_y_mae": y_mae,
        "steady_velocity_xy_mae": (x_mae + y_mae) / 2.0 if available else None,
        "steady_velocity_xy_l2_mean": (
            sum(math.hypot(row[2], row[3]) for row in steady) / steady_count if available else None
        ),
        "initial_x": float(initial_x),
        "final_x": final_x,
        "net_forward_m": final_x - float(initial_x),
        "steady_start_x": boundary_x,
        "steady_net_forward_m": final_x - boundary_x if available and boundary_x is not None else None,
        "full_sample_budget": full_budget,
        "post_reset_samples_excluded": len(rows) - sample_count,
        "comparison_eligible": bool(
            available and full_budget and not fell and not safety_violation and reset_count == 0
        ),
    }


__all__ = ["FLAT_RETENTION_SCHEMA", "configure_flat_retention_plane", "summarize_flat_retention"]
