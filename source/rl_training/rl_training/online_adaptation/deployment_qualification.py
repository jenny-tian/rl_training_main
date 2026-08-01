from __future__ import annotations

import math
from typing import Any


_COUNT_FIELDS = (
    "attempted_rows",
    "completed_episodes",
    "timeouts",
    "falls",
    "premature_terminations",
    "paused_rows",
    "projected_rows",
    "nonfinite_rows",
)


def _count(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"deployment qualification {field} must be an exact nonnegative integer")
    return value


def _finite_float(record: dict[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"deployment qualification {field} must be finite")
    return float(value)


def build_deployment_qualification_report(
    *,
    task: str,
    base_checkpoint: str,
    record: dict[str, Any],
    expected_seed: int,
    expected_steps: int,
    expected_episodes: int,
    max_mean_velocity_error: float,
    max_projected_gravity_xy: float,
) -> dict[str, Any]:
    if record.get("task") != task:
        raise ValueError("deployment qualification record task does not match task argument")
    if type(record.get("seed")) is not int or record["seed"] != expected_seed:
        raise ValueError("deployment qualification record seed does not match expected seed")
    counts = {field: _count(record, field) for field in _COUNT_FIELDS}
    max_tilt = _finite_float(record, "max_projected_gravity_xy")
    min_height = _finite_float(record, "min_base_height")
    velocity_error = _finite_float(record, "mean_velocity_error")
    if min_height < 0.0:
        raise ValueError("deployment qualification min_base_height must be nonnegative")
    passed = (
        counts["attempted_rows"] == expected_steps
        and counts["completed_episodes"] == expected_episodes
        and counts["timeouts"] == expected_episodes
        and all(counts[field] == 0 for field in _COUNT_FIELDS[3:])
        and max_tilt <= max_projected_gravity_xy
        and velocity_error <= max_mean_velocity_error
    )
    return {
        "task": task,
        "base_checkpoint": base_checkpoint,
        "thresholds": {
            "expected_seed": expected_seed,
            "expected_steps": expected_steps,
            "expected_episodes": expected_episodes,
            "max_mean_velocity_error": max_mean_velocity_error,
            "max_projected_gravity_xy": max_projected_gravity_xy,
        },
        "record": dict(record),
        "passed": passed,
    }
