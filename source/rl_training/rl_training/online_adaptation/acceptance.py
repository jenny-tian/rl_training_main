from __future__ import annotations

from typing import Any


def _relative_improvement(first: float, second: float) -> float:
    if first <= 0.0:
        return 0.0
    return (first - second) / first


def evaluate_run(summary: dict[str, Any]) -> dict[str, Any]:
    first_velocity = float(summary["first_half_velocity_error"])
    second_velocity = float(summary["second_half_velocity_error"])
    first_tilt = float(summary["first_half_tilt_rms"])
    second_tilt = float(summary["second_half_tilt_rms"])
    velocity_improvement = _relative_improvement(first_velocity, second_velocity)
    tilt_improvement = _relative_improvement(first_tilt, second_tilt)
    checks = {
        "no_reset": int(summary["reset_count"]) == 0,
        "no_fall": not bool(summary["fell"]),
        "velocity_improvement": velocity_improvement >= 0.30,
        "tilt_improvement": tilt_improvement >= 0.20,
        "nonfoot_contacts_not_increased": float(summary["second_half_nonfoot_contacts"])
        <= float(summary["first_half_nonfoot_contacts"]),
        "parameter_changed": bool(summary["parameter_changed"]),
        "safety_compliance": not bool(summary["safety_violation"]),
    }
    return {
        "passed": all(checks.values()),
        "velocity_improvement": velocity_improvement,
        "tilt_improvement": tilt_improvement,
        "checks": checks,
    }
