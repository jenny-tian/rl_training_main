from __future__ import annotations

from typing import Any, Sequence


def build_qualification_report(
    task: str,
    base_checkpoint: str,
    records: Sequence[dict[str, Any]],
    max_mean_velocity_error: float,
) -> dict[str, Any]:
    if not records:
        raise ValueError("qualification requires at least one seed record")
    aggregate_error = sum(float(record["mean_velocity_error"]) for record in records) / len(records)
    aggregate_falls = sum(int(record["falls"]) for record in records)
    passed = aggregate_falls == 0 and aggregate_error <= max_mean_velocity_error
    return {
        "task": task,
        "base_checkpoint": base_checkpoint,
        "records": list(records),
        "aggregate": {
            "falls": aggregate_falls,
            "mean_velocity_error": aggregate_error,
            "max_mean_velocity_error": max_mean_velocity_error,
            "passed": passed,
        },
    }
