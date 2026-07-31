from __future__ import annotations

from typing import Any, Sequence


_COUNT_FIELDS = ("falls", "paused_rows", "projected_rows", "nonfinite_rows")


def _validated_count(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"qualification record {field} must be an exact nonnegative integer")
    return value


def build_qualification_report(
    task: str,
    base_checkpoint: str,
    records: Sequence[dict[str, Any]],
    max_mean_velocity_error: float,
) -> dict[str, Any]:
    if not records:
        raise ValueError("qualification requires at least one seed record")
    aggregate_error = sum(float(record["mean_velocity_error"]) for record in records) / len(records)
    aggregate_counts = {
        field: sum(_validated_count(record, field) for record in records)
        for field in _COUNT_FIELDS
    }
    passed = all(count == 0 for count in aggregate_counts.values()) and (
        aggregate_error <= max_mean_velocity_error
    )
    return {
        "task": task,
        "base_checkpoint": base_checkpoint,
        "records": list(records),
        "aggregate": {
            **aggregate_counts,
            "mean_velocity_error": aggregate_error,
            "max_mean_velocity_error": max_mean_velocity_error,
            "passed": passed,
        },
    }
