from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .terrain_protocol import default_flat_online_protocol


_REQUIRED_SUMMARY_FIELDS = (
    "reset_count",
    "fell",
    "first_half_velocity_error",
    "second_half_velocity_error",
    "first_half_tilt_rms",
    "second_half_tilt_rms",
    "first_half_nonfoot_contacts",
    "second_half_nonfoot_contacts",
    "parameter_changed",
    "safety_violation",
    "actor_updates_before_novelty",
    "actor_updates_after_novelty",
    "novelty_entries",
    "rollbacks",
)


def _relative_improvement(first: float, second: float) -> float:
    if first <= 0.0:
        return 0.0
    return (first - second) / first


def evaluate_run(summary: dict[str, Any]) -> dict[str, Any]:
    if any(field not in summary for field in _REQUIRED_SUMMARY_FIELDS):
        return {
            "passed": False,
            "velocity_improvement": None,
            "tilt_improvement": None,
            "checks": {"complete_summary": False},
        }

    first_velocity = float(summary["first_half_velocity_error"])
    second_velocity = float(summary["second_half_velocity_error"])
    first_tilt = float(summary["first_half_tilt_rms"])
    second_tilt = float(summary["second_half_tilt_rms"])
    velocity_improvement = _relative_improvement(first_velocity, second_velocity)
    tilt_improvement = _relative_improvement(first_tilt, second_tilt)
    checks = {
        "complete_summary": True,
        "no_reset": int(summary["reset_count"]) == 0,
        "no_fall": not bool(summary["fell"]),
        "velocity_improvement": velocity_improvement >= 0.30,
        "tilt_improvement": tilt_improvement >= 0.20,
        "nonfoot_contacts_not_increased": float(summary["second_half_nonfoot_contacts"])
        <= float(summary["first_half_nonfoot_contacts"]),
        "parameter_changed": bool(summary["parameter_changed"]),
        "safety_compliance": not bool(summary["safety_violation"]),
        "no_actor_updates_before_novelty": int(summary["actor_updates_before_novelty"]) == 0,
        "actor_updated_after_novelty": int(summary["actor_updates_after_novelty"]) > 0,
        "novelty_entered": int(summary["novelty_entries"]) > 0,
        "rollback_count_recorded": int(summary["rollbacks"]) >= 0,
    }
    return {
        "passed": all(checks.values()),
        "velocity_improvement": velocity_improvement,
        "tilt_improvement": tilt_improvement,
        "checks": checks,
    }


def evaluate_flat_online_experiment(
    *,
    mode_records: Mapping[str, object] | Sequence[Mapping[str, object]],
    flat_base: Mapping[str, object],
    flat_learned: Mapping[str, object],
) -> dict[str, Any]:
    """Evaluate the formal flat-online experiment without reading or writing artifacts."""
    protocol = default_flat_online_protocol()
    evaluated_modes = ("base", "initial_zero_residual", "learned_frozen")
    if isinstance(mode_records, Mapping):
        grouped: Mapping[str, object] = mode_records
    elif isinstance(mode_records, Sequence) and not isinstance(mode_records, (str, bytes)):
        mutable_grouped: dict[str, object] = {mode: [] for mode in evaluated_modes}
        online_records = []
        for item in mode_records:
            if not isinstance(item, Mapping):
                continue
            mode = item.get("mode")
            if mode in evaluated_modes:
                mutable_grouped[mode].append(item)
            elif mode == "online_learning":
                online_records.append(item)
        mutable_grouped["online_learning"] = online_records[-1] if online_records else {}
        grouped = mutable_grouped
    else:
        grouped = {}

    def records_for(mode: str) -> list[Mapping[str, object]]:
        value = grouped.get(mode)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        return [item for item in value if isinstance(item, Mapping)]

    records = {mode: records_for(mode) for mode in evaluated_modes}
    exact_seeds = True
    for mode in evaluated_modes:
        mode_values = records[mode]
        seeds = [item.get("seed") for item in mode_values]
        exact_seeds = exact_seeds and (
            len(mode_values) == len(protocol.sealed_seeds)
            and len(seeds) == len(set(seeds))
            and set(seeds) == set(protocol.sealed_seeds)
            and all(item.get("mode") == mode for item in mode_values)
            and all(item.get("deterministic") is True for item in mode_values)
            and all(item.get("updates") == 0 for item in mode_values)
        )

    def passed_value(item: Mapping[str, object]) -> object:
        return item.get("passed", item.get("success"))

    def success_rate(mode: str) -> float:
        values = records[mode]
        if not values or any(type(passed_value(item)) is not bool for item in values):
            return 0.0
        return sum(bool(passed_value(item)) for item in values) / len(values)

    learned_rate = success_rate("learned_frozen")
    base_rate = success_rate("base")
    initial_rate = success_rate("initial_zero_residual")

    def finite_number(record: Mapping[str, object], name: str) -> float | None:
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    base_completion = finite_number(flat_base, "completion_rate")
    learned_completion = finite_number(flat_learned, "completion_rate")
    base_velocity = finite_number(flat_base, "velocity_error")
    learned_velocity = finite_number(flat_learned, "velocity_error")
    base_tilt = finite_number(flat_base, "tilt_rms")
    learned_tilt = finite_number(flat_learned, "tilt_rms")
    base_contacts = finite_number(flat_base, "contacts")
    learned_contacts = finite_number(flat_learned, "contacts")

    evaluation_falls = []
    for mode in evaluated_modes:
        evaluation_falls.extend(item.get("fell", item.get("falls")) for item in records[mode])
    evaluation_falls.extend((flat_base.get("falls"), flat_learned.get("falls")))

    training = grouped.get("online_learning")
    training = training if isinstance(training, Mapping) else {}
    total_episodes = training.get("total_episodes")
    total_steps = training.get("total_steps")
    checks = {
        "twenty_sealed_seeds": exact_seeds,
        "learned_success_at_least_80_percent": learned_rate >= 0.80,
        "improves_base_by_30_points": learned_rate - base_rate >= 0.30,
        "improves_initial_zero_by_30_points": learned_rate - initial_rate >= 0.30,
        "zero_evaluation_falls": bool(evaluation_falls)
        and all(type(value) in {bool, int} and int(value) == 0 for value in evaluation_falls),
        "flat_completion_retained": base_completion is not None
        and learned_completion is not None
        and learned_completion + 0.05 >= base_completion,
        "flat_velocity_retained": base_velocity is not None
        and learned_velocity is not None
        and learned_velocity <= 1.05 * base_velocity,
        "flat_tilt_retained": base_tilt is not None
        and learned_tilt is not None
        and learned_tilt <= 1.05 * base_tilt,
        "flat_contacts_not_increased": base_contacts is not None
        and learned_contacts is not None
        and learned_contacts <= base_contacts,
        "within_training_budget": type(total_episodes) is int
        and type(total_steps) is int
        and 0 <= total_episodes <= protocol.max_total_episodes
        and 0 <= total_steps <= protocol.max_total_steps,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "success_rates": {
            "base": base_rate,
            "initial_zero_residual": initial_rate,
            "learned_frozen": learned_rate,
        },
    }
