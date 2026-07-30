from __future__ import annotations

from copy import deepcopy

import pytest

from rl_training.online_adaptation.acceptance import evaluate_flat_online_experiment
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol


def mode_records():
    seeds = default_flat_online_protocol().sealed_seeds

    def records(mode, successes):
        return [
            {
                "mode": mode,
                "seed": seed,
                "passed": index < successes,
                "fell": False,
                "updates": 0,
                "deterministic": True,
            }
            for index, seed in enumerate(seeds)
        ]

    return {
        "base": records("base", 8),
        "initial_zero_residual": records("initial_zero_residual", 8),
        "learned_frozen": records("learned_frozen", 16),
        "online_learning": {
            "total_episodes": 100,
            "total_steps": 200_000,
            "falls": 0,
        },
    }


def flat_record(**overrides):
    value = {
        "completion_rate": 1.0,
        "velocity_error": 0.10,
        "tilt_rms": 0.05,
        "contacts": 0,
        "falls": 0,
    }
    value.update(overrides)
    return value


def test_strict_acceptance_requires_all_confirmed_thresholds():
    report = evaluate_flat_online_experiment(
        mode_records=mode_records(),
        flat_base=flat_record(),
        flat_learned=flat_record(
            completion_rate=0.95, velocity_error=0.105, tilt_rms=0.0525
        ),
    )

    assert report["passed"]
    assert report["checks"] == {
        "twenty_sealed_seeds": True,
        "learned_success_at_least_80_percent": True,
        "improves_base_by_30_points": True,
        "improves_initial_zero_by_30_points": True,
        "zero_evaluation_falls": True,
        "flat_completion_retained": True,
        "flat_velocity_retained": True,
        "flat_tilt_retained": True,
        "flat_contacts_not_increased": True,
        "within_training_budget": True,
    }


def test_acceptance_accepts_flat_record_sequence_with_success_field():
    grouped = mode_records()
    sequence = []
    for mode in ("base", "initial_zero_residual", "learned_frozen"):
        for record in grouped[mode]:
            converted = dict(record)
            converted["success"] = converted.pop("passed")
            sequence.append(converted)
    sequence.append({"mode": "online_learning", **grouped["online_learning"]})

    report = evaluate_flat_online_experiment(
        mode_records=sequence,
        flat_base=flat_record(),
        flat_learned=flat_record(
            completion_rate=0.95, velocity_error=0.105, tilt_rms=0.0525
        ),
    )

    assert report["passed"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_seed",
        "learned_success",
        "base_improvement",
        "initial_improvement",
        "fall",
        "flat_completion",
        "flat_velocity",
        "flat_tilt",
        "flat_contacts",
        "budget",
    ),
)
def test_each_failed_check_rejects_aggregate(mutation):
    modes = mode_records()
    base = flat_record()
    learned = flat_record(completion_rate=0.95, velocity_error=0.105, tilt_rms=0.0525)
    if mutation == "missing_seed":
        modes["base"].pop()
    elif mutation == "learned_success":
        modes["learned_frozen"][15]["passed"] = False
    elif mutation == "base_improvement":
        for item in modes["base"][8:11]:
            item["passed"] = True
    elif mutation == "initial_improvement":
        for item in modes["initial_zero_residual"][8:11]:
            item["passed"] = True
    elif mutation == "fall":
        modes["learned_frozen"][0]["fell"] = True
    elif mutation == "flat_completion":
        learned["completion_rate"] = 0.94
    elif mutation == "flat_velocity":
        learned["velocity_error"] = 0.106
    elif mutation == "flat_tilt":
        learned["tilt_rms"] = 0.053
    elif mutation == "flat_contacts":
        learned["contacts"] = 1
    elif mutation == "budget":
        modes["online_learning"]["total_steps"] = 200_001

    report = evaluate_flat_online_experiment(
        mode_records=deepcopy(modes), flat_base=base, flat_learned=learned
    )

    assert report["passed"] is False
    assert list(report["checks"].values()).count(False) >= 1
