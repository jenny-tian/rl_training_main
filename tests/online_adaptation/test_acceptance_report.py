import pytest

from rl_training.online_adaptation.acceptance import evaluate_run


def test_acceptance_requires_velocity_tilt_and_no_reset():
    result = evaluate_run(
        {
            "reset_count": 0,
            "fell": False,
            "first_half_velocity_error": 1.0,
            "second_half_velocity_error": 0.65,
            "first_half_tilt_rms": 0.5,
            "second_half_tilt_rms": 0.39,
            "first_half_nonfoot_contacts": 2,
            "second_half_nonfoot_contacts": 2,
            "parameter_changed": True,
            "safety_violation": False,
            "actor_updates_before_novelty": 0,
            "actor_updates_after_novelty": 1,
            "novelty_entries": 1,
            "rollbacks": 0,
        }
    )
    assert result["passed"]
    assert result["velocity_improvement"] == pytest.approx(0.35)
    assert result["tilt_improvement"] == pytest.approx(0.22)


def test_acceptance_rejects_actor_updates_before_novelty():
    result = evaluate_run(
        {
            "reset_count": 0,
            "fell": False,
            "first_half_velocity_error": 1.0,
            "second_half_velocity_error": 0.65,
            "first_half_tilt_rms": 0.5,
            "second_half_tilt_rms": 0.39,
            "first_half_nonfoot_contacts": 2,
            "second_half_nonfoot_contacts": 2,
            "parameter_changed": True,
            "safety_violation": False,
            "actor_updates_before_novelty": 1,
            "actor_updates_after_novelty": 1,
            "novelty_entries": 1,
            "rollbacks": 0,
        }
    )

    assert not result["passed"]
    assert result["checks"]["no_actor_updates_before_novelty"] is False


def test_acceptance_reports_incomplete_novelty_summary_without_raising():
    result = evaluate_run(
        {
            "reset_count": 0,
            "fell": False,
            "first_half_velocity_error": 1.0,
            "second_half_velocity_error": 0.65,
            "first_half_tilt_rms": 0.5,
            "second_half_tilt_rms": 0.39,
            "first_half_nonfoot_contacts": 2,
            "second_half_nonfoot_contacts": 2,
            "parameter_changed": True,
            "safety_violation": False,
        }
    )

    assert not result["passed"]
    assert result["checks"]["complete_summary"] is False
