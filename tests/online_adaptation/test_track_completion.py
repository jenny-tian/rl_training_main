from __future__ import annotations

import math

import pytest

from rl_training.online_adaptation.track_completion import TrackCompletion


def _start(stage: str) -> TrackCompletion:
    return TrackCompletion.start(
        stage_name=stage,
        initial_x=10.0,
        flat_length=3.0,
        step_length=2.0,
        track_length=20.0,
        step_clearance_margin_m=0.5,
        track_end_buffer_m=0.5,
    )


def test_step_target_clears_the_far_edge_of_the_step():
    completion = _start("step_2cm")

    assert completion.target_x == pytest.approx(14.0)
    assert completion.target_distance == pytest.approx(4.0)
    assert completion.step_edge_x == pytest.approx(11.5)
    assert not completion.past_step_edge
    completion.observe(13.5)
    assert not completion.reached
    assert completion.past_step_edge
    assert completion.remaining_m == pytest.approx(0.5)

    completion.observe(14.0)
    assert completion.reached
    assert completion.remaining_m == 0.0
    assert completion.cost() == 0.0


def test_rough_target_is_before_the_track_end_buffer():
    completion = _start("rough_4cm")

    assert completion.target_x == pytest.approx(28.0)
    assert completion.target_distance == pytest.approx(18.0)
    completion.observe(27.99)
    assert not completion.reached
    completion.observe(29.0)
    assert completion.reached


def test_progress_is_monotonic_when_robot_moves_backwards():
    completion = _start("step_4cm")
    completion.observe(12.75)
    completion.observe(12.0)

    assert completion.max_x == pytest.approx(12.75)
    assert completion.remaining_m == pytest.approx(1.25)


@pytest.mark.parametrize("stage", ["step_2cm", "rough_2cm"])
def test_fall_or_safety_violation_adds_one_target_distance_penalty(stage: str):
    completion = _start(stage)
    completion.observe(completion.initial_x + 1.0)

    expected_remaining = completion.target_distance - 1.0
    assert completion.cost(fell=True) == pytest.approx(
        expected_remaining + completion.target_distance
    )
    assert completion.cost(safety_violation=True) == pytest.approx(
        expected_remaining + completion.target_distance
    )
    assert completion.cost(fell=True, safety_violation=True) == pytest.approx(
        expected_remaining + completion.target_distance
    )


def test_invalid_geometry_and_observations_fail_closed():
    with pytest.raises(ValueError):
        _start("unknown")
    with pytest.raises(ValueError):
        _start("step_unknown")
    with pytest.raises(ValueError):
        TrackCompletion.start(
            stage_name="step_2cm",
            initial_x=0.0,
            flat_length=3.0,
            step_length=2.0,
            track_length=5.0,
            step_clearance_margin_m=0.5,
            track_end_buffer_m=0.5,
        )
    completion = _start("step_2cm")
    with pytest.raises(ValueError):
        completion.observe(math.nan)
