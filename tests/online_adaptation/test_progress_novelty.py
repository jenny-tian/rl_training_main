import math

import pytest

from rl_training.online_adaptation.progress_novelty import ForwardProgressMonitor


def test_commanded_stall_becomes_persistent_novelty():
    monitor = ForwardProgressMonitor(min_delta_m=0.05, stall_steps=3)

    assert not monitor.observe(base_x=0.0, commanded_forward=True).stalled
    assert not monitor.observe(base_x=0.01, commanded_forward=True).stalled
    decision = monitor.observe(base_x=0.01, commanded_forward=True)

    assert decision.stalled
    assert decision.steps_without_progress == 3
    assert decision.max_x == pytest.approx(0.01)


def test_meaningful_progress_resets_stall_counter():
    monitor = ForwardProgressMonitor(min_delta_m=0.05, stall_steps=3)
    monitor.observe(base_x=0.0, commanded_forward=True)
    monitor.observe(base_x=0.01, commanded_forward=True)

    progressed = monitor.observe(base_x=0.05, commanded_forward=True)

    assert not progressed.stalled
    assert progressed.steps_without_progress == 0
    assert progressed.max_x == pytest.approx(0.05)


def test_pause_and_zero_command_freeze_stall_counter():
    monitor = ForwardProgressMonitor(min_delta_m=0.05, stall_steps=2)
    first = monitor.observe(base_x=0.0, commanded_forward=True)

    paused = monitor.observe(base_x=0.0, commanded_forward=True, paused=True)
    stopped = monitor.observe(base_x=0.0, commanded_forward=False)

    assert paused.steps_without_progress == first.steps_without_progress
    assert stopped.steps_without_progress == first.steps_without_progress
    assert not paused.stalled
    assert not stopped.stalled


def test_backward_motion_does_not_reduce_monotonic_maximum():
    monitor = ForwardProgressMonitor(min_delta_m=0.05, stall_steps=2)
    monitor.observe(base_x=1.0, commanded_forward=True)

    decision = monitor.observe(base_x=0.5, commanded_forward=True)

    assert decision.max_x == pytest.approx(1.0)
    assert decision.stalled


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_position_is_rejected(value):
    monitor = ForwardProgressMonitor()
    with pytest.raises(ValueError, match="base_x"):
        monitor.observe(base_x=value, commanded_forward=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_delta_m": 0.0},
        {"min_delta_m": math.nan},
        {"stall_steps": 0},
        {"stall_steps": True},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ForwardProgressMonitor(**kwargs)
