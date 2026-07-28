import pytest

from rl_training.online_adaptation.metrics import AdaptationPerformance


def add_window(monitor, velocity_error, tilt_rms, contact=0):
    first = monitor.add(
        velocity_error=velocity_error,
        tilt_rms=tilt_rms,
        nonfoot_contact=contact,
    )
    assert first is None
    return monitor.add(
        velocity_error=velocity_error,
        tilt_rms=tilt_rms,
        nonfoot_contact=contact,
    )


def test_first_complete_window_establishes_episode_reference():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()

    decision = add_window(monitor, velocity_error=0.20, tilt_rms=0.10)

    assert decision.cost == pytest.approx(0.30)
    assert monitor.best_cost == pytest.approx(0.30)
    assert not decision.improved
    assert not decision.degraded
    assert not decision.should_rollback


def test_five_percent_improvement_updates_best_cost():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)

    decision = add_window(monitor, velocity_error=0.18, tilt_rms=0.10)

    assert decision.improved
    assert monitor.best_cost == pytest.approx(0.28)
    assert monitor.degradation_count == 0


def test_two_consecutive_degraded_windows_request_rollback():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)

    first = add_window(monitor, velocity_error=0.25, tilt_rms=0.10)
    second = add_window(monitor, velocity_error=0.25, tilt_rms=0.10)

    assert first.degraded and not first.should_rollback
    assert second.degraded and second.should_rollback
    assert second.rollback_reason == "performance_degradation"


def test_neutral_window_breaks_consecutive_degradation():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    add_window(monitor, velocity_error=0.25, tilt_rms=0.10)

    neutral = add_window(monitor, velocity_error=0.21, tilt_rms=0.10)

    assert not neutral.degraded
    assert monitor.degradation_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 0},
        {"patience": 0},
        {"improvement_ratio": -0.1},
        {"degradation_ratio": -0.1},
    ],
)
def test_invalid_adaptation_performance_config_is_rejected(kwargs):
    defaults = {"window": 2, "improvement_ratio": 0.05, "degradation_ratio": 0.10, "patience": 2}
    with pytest.raises(ValueError):
        AdaptationPerformance(**{**defaults, **kwargs})
