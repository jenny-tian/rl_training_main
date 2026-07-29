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


def test_pre_update_windows_replace_baseline_without_degradation():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()

    first = add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    second = add_window(monitor, velocity_error=0.40, tilt_rms=0.10)
    third = add_window(monitor, velocity_error=0.60, tilt_rms=0.10)

    assert first.cost == pytest.approx(0.30)
    assert second.cost == pytest.approx(0.50)
    assert third.cost == pytest.approx(0.70)
    assert monitor.best_cost == pytest.approx(0.70)
    assert monitor.degradation_count == 0
    assert all(not decision.degraded and not decision.should_rollback for decision in (first, second, third))
    assert not monitor.evaluation_enabled


def test_first_actor_update_preserves_baseline_and_clears_partial_window():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    assert monitor.add(velocity_error=9.0, tilt_rms=0.0, nonfoot_contact=0) is None

    monitor.mark_actor_updated()

    assert monitor.evaluation_enabled
    assert monitor.best_cost == pytest.approx(0.30)
    assert monitor.add(velocity_error=0.25, tilt_rms=0.10, nonfoot_contact=0) is None
    decision = monitor.add(velocity_error=0.25, tilt_rms=0.10, nonfoot_contact=0)
    assert decision.degraded
    assert not decision.should_rollback


def test_repeated_actor_update_notification_is_idempotent():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    monitor.mark_actor_updated()
    assert monitor.add(velocity_error=0.25, tilt_rms=0.10, nonfoot_contact=0) is None

    monitor.mark_actor_updated()

    decision = monitor.add(velocity_error=0.25, tilt_rms=0.10, nonfoot_contact=0)
    assert decision.degraded


def test_first_post_update_window_establishes_reference_without_pre_update_window():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    monitor.add(velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0)
    monitor.mark_actor_updated()

    decision = add_window(monitor, velocity_error=0.30, tilt_rms=0.10)

    assert decision.cost == pytest.approx(0.40)
    assert monitor.best_cost == pytest.approx(0.40)
    assert not decision.degraded


def test_five_percent_improvement_updates_best_cost():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    monitor.mark_actor_updated()

    decision = add_window(monitor, velocity_error=0.18, tilt_rms=0.10)

    assert decision.improved
    assert monitor.best_cost == pytest.approx(0.28)
    assert monitor.degradation_count == 0


def test_two_complete_post_update_degraded_windows_request_rollback():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    monitor.mark_actor_updated()

    first = add_window(monitor, velocity_error=0.25, tilt_rms=0.10)
    second = add_window(monitor, velocity_error=0.25, tilt_rms=0.10)

    assert first.degraded and not first.should_rollback
    assert second.degraded and second.should_rollback
    assert second.rollback_reason == "performance_degradation"


def test_neutral_window_breaks_consecutive_degradation():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    monitor.mark_actor_updated()
    add_window(monitor, velocity_error=0.25, tilt_rms=0.10)

    neutral = add_window(monitor, velocity_error=0.21, tilt_rms=0.10)

    assert not neutral.degraded
    assert monitor.degradation_count == 0


def test_new_episode_disables_post_update_evaluation():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    monitor.mark_actor_updated()

    monitor.start_episode()

    assert not monitor.evaluation_enabled
    assert monitor.best_cost is None
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
