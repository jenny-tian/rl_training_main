import pytest

from rl_training.online_adaptation.curriculum import (
    CurriculumCoordinator,
    EpisodeOutcome,
    ResidualGainScheduler,
)
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol


def test_gain_starts_zero_opens_to_quarter_and_advances_after_five_safe_episodes():
    scheduler = ResidualGainScheduler()
    assert scheduler.gain == 0.0
    scheduler.open_for_adaptation()
    assert scheduler.gain == 0.25
    for _ in range(5):
        scheduler.record_episode(
            safe=True, rolled_back=False, performance_degraded=False
        )
    assert scheduler.gain == 0.5


def test_any_rollback_reduces_one_gain_tier():
    scheduler = ResidualGainScheduler(initial_gain=0.75)
    scheduler.record_episode(
        safe=False, rolled_back=True, performance_degraded=True
    )
    assert scheduler.gain == 0.5


def test_gain_scheduler_rejects_boolean_initial_gain():
    with pytest.raises(ValueError, match="initial_gain"):
        ResidualGainScheduler(initial_gain=True)


def test_coordinator_advances_after_ten_episode_eighty_percent_window():
    coordinator = CurriculumCoordinator(default_flat_online_protocol())
    for passed in (True, True, True, True, True, True, True, True, False, False):
        coordinator.record_episode(
            EpisodeOutcome(passed=passed, fell=False, steps=1000, cost=0.2)
        )
    assert coordinator.stage.name == "step_4cm"


def test_fall_or_flat_retention_failure_blocks_stage_advancement():
    for outcome in (
        EpisodeOutcome(True, True, 1000, 0.2),
        EpisodeOutcome(True, False, 1000, 0.2, flat_retention_passed=False),
    ):
        coordinator = CurriculumCoordinator(default_flat_online_protocol())
        coordinator.record_episode(outcome)
        for _ in range(9):
            coordinator.record_episode(EpisodeOutcome(True, False, 1000, 0.2))
        assert coordinator.stage.name == "step_2cm"


def test_coordinator_stops_at_either_budget():
    by_episode = CurriculumCoordinator(default_flat_online_protocol())
    by_episode.load_counts(total_episodes=100, total_steps=10_000)
    assert by_episode.stop_reason == "episode_budget_exhausted"
    by_steps = CurriculumCoordinator(default_flat_online_protocol())
    by_steps.load_counts(total_episodes=10, total_steps=200_000)
    assert by_steps.stop_reason == "step_budget_exhausted"


def test_curriculum_and_gain_state_round_trip_is_exact():
    protocol = default_flat_online_protocol()
    coordinator = CurriculumCoordinator(protocol)
    coordinator.record_episode(EpisodeOutcome(True, False, 321, 0.4))
    assert coordinator.next_development_seed() == protocol.development_seeds[0]
    state = coordinator.state_dict()
    restored = CurriculumCoordinator(protocol)
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    scheduler = ResidualGainScheduler()
    scheduler.open_for_adaptation()
    scheduler.record_episode(True, False, False)
    gain_state = scheduler.state_dict()
    restored_gain = ResidualGainScheduler()
    restored_gain.load_state_dict(gain_state)
    assert restored_gain.state_dict() == gain_state


def test_ten_episodes_without_new_best_stops_curriculum():
    coordinator = CurriculumCoordinator(default_flat_online_protocol())
    coordinator.record_episode(EpisodeOutcome(False, False, 10, 0.2))
    for _ in range(10):
        coordinator.record_episode(EpisodeOutcome(False, False, 10, 0.3))
    assert coordinator.stop_reason == "no_improvement"


def test_episode_outcome_rejects_nonfinite_cost():
    with pytest.raises(ValueError, match="cost"):
        EpisodeOutcome(True, False, 10, float("nan"))
