from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import (
    AdaptationCommand,
    AdaptationRuntimeOrchestrator,
    RuntimeCounters,
    resolve_track_boundary_x,
)
from rl_training.online_adaptation.metrics import AdaptationPerformance
from rl_training.online_adaptation.novelty import NoveltyConfig, NoveltyGate


def make_orchestrator():
    return AdaptationRuntimeOrchestrator(
        mode="online",
        novelty_gate=NoveltyGate(
            NoveltyConfig(
                calibration_steps=3,
                enter_threshold=2.5,
                exit_threshold=1.0,
                enter_count=2,
                enter_window=3,
                recovery_steps=2,
                cooldown_steps=2,
                contact_count=2,
                contact_window=3,
            )
        ),
        adaptation_performance=AdaptationPerformance(
            window=2,
            improvement_ratio=0.05,
            degradation_ratio=0.10,
            patience=2,
        ),
    )


class EnvCfg:
    pass


def test_resolve_track_boundary_prefers_metric_boundary_over_terrain_start():
    cfg = EnvCfg()
    cfg.heldout_start_x = -7.0
    cfg.track_boundary_x = -7.35

    assert resolve_track_boundary_x(cfg) == -7.35


def test_resolve_track_boundary_falls_back_to_heldout_start_for_legacy_configs():
    cfg = EnvCfg()
    cfg.heldout_start_x = -6.5

    assert resolve_track_boundary_x(cfg) == -6.5


def test_runtime_counters_separate_pre_and_post_novelty_actor_updates():
    counters = RuntimeCounters()
    counters.record_state("calibrating")
    counters.record_critic_update()
    counters.record_actor_update(step=100, adaptation_episode=0)
    counters.record_novelty_entry()
    counters.record_state("adapting")
    counters.record_actor_update(step=300, adaptation_episode=1)

    assert counters.critic_updates == 1
    assert counters.actor_updates == 2
    assert counters.actor_updates_before_novelty == 1
    assert counters.actor_updates_after_novelty == 1
    assert counters.first_actor_update_step == 100
    assert counters.novelty_entries == 1
    assert counters.state_durations == {"calibrating": 1, "adapting": 1}


def test_runtime_counters_record_distinct_rollback_reasons():
    counters = RuntimeCounters()

    counters.record_rollback("performance_degradation")
    counters.record_rollback("control_timeout")

    assert counters.rollbacks == 2
    assert counters.performance_rollbacks == 1
    assert counters.safety_rollbacks == 1


def test_adaptation_command_defaults_to_no_reason():
    command = AdaptationCommand(kind="start_episode", episode=3)

    assert command.kind == "start_episode"
    assert command.episode == 3
    assert command.reason == ""


def test_runtime_orchestrator_gates_actor_until_persistent_novelty():
    runtime = make_orchestrator()
    rows = []
    for step in range(1, 6):
        result = runtime.observe_step(
            step=step,
            velocity_error=0.10,
            tilt_rms=0.02,
            nonfoot_contact=0,
            reward=1.0,
            paused=False,
        )
        runtime.record_critic_update()
        rows.append(result.record_fields)

    first = runtime.observe_step(
        step=6,
        velocity_error=0.50,
        tilt_rms=0.02,
        nonfoot_contact=0,
        reward=0.2,
        paused=False,
    )
    second = runtime.observe_step(
        step=7,
        velocity_error=0.50,
        tilt_rms=0.02,
        nonfoot_contact=0,
        reward=0.2,
        paused=False,
    )
    runtime.record_actor_update(step=7)
    rows.extend([first.record_fields, second.record_fields])

    assert runtime.counters.actor_updates_before_novelty == 0
    assert runtime.counters.critic_updates > 0
    assert runtime.counters.actor_updates_after_novelty == 1
    assert not first.adaptation_enabled
    assert second.adaptation_enabled
    assert second.clear_adaptation_replay
    assert second.commands == (AdaptationCommand("start_episode", 1),)
    assert all("adaptation_state" in row for row in rows)
    assert all("novelty_score" in row for row in rows)


def test_runtime_orchestrator_requests_performance_rollback():
    runtime = make_orchestrator()
    for step in range(1, 6):
        runtime.observe_step(
            step=step,
            velocity_error=0.10,
            tilt_rms=0.02,
            nonfoot_contact=0,
            reward=1.0,
            paused=False,
        )
    runtime.observe_step(step=6, velocity_error=0.50, tilt_rms=0.02, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(step=7, velocity_error=0.50, tilt_rms=0.02, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(step=8, velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(step=9, velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(step=10, velocity_error=0.30, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(
        step=11,
        velocity_error=0.30,
        tilt_rms=0.10,
        nonfoot_contact=0,
        reward=0.2,
    )
    runtime.observe_step(step=12, velocity_error=0.30, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(step=13, velocity_error=0.30, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    rollback = runtime.observe_step(step=14, velocity_error=0.30, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)

    assert rollback.commands == (AdaptationCommand("rollback", 1, "performance_degradation"),)
    assert rollback.clear_adaptation_replay
    assert rollback.record_fields["rollback_reason"] == "performance_degradation"
    assert rollback.record_fields["adaptation_state"] == "cooldown"
    assert runtime.counters.rollbacks == 1
    assert runtime.counters.performance_rollbacks == 1
