from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import (
    AdaptationCommand,
    AdaptationRuntimeOrchestrator,
    RuntimeCounters,
    resolve_track_boundary_x,
    warm_up_control_path,
)
from rl_training.online_adaptation.metrics import AdaptationPerformance
from rl_training.online_adaptation.novelty import NoveltyConfig, NoveltyGate
import torch


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


def enter_adapting(runtime):
    for step in range(1, 6):
        runtime.observe_step(
            step=step,
            velocity_error=0.10,
            tilt_rms=0.02,
            nonfoot_contact=0,
            reward=1.0,
        )
    runtime.observe_step(step=6, velocity_error=0.50, tilt_rms=0.02, nonfoot_contact=0, reward=0.2)
    result = runtime.observe_step(
        step=7,
        velocity_error=0.50,
        tilt_rms=0.02,
        nonfoot_contact=0,
        reward=0.2,
    )
    assert result.commands == (AdaptationCommand("start_episode", 1),)


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


def test_warm_up_control_path_primes_policy_actor_and_safety_without_agent_step():
    calls = {}

    class BasePolicy:
        def __call__(self, observation):
            calls["base_observation"] = observation
            return torch.ones(1, 12)

    class Actor:
        def sample(self, history, frame, base_action, *, deterministic):
            calls["actor_history_shape"] = tuple(history.shape)
            calls["actor_frame_shape"] = tuple(frame.shape)
            calls["actor_base_action_shape"] = tuple(base_action.shape)
            calls["actor_deterministic"] = deterministic
            return torch.zeros_like(base_action), None, None

    class Publisher:
        def current(self):
            return Actor(), 11

    class Config:
        history_length = 25

    class Agent:
        config = Config()
        publisher = Publisher()

        def begin_step(self, *args, **kwargs):
            raise AssertionError("warmup must not mutate the agent step state")

    class Adapter:
        def frame(self, previous_action):
            calls["previous_action_shape"] = tuple(previous_action.shape)
            return torch.zeros(1, 49)

        def safety_state(self, timeout):
            calls["timeout_shape"] = tuple(timeout.shape)
            return "safety_state"

    class SafetyFilter:
        def __init__(self, limits):
            calls["safety_limits"] = limits

        def project(self, base_action, residual, safety_state):
            calls["safety_project"] = (tuple(base_action.shape), tuple(residual.shape), safety_state)

    class Device:
        type = "cpu"

    safety_limits = object()

    warm_up_control_path(
        base_policy=BasePolicy(),
        online_agent=Agent(),
        adapter=Adapter(),
        observation="observation",
        previous_action=torch.zeros(1, 12),
        safety_limits=safety_limits,
        safety_filter_cls=SafetyFilter,
        deterministic=True,
        device=Device(),
    )

    assert calls["base_observation"] == "observation"
    assert calls["previous_action_shape"] == (1, 12)
    assert calls["actor_history_shape"] == (1, 25, 49)
    assert calls["actor_frame_shape"] == (1, 49)
    assert calls["actor_base_action_shape"] == (1, 12)
    assert calls["actor_deterministic"] is True
    assert calls["timeout_shape"] == (1, 1)
    assert calls["safety_limits"] is safety_limits
    assert calls["safety_project"] == ((1, 12), (1, 12), "safety_state")


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
    assert counters.actor_updates_by_episode == {1: 1}
    assert counters.first_actor_update_step == 100
    assert counters.novelty_entries == 1
    assert counters.state_durations == {"calibrating": 1, "adapting": 1}


def test_runtime_counters_isolate_actor_updates_by_episode():
    counters = RuntimeCounters()
    counters.record_actor_update(step=100, adaptation_episode=1)
    counters.record_actor_update(step=101, adaptation_episode=1)
    counters.record_actor_update(step=200, adaptation_episode=2)

    assert counters.actor_updates_for_episode(1) == 2
    assert counters.actor_updates_for_episode(2) == 1
    assert counters.actor_updates_for_episode(3) == 0


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
    assert all("performance_evaluation_enabled" in row for row in rows)


def test_runtime_does_not_roll_back_before_first_actor_update():
    runtime = make_orchestrator()
    enter_adapting(runtime)

    results = [
        runtime.observe_step(
            step=step,
            velocity_error=0.80,
            tilt_rms=0.20,
            nonfoot_contact=0,
            reward=0.1,
        )
        for step in range(8, 16)
    ]

    assert runtime.active_episode == 1
    assert runtime.counters.performance_rollbacks == 0
    assert all(not result.commands for result in results)
    assert not runtime.adaptation_performance.evaluation_enabled


def test_runtime_rolls_back_after_two_complete_post_update_windows():
    runtime = make_orchestrator()
    enter_adapting(runtime)
    runtime.observe_step(step=8, velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(step=9, velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.record_actor_update(step=9)

    runtime.observe_step(step=10, velocity_error=0.50, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    first = runtime.observe_step(
        step=11,
        velocity_error=0.50,
        tilt_rms=0.10,
        nonfoot_contact=0,
        reward=0.2,
    )
    runtime.observe_step(step=12, velocity_error=0.50, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    second = runtime.observe_step(
        step=13,
        velocity_error=0.50,
        tilt_rms=0.10,
        nonfoot_contact=0,
        reward=0.2,
    )

    assert not first.commands
    assert first.record_fields["performance_evaluation_enabled"]
    assert second.commands == (AdaptationCommand("rollback", 1, "performance_degradation"),)
    assert second.clear_adaptation_replay
    assert second.record_fields["rollback_reason"] == "performance_degradation"
    assert second.record_fields["adaptation_state"] == "cooldown"
    assert not second.record_fields["performance_evaluation_enabled"]
    assert not runtime.adaptation_performance.evaluation_enabled
    assert runtime.counters.rollbacks == 1
    assert runtime.counters.performance_rollbacks == 1


def test_old_episode_update_does_not_enable_current_episode():
    runtime = make_orchestrator()
    enter_adapting(runtime)
    runtime.active_episode = 2
    runtime.adaptation_episode = 2
    runtime.adaptation_performance.start_episode()
    runtime.counters.record_actor_update(step=10, adaptation_episode=1)

    runtime.observe_step(step=11, velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    result = runtime.observe_step(
        step=12,
        velocity_error=0.20,
        tilt_rms=0.10,
        nonfoot_contact=0,
        reward=0.2,
    )

    assert not runtime.adaptation_performance.evaluation_enabled
    assert not result.record_fields["performance_evaluation_enabled"]


def test_leaving_adapting_resets_performance_without_end_episode_command():
    runtime = make_orchestrator()
    enter_adapting(runtime)
    runtime.adaptation_performance.mark_actor_updated()

    runtime.observe_step(step=8, velocity_error=0.10, tilt_rms=0.02, nonfoot_contact=0, reward=1.0)
    result = runtime.observe_step(
        step=9,
        velocity_error=0.10,
        tilt_rms=0.02,
        nonfoot_contact=0,
        reward=1.0,
    )

    assert runtime.active_episode == 0
    assert result.clear_adaptation_replay
    assert not result.commands
    assert not runtime.adaptation_performance.evaluation_enabled
    assert not result.record_fields["performance_evaluation_enabled"]


def test_safety_rollback_remains_active_before_actor_update():
    runtime = make_orchestrator()
    enter_adapting(runtime)

    result = runtime.observe_step(
        step=8,
        velocity_error=0.50,
        tilt_rms=0.20,
        nonfoot_contact=1,
        reward=0.1,
        paused=True,
        safety_reasons=("excessive_tilt",),
    )

    assert result.commands == (AdaptationCommand("rollback", 1, "excessive_tilt"),)
    assert not result.record_fields["performance_evaluation_enabled"]


def test_control_timeout_rollback_remains_active_before_actor_update():
    runtime = make_orchestrator()
    enter_adapting(runtime)

    result = runtime.observe_step(
        step=8,
        velocity_error=0.50,
        tilt_rms=0.20,
        nonfoot_contact=0,
        reward=0.1,
        control_timed_out=True,
    )

    assert result.commands == (AdaptationCommand("rollback", 1, "control_timeout"),)
    assert not result.record_fields["performance_evaluation_enabled"]
