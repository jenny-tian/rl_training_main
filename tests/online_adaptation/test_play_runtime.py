import json
import threading

import pytest

from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import (
    AdaptationCommand,
    AdaptationRuntimeOrchestrator,
    RuntimeCounters,
    enable_actor_gate_if_active,
    enforce_prepared_rollback_fallback,
    actor_version_for_record,
    policy_drift_record,
    prepare_rollback_fallback,
    apply_adaptation_command,
    cutoff_before_final_step,
    record_update_result,
    rollback_record_fields,
    resolve_track_boundary_x,
    transition_episode_publication,
    warm_up_control_path,
    write_actor_update_event,
    resolve_worker_geometry,
    split_worker_endings,
    write_episode_summary,
)
from rl_training.online_adaptation.metrics import AdaptationPerformance
from rl_training.online_adaptation.novelty import NoveltyConfig, NoveltyGate
from rl_training.online_adaptation.policy_drift import PolicyDrift
from rl_training.online_adaptation.types import UpdateStats
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


class FakeEvent:
    def __init__(self):
        self.enabled = False

    def set(self):
        self.enabled = True

    def clear(self):
        self.enabled = False

    def is_set(self):
        return self.enabled


class FakeStableManager:
    def __init__(self):
        self.checkpoints = []
        self.rollbacks = []

    def checkpoint(self, updater):
        self.checkpoints.append(updater.version)

    def rollback(self, updater):
        self.rollbacks.append(updater.version)
        return True


class FakeDriftMonitor:
    def __init__(self):
        self.started = []
        self.cleared = []

    def start_episode(self, actor, *, adaptation_episode):
        self.started.append((actor, adaptation_episode))

    def clear_episode(self, *, adaptation_episode):
        self.cleared.append(adaptation_episode)


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, actor, version):
        self.published.append((actor, version))


class FakeUpdater:
    actor = object()
    version = 7


class EventLock:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("lock_enter")

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append("lock_exit")


def test_episode_transition_clears_gate_writes_state_and_enqueues_under_one_lock():
    events = []
    active = {"episode": 4}

    previous = transition_episode_publication(
        shared_lock=EventLock(events),
        actor_gate_event=type("Gate", (), {"clear": lambda self: events.append("gate_clear")})(),
        read_active_episode=lambda: events.append("read") or active["episode"],
        write_active_episode=lambda episode: events.append(("write", episode))
        or active.__setitem__("episode", episode),
        next_episode=0,
        after_transition=lambda episode: events.append(("enqueue_end", episode)),
    )

    assert previous == 4
    assert active["episode"] == 0
    assert events == [
        "lock_enter",
        "read",
        "gate_clear",
        ("write", 0),
        ("enqueue_end", 4),
        "lock_exit",
    ]


def test_actor_gate_enable_checks_episode_and_sets_gate_under_one_lock():
    events = []
    gate = type("Gate", (), {"set": lambda self: events.append("gate_set")})()

    enabled = enable_actor_gate_if_active(
        shared_lock=EventLock(events),
        actor_gate_event=gate,
        read_active_episode=lambda: events.append("read") or 3,
        adaptation_episode=3,
    )

    assert enabled
    assert events == ["lock_enter", "read", "gate_set", "lock_exit"]


def make_update_stats(**changes):
    values = {
        "critic_loss": 1.0,
        "actor_loss": 2.0,
        "alpha_loss": 3.0,
        "alpha": 0.2,
        "mean_q": 4.0,
        "grad_norm": 5.0,
        "accepted": True,
        "version": 7,
        "critic_updated": True,
        "actor_updated": True,
        "rejection_reason": None,
        "mean_target_q": 6.0,
        "mean_td_error": 7.0,
        "mean_flat_q": None,
        "mean_online_q": None,
        "flat_anchor_loss": None,
        "flat_batch_fraction": 0.0,
        "n_step": 1,
    }
    values.update(changes)
    return UpdateStats(**values)


def test_worker_geometry_uses_development_or_sealed_protocol_paths():
    development = resolve_worker_geometry("step_2cm", 101, evaluation_only=False)
    sealed = resolve_worker_geometry("step_2cm", 1009, evaluation_only=True)
    assert development.step_height == 0.02
    assert development.seed == 101
    assert sealed.seed == 1009


def test_worker_endings_support_gymnasium_and_reject_ambiguous_rsl_rows():
    terminated, truncated, bootstrap = split_worker_endings(
        (None, None, torch.tensor([True]), torch.tensor([False]), {})
    )
    assert terminated.item() and not truncated.item() and not bootstrap.item()
    with pytest.raises(RuntimeError, match="cannot distinguish termination from truncation"):
        split_worker_endings((None, None, torch.tensor([True]), {}))


def test_episode_summary_is_strict_json_and_immutable(tmp_path):
    path = tmp_path / "episode_summary.json"
    write_episode_summary(path, {"steps": 10, "gain": 0.25})
    assert json.loads(path.read_text()) == {"steps": 10, "gain": 0.25}
    with pytest.raises(FileExistsError):
        write_episode_summary(path, {"steps": 11})
    with pytest.raises(ValueError):
        write_episode_summary(tmp_path / "invalid.json", {"cost": float("nan")})


def test_policy_drift_record_is_flat_and_scalar_only():
    drift = PolicyDrift(
        residual_mse=0.01,
        joint_abs_change_mean=0.02,
        joint_abs_change_max=0.03,
        parameter_l2=0.04,
        adaptation_episode=2,
        actor_version=9,
        update_step=700,
    )

    record = policy_drift_record(drift)

    assert record == {
        "policy_residual_mse": 0.01,
        "policy_joint_abs_change_mean": 0.02,
        "policy_joint_abs_change_max": 0.03,
        "policy_parameter_l2": 0.04,
        "policy_drift_episode": 2,
        "policy_drift_actor_version": 9,
        "policy_drift_update_step": 700,
    }
    assert all(isinstance(value, (float, int)) for value in record.values())


def test_old_rollback_does_not_clear_new_episode_gate_or_reference():
    stable_manager = FakeStableManager()
    drift_monitor = FakeDriftMonitor()
    publisher = FakePublisher()
    updater = FakeUpdater()
    gate = FakeEvent()
    replay_clears = []
    checkpointed_episode, _ = apply_adaptation_command(
        AdaptationCommand("start_episode", 2),
        checkpointed_episode=0,
        updater=updater,
        publisher=publisher,
        stable_manager=stable_manager,
        drift_monitor=drift_monitor,
        actor_gate_event=gate,
        clear_adaptation_replay=lambda: replay_clears.append(2),
        enable_actor_if_episode_active=lambda episode: gate.set() or episode == 2,
    )

    completion = threading.Event()
    checkpointed_episode, rollback_reason = apply_adaptation_command(
        AdaptationCommand("rollback", 1, "performance_degradation", completion),
        checkpointed_episode=checkpointed_episode,
        updater=updater,
        publisher=publisher,
        stable_manager=stable_manager,
        drift_monitor=drift_monitor,
        actor_gate_event=gate,
        clear_adaptation_replay=lambda: replay_clears.append(1),
        enable_actor_if_episode_active=lambda episode: False,
    )

    assert checkpointed_episode == 2
    assert rollback_reason is None
    assert gate.is_set()
    assert drift_monitor.started == [(updater.actor, 2)]
    assert drift_monitor.cleared == []
    assert stable_manager.rollbacks == []
    assert publisher.published == []
    assert replay_clears == [2]
    assert completion.is_set()


def test_rollback_command_completion_does_not_affect_equality_or_repr():
    command = AdaptationCommand(
        "rollback",
        1,
        "performance_degradation",
        threading.Event(),
    )

    assert command == AdaptationCommand("rollback", 1, "performance_degradation")
    assert "completion" not in repr(command)


def test_matching_rollback_acknowledges_only_after_restored_actor_is_published():
    completion = threading.Event()
    publisher = FakePublisher()

    def publish(actor, version):
        assert not completion.is_set()
        publisher.published.append((actor, version))

    publisher.publish = publish

    checkpointed_episode, rollback_reason = apply_adaptation_command(
        AdaptationCommand("rollback", 2, "performance_degradation", completion),
        checkpointed_episode=2,
        updater=FakeUpdater(),
        publisher=publisher,
        stable_manager=FakeStableManager(),
        drift_monitor=FakeDriftMonitor(),
        actor_gate_event=FakeEvent(),
        clear_adaptation_replay=lambda: None,
        enable_actor_if_episode_active=lambda episode: False,
    )

    assert checkpointed_episode == 0
    assert rollback_reason == "performance_degradation"
    assert publisher.published == [(FakeUpdater.actor, FakeUpdater.version)]
    assert completion.is_set()


def test_rollback_exception_does_not_acknowledge_completion():
    completion = threading.Event()

    class FailingPublisher:
        def publish(self, actor, version):
            raise RuntimeError("publish failed")

    with pytest.raises(RuntimeError, match="publish failed"):
        apply_adaptation_command(
            AdaptationCommand("rollback", 2, "performance_degradation", completion),
            checkpointed_episode=2,
            updater=FakeUpdater(),
            publisher=FailingPublisher(),
            stable_manager=FakeStableManager(),
            drift_monitor=FakeDriftMonitor(),
            actor_gate_event=FakeEvent(),
            clear_adaptation_replay=lambda: None,
            enable_actor_if_episode_active=lambda episode: False,
        )

    assert not completion.is_set()


def test_unavailable_stable_rollback_raises_without_acknowledging_completion():
    completion = threading.Event()
    stable_manager = FakeStableManager()
    stable_manager.rollback = lambda updater: False
    publisher = FakePublisher()
    drift_monitor = FakeDriftMonitor()
    gate = FakeEvent()
    gate.set()
    replay_clears = []

    with pytest.raises(RuntimeError, match="^stable rollback unavailable$"):
        apply_adaptation_command(
            AdaptationCommand("rollback", 2, "performance_degradation", completion),
            checkpointed_episode=2,
            updater=FakeUpdater(),
            publisher=publisher,
            stable_manager=stable_manager,
            drift_monitor=drift_monitor,
            actor_gate_event=gate,
            clear_adaptation_replay=lambda: replay_clears.append(2),
            enable_actor_if_episode_active=lambda episode: False,
        )

    assert not completion.is_set()
    assert publisher.published == []
    assert drift_monitor.cleared == []
    assert replay_clears == []
    assert gate.is_set()


def test_step_start_snapshot_forces_fallback_even_if_ack_arrives_after_begin():
    pending = threading.Event()
    normal_decision = object()
    fallback_decision = object()

    class Agent:
        def fallback_pending(self, safety_state):
            assert safety_state == "safe"
            return fallback_decision

    pending_after_step, fallback_required = prepare_rollback_fallback(pending)
    decision = normal_decision
    pending.set()
    decision, fallback_used = enforce_prepared_rollback_fallback(
        online_agent=Agent(),
        decision=decision,
        safety_state="safe",
        fallback_required=fallback_required,
    )

    assert decision is fallback_decision
    assert pending_after_step is pending
    assert fallback_used

    pending_after_step, fallback_required = prepare_rollback_fallback(pending)
    decision, fallback_used = enforce_prepared_rollback_fallback(
        online_agent=Agent(),
        decision=normal_decision,
        safety_state="safe",
        fallback_required=fallback_required,
    )

    assert decision is normal_decision
    assert pending_after_step is None
    assert not fallback_used


def test_old_end_does_not_clear_new_episode_gate_or_reference():
    stable_manager = FakeStableManager()
    drift_monitor = FakeDriftMonitor()
    publisher = FakePublisher()
    updater = FakeUpdater()
    gate = FakeEvent()
    replay_clears = []
    checkpointed_episode, _ = apply_adaptation_command(
        AdaptationCommand("start_episode", 2),
        checkpointed_episode=0,
        updater=updater,
        publisher=publisher,
        stable_manager=stable_manager,
        drift_monitor=drift_monitor,
        actor_gate_event=gate,
        clear_adaptation_replay=lambda: replay_clears.append(2),
        enable_actor_if_episode_active=lambda episode: gate.set() or episode == 2,
    )

    checkpointed_episode, rollback_reason = apply_adaptation_command(
        AdaptationCommand("end_episode", 1),
        checkpointed_episode=checkpointed_episode,
        updater=updater,
        publisher=publisher,
        stable_manager=stable_manager,
        drift_monitor=drift_monitor,
        actor_gate_event=gate,
        clear_adaptation_replay=lambda: replay_clears.append(1),
        enable_actor_if_episode_active=lambda episode: False,
    )

    assert checkpointed_episode == 2
    assert rollback_reason is None
    assert gate.is_set()
    assert drift_monitor.cleared == []
    assert replay_clears == [2]


def test_start_command_clears_replay_before_enabling_actor_gate():
    events = []
    stable_manager = FakeStableManager()
    drift_monitor = FakeDriftMonitor()
    gate = FakeEvent()
    stable_manager.checkpoint = lambda updater: events.append("checkpoint")
    drift_monitor.start_episode = lambda actor, *, adaptation_episode: events.append("reference")
    gate.set = lambda: events.append("gate")

    checkpointed_episode, rollback_reason = apply_adaptation_command(
        AdaptationCommand("start_episode", 3),
        checkpointed_episode=0,
        updater=FakeUpdater(),
        publisher=FakePublisher(),
        stable_manager=stable_manager,
        drift_monitor=drift_monitor,
        actor_gate_event=gate,
        clear_adaptation_replay=lambda: events.append("clear"),
        enable_actor_if_episode_active=lambda episode: gate.set() or episode == 3,
    )

    assert checkpointed_episode == 3
    assert rollback_reason is None
    assert events == ["clear", "checkpoint", "reference", "gate"]


def test_inactive_start_command_installs_lifecycle_without_enabling_gate():
    stable_manager = FakeStableManager()
    drift_monitor = FakeDriftMonitor()
    replay_clears = []
    gate = FakeEvent()

    checkpointed_episode, rollback_reason = apply_adaptation_command(
        AdaptationCommand("start_episode", 3),
        checkpointed_episode=2,
        updater=FakeUpdater(),
        publisher=FakePublisher(),
        stable_manager=stable_manager,
        drift_monitor=drift_monitor,
        actor_gate_event=gate,
        clear_adaptation_replay=lambda: replay_clears.append(3),
        enable_actor_if_episode_active=lambda episode: False,
    )

    assert checkpointed_episode == 3
    assert rollback_reason is None
    assert replay_clears == [3]
    assert stable_manager.checkpoints == [FakeUpdater.version]
    assert drift_monitor.started == [(FakeUpdater.actor, 3)]
    assert not gate.is_set()


def test_matching_rollback_and_end_commands_clear_only_their_replay():
    for kind in ("rollback", "end_episode"):
        stable_manager = FakeStableManager()
        drift_monitor = FakeDriftMonitor()
        gate = FakeEvent()
        gate.set()
        replay_clears = []

        checkpointed_episode, _ = apply_adaptation_command(
            AdaptationCommand(kind, 2, "performance_degradation"),
            checkpointed_episode=2,
            updater=FakeUpdater(),
            publisher=FakePublisher(),
            stable_manager=stable_manager,
            drift_monitor=drift_monitor,
            actor_gate_event=gate,
            clear_adaptation_replay=lambda: replay_clears.append(2),
            enable_actor_if_episode_active=lambda episode: False,
        )

        assert checkpointed_episode == 0
        assert replay_clears == [2]
        assert drift_monitor.cleared == [2]
        assert not gate.is_set()


def test_final_step_cutoff_preserves_episode_for_safety_handling():
    gate = FakeEvent()
    gate.set()
    commands = []

    active_episode, cutoff_applied = cutoff_before_final_step(
        step=100,
        max_steps=100,
        active_episode=4,
        actor_gate_event=gate,
    )

    assert cutoff_applied
    assert active_episode == 4
    assert not gate.is_set()
    assert commands == []


def test_final_step_cutoff_closes_gate_without_active_episode():
    gate = FakeEvent()
    gate.set()
    commands = []

    active_episode, cutoff_applied = cutoff_before_final_step(
        step=100,
        max_steps=100,
        active_episode=0,
        actor_gate_event=gate,
    )

    assert cutoff_applied
    assert active_episode == 0
    assert not gate.is_set()
    assert commands == []


def test_stale_update_result_preserves_current_log_and_counters():
    counters = RuntimeCounters()
    last_update = {"version": 8, "policy_drift_episode": 2}
    stale_stats = make_update_stats(
        accepted=False,
        version=7,
        critic_updated=False,
        actor_updated=False,
        rejection_reason="stale_adaptation_episode",
    )

    event = record_update_result(
        stats=stale_stats,
        policy_drift=None,
        last_update=last_update,
        counters=counters,
        step=700,
        adaptation_episode=1,
    )

    assert event is None
    assert last_update == {"version": 8, "policy_drift_episode": 2}
    assert counters.critic_updates == 0
    assert counters.actor_updates == 0


def test_critic_update_does_not_enqueue_actor_event():
    counters = RuntimeCounters()
    last_update = {}

    event = record_update_result(
        stats=make_update_stats(actor_updated=False),
        policy_drift=None,
        last_update=last_update,
        counters=counters,
        step=701,
        adaptation_episode=1,
    )

    assert event is None
    assert counters.critic_updates == 1
    assert counters.actor_updates == 0


def test_actor_updates_return_complete_events_across_intervening_critic_update():
    counters = RuntimeCounters()
    last_update = {}
    first_drift = PolicyDrift(0.01, 0.02, 0.03, 0.04, 1, 8, 702)
    second_drift = PolicyDrift(0.05, 0.06, 0.07, 0.08, 1, 9, 704)

    first_event = record_update_result(
        stats=make_update_stats(version=8),
        policy_drift=first_drift,
        last_update=last_update,
        counters=counters,
        step=702,
        adaptation_episode=1,
    )
    critic_event = record_update_result(
        stats=make_update_stats(version=8, actor_updated=False),
        policy_drift=None,
        last_update=last_update,
        counters=counters,
        step=703,
        adaptation_episode=1,
    )
    second_event = record_update_result(
        stats=make_update_stats(version=9),
        policy_drift=second_drift,
        last_update=last_update,
        counters=counters,
        step=704,
        adaptation_episode=1,
    )

    events = [first_event, second_event]

    assert critic_event is None
    assert [event["version"] for event in events] == [8, 9]
    assert [event["policy_drift_update_step"] for event in events] == [702, 704]
    assert all(event["accepted"] and event["actor_updated"] for event in events)
    assert all(event["mean_target_q"] == 6.0 for event in events)
    assert all(event["mean_td_error"] == 7.0 for event in events)
    assert all(event["n_step"] == 1 for event in events)
    assert all(
        torch.isfinite(torch.tensor(value))
        for event in events
        for key, value in event.items()
        if key.startswith("policy_") and isinstance(value, float)
    )
    assert len(events) == counters.actor_updates == 2


class RecordingEventStream:
    def __init__(self):
        self.data = ""
        self.flush_count = 0

    def write(self, value):
        self.data += value

    def flush(self):
        self.flush_count += 1


def test_actor_event_is_written_and_flushed_before_later_updater_failure():
    stream = RecordingEventStream()
    event = record_update_result(
        stats=make_update_stats(version=8),
        policy_drift=PolicyDrift(0.01, 0.02, 0.03, 0.04, 1, 8, 702),
        last_update={},
        counters=RuntimeCounters(),
        step=702,
        adaptation_episode=1,
    )

    write_actor_update_event(stream, event)
    try:
        raise RuntimeError("second updater iteration failed")
    except RuntimeError:
        pass

    assert stream.flush_count == 1
    assert json.loads(stream.data) == event


def test_actor_event_serialization_rejects_nonfinite_values():
    stream = RecordingEventStream()

    try:
        write_actor_update_event(stream, {"policy_residual_mse": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite Actor event must not be serialized")

    assert stream.data == ""
    assert stream.flush_count == 0


def test_accepted_actor_update_requires_policy_drift():
    counters = RuntimeCounters()

    try:
        record_update_result(
            stats=make_update_stats(),
            policy_drift=None,
            last_update={},
            counters=counters,
            step=705,
            adaptation_episode=1,
        )
    except AssertionError as error:
        assert "policy drift" in str(error).lower()
    else:
        raise AssertionError("accepted Actor update without policy drift must fail")

    assert counters.actor_updates == 0


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


def test_rollback_record_fields_keep_current_event_separate_from_history():
    assert rollback_record_fields("", "performance_degradation") == {
        "rollback_reason": "",
        "last_rollback_reason": "performance_degradation",
    }


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

    normal = runtime.observe_step(
        step=14,
        velocity_error=0.10,
        tilt_rms=0.02,
        nonfoot_contact=0,
        reward=1.0,
    )
    assert normal.record_fields["rollback_reason"] == ""
    assert normal.record_fields["last_rollback_reason"] == "performance_degradation"


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


def test_leaving_adapting_resets_performance_and_emits_end_episode_command():
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
    assert result.commands == (AdaptationCommand("end_episode", 1),)
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
