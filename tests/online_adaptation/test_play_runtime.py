import json
import threading
from argparse import Namespace

import pytest
from tensordict import TensorDict

from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import (
    AdaptationCommand,
    AdaptationRuntimeOrchestrator,
    PostStepSnapshot,
    actor_replay_transition_eligible,
    PostStepRecoveryRamp,
    RuntimeCounters,
    TransitionUpdateBudget,
    enable_actor_gate_if_active,
    evaluate_track_completion_step,
    enforce_prepared_rollback_fallback,
    actor_version_for_record,
    policy_drift_record,
    actor_update_options,
    prepare_control_inputs,
    validate_terrain_checkpoint_metadata,
    pin_forward_command,
    normalized_forward_progress,
    observe_completion_snapshot,
    prepare_rollback_fallback,
    apply_adaptation_command,
    build_track_completion,
    close_active_episode_for_worker_boundary,
    cutoff_before_final_step,
    record_update_result,
    replace_base_policy_group,
    rollback_record_fields,
    resolve_track_boundary_x,
    select_post_step_snapshot,
    control_phase,
    control_is_deterministic,
    start_control_timing,
    transition_episode_publication,
    update_pause_event,
    warm_up_control_path,
    write_actor_update_event,
    resolve_worker_geometry,
    split_worker_endings,
    step_clearance_reward_active,
    termination_reasons_from_step_result,
    write_episode_summary,
)
from rl_training.online_adaptation.metrics import AdaptationPerformance
from rl_training.online_adaptation.novelty import NoveltyConfig, NoveltyGate, NoveltyState
from rl_training.online_adaptation.policy_drift import PolicyDrift
from rl_training.online_adaptation.types import UpdateStats
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from rl_training.online_adaptation.track_completion import TrackCompletion
import torch


def test_stalled_transitions_are_thinned_after_actor_replay_warmup():
    assert actor_replay_transition_eligible(
        progress_stalled=False,
        steps_without_progress=1,
        replay_size=500,
        minimum_replay_size=128,
        stalled_replay_stride=16,
    )
    assert actor_replay_transition_eligible(
        progress_stalled=True,
        steps_without_progress=127,
        replay_size=127,
        minimum_replay_size=128,
        stalled_replay_stride=16,
    )
    assert not actor_replay_transition_eligible(
        progress_stalled=True,
        steps_without_progress=129,
        replay_size=128,
        minimum_replay_size=128,
        stalled_replay_stride=16,
    )
    assert actor_replay_transition_eligible(
        progress_stalled=True,
        steps_without_progress=144,
        replay_size=128,
        minimum_replay_size=128,
        stalled_replay_stride=16,
    )

    with pytest.raises(TypeError, match="progress_stalled"):
        actor_replay_transition_eligible(
            progress_stalled=1,
            steps_without_progress=0,
            replay_size=0,
            minimum_replay_size=128,
            stalled_replay_stride=16,
        )
    with pytest.raises(ValueError, match="replay_size"):
        actor_replay_transition_eligible(
            progress_stalled=True,
            steps_without_progress=0,
            replay_size=-1,
            minimum_replay_size=128,
            stalled_replay_stride=16,
        )
    with pytest.raises(ValueError, match="steps_without_progress"):
        actor_replay_transition_eligible(
            progress_stalled=True,
            steps_without_progress=-1,
            replay_size=0,
            minimum_replay_size=128,
            stalled_replay_stride=16,
        )


def test_terrain_checkpoint_metadata_rejects_source_mismatch():
    geometry = {"terrain_feature_source": "geometry", "terrain_feature_schema_version": 1}
    height_scan = {
        "terrain_feature_source": "height_scan",
        "terrain_feature_schema_version": 1,
        "height_scanner_name": "height_scanner",
        "height_scan": {"rows": 23, "cols": 15},
    }
    with pytest.raises(ValueError, match="terrain metadata"):
        validate_terrain_checkpoint_metadata(geometry, height_scan)
    with pytest.raises(ValueError, match="lacks terrain metadata"):
        validate_terrain_checkpoint_metadata(None, height_scan)
    validate_terrain_checkpoint_metadata(None, geometry)


def test_post_step_recovery_ramp_reaches_target_in_exact_number_of_steps():
    ramp = PostStepRecoveryRamp(transition_steps=4, target_gain=0.25)

    values = [
        ramp.step(reached=True, current_gain=0.5)
        for _ in range(4)
    ]

    assert [value[0] for value in values] == pytest.approx([0.4375, 0.375, 0.3125, 0.25])
    assert [value[1] for value in values] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert [value[2] for value in values] == [True, True, True, False]


def test_actor_update_options_forward_policy_drift_limit():
    assert actor_update_options(
        Namespace(max_policy_residual_mse=0.125)
    ) == {"max_residual_mse": 0.125}


def test_post_step_recovery_ramp_does_not_start_before_crossing():
    ramp = PostStepRecoveryRamp(transition_steps=4, target_gain=0.25)

    assert ramp.step(reached=False, current_gain=0.5) == (0.5, 0.0, False)
    assert ramp.elapsed_steps == 0


@pytest.mark.parametrize("base_x", [-7.49, -7.0, -6.51])
def test_step_clearance_reward_covers_stalls_around_geometric_edge(base_x):
    assert step_clearance_reward_active(
        learning_enabled=True,
        progress_stalled=True,
        base_x=base_x,
        step_edge_x=-7.0,
        interaction_distance=0.5,
    )


def test_step_clearance_reward_requires_learning_and_stall():
    options = {
        "base_x": -6.9,
        "step_edge_x": -7.0,
        "interaction_distance": 0.5,
    }

    assert not step_clearance_reward_active(
        learning_enabled=False, progress_stalled=True, **options
    )
    assert not step_clearance_reward_active(
        learning_enabled=True, progress_stalled=False, **options
    )


@pytest.mark.parametrize(
    ("state", "learning_enabled", "expected"),
    [
        (NoveltyState.ADAPTING, True, False),
        (NoveltyState.MONITORING, True, True),
        (NoveltyState.COOLDOWN, True, True),
        (NoveltyState.ADAPTING, False, True),
    ],
)
def test_adapting_stochastic_control_is_scoped_to_learning(state, learning_enabled, expected):
    assert control_is_deterministic(
        stochastic_control=False,
        adapting_stochastic_control=True,
        learning_enabled=learning_enabled,
        novelty_state=state,
    ) is expected


def test_global_stochastic_control_remains_an_explicit_override():
    assert not control_is_deterministic(
        stochastic_control=True,
        adapting_stochastic_control=False,
        learning_enabled=False,
        novelty_state=NoveltyState.MONITORING,
    )


class _TrackGenerator:
    size = (20.0, 4.0)
    sub_terrains = {"flat_online": type("Flat", (), {"flat_length": 3.0, "step_length": 2.0})()}


class _TrackEnvCfg:
    class scene:
        class terrain:
            terrain_generator = _TrackGenerator()


def test_build_track_completion_uses_generated_dimensions_and_reset_x():
    completion = build_track_completion(
        _TrackEnvCfg(),
        stage_name="rough_4cm",
        initial_x=-8.5,
        protocol=default_flat_online_protocol(),
    )

    assert completion.target_x == pytest.approx(9.5)
    assert completion.initial_x == pytest.approx(-8.5)


def test_track_completion_stops_only_for_safe_nonterminal_crossing():
    completion = TrackCompletion.start(
        stage_name="step_2cm",
        initial_x=-8.5,
        flat_length=3.0,
        step_length=2.0,
        track_length=20.0,
        step_clearance_margin_m=0.5,
        track_end_buffer_m=0.5,
    )
    completion.observe(-4.5)

    safe = evaluate_track_completion_step(
        completion, environment_done=False, safety_violation=False
    )
    terminal = evaluate_track_completion_step(
        completion, environment_done=True, safety_violation=False
    )
    unsafe = evaluate_track_completion_step(
        completion, environment_done=False, safety_violation=True
    )

    assert safe.reached and safe.passed and safe.should_stop
    assert terminal.reached and not terminal.passed and not terminal.should_stop
    assert unsafe.reached and not unsafe.passed and not unsafe.should_stop


def test_track_completion_does_not_pass_target_without_strict_qualification():
    completion = TrackCompletion.start(
        stage_name="step_2cm",
        initial_x=-8.5,
        flat_length=3.0,
        step_length=2.0,
        track_length=20.0,
        step_clearance_margin_m=0.5,
        track_end_buffer_m=0.5,
        initial_y=2.0,
        lane_width_m=2.0,
        track_width_m=4.0,
    )
    completion.observe(completion.target_x)

    decision = evaluate_track_completion_step(
        completion, environment_done=False, safety_violation=False
    )

    assert decision.reached
    assert not decision.passed
    assert not decision.should_stop


def test_worker_boundary_closes_active_episode_and_forces_novelty_cooldown():
    gate = FakeEvent()
    gate.set()
    active_episode = 3
    commands = []

    class Gate:
        def force_cooldown(self):
            return "cooldown"

    class Performance:
        starts = 0

        def start_episode(self):
            self.starts += 1

    performance = Performance()

    def write_active_episode(value):
        nonlocal active_episode
        active_episode = value

    decision = close_active_episode_for_worker_boundary(
        active_episode=active_episode,
        shared_lock=threading.Lock(),
        actor_gate_event=gate,
        read_active_episode=lambda: active_episode,
        write_active_episode=write_active_episode,
        enqueue_command=commands.append,
        novelty_gate=Gate(),
        adaptation_performance=performance,
    )

    assert decision == "cooldown"
    assert active_episode == 0
    assert not gate.is_set()
    assert commands == [AdaptationCommand("end_episode", 3)]
    assert performance.starts == 1


def test_worker_boundary_without_active_episode_still_forces_novelty_cooldown():
    gate = FakeEvent()
    gate.set()
    commands = []

    class Gate:
        def force_cooldown(self):
            return "cooldown"

    class Performance:
        starts = 0

        def start_episode(self):
            self.starts += 1

    performance = Performance()

    decision = close_active_episode_for_worker_boundary(
        active_episode=0,
        shared_lock=threading.Lock(),
        actor_gate_event=gate,
        read_active_episode=lambda: 0,
        write_active_episode=lambda _value: None,
        enqueue_command=commands.append,
        novelty_gate=Gate(),
        adaptation_performance=performance,
    )

    assert decision == "cooldown"
    assert not gate.is_set()
    assert commands == []
    assert performance.starts == 1


def test_start_control_timing_synchronizes_cuda_before_reading_clock(monkeypatch):
    events = []

    class Device:
        type = "cuda"

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: events.append("synchronize"),
    )

    started = start_control_timing(
        device=Device(),
        clock=lambda: events.append("clock") or 12.5,
    )

    assert started == 12.5
    assert events == ["synchronize", "clock"]


def test_control_phase_serializes_cuda_work_before_and_during_control(monkeypatch):
    events = []

    class Device:
        type = "cuda"

    class Lock:
        def __enter__(self):
            events.append("acquire")

        def __exit__(self, *_args):
            events.append("release")

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: events.append("synchronize"),
    )

    with control_phase(
        control_update_lock=Lock(),
        device=Device(),
        clock=lambda: events.append("clock") or 5.0,
    ) as started:
        events.append(("body", started))

    assert events == ["acquire", "synchronize", "clock", ("body", 5.0), "release"]


def test_prepare_control_inputs_reads_frame_and_safety_state_before_control_phase():
    calls = []

    class Adapter:
        def frame(self, previous_action):
            calls.append(("frame", tuple(previous_action.shape)))
            return torch.zeros(1, 49)

        def safety_state(self, timeout):
            calls.append(("safety", tuple(timeout.shape)))
            return "safety_state"

    frame, timeout, safety_state = prepare_control_inputs(
        adapter=Adapter(),
        previous_action=torch.zeros(1, 12),
        device=torch.device("cpu"),
    )

    assert frame.shape == (1, 49)
    assert timeout.shape == (1, 1)
    assert timeout.dtype is torch.bool
    assert safety_state == "safety_state"
    assert calls == [("frame", (1, 12)), ("safety", (1, 1))]


def test_pin_forward_command_overrides_reset_state_and_special_modes():
    class Term:
        vel_command_b = torch.zeros(2, 3)
        target_vel_command_b = torch.zeros(2, 3)
        is_standing_env = torch.ones(2, dtype=torch.bool)
        is_heading_env = torch.ones(2, dtype=torch.bool)

    term = Term()

    class Manager:
        def get_command(self, name):
            assert name == "base_velocity"
            return term.vel_command_b

        def get_term(self, name):
            assert name == "base_velocity"
            return term

    env = type("Env", (), {"command_manager": Manager()})()

    pin_forward_command(env, 0.5)

    assert torch.equal(term.vel_command_b, torch.tensor([[0.5, 0.0, 0.0]]).expand(2, -1))
    assert torch.equal(
        term.target_vel_command_b, torch.tensor([[0.5, 0.0, 0.0]]).expand(2, -1)
    )
    assert not term.is_standing_env.any()
    assert not term.is_heading_env.any()


def test_normalized_forward_progress_masks_reset_jump_and_clamps_motion():
    progress = normalized_forward_progress(
        torch.tensor([[0.01], [-10.0], [0.5]]),
        forward_speed=0.5,
        step_dt=0.02,
        valid_mask=torch.tensor([[True], [False], [True]]),
    )

    assert torch.equal(progress, torch.tensor([[1.0], [0.0], [1.0]]))


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


def test_transition_update_budget_is_charged_by_committed_samples_and_capped():
    budget = TransitionUpdateBudget(
        critic_updates_per_transition=0.5,
        actor_updates_per_transition=0.005,
        max_pending_critic_updates=4,
        max_pending_actor_updates=2,
    )
    budget.add(critic_transitions=10, actor_transitions=300)
    assert budget.snapshot() == {"critic": 4.0, "actor": 1.5}
    assert budget.take_critic()
    assert budget.take_actor()
    assert budget.snapshot() == {"critic": 3.0, "actor": 0.5}
    assert not budget.take_actor()


def test_transition_update_budget_clears_episode_local_actor_budget():
    budget = TransitionUpdateBudget(
        critic_updates_per_transition=0.5,
        actor_updates_per_transition=0.01,
    )
    budget.add(critic_transitions=4, actor_transitions=100)
    assert budget.snapshot() == {"critic": 2.0, "actor": 1.0}
    budget.clear_actor()
    assert budget.snapshot() == {"critic": 2.0, "actor": 0.0}


def test_transition_update_budget_rejects_invalid_transition_counts():
    budget = TransitionUpdateBudget(
        critic_updates_per_transition=0.5,
        actor_updates_per_transition=0.005,
    )
    with pytest.raises(ValueError):
        budget.add(critic_transitions=-1, actor_transitions=0)
    with pytest.raises(ValueError):
        budget.add(critic_transitions=0, actor_transitions=True)


def test_runtime_counters_track_committed_transitions_and_rejections():
    counters = RuntimeCounters()
    counters.record_transition_commits(critic_transitions=3, actor_transitions=2)
    counters.record_transition_commits(critic_transitions=1, actor_transitions=0)
    assert counters.critic_transitions_committed == 4
    assert counters.actor_transitions_committed == 2

    counters.record_update_result(
        make_update_stats(accepted=False, rejection_reason="nonfinite_critic_loss")
    )
    assert counters.update_attempts == 1
    assert counters.update_rejections == {"nonfinite_critic_loss": 1}


def test_worker_geometry_uses_development_or_sealed_protocol_paths():
    development = resolve_worker_geometry("step_2cm", 101, terrain_split="development")
    sealed = resolve_worker_geometry("step_2cm", 1009, terrain_split="sealed")
    assert development.step_height == 0.02
    assert development.seed == 101
    assert sealed.seed == 1009


def test_worker_geometry_rejects_unknown_split():
    with pytest.raises(ValueError, match="unknown terrain split"):
        resolve_worker_geometry("step_2cm", 101, terrain_split="unknown")


def test_worker_endings_support_gymnasium_and_reject_ambiguous_rsl_rows():
    terminated, truncated, bootstrap = split_worker_endings(
        (None, None, torch.tensor([True]), torch.tensor([False]), {})
    )
    assert terminated.item() and not truncated.item() and not bootstrap.item()
    with pytest.raises(RuntimeError, match="cannot distinguish termination from truncation"):
        split_worker_endings((None, None, torch.tensor([True]), {}))


@pytest.mark.parametrize(
    "last_y,environment_done,safety_violation,expected",
    [(0.0, False, False, True), (1.0, False, False, False),
     (0.0, True, False, False), (0.0, False, True, False)],
)
def test_completion_reward_uses_current_pose_and_ending_status(
    last_y, environment_done, safety_violation, expected,
):
    from types import SimpleNamespace

    completion = TrackCompletion.start(
        stage_name="rough_4cm", initial_x=0.0, flat_length=3.0,
        step_length=2.0, track_length=10.0, step_clearance_margin_m=0.5,
        track_end_buffer_m=0.5, initial_y=0.0, lane_width_m=2.0,
        track_width_m=4.0, min_challenge_observation_steps=2,
        min_challenge_swing_steps=2, min_stable_completion_steps=2,
    )
    completion.observe_pose(
        4.0, 0.0, foot_y=[0.0] * 4, swing_lift=[0.03] * 4,
        velocity_error=0.1, tilt_rms=0.1, foot_slip=0.1,
    )
    assert not completion.qualification_satisfied
    snapshot = PostStepSnapshot(
        mask=torch.tensor([[environment_done]]), policy_observation=None,
        frame=None, signals=SimpleNamespace(foot_slip=torch.tensor([[0.1]])),
        base_x=torch.tensor([[completion.target_x]]),
        base_y=torch.tensor([[last_y]]), foot_y=torch.full((1, 4), last_y),
    )
    assert observe_completion_snapshot(
        completion, snapshot, swing_lift=torch.full((1, 4), 0.03),
        velocity_error=0.1, tilt_rms=0.1, environment_done=environment_done,
        safety_violation=safety_violation,
    ) is expected
    assert completion.reached
    assert completion.lateral_violation_steps == int(last_y > 0.65)


def test_terminal_step_uses_pre_reset_snapshot_instead_of_reset_state():
    terminal = PostStepSnapshot(
        mask=torch.tensor([[True]]),
        policy_observation=torch.tensor([[1.0]]),
        frame=torch.tensor([[2.0]]),
        signals="terminal-signals",
        base_x=torch.tensor([[3.0]]),
    )
    reset = PostStepSnapshot(
        mask=torch.tensor([[False]]),
        policy_observation=torch.tensor([[10.0]]),
        frame=torch.tensor([[20.0]]),
        signals="reset-signals",
        base_x=torch.tensor([[-8.5]]),
    )

    selected = select_post_step_snapshot(
        done=torch.tensor([[True]]),
        terminal=terminal,
        post_reset=reset,
    )

    assert selected is terminal
    assert selected.signals == "terminal-signals"
    torch.testing.assert_close(selected.frame, torch.tensor([[2.0]]))
    torch.testing.assert_close(selected.base_x, torch.tensor([[3.0]]))


def test_terminal_policy_group_is_restored_in_rsl_observation_container():
    terminal = PostStepSnapshot(
        mask=torch.tensor([[True]]),
        policy_observation=torch.tensor([[1.0, 2.0]]),
        frame=torch.zeros(1, 1),
        signals=object(),
        base_x=torch.zeros(1, 1),
    )
    reset_observation = TensorDict(
        {
            "policy": torch.tensor([[10.0, 20.0]]),
            "critic": torch.tensor([[30.0, 40.0, 50.0]]),
        },
        batch_size=[1],
    )
    reset = PostStepSnapshot(
        mask=torch.tensor([[False]]),
        policy_observation=reset_observation["policy"],
        frame=torch.zeros(1, 1),
        signals=object(),
        base_x=torch.zeros(1, 1),
    )

    selected = select_post_step_snapshot(
        done=torch.tensor([[True]]),
        terminal=terminal,
        post_reset=reset,
    )
    restored = replace_base_policy_group(
        reset_observation,
        selected.policy_observation,
    )

    assert isinstance(restored, TensorDict)
    torch.testing.assert_close(restored["policy"], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(
        restored["critic"], torch.tensor([[30.0, 40.0, 50.0]])
    )
    torch.testing.assert_close(
        reset_observation["policy"], torch.tensor([[10.0, 20.0]])
    )

    def grouped_policy(obs):
        return obs["policy"].sum(dim=-1)

    torch.testing.assert_close(grouped_policy(restored), torch.tensor([3.0]))


def test_terminal_policy_group_must_match_reset_observation_schema():
    reset_observation = TensorDict(
        {"policy": torch.zeros(1, 2)},
        batch_size=[1],
    )

    with pytest.raises(RuntimeError, match="must match"):
        replace_base_policy_group(reset_observation, torch.zeros(1, 3))


def test_terminal_snapshot_mask_must_match_done_rows():
    terminal = PostStepSnapshot(
        mask=torch.tensor([[False]]),
        policy_observation=torch.zeros(1, 1),
        frame=torch.zeros(1, 1),
        signals=object(),
        base_x=torch.zeros(1, 1),
    )
    reset = PostStepSnapshot(
        mask=torch.tensor([[False]]),
        policy_observation=torch.zeros(1, 1),
        frame=torch.zeros(1, 1),
        signals=object(),
        base_x=torch.zeros(1, 1),
    )

    with pytest.raises(RuntimeError, match="capture mask"):
        select_post_step_snapshot(
            done=torch.tensor([[True]]),
            terminal=terminal,
            post_reset=reset,
        )


def test_nonterminal_step_uses_live_post_step_state_without_capture():
    reset = PostStepSnapshot(
        mask=torch.tensor([[False]]),
        policy_observation=torch.tensor([[10.0]]),
        frame=torch.tensor([[20.0]]),
        signals="live-signals",
        base_x=torch.tensor([[30.0]]),
    )

    assert select_post_step_snapshot(
        done=torch.tensor([[False]]),
        terminal=None,
        post_reset=reset,
    ) is reset


def test_termination_reasons_read_episode_log_from_rsl_step_result():
    step_result = (
        None,
        None,
        torch.tensor([1]),
        {
            "log": {
                "Episode_Termination/illegal_contact": 1.0,
                "Episode_Termination/bad_orientation_2": 0.0,
            }
        },
    )

    assert termination_reasons_from_step_result(step_result) == ("illegal_contact",)


def test_termination_reasons_ignore_inactive_and_malformed_terms():
    step_result = (
        None,
        None,
        torch.tensor([1]),
        {
            "log": {
                "Episode_Termination/illegal_contact": 0.0,
                "Episode_Termination/bad_orientation_2": "invalid",
                "unrelated": 1.0,
            }
        },
    )

    assert termination_reasons_from_step_result(step_result) == ()


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


def test_fall_reset_preserves_live_actor_and_clears_episode_state():
    stable_manager = FakeStableManager()
    drift_monitor = FakeDriftMonitor()
    publisher = FakePublisher()
    updater = FakeUpdater()
    gate = FakeEvent()
    gate.set()
    replay_clears = []

    checkpointed_episode, reason = apply_adaptation_command(
        AdaptationCommand("fall_reset", 2, "fall_reset"),
        checkpointed_episode=2,
        updater=updater,
        publisher=publisher,
        stable_manager=stable_manager,
        drift_monitor=drift_monitor,
        actor_gate_event=gate,
        clear_adaptation_replay=lambda: replay_clears.append(True),
        enable_actor_if_episode_active=lambda episode: False,
    )

    assert checkpointed_episode == 0
    assert reason == "fall_reset"
    assert stable_manager.rollbacks == []
    assert publisher.published == [(FakeUpdater.actor, FakeUpdater.version)]
    assert drift_monitor.cleared == [2]
    assert replay_clears == [True]
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


def test_critic_only_update_makes_uncomputed_losses_json_null(tmp_path):
    counters = RuntimeCounters()
    last_update = {}

    record_update_result(
        stats=make_update_stats(
            actor_loss=float("nan"),
            alpha_loss=float("nan"),
            actor_updated=False,
        ),
        policy_drift=None,
        last_update=last_update,
        counters=counters,
        step=701,
        adaptation_episode=0,
    )

    assert last_update["actor_loss"] is None
    assert last_update["alpha_loss"] is None
    output = tmp_path / "episode_summary.json"
    write_episode_summary(output, {"last_update": last_update})
    assert json.loads(output.read_text()) == {
        "last_update": {**last_update, "actor_loss": None, "alpha_loss": None}
    }


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


def test_fall_ends_adaptation_without_counting_as_safety_rollback():
    runtime = make_orchestrator()
    enter_adapting(runtime)

    result = runtime.observe_step(
        step=8,
        velocity_error=0.50,
        tilt_rms=0.20,
        nonfoot_contact=0,
        reward=0.1,
        done=True,
    )

    assert result.commands == (AdaptationCommand("fall_reset", 1, "fall_reset"),)
    assert result.clear_adaptation_replay
    assert runtime.active_episode == 0
    assert runtime.counters.rollbacks == 0
    assert runtime.counters.fall_resets == 1
    assert result.record_fields["rollback_reason"] == "fall_reset"
    assert result.record_fields["adaptation_state"] == "cooldown"


def test_terminal_fall_takes_soft_reset_precedence_over_same_step_pause():
    runtime = make_orchestrator()
    enter_adapting(runtime)

    result = runtime.observe_step(
        step=8,
        velocity_error=0.50,
        tilt_rms=0.40,
        nonfoot_contact=1,
        reward=-1.0,
        paused=True,
        done=True,
        safety_reasons=("height",),
    )

    assert result.commands == (AdaptationCommand("fall_reset", 1, "fall_reset"),)
    assert runtime.counters.fall_resets == 1
    assert runtime.counters.safety_rollbacks == 0


def test_terminal_without_active_episode_cannot_open_adaptation_on_reset_sample():
    runtime = make_orchestrator()
    for step in range(1, 6):
        runtime.observe_step(
            step=step,
            velocity_error=0.10,
            tilt_rms=0.02,
            nonfoot_contact=0,
            reward=1.0,
        )
    first_crossing = runtime.observe_step(
        step=6,
        velocity_error=0.50,
        tilt_rms=0.02,
        nonfoot_contact=0,
        reward=0.2,
    )
    assert first_crossing.record_fields["adaptation_state"] == "monitoring"

    terminal = runtime.observe_step(
        step=7,
        velocity_error=0.50,
        tilt_rms=0.02,
        nonfoot_contact=0,
        reward=0.2,
        done=True,
    )

    assert terminal.commands == ()
    assert terminal.record_fields["adaptation_state"] == "cooldown"
    assert not terminal.adaptation_enabled
    assert runtime.active_episode == 0
    assert runtime.novelty_gate.state_dict()["state"] == "cooldown"


def test_terminal_with_timeout_still_closes_active_adaptation_episode():
    runtime = make_orchestrator()
    enter_adapting(runtime)

    result = runtime.observe_step(
        step=8,
        velocity_error=0.50,
        tilt_rms=0.20,
        nonfoot_contact=0,
        reward=0.1,
        control_timed_out=True,
        done=True,
    )

    assert result.commands == (AdaptationCommand("fall_reset", 1, "fall_reset"),)
    assert result.record_fields["adaptation_state"] == "cooldown"
    assert runtime.active_episode == 0


def test_control_timeout_preserves_active_actor_episode_before_actor_update():
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

    assert result.commands == ()
    assert runtime.active_episode == 1
    assert result.record_fields["control_timeout_preserved_actor"] is True
    assert result.record_fields["rollback_reason"] == ""
    assert not result.record_fields["performance_evaluation_enabled"]


def test_timeout_fallback_does_not_latch_pause_for_next_control_step():
    paused_event = threading.Event()
    paused_event.set()

    update_pause_event(paused_event, paused=True, control_timed_out=True)

    assert not paused_event.is_set()


def test_new_physical_trial_restores_critic_credit_but_discards_episode_actor_credit():
    budget = TransitionUpdateBudget(critic_updates_per_transition=1., actor_updates_per_transition=.5)
    budget.restore_for_new_trial({"critic": 12.5, "actor": 3.})
    assert budget.snapshot() == {"critic": 12.5, "actor": 0.}
    assert budget.take_critic()
    assert budget.snapshot()["critic"] == 11.5
    assert not budget.take_actor()
    for invalid in (None, {"critic": 300., "actor": 0.}, {"critic": float("nan"), "actor": 0.}):
        with pytest.raises(ValueError, match="update budget"):
            budget.restore_for_new_trial(invalid)
    assert budget.snapshot()["critic"] == 11.5


def test_target_warmup_records_and_summary_share_strict_json_null_semantics(tmp_path):
    from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import record_target_critic_update
    records, last = [], {}
    counters = RuntimeCounters()
    path = tmp_path / 'updates.jsonl'
    for _ in range(2):
        record_target_critic_update(
            stats=make_update_stats(actor_loss=float('nan'), alpha_loss=float('nan'), actor_updated=False),
            records=records, last_update=last, counters=counters,
            step=4000, adaptation_episode=0, path=path, residual_gain=.5,
        )
    summary = tmp_path / 'episode_summary.json'
    write_episode_summary(summary, {'warmup_critic_diagnostics': records})
    assert 'NaN' not in path.read_text() and 'NaN' not in summary.read_text()
    assert len(records) == counters.critic_updates == 2 and counters.actor_updates == 0
    assert [json.loads(s) for s in path.read_text().splitlines()] == records
    assert all(r['actor_loss'] is None and r['alpha_loss'] is None for r in records)
    assert all(r['residual_gain'] == .5 for r in records)
    last['actor_loss'] = 99.
    assert records[0]['actor_loss'] is None


@pytest.mark.parametrize('mode,frozen,expected', [
    ('base',True,0.),('residual_frozen',True,.5),('online',False,0.),
    ('residual_frozen',False,0.),
])
def test_managed_frozen_candidate_uses_adaptation_gain_instead_of_zero_waiting_gain(mode,frozen,expected):
    from scripts.reinforcement_learning.online_adaptation.play_online_adaptation import resolve_control_gain
    assert resolve_control_gain(mode=mode,managed_frozen=frozen,residual_gain=0.,adapting_residual_gain=.5)==expected
