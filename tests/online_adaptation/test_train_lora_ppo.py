import pytest

from scripts.reinforcement_learning.online_adaptation.train_lora_ppo import (
    CONTACT_TERMINATION_PROTOCOL,
    _assert_forward_command,
    _capture_step_state,
    _completion_passed,
    _configure_forward_command,
    _evaluation_mode,
    _finalize_checkpoint,
    _lateral_deviation,
    _learning_boundary,
    _pin_forward_observation,
    _prepare_evaluation_args,
    _rollback_reason,
    _task_failure_reason,
    build_parser,
)


def test_flat_probe_forces_frozen_sampling_without_track_or_task_deadline():
    args = build_parser().parse_args([
        "--base-checkpoint", "base.pt", "--flat-retention-probe",
        "--require-track-completion", "--trial-deadline-steps", "2000",
    ])
    _prepare_evaluation_args(args)
    assert args.evaluation_only
    assert not args.require_track_completion
    assert args.trial_deadline_steps is None
    assert _evaluation_mode(args) == "base_frozen"


def test_flat_probe_rejects_continuing_across_physical_resets():
    args = build_parser().parse_args([
        "--base-checkpoint", "base.pt", "--flat-retention-probe", "--continue-after-termination",
    ])
    with pytest.raises(SystemExit, match="cannot continue"):
        _prepare_evaluation_args(args)


def test_regular_terrain_training_keeps_its_existing_deadline_and_qualification():
    args = build_parser().parse_args([
        "--base-checkpoint", "base.pt", "--require-track-completion", "--trial-deadline-steps", "2000",
    ])
    _prepare_evaluation_args(args)
    assert not args.evaluation_only
    assert args.require_track_completion
    assert args.trial_deadline_steps == 2000
    assert _evaluation_mode(args) == "online_learning"


@pytest.mark.parametrize("checkpoint", [None, "learned.pt"])
@pytest.mark.parametrize("flat", [False, True])
def test_frozen_base_or_learned_evaluation_passes_main_validation_without_launching_simulator(
    monkeypatch, checkpoint, flat,
):
    from types import SimpleNamespace
    from scripts.reinforcement_learning.online_adaptation import train_lora_ppo as worker

    options = ["--base-checkpoint", "base.pt", "--evaluation-only"]
    if checkpoint:
        options.extend(("--checkpoint-in", checkpoint))
    if flat:
        options.append("--flat-retention-probe")
    args = build_parser().parse_args(options)
    seen = []

    class LaunchReached(Exception):
        pass

    def fake_launcher(prepared):
        seen.append(prepared)
        raise LaunchReached

    monkeypatch.setattr(worker, "build_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(worker, "AppLauncher", fake_launcher)
    with pytest.raises(LaunchReached):
        worker.main()
    assert seen == [args]
    assert args.evaluation_only
    assert _evaluation_mode(args) == ("learned_frozen" if checkpoint else "base_frozen")


def test_plane_configuration_keeps_robot_material_command_and_physical_safety():
    from types import SimpleNamespace
    from rl_training.online_adaptation.flat_retention import configure_flat_retention_plane

    material, robot, command, observation = object(), object(), object(), object()
    contact, orientation, gait_curriculum = object(), object(), object()
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=20, robot=robot, terrain=SimpleNamespace(
            terrain_type="generator", terrain_generator=object(), physics_material=material,
        )),
        commands=command, observations=observation,
        terminations=SimpleNamespace(
            time_out=object(), terrain_out_of_bounds=object(), illegal_contact=contact,
            bad_orientation_2=orientation,
        ),
        curriculum=SimpleNamespace(terrain_levels=object(), gait_level=gait_curriculum),
    )
    geometry = configure_flat_retention_plane(cfg)
    assert geometry == {"terrain_type": "plane", "terrain_generator": None}
    assert cfg.scene.num_envs == 1
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert cfg.terminations.terrain_out_of_bounds is cfg.terminations.time_out is None
    assert cfg.curriculum.terrain_levels is None
    assert cfg.scene.robot is robot and cfg.scene.terrain.physics_material is material
    assert cfg.commands is command and cfg.observations is observation
    assert cfg.terminations.illegal_contact is contact
    assert cfg.terminations.bad_orientation_2 is orientation
    assert cfg.curriculum.gait_level is gait_curriculum


def _flat_rows(count, *, step_dt=0.02, initial_x=0.0):
    return [{
        "step": index,
        "base_x": initial_x + 0.3 * step_dt * index,
        "base_lin_vel_xy": [0.3, 0.0],
        "command_lin_vel_xy": [0.3, 0.0],
        "done": False,
    } for index in range(1, count + 1)]


def _flat_metrics(rows, *, step_dt=0.02, initial_x=0.0, max_steps=None, **physical_flags):
    from rl_training.online_adaptation.flat_retention import summarize_flat_retention

    return summarize_flat_retention(
        rows, initial_x=initial_x, step_dt=step_dt,
        max_steps=len(rows) if max_steps is None else max_steps,
        **{"fell": False, "safety_violation": False, "reset_count": 0, **physical_flags},
    )


def test_flat_2000_step_trial_has_1750_steady_samples_and_no_track_end():
    metrics = _flat_metrics(_flat_rows(2000))
    assert metrics["sample_count"] == 2000
    assert metrics["steady_sample_count"] == 1750
    assert metrics["steady_duration_seconds"] == 35.0
    assert metrics["steady_velocity_x_mae"] == metrics["steady_velocity_y_mae"] == 0.0
    assert metrics["net_forward_m"] == pytest.approx(12.0)
    assert metrics["steady_net_forward_m"] == pytest.approx(10.5)
    assert metrics["full_sample_budget"] and metrics["comparison_eligible"]
    assert "passed" not in metrics


def test_flat_metrics_exclude_warmup_and_include_displacement_from_its_boundary():
    rows = _flat_rows(252, initial_x=10.0)
    for row in rows[:250]:
        row["base_lin_vel_xy"] = [30.0, -30.0]
        row["base_x"] = 10.0 + row["step"] / 250.0
    rows[250].update(base_x=11.10, base_lin_vel_xy=[0.4, 0.2])
    rows[251].update(base_x=11.05, base_lin_vel_xy=[0.2, -0.2])

    metrics = _flat_metrics(rows, initial_x=10.0)

    assert metrics["steady_sample_count"] == 2
    assert metrics["steady_velocity_x_mae"] == pytest.approx(0.1)
    assert metrics["steady_velocity_y_mae"] == pytest.approx(0.2)
    assert metrics["steady_velocity_xy_mae"] == pytest.approx(0.15)
    assert metrics["steady_velocity_xy_l2_mean"] == pytest.approx(0.05 ** 0.5)
    assert metrics["net_forward_m"] == pytest.approx(1.05)  # Final, not maximum x.
    assert metrics["steady_start_x"] == pytest.approx(11.0)
    assert metrics["steady_net_forward_m"] == pytest.approx(0.05)  # Includes step 251.


def test_flat_metrics_interpolate_the_five_second_boundary_between_samples():
    metrics = _flat_metrics(_flat_rows(170, step_dt=0.03), step_dt=0.03)
    assert metrics["steady_sample_count"] == 4
    assert metrics["steady_start_x"] == pytest.approx(1.5)
    assert metrics["steady_net_forward_m"] == pytest.approx(0.03)
    assert metrics["steady_duration_seconds"] == pytest.approx(0.1)


@pytest.mark.parametrize("steps", [100, 250])
def test_flat_trial_without_samples_after_five_seconds_is_not_comparable(steps):
    import json

    metrics = _flat_metrics(_flat_rows(steps))
    assert metrics["steady_sample_count"] == 0
    assert not metrics["steady_data_available"]
    assert not metrics["comparison_eligible"]
    assert metrics["steady_velocity_x_mae"] is None
    assert metrics["steady_velocity_xy_mae"] is None
    assert metrics["steady_net_forward_m"] is None
    json.dumps(metrics, allow_nan=False)


def test_flat_metrics_do_not_include_positions_or_errors_after_a_reset():
    rows = _flat_rows(260)
    rows[252]["done"] = True
    for row in rows[253:]:
        row["base_x"] = 100.0
        row["base_lin_vel_xy"] = [30.0, 30.0]
    metrics = _flat_metrics(rows, reset_count=1)
    assert metrics["sample_count"] == 253
    assert metrics["steady_sample_count"] == 3
    assert metrics["post_reset_samples_excluded"] == 7
    assert metrics["net_forward_m"] == pytest.approx(1.518)
    assert metrics["steady_net_forward_m"] == pytest.approx(0.018)
    assert metrics["steady_velocity_x_mae"] == 0.0
    assert not metrics["comparison_eligible"]


@pytest.mark.parametrize("physical_flags", [
    {"fell": True}, {"safety_violation": True}, {"reset_count": 1}, {"max_steps": 300},
])
def test_flat_comparison_requires_full_budget_and_no_physical_failure(physical_flags):
    metrics = _flat_metrics(_flat_rows(260), **physical_flags)
    assert metrics["steady_data_available"]
    assert not metrics["comparison_eligible"]


@pytest.mark.parametrize("evaluation_only", [False, True])
def test_checkpoint_finalization_keeps_exact_frozen_candidate_and_training_default(evaluation_only):
    import copy
    import torch
    from torch import nn
    from rl_training.online_adaptation.online_session import OnlinePPOSession
    from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater

    session = OnlinePPOSession(OnlinePPOUpdater(nn.Linear(2, 1), nn.Linear(2, 1)))
    stable = copy.deepcopy(session.updater.policy.state_dict())
    state = session.state_dict()
    state["updater"]["policy"]["bias"].add_(1.0)
    state["updater"]["policy_version"] = 1
    state["has_unverified_candidate"] = True
    session.load_state_dict(state)
    candidate = copy.deepcopy(session.updater.policy.state_dict())

    result = _finalize_checkpoint(
        session, evaluation_only=evaluation_only,
        retain_terminal_learning=False, completion_stop=False,
    )

    assert result.restored_unverified_candidate is (not evaluation_only)
    expected = candidate if evaluation_only else stable
    for key, value in expected.items():
        torch.testing.assert_close(session.updater.policy.state_dict()[key], value)
    assert session.counters.update_attempts == 0


def test_fixed_command_disables_independent_special_modes_and_low_speed_deadband():
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        ranges=SimpleNamespace(), rel_zero_vel_envs=0.07,
        rel_only_lin_y_envs=0.07, rel_only_lin_x_envs=0.07,
        rel_only_ang_z_envs=0.07, planar_velocity_deadband=0.2,
        command_acceleration_limits=(0.1, 0.1, 0.1),
        only_lin_x_magnitude_range=(0.1, 0.6),
    )
    _configure_forward_command(cfg, 0.1)
    assert cfg.ranges.lin_vel_x == (0.1, 0.1)
    assert cfg.ranges.lin_vel_y == cfg.ranges.ang_vel_z == (0.0, 0.0)
    assert (cfg.rel_zero_vel_envs, cfg.rel_only_lin_y_envs,
            cfg.rel_only_lin_x_envs, cfg.rel_only_ang_z_envs) == (0.0,) * 4
    assert cfg.planar_velocity_deadband == 0.0
    assert cfg.command_acceleration_limits is None
    assert cfg.only_lin_x_magnitude_range is None


def test_pin_refreshes_stale_policy_observation_after_reset_only_when_needed():
    from types import SimpleNamespace
    import torch

    term = SimpleNamespace(
        vel_command_b=torch.zeros(1, 3), target_vel_command_b=torch.zeros(1, 3),
        is_standing_env=torch.ones(1, dtype=torch.bool),
        is_heading_env=torch.ones(1, dtype=torch.bool),
    )
    manager = SimpleNamespace(
        get_command=lambda _: term.vel_command_b, get_term=lambda _: term,
    )
    calls = []

    def compute():
        calls.append(True)
        return {"policy": term.vel_command_b.clone()}

    wrapper = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=manager),
                              get_observations=compute)
    stale = {"policy": torch.zeros(1, 3)}
    refreshed = _pin_forward_observation(wrapper, stale, 0.3)
    assert refreshed["policy"].tolist()[0] == pytest.approx([0.3, 0.0, 0.0])
    assert not term.is_standing_env.any() and not term.is_heading_env.any()
    assert _pin_forward_observation(wrapper, refreshed, 0.3) is refreshed
    assert len(calls) == 1  # no duplicate history/noise sampling on ordinary steps


@pytest.mark.parametrize("command", [(0.0, 0.0, 0.0), (0.3, 0.1, 0.0),
                                     (0.3, 0.0, 0.1), (float("nan"), 0.0, 0.0)])
def test_command_override_during_physics_is_rejected_before_learning(command):
    from types import SimpleNamespace
    import torch

    signals = SimpleNamespace(command_lin_vel_xy=torch.tensor([command[:2]]),
                              command_ang_vel_z=torch.tensor([[command[2]]]))
    with pytest.raises(RuntimeError, match="refusing PPO data"):
        _assert_forward_command(signals, 0.3)
    signals.command_lin_vel_xy = torch.tensor([[0.3, 0.0]])
    signals.command_ang_vel_z.zero_()
    _assert_forward_command(signals, 0.3)


@pytest.mark.parametrize("stage, failure", [("rough_4cm", "lateral_violation"),
                                          ("step_4cm", "edge_slip")])
def test_first_irreversible_qualification_failure_is_a_terminal_learning_boundary(stage, failure):
    import torch
    from rl_training.online_adaptation.track_completion import TrackCompletion

    completion = TrackCompletion.start(
        stage_name=stage, initial_x=0.0, initial_y=0.0,
        flat_length=3.0, step_length=2.0, track_length=20.0,
        step_clearance_margin_m=0.5, track_end_buffer_m=0.5,
        lane_width_m=2.0, track_width_m=4.0,
    )
    assert not completion.qualification_satisfied
    assert _task_failure_reason(completion) is None  # approach still needs gait evidence
    feet = [0.2, -0.2, 0.2, -0.2]
    slip = 0.0
    if failure == "lateral_violation":
        feet[0] = completion.allowed_lateral_deviation_m + 0.01
    else:
        slip = completion.max_completion_foot_slip + 0.01
    completion.observe_pose(completion.challenge_start_x, 0.0, foot_y=feet, foot_slip=slip)
    reason = _task_failure_reason(completion)
    assert reason == failure
    assert _learning_boundary(
        retain_failures=True, rollback_reason=None, completion_passed=False,
        environment_done=False, task_failure_reason=reason,
    ) == failure
    assert _learning_boundary(
        retain_failures=False, rollback_reason=None, completion_passed=False,
        environment_done=False, task_failure_reason=reason,
    ) is None  # conservative mode still rolls back instead of retaining the candidate
    deviation = _lateral_deviation(torch.zeros(1, 1), torch.tensor([feet]), 0.0)
    assert deviation.item() == pytest.approx(max(abs(y) for y in feet))
    # Returning inside the lane or eventually reaching the end cannot erase failure.
    completion.observe_pose(completion.target_x, 0.0, foot_y=[0.0] * 4, foot_slip=0.0)
    assert _task_failure_reason(completion) == failure
    assert not _completion_passed(completion, fell=False, safety_violation=False, reset_count=0)


def test_terminal_diagnostics_survive_in_place_simulator_reset():
    from dataclasses import fields
    from types import SimpleNamespace
    import torch
    from rl_training.online_adaptation.types import RewardSignals

    signals = RewardSignals(**{
        f.name: torch.full((1, 4 if f.name in ("foot_height", "foot_contact") else 1), 0.3)
        for f in fields(RewardSignals)
    })
    x, y, feet = torch.tensor([[4.6]]), torch.tensor([[1.9]]), torch.full((1, 4), 1.8)
    adapter = SimpleNamespace(
        reward_signals=lambda: signals, base_x=lambda: x, base_y=lambda: y,
        foot_y=lambda: feet, observe_nonfoot_contact=lambda: {"max_nonfoot_contact_force_n": 115.0},
    )
    snapshot = _capture_step_state(adapter)
    x.fill_(-3.5)
    y.zero_()
    feet.zero_()
    for f in fields(signals):
        getattr(signals, f.name).zero_()
    assert snapshot["base_x"].item() == pytest.approx(4.6)
    assert snapshot["base_y"].item() == pytest.approx(1.9)
    assert snapshot["signals"].projected_gravity_xy.item() == pytest.approx(0.3)
    assert snapshot["contact"]["max_nonfoot_contact_force_n"] == 115.0


@pytest.mark.parametrize("failure", ["fell", "safety_violation", "reset_count"])
def test_reaching_finish_after_earlier_failure_is_not_a_pass(failure):
    from types import SimpleNamespace
    completion = SimpleNamespace(strict_qualification_enabled=True, reached=True, qualification_satisfied=True)
    flags = dict(fell=False, safety_violation=False, reset_count=0)
    assert _completion_passed(completion, **flags)
    flags[failure] = 1 if failure == "reset_count" else True
    assert not _completion_passed(completion, **flags)
    assert not _completion_passed(None, fell=False, safety_violation=False, reset_count=0)


@pytest.mark.parametrize("reason", ["severe_nonfoot_contact", "terminated"])
def test_failure_learning_finishes_before_rollback_can_erase_the_buffer(reason):
    assert _learning_boundary(
        retain_failures=True, rollback_reason=reason,
        completion_passed=False, environment_done=reason == "terminated",
    ) == reason
    assert _learning_boundary(
        retain_failures=False, rollback_reason=reason,
        completion_passed=False, environment_done=reason == "terminated",
    ) is None


def test_completion_and_timeout_flush_partial_rollouts():
    assert _learning_boundary(
        retain_failures=False, rollback_reason=None, completion_passed=True, environment_done=False,
    ) == "completion"
    assert _learning_boundary(
        retain_failures=True, rollback_reason=None, completion_passed=False, environment_done=True,
    ) == "truncated"


def test_declared_task_deadline_is_distinct_from_a_short_sampling_budget():
    from scripts.reinforcement_learning.online_adaptation.train_lora_ppo import _deadline_failure

    # A diagnostic cut at 1200 steps must not invent a failure at the formal
    # 2000-step deadline; a genuine deadline has no future task continuation.
    assert _deadline_failure(completed_steps=1200, deadline_steps=2000, completion_passed=False) is None
    assert _deadline_failure(completed_steps=2000, deadline_steps=None, completion_passed=False) is None
    reason = _deadline_failure(completed_steps=2000, deadline_steps=2000, completion_passed=False)
    assert reason == "task_timeout"
    assert _learning_boundary(
        retain_failures=True, rollback_reason=None, completion_passed=False,
        environment_done=False, task_failure_reason=reason,
    ) == "task_timeout"
    assert _deadline_failure(completed_steps=2000, deadline_steps=2000, completion_passed=True) is None


def test_short_sampling_cut_keeps_declared_deadline_context():
    import torch
    from rl_training.online_adaptation.ppo_critic_context import build_critic_observation
    from scripts.reinforcement_learning.online_adaptation.train_lora_ppo import _critic_time_horizon

    horizon = _critic_time_horizon(max_steps=1200, deadline_steps=2000)
    observation = build_critic_observation(
        torch.zeros(1, 45), torch.zeros(1, 2), step=1200, max_steps=horizon,
    )
    assert observation[0, -1].item() == pytest.approx(0.6)
    assert _critic_time_horizon(max_steps=1200, deadline_steps=None) == 1200


def test_task_termination_never_bootstraps_even_when_failure_learning_is_disabled():
    import torch
    from rl_training.online_adaptation.ppo_rollout import PPORollout
    from scripts.reinforcement_learning.online_adaptation.train_lora_ppo import _learning_termination

    assert _learning_boundary(
        retain_failures=False, rollback_reason=None, task_failure_reason="task_timeout",
        completion_passed=False, environment_done=False,
    ) is None
    terminal = _learning_termination(
        torch.zeros(1, 1, dtype=torch.bool), rollback_reason=None,
        task_failure_reason="task_timeout", completion_passed=False,
    )
    rollout = PPORollout(
        observations=torch.zeros(1, 45), actions=torch.zeros(1, 12),
        old_log_probs=torch.zeros(1, 1), old_values=torch.ones(1, 1),
        rewards=torch.full((1, 1), -8.), terminated=terminal,
        truncated=torch.zeros(1, 1), next_values=torch.full((1, 1), 999.),
    )
    advantage, returns = rollout.compute_returns()
    assert returns.item() == -8.
    assert advantage.item() == -9.


def test_parser_has_planned_low_rank_ppo_defaults():
    args = build_parser().parse_args(["--base-checkpoint", "flat.pt"])
    assert args.rank == 2
    assert args.rollout_steps == 256
    assert args.action_std == 0.01
    assert args.actor_lr == 5e-5
    assert args.critic_lr == 5e-5
    assert args.nonfoot_contact_rollback_force == 20.0
    assert args.target_kl == 0.01
    assert args.reference_anchor_coef == 1.0
    assert args.max_reference_mse == 1e-4
    assert args.terminal_failure_penalty == -8.0
    assert args.safety_projection_penalty == 0.0
    assert args.retain_terminal_learning is False
    assert args.terrain_stage == "rough_4cm"
    assert args.terrain_seed == 101
    assert args.protocol_sha256 is None
    assert args.require_track_completion is False
    assert args.stop_after_termination is False
    assert CONTACT_TERMINATION_PROTOCOL == "sustained_nonfoot_contact_5n_3steps_v1"


def test_parser_accepts_explicit_app_and_learning_options():
    args = build_parser().parse_args(
        ["--base-checkpoint", "flat.pt", "--headless", "--device", "cuda:1", "--rank", "4"]
    )
    assert args.headless
    assert args.device == "cuda:1"
    assert args.rank == 4


def test_parser_accepts_formal_completion_protocol_options():
    protocol_sha256 = "a" * 64
    args = build_parser().parse_args(
        [
            "--base-checkpoint",
            "flat.pt",
            "--protocol-sha256",
            protocol_sha256,
            "--require-track-completion",
        ]
    )
    assert args.protocol_sha256 == protocol_sha256
    assert args.require_track_completion is True


def test_parser_accepts_frozen_evaluation_options():
    args = build_parser().parse_args(
        [
            "--base-checkpoint",
            "flat.pt",
            "--checkpoint-in",
            "learned.pt",
            "--evaluation-only",
            "--video",
            "--video-length",
            "128",
        ]
    )
    assert args.checkpoint_in == "learned.pt"
    assert args.evaluation_only
    assert args.video
    assert args.video_length == 128


def test_parser_accepts_physical_episode_boundary():
    args = build_parser().parse_args(
        ["--base-checkpoint", "flat.pt", "--stop-after-termination"]
    )
    assert args.stop_after_termination


def test_video_flag_is_available_before_simulation_launch():
    args = build_parser().parse_args(
        ["--base-checkpoint", "flat.pt", "--video"]
    )
    # ``main`` promotes this to True before AppLauncher is constructed.  The
    # parser must retain the flag so camera provisioning can happen at launch.
    assert args.video is True


def test_rollback_reason_prioritizes_termination_and_requires_force_threshold():
    assert _rollback_reason(
        terminated=True,
        nonfoot_contact=True,
        max_nonfoot_contact_force_n=100.0,
        force_threshold_n=20.0,
    ) == "terminated"
    assert _rollback_reason(
        terminated=False,
        nonfoot_contact=True,
        max_nonfoot_contact_force_n=20.0,
        force_threshold_n=20.0,
    ) == "severe_nonfoot_contact"
    assert _rollback_reason(
        terminated=False,
        nonfoot_contact=True,
        max_nonfoot_contact_force_n=19.99,
        force_threshold_n=20.0,
    ) is None
    with pytest.raises(ValueError, match="threshold"):
        _rollback_reason(
            terminated=False,
            nonfoot_contact=False,
            max_nonfoot_contact_force_n=0.0,
            force_threshold_n=0.0,
        )
