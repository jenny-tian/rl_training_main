import pytest

from scripts.reinforcement_learning.online_adaptation.train_lora_ppo import (
    CONTACT_TERMINATION_PROTOCOL,
    _assert_forward_command,
    _capture_step_state,
    _completion_passed,
    _configure_forward_command,
    _lateral_deviation,
    _learning_boundary,
    _pin_forward_observation,
    _rollback_reason,
    _task_failure_reason,
    build_parser,
)


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
