import pytest

from scripts.reinforcement_learning.online_adaptation.train_lora_ppo import (
    CONTACT_TERMINATION_PROTOCOL,
    _capture_step_state,
    _completion_passed,
    _learning_boundary,
    _rollback_reason,
    build_parser,
)


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
