from argparse import Namespace
from pathlib import Path

import pytest

from scripts.reinforcement_learning.online_adaptation.measure_step_completion import (
    _comparison_command,
    _ensure_novelty_baseline,
    _successful_completion,
    _target_configuration,
    _worker_args,
)


def test_step_measurement_uses_a_compute_cap_not_the_formal_hundred_episode_budget():
    args = Namespace(
        max_episodes=500,
        max_steps_per_episode=2_000,
        base_checkpoint="base.pt",
        base_manifest="base_manifest.json",
        initial_residual_checkpoint="residual.pt",
        flat_replay="flat_replay.pt",
        output_dir="output",
        device="cuda:0",
        headless=True,
        video=True,
        terrain_stage="step_2cm",
        residual_gain=0.1,
        adapting_residual_gain=0.1,
        post_step_residual_gain=0.1,
        post_step_stability_weight=1.0,
        post_step_progress_weight=2.0,
        stall_penalty_weight=2.0,
        completion_bonus=5.0,
        pre_step_clearance_weight=2.0,
        pre_step_clearance_target=0.065,
        pre_step_approach_distance=0.5,
        max_critic_grad_norm=5_000.0,
        critic_huber_delta=1.0,
        max_policy_residual_mse=0.01,
        actor_updates_per_second=0.5,
        adaptation_min_replay_size=128,
        stalled_actor_replay_stride=16,
        control_timeout_ms=50.0,
        forward_speed=0.3,
        comparison_seed=101,
    )

    configuration = _target_configuration(args)
    worker_args = _worker_args(args, configuration)

    assert configuration.max_episodes == 500
    assert configuration.max_stage_episodes == 500
    assert configuration.max_total_steps == 1_000_000
    assert worker_args.initial_residual_checkpoint == "residual.pt"
    assert worker_args.video is True
    assert worker_args.forward_speed == 0.3
    assert worker_args.adapting_residual_gain == 0.1
    assert worker_args.post_step_residual_gain == 0.1
    assert worker_args.post_step_stability_weight == 1.0
    assert worker_args.post_step_progress_weight == 2.0
    assert worker_args.stall_penalty_weight == 2.0
    assert worker_args.completion_bonus == 5.0
    assert worker_args.pre_step_clearance_weight == 2.0
    assert worker_args.pre_step_clearance_target == 0.065
    assert worker_args.pre_step_approach_distance == 0.5
    assert worker_args.max_critic_grad_norm == 5_000.0
    assert worker_args.critic_huber_delta == 1.0
    assert worker_args.max_policy_residual_mse == 0.01
    assert worker_args.target_q_min == -1_000.0
    assert worker_args.target_q_max == 500.0
    assert worker_args.deterministic_actor_weight == 0.25
    assert worker_args.actor_updates_per_second == 0.5
    assert worker_args.adaptation_min_replay_size == 128
    assert worker_args.stalled_actor_replay_stride == 16
    assert worker_args.control_timeout_ms == 50.0

    command = _comparison_command(
        args,
        configuration,
        mode="base",
        run_dir=Path("base-control"),
    )
    assert command[command.index("--control-timeout-ms") + 1] == "50.0"


def test_step_measurement_does_not_count_an_unsafe_crossing_as_completion():
    assert not _successful_completion(
        completion_reached=True,
        passed=False,
        fell=True,
        safety_violation=True,
    )


def test_step_measurement_counts_only_a_safe_pass():
    assert _successful_completion(
        completion_reached=True,
        passed=True,
        fell=False,
        safety_violation=False,
    )


def test_unsafe_novelty_probe_does_not_leave_a_reusable_baseline(
    tmp_path, monkeypatch
):
    output = tmp_path / "run"
    run_dir = output / "novelty_baseline" / "seed-7"
    baseline = output / "novelty_baseline.json"

    def fake_run(command, check):
        assert check is False
        baseline.write_text("{}", encoding="utf-8")
        (run_dir / "episode_summary.json").write_text(
            '{"fell": false, "safety_violation": true}', encoding="utf-8"
        )
        return Namespace(returncode=0)

    monkeypatch.setattr(
        "scripts.reinforcement_learning.online_adaptation.measure_step_completion.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "scripts.reinforcement_learning.online_adaptation.measure_step_completion.build_novelty_baseline_probe_command",
        lambda *args, **kwargs: ["probe"],
    )

    with pytest.raises(RuntimeError, match="unsafe"):
        _ensure_novelty_baseline(Namespace(video=False), output)

    assert not baseline.exists()
