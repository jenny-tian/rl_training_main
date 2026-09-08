from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.reinforcement_learning.online_adaptation.run_rocky_road_multiepisode import (
    _append_jsonl,
    _archive_incomplete,
    _initial_state,
    _target_configuration,
    build_development_control_command,
    build_parser,
    build_sealed_evaluation_command,
    build_training_command,
    stable_success_reached,
    summarize_actor_drift,
    update_learning_state,
    worker_environment,
)


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "base_checkpoint": str(tmp_path / "base.pt"),
        "base_manifest": str(tmp_path / "base-manifest.json"),
        "initial_residual_checkpoint": str(tmp_path / "residual.pt"),
        "flat_replay": str(tmp_path / "flat-replay.pt"),
        "novelty_baseline": str(tmp_path / "novelty-baseline.json"),
        "frozen_benchmark_dir": str(tmp_path / "frozen-benchmark"),
        "output_dir": str(tmp_path / "run"),
        "terrain_seed": 101,
        "simulation_seed": 7,
        "terrain_stage": "rough_4cm",
        "terrain_rough_amplitude": 0.04,
        "max_episodes": 500,
        "max_steps_per_episode": 2_000,
        "forward_speed": 0.3,
        "residual_gain": 0.05,
        "adapting_residual_gain": 0.10,
        "post_step_residual_gain": 0.10,
        "actor_updates_per_second": 0.5,
        "control_timeout_ms": 50.0,
        "target_q_min": -500.0,
        "target_q_max": 300.0,
        "deterministic_actor_weight": 0.25,
        "max_policy_residual_mse": 1.0e-5,
        "flat_probe_interval": 10,
        "stability_window": 10,
        "stability_min_success_rate": 0.8,
        "stability_min_consecutive_successes": 3,
        "physical_gpu": 0,
        "headless": True,
        "video": True,
        "sealed_evaluation": True,
    }
    values.update(overrides)
    return Namespace(**values)


def _summary(*, passed: bool, fell: bool = False) -> dict[str, object]:
    return {
        "completion_reached": passed,
        "passed": passed,
        "fell": fell,
        "safety_violation": False,
        "interaction_steps": 123,
        "actor_updates": 4,
        "critic_updates": 30,
        "rollbacks": 1,
        "performance_rollbacks": 1,
        "safety_rollbacks": 0,
        "max_x": 1.25,
        "completion_remaining_m": 0.75,
        "second_half_velocity_error": 0.2,
        "second_half_tilt_rms": 0.1,
        "actor_version": 4,
    }


def _value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_parser_defaults_define_bounded_stable_multiepisode_experiment():
    args = build_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--base-manifest", "base.json",
            "--initial-residual-checkpoint", "residual.pt",
            "--flat-replay", "flat.pt",
            "--novelty-baseline", "novelty.json",
            "--frozen-benchmark-dir", "frozen-benchmark",
            "--output-dir", "run",
        ]
    )

    assert args.max_episodes == 500
    assert args.terrain_seed == 101
    assert args.simulation_seed == 7
    assert args.terrain_stage == "rough_4cm"
    assert args.terrain_rough_amplitude == pytest.approx(0.04)
    assert args.forward_speed == pytest.approx(0.3)
    assert args.flat_probe_interval == 10
    assert args.stability_window == 10
    assert args.stability_min_success_rate == pytest.approx(0.8)
    assert args.stability_min_consecutive_successes == 3
    assert args.video is True
    assert args.sealed_evaluation is True


def test_worker_environment_hides_gpu_two_and_three(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("DISPLAY", ":1")

    environment = worker_environment(1)

    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "0,1"
    assert "DISPLAY" not in environment


def test_parser_rejects_gpu_two():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--base-checkpoint", "base.pt",
                "--base-manifest", "base.json",
                "--initial-residual-checkpoint", "residual.pt",
                "--flat-replay", "flat.pt",
                "--novelty-baseline", "novelty.json",
                "--frozen-benchmark-dir", "frozen-benchmark",
                "--output-dir", "run",
                "--physical-gpu", "2",
            ]
        )


def test_training_command_keeps_fixed_development_geometry_and_checkpoint_chain(tmp_path):
    args = _args(tmp_path)
    output = Path(args.output_dir)
    output.mkdir()
    (output / "novelty_baseline.json").write_text("{}", encoding="utf-8")
    configuration = _target_configuration(args)

    first = build_training_command(args, configuration, output=output, episode=1)

    assert _value(first, "--task") == "RockyRoad-Deeprobotics-Lite3-v0"
    assert _value(first, "--terrain-split") == "development"
    assert _value(first, "--terrain-stage") == "rough_4cm"
    assert _value(first, "--terrain-seed") == "101"
    assert _value(first, "--seed") == "7"
    assert _value(first, "--terrain-rough-amplitude") == "0.04"
    assert _value(first, "--residual-frame-dim") == "49"
    assert _value(first, "--forward-speed") == "0.3"
    assert _value(first, "--device") == "cuda:0"
    assert _value(first, "--residual-checkpoint") == args.initial_residual_checkpoint
    assert "--video" in first
    assert "--require-track-completion" in first

    (output / "experiment.pt").write_bytes(b"checkpoint")
    resumed = build_training_command(args, configuration, output=output, episode=2)

    assert "--residual-checkpoint" not in resumed
    assert _value(resumed, "--experiment-checkpoint-in") == str(
        output / "experiment.pt"
    )
    assert _value(resumed, "--terrain-seed") == "101"


def test_learning_state_records_first_success_running_rates_and_updates():
    state = _initial_state()
    for episode, passed in enumerate((False, True, False, True), start=1):
        state = update_learning_state(
            state,
            episode=episode,
            summary=_summary(passed=passed, fell=not passed),
            drift={
                "max_policy_residual_mse": episode * 1.0e-6,
                "last_policy_parameter_l2": episode * 0.1,
            },
            rolling_window=3,
        )

    assert state["episodes_completed"] == 4
    assert state["first_success_episode"] == 2
    assert state["successes"] == 2
    assert state["cumulative_success_rate"] == pytest.approx(0.5)
    assert state["recent_successes"] == [True, False, True]
    assert state["recent_success_rate"] == pytest.approx(2 / 3)
    assert state["consecutive_successes"] == 1
    assert state["max_consecutive_successes"] == 1
    assert state["falls"] == 2
    assert state["actor_updates"] == 16
    assert state["critic_updates"] == 120
    assert state["rollbacks"] == 4
    assert state["max_policy_residual_mse"] == pytest.approx(4.0e-6)


def test_stable_success_requires_full_window_current_flat_probe_and_streak():
    state = _initial_state()
    outcomes = (False, False, True, True, True, True, True, True, True, True)
    for episode, passed in enumerate(outcomes, start=1):
        state = update_learning_state(
            state,
            episode=episode,
            summary=_summary(passed=passed),
            drift={},
            rolling_window=10,
        )
    state["last_flat_probe_episode"] = 10
    state["latest_flat_retention_passed"] = True
    state["base_control_passed"] = False
    state["initial_residual_control_passed"] = False

    assert stable_success_reached(
        state,
        window=10,
        min_success_rate=0.8,
        min_consecutive_successes=3,
    )
    state["last_flat_probe_episode"] = 9
    assert not stable_success_reached(
        state,
        window=10,
        min_success_rate=0.8,
        min_consecutive_successes=3,
    )


def test_actor_drift_summary_reads_only_finite_accepted_update_records(tmp_path):
    path = tmp_path / "actor_updates.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "accepted": True,
                        "policy_residual_mse": 1.0e-6,
                        "policy_joint_abs_change_mean": 0.01,
                        "policy_joint_abs_change_max": 0.02,
                        "policy_parameter_l2": 0.1,
                    }
                ),
                json.dumps(
                    {
                        "accepted": True,
                        "policy_residual_mse": 3.0e-6,
                        "policy_joint_abs_change_mean": 0.03,
                        "policy_joint_abs_change_max": 0.04,
                        "policy_parameter_l2": 0.2,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    drift = summarize_actor_drift(path)

    assert drift["accepted_actor_update_records"] == 2
    assert drift["max_policy_residual_mse"] == pytest.approx(3.0e-6)
    assert drift["last_policy_parameter_l2"] == pytest.approx(0.2)


def test_sealed_evaluation_is_frozen_and_uses_unseen_partition(tmp_path):
    args = _args(tmp_path)
    output = Path(args.output_dir)
    output.mkdir()
    checkpoint = output / "experiment.pt"
    checkpoint.write_bytes(b"checkpoint")
    command = build_sealed_evaluation_command(
        args,
        _target_configuration(args),
        output=output,
        terrain_seed=1009,
        run_dir=output / "sealed" / "seed-1009",
    )

    assert _value(command, "--mode") == "residual_frozen"
    assert _value(command, "--task") == "RockyRoad-Deeprobotics-Lite3-v0"
    assert _value(command, "--terrain-split") == "sealed"
    assert _value(command, "--terrain-seed") == "1009"
    assert _value(command, "--seed") == "7"
    assert _value(command, "--experiment-checkpoint-in") == str(checkpoint)
    assert "--evaluation-only" in command
    assert "--require-track-completion" in command
    assert "--flat-retention-probe" not in command


def test_development_no_update_control_uses_clean_initializer(tmp_path):
    args = _args(tmp_path)
    output = Path(args.output_dir)
    output.mkdir()
    command = build_development_control_command(
        args,
        _target_configuration(args),
        output=output,
        mode="initial_residual",
        run_dir=output / "controls" / "initial",
    )

    assert _value(command, "--mode") == "residual_frozen"
    assert _value(command, "--terrain-split") == "development"
    assert _value(command, "--terrain-seed") == "101"
    assert _value(command, "--residual-checkpoint") == args.initial_residual_checkpoint
    assert _value(command, "--control-timeout-ms") == "50.0"
    assert "--experiment-checkpoint-in" not in command
    assert "--evaluation-only" in command


def test_episode_record_commit_is_idempotent_and_rejects_conflict(tmp_path):
    path = tmp_path / "episodes.jsonl"
    record = {"episode": 1, "passed": False}

    _append_jsonl(path, record)
    _append_jsonl(path, record)

    assert path.read_text(encoding="utf-8").count("\n") == 1
    with pytest.raises(RuntimeError, match="differs"):
        _append_jsonl(path, {"episode": 1, "passed": True})


def test_incomplete_attempt_is_archived_without_deletion(tmp_path):
    run_dir = tmp_path / "episodes" / "episode-0001"
    run_dir.mkdir(parents=True)
    (run_dir / "runtime.log").write_text("partial", encoding="utf-8")

    _archive_incomplete(run_dir)

    archived = tmp_path / "episodes" / "incomplete" / "episode-0001-attempt-01"
    assert not run_dir.exists()
    assert (archived / "runtime.log").read_text(encoding="utf-8") == "partial"
