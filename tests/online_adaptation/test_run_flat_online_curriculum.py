from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from rl_training.online_adaptation.curriculum import CurriculumCoordinator
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from scripts.reinforcement_learning.online_adaptation.run_flat_online_curriculum import (
    EpisodeRequest,
    TargetConfiguration,
    build_worker_command,
    classify_worker_result,
    flat_retention_passes,
    load_curriculum_state,
    resolve_target_configuration,
    worker_step_budget,
    write_curriculum_state,
)


def make_args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "base_checkpoint": str(tmp_path / "base.pt"),
        "base_manifest": str(tmp_path / "base-manifest.json"),
        "output_dir": str(tmp_path / "run"),
        "flat_replay": str(tmp_path / "flat-replay"),
        "device": "cuda:0",
        "headless": True,
        "target_semantics": "timeout-aware",
        "n_step": 3,
        "critic_replay_mode": "dual",
        "flat_fraction_initial": 0.75,
        "flat_fraction_ready": 0.50,
        "flat_anchor_weight": 1.0,
        "flat_critic_warmup_updates": 1_000,
        "max_episodes": 100,
        "max_total_steps": 200_000,
        "max_stage_episodes": 20,
    }
    values.update(overrides)
    return Namespace(**values)


def command_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_worker_command_contains_exact_stage_seed_budget_and_checkpoint_paths(tmp_path):
    request = EpisodeRequest(stage="step_2cm", seed=101, episode=1, gain=0.25)

    command = build_worker_command(make_args(tmp_path), request)

    assert command_value(command, "--terrain-stage") == "step_2cm"
    assert command_value(command, "--terrain-seed") == "101"
    assert command_value(command, "--max-steps") == "2000"
    assert command_value(command, "--residual-gain") == "0.25"
    assert command_value(command, "--n-step") == "3"
    assert command_value(command, "--critic-replay-mode") == "dual"
    assert command_value(command, "--flat-anchor-weight") == "1.0"
    assert command_value(command, "--run-dir").endswith("episodes/episode-0001")
    assert command_value(command, "--experiment-checkpoint-out").endswith(
        "episodes/episode-0001/experiment.pt"
    )


def test_worker_budget_is_clipped_and_refuses_exhausted_budget():
    assert worker_step_budget(total_episodes=99, total_steps=199_250) == 750
    with pytest.raises(RuntimeError, match="episode budget"):
        worker_step_budget(total_episodes=100, total_steps=0)
    with pytest.raises(RuntimeError, match="step budget"):
        worker_step_budget(total_episodes=0, total_steps=200_000)


def test_configuration_allows_partial_fill_but_rejects_conflict(tmp_path):
    config_path = tmp_path / "configuration.json"
    config_path.write_text(json.dumps({"n_step": 5}), encoding="utf-8")

    resolved = resolve_target_configuration(config_path, {"flat_anchor_weight": 2.0})

    assert resolved.n_step == 5
    assert resolved.flat_anchor_weight == 2.0
    assert resolved.critic_replay_mode == "dual"
    with pytest.raises(ValueError, match="conflicting.*n_step"):
        resolve_target_configuration(config_path, {"n_step": 3})


def test_resume_round_trip_refuses_different_protocol_hash(tmp_path):
    protocol = default_flat_online_protocol()
    coordinator = CurriculumCoordinator(protocol)
    state_path = tmp_path / "curriculum_state.json"
    write_curriculum_state(state_path, coordinator, evaluation_steps=123)

    restored, evaluation_steps = load_curriculum_state(state_path, protocol)

    assert restored.state_dict() == coordinator.state_dict()
    assert evaluation_steps == 123
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["protocol_sha256"] = "0" * 64
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        load_curriculum_state(state_path, protocol)


def test_valid_failure_record_counts_failed_episode_but_missing_outputs_block(tmp_path):
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    (episode_dir / "failure.json").write_text(
        json.dumps({"error_type": "RuntimeError", "message": "boom", "traceback": "..."}),
        encoding="utf-8",
    )

    failed = classify_worker_result(episode_dir, returncode=2)

    assert failed.passed is False
    assert failed.fell is True
    assert failed.steps == 1
    with pytest.raises(RuntimeError, match="valid summary or failure"):
        classify_worker_result(tmp_path / "missing", returncode=2)


def test_target_configuration_defaults_match_formal_experiment():
    assert TargetConfiguration() == TargetConfiguration(
        target_semantics="timeout-aware",
        n_step=3,
        critic_replay_mode="dual",
        flat_fraction_initial=0.75,
        flat_fraction_ready=0.50,
        flat_anchor_weight=1.0,
        flat_critic_warmup_updates=1_000,
        max_episodes=100,
        max_total_steps=200_000,
        max_stage_episodes=20,
    )


def test_flat_retention_requires_all_relative_thresholds():
    base = [
        {
            "passed": True,
            "fell": False,
            "contacts": 0,
            "second_half_velocity_error": 0.20,
            "second_half_tilt_rms": 0.10,
        }
        for _ in range(5)
    ]
    learned = [dict(item) for item in base]
    learned[0]["passed"] = False

    assert flat_retention_passes(base, learned) is False
    learned[0]["passed"] = True
    learned[0]["second_half_velocity_error"] = 0.21
    assert flat_retention_passes(base, learned) is True
    learned[0]["contacts"] = 1
    assert flat_retention_passes(base, learned) is False
