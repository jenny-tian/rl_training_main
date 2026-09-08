from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from rl_training.online_adaptation.curriculum import CurriculumCoordinator
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from scripts.reinforcement_learning.online_adaptation.run_flat_online_curriculum import (
    EpisodeRequest,
    TargetConfiguration,
    build_retention_probe_command,
    build_novelty_baseline_probe_command,
    build_worker_command,
    classify_worker_result,
    flat_retention_passes,
    load_curriculum_state,
    resolve_target_configuration,
    validate_track_completion_summary,
    worker_step_budget,
    protocol_for_configuration,
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
        "max_policy_residual_mse": None,
        "terrain_feature_source": "geometry",
        "terrain_min_valid_fraction": 0.90,
        "flat_critic_warmup_updates": 1_000,
        "forward_speed": 0.3,
        "adapting_residual_gain": 0.1,
        "post_step_residual_gain": 0.1,
        "target_q_min": -1_000.0,
        "target_q_max": 500.0,
        "deterministic_actor_weight": 0.25,
        "critic_huber_delta": None,
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
    baseline = Path(make_args(tmp_path).output_dir) / "novelty_baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("{}", encoding="utf-8")

    command = build_worker_command(make_args(tmp_path), request)

    assert command_value(command, "--terrain-stage") == "step_2cm"
    assert command_value(command, "--terrain-seed") == "101"
    assert command_value(command, "--max-steps") == "2000"
    assert command_value(command, "--residual-gain") == "0.25"
    assert command_value(command, "--control-timeout-ms") == "20.0"
    assert command_value(command, "--forward-speed") == "0.3"
    assert command_value(command, "--adapting-residual-gain") == "0.1"
    assert command_value(command, "--post-step-residual-gain") == "0.1"
    assert command_value(command, "--target-q-min") == "-1000.0"
    assert command_value(command, "--target-q-max") == "500.0"
    assert command_value(command, "--deterministic-actor-weight") == "0.25"
    assert command_value(command, "--actor-updates-per-second") == "2.0"
    assert command_value(command, "--adaptation-min-replay-size") == "128"
    assert command_value(command, "--stalled-actor-replay-stride") == "16"
    assert command_value(command, "--control-timeout-ms") == "20.0"
    assert "--require-track-completion" in command
    assert command_value(command, "--n-step") == "3"
    assert command_value(command, "--critic-replay-mode") == "dual"
    assert command_value(command, "--flat-anchor-weight") == "1.0"
    assert command_value(command, "--pre-step-clearance-weight") == "4.0"
    assert command_value(command, "--post-step-progress-weight") == "2.0"
    assert command_value(command, "--stall-penalty-weight") == "2.0"
    assert command_value(command, "--completion-bonus") == "5.0"
    assert command_value(command, "--pre-step-clearance-target") == "0.045"
    assert command_value(command, "--pre-step-approach-distance") == "0.5"
    assert command_value(command, "--post-step-recovery-steps") == "25"
    assert "--critic-huber-delta" not in command
    assert command_value(command, "--run-dir").endswith("episodes/episode-0001")
    assert command_value(command, "--experiment-checkpoint-out").endswith(
        "episodes/episode-0001/experiment.pt"
    )
    assert command_value(command, "--novelty-baseline-in") == str(baseline)
    assert command_value(command, "--novelty-baseline-sha256") == hashlib.sha256(
        baseline.read_bytes()
    ).hexdigest()


def test_worker_command_forwards_frozen_policy_residual_mse_limit(tmp_path):
    request = EpisodeRequest(stage="step_2cm", seed=101, episode=1, gain=0.25)
    baseline = Path(make_args(tmp_path).output_dir) / "novelty_baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("{}", encoding="utf-8")

    command = build_worker_command(
        make_args(tmp_path, max_policy_residual_mse=0.125), request
    )

    assert command_value(command, "--max-policy-residual-mse") == "0.125"


def test_worker_command_forwards_critic_huber_delta(tmp_path):
    request = EpisodeRequest(stage="step_2cm", seed=101, episode=1, gain=0.25)
    baseline = Path(make_args(tmp_path).output_dir) / "novelty_baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("{}", encoding="utf-8")

    command = build_worker_command(
        make_args(tmp_path, critic_huber_delta=1.0), request
    )

    assert command_value(command, "--critic-huber-delta") == "1.0"


def test_worker_command_uses_initial_residual_only_before_first_experiment_checkpoint(tmp_path):
    request = EpisodeRequest(stage="step_2cm", seed=101, episode=1, gain=0.25)
    args = make_args(tmp_path, initial_residual_checkpoint=str(tmp_path / "initial.pt"), video=True)
    baseline = Path(args.output_dir) / "novelty_baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("{}", encoding="utf-8")

    initial = build_worker_command(args, request)

    assert command_value(initial, "--residual-checkpoint") == str(tmp_path / "initial.pt")
    assert "--video" in initial
    assert command_value(initial, "--video-length") == "2000"

    latest = Path(args.output_dir) / "experiment.pt"
    latest.write_text("checkpoint", encoding="utf-8")
    resumed = build_worker_command(args, request)

    assert "--residual-checkpoint" not in resumed
    assert command_value(resumed, "--experiment-checkpoint-in") == str(latest)


def test_worker_command_forwards_height_scan_configuration_only_when_enabled(tmp_path):
    request = EpisodeRequest(stage="step_2cm", seed=101, episode=1, gain=0.25)
    baseline = Path(make_args(tmp_path).output_dir) / "novelty_baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("{}", encoding="utf-8")

    geometry = build_worker_command(make_args(tmp_path), request)
    assert "--terrain-feature-source" not in geometry
    height_scan = build_worker_command(
        make_args(tmp_path, terrain_feature_source="height_scan", terrain_min_valid_fraction=0.8),
        request,
    )
    assert command_value(height_scan, "--terrain-feature-source") == "height_scan"
    assert command_value(height_scan, "--terrain-min-valid-fraction") == "0.8"
    proprio = build_worker_command(make_args(tmp_path, terrain_feature_source="proprio"), request)
    assert command_value(proprio, "--terrain-feature-source") == "proprio"


def test_novelty_baseline_probe_is_strict_flat_base_mode(tmp_path):
    command = build_novelty_baseline_probe_command(
        make_args(tmp_path),
        run_dir=tmp_path / "baseline-run",
        baseline_out=tmp_path / "baseline.json",
    )

    assert command_value(command, "--mode") == "base"
    assert command_value(command, "--terrain-seed") == "7"
    assert command_value(command, "--max-steps") == "384"
    assert command_value(command, "--forward-speed") == "0.3"
    assert command_value(command, "--novelty-baseline-out") == str(
        tmp_path / "baseline.json"
    )
    assert "--flat-retention-probe" in command


def test_retention_command_preserves_checkpoint_sensitive_agent_configuration(tmp_path):
    args = make_args(
        tmp_path,
        n_step=5,
        critic_replay_mode="online-only",
        flat_fraction_initial=0.6,
        flat_fraction_ready=0.4,
        flat_anchor_weight=2.0,
        flat_critic_warmup_updates=17,
    )

    command = build_retention_probe_command(
        args,
        mode="learned_frozen",
        seed=7,
        run_dir=tmp_path / "retention",
        learned_checkpoint=tmp_path / "experiment.pt",
    )

    assert command_value(command, "--n-step") == "5"
    assert command_value(command, "--residual-gain") == "0.25"
    assert command_value(command, "--critic-replay-mode") == "online-only"
    assert command_value(command, "--flat-fraction-initial") == "0.6"
    assert command_value(command, "--flat-fraction-ready") == "0.4"
    assert command_value(command, "--flat-anchor-weight") == "2.0"
    assert command_value(command, "--flat-critic-warmup-updates") == "17"
    assert command_value(command, "--residual-frame-dim") == "59"


def test_base_retention_does_not_force_learned_residual_frame_dimension(tmp_path):
    command = build_retention_probe_command(
        make_args(tmp_path),
        mode="base",
        seed=7,
        run_dir=tmp_path / "retention",
        learned_checkpoint=tmp_path / "experiment.pt",
    )

    assert "--residual-frame-dim" not in command


def test_proprio_retention_keeps_49_dimensional_checkpoint_contract(tmp_path):
    command = build_retention_probe_command(
        make_args(tmp_path, terrain_feature_source="proprio"),
        mode="learned_frozen",
        seed=7,
        run_dir=tmp_path / "retention",
        learned_checkpoint=tmp_path / "experiment.pt",
    )

    assert command_value(command, "--terrain-feature-source") == "proprio"
    assert command_value(command, "--residual-frame-dim") == "49"


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


def _write_strict_worker_result(tmp_path: Path, **overrides) -> tuple[Path, EpisodeRequest, str]:
    episode_dir = tmp_path / "strict-episode"
    episode_dir.mkdir()
    checkpoint = episode_dir / "experiment.pt"
    checkpoint.write_bytes(b"strict checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (episode_dir / "experiment.pt.sha256").write_text(
        checkpoint_sha256 + "\n", encoding="ascii"
    )
    protocol_sha256 = default_flat_online_protocol().sha256()
    summary = {
        "terrain_stage": "step_2cm",
        "terrain_seed": 101,
        "interaction_steps": 500,
        "passed": True,
        "fell": False,
        "safety_violation": False,
        "initial_x": -8.5,
        "max_x": -4.5,
        "completion_target_x": -4.5,
        "completion_remaining_m": 0.0,
        "completion_reached": True,
        "completion_mode": "full_track_v1",
        "progress_cost_version": "remaining_plus_safety_penalty_v1",
        "cost": 0.0,
        "protocol_sha256": protocol_sha256,
        "experiment_checkpoint_out_sha256": checkpoint_sha256,
    }
    summary.update(overrides)
    (episode_dir / "episode_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return episode_dir, EpisodeRequest("step_2cm", 101, 1, 0.25), protocol_sha256


def test_classifier_accepts_consistent_strict_completion(tmp_path):
    episode_dir, request, protocol_sha256 = _write_strict_worker_result(tmp_path)

    outcome = classify_worker_result(
        episode_dir,
        returncode=0,
        request=request,
        expected_protocol_sha256=protocol_sha256,
    )

    assert outcome.passed is True
    assert outcome.fell is False
    assert outcome.cost == 0.0


def test_classifier_accepts_safe_partial_progress_as_finite_failure(tmp_path):
    episode_dir, request, protocol_sha256 = _write_strict_worker_result(
        tmp_path,
        passed=False,
        max_x=-5.5,
        completion_remaining_m=1.0,
        completion_reached=False,
        cost=1.0,
    )

    outcome = classify_worker_result(
        episode_dir,
        returncode=0,
        request=request,
        expected_protocol_sha256=protocol_sha256,
    )

    assert outcome.passed is False
    assert outcome.cost == pytest.approx(1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"cost": None},
        {"cost": float("nan")},
        {"completion_mode": "disabled"},
        {"completion_reached": False},
        {"completion_remaining_m": 1.0},
        {"passed": False},
        {"safety_violation": True},
        {"max_x": -9.0},
        {"completion_target_x": -8.5},
    ],
)
def test_classifier_rejects_contradictory_or_missing_completion_evidence(
    tmp_path, overrides
):
    episode_dir, request, protocol_sha256 = _write_strict_worker_result(
        tmp_path, **overrides
    )

    with pytest.raises(RuntimeError):
        classify_worker_result(
            episode_dir,
            returncode=0,
            request=request,
            expected_protocol_sha256=protocol_sha256,
        )


def test_shared_validator_rejects_nonboolean_safety_and_zero_incomplete_cost():
    summary = {
        "passed": False,
        "fell": False,
        "safety_violation": False,
        "initial_x": -8.5,
        "max_x": -4.5000005,
        "completion_target_x": -4.5,
        "completion_remaining_m": 0.0000005,
        "completion_reached": False,
        "completion_mode": "full_track_v1",
        "progress_cost_version": "remaining_plus_safety_penalty_v1",
        "cost": 0.0,
    }

    with pytest.raises(RuntimeError):
        validate_track_completion_summary(summary)
    summary["cost"] = 0.0000005
    summary["fell"] = 0
    with pytest.raises(RuntimeError):
        validate_track_completion_summary(summary)


def test_target_configuration_defaults_match_formal_experiment():
    assert TargetConfiguration() == TargetConfiguration(
        target_semantics="timeout-aware",
        n_step=3,
        critic_replay_mode="dual",
        flat_fraction_initial=0.75,
        flat_fraction_ready=0.25,
        flat_anchor_weight=0.5,
        max_policy_residual_mse=None,
        terrain_feature_source="geometry",
        terrain_min_valid_fraction=0.90,
        flat_critic_warmup_updates=1_000,
        max_episodes=100,
        max_total_steps=200_000,
        max_stage_episodes=20,
    )


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_target_configuration_rejects_invalid_policy_residual_mse_limit(value):
    with pytest.raises(ValueError, match="max_policy_residual_mse"):
        TargetConfiguration(max_policy_residual_mse=value)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_target_configuration_rejects_invalid_height_scan_fraction(value):
    with pytest.raises(ValueError, match="terrain_min_valid_fraction"):
        TargetConfiguration(terrain_min_valid_fraction=value)


def test_full_budget_configuration_targets_only_the_first_stage():
    configuration = TargetConfiguration(max_episodes=100, max_stage_episodes=100)
    protocol = protocol_for_configuration(configuration)

    assert [stage.name for stage in protocol.stages] == ["step_2cm"]
    assert protocol.stages[0].max_episodes == 100
    assert protocol.max_total_episodes == 100


def test_full_budget_configuration_can_target_a_later_stage():
    configuration = TargetConfiguration(max_episodes=100, max_stage_episodes=100)

    protocol = protocol_for_configuration(configuration, target_stage="step_4cm")

    assert [stage.name for stage in protocol.stages] == ["step_4cm"]
    assert protocol.stages[0].step_height == 0.04
    assert protocol.stages[0].max_episodes == 100


def test_flat_retention_requires_all_relative_thresholds():
    base = [
        {
            "passed": True,
            "fell": False,
            "completion_mode": "disabled",
            "completion_reached": False,
            "completion_target_x": None,
            "completion_remaining_m": None,
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


def test_flat_retention_rejects_strict_or_missing_completion_labels():
    disabled = {
        "passed": True,
        "fell": False,
        "completion_mode": "disabled",
        "completion_reached": False,
        "completion_target_x": None,
        "completion_remaining_m": None,
        "contacts": 0,
        "second_half_velocity_error": 0.20,
        "second_half_tilt_rms": 0.10,
    }
    base = [dict(disabled) for _ in range(5)]
    learned = [dict(disabled) for _ in range(5)]

    assert flat_retention_passes(base, learned)
    learned[0]["completion_mode"] = "full_track_v1"
    assert not flat_retention_passes(base, learned)
    learned[0].pop("completion_mode")
    assert not flat_retention_passes(base, learned)
