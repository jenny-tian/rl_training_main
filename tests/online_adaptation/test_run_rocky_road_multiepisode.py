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
    _manifest,
    _validate_input_lineage,
    build_flat_probe_command,
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
        "initial_alpha": 0.02,
        "alpha_max": 0.05,
        "critic_huber_delta": 1.0,
        "rough_swing_lift_weight": 1.0,
        "rough_swing_lift_target": 0.045,
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


@pytest.mark.parametrize("mismatch", [None, "missing", "semantics", "updater_version", "replay_version"])
def test_initializer_lineage_requires_versioned_action_contract(tmp_path, mismatch):
    from rl_training.online_adaptation.provenance import sha256_file
    from rl_training.online_adaptation.sac import SACUpdater

    args = _args(tmp_path)
    base = Path(args.base_checkpoint)
    flat = Path(args.flat_replay)
    initializer = Path(args.initial_residual_checkpoint)
    for path in (base, flat, initializer):
        path.write_bytes(b"test artifact")
    base_manifest = Path(args.base_manifest)
    base_manifest.write_text(json.dumps({"files": {
        str(base.resolve()): {"sha256": sha256_file(base), "bytes": base.stat().st_size}
    }}), encoding="utf-8")
    flat.with_name("flat_replay_manifest.json").write_text(json.dumps({
        "frozen": True,
        "transition_count": 100_000,
        "replay_sha256": sha256_file(flat),
        "baseline_manifest_sha256": sha256_file(base_manifest),
    }), encoding="utf-8")
    manifest = {
        "updater_sha256": sha256_file(initializer),
        "flat_replay_sha256": sha256_file(flat),
        "frame_dim": 49,
        "actor_source": "fresh_zero_mean_actor",
        "critic_action_semantics": SACUpdater.CRITIC_ACTION_SEMANTICS,
        "updater_schema_version": SACUpdater.CHECKPOINT_SCHEMA_VERSION,
        "replay_schema_version": 2,
        "sac_config": {"initial_alpha": 0.02, "alpha_max": 0.05},
    }
    if mismatch == "missing":
        manifest.pop("updater_schema_version")
    elif mismatch == "semantics":
        manifest["critic_action_semantics"] = "executed_residual_after_safety_v1"
    elif mismatch == "updater_version":
        manifest["updater_schema_version"] = True
    elif mismatch == "replay_version":
        manifest["replay_schema_version"] = 1
    initializer.with_name("initializer.json").write_text(json.dumps(manifest), encoding="utf-8")
    if mismatch is None:
        result = _validate_input_lineage(args)
        assert result["initializer_manifest"] == str(initializer.with_name("initializer.json"))
    else:
        with pytest.raises(ValueError, match="action semantics/schema"):
            _validate_input_lineage(args)


def test_training_and_all_evaluations_share_rate_limit(tmp_path):
    args = _args(tmp_path, residual_rate_limit=0.05)
    output = Path(args.output_dir)
    output.mkdir()
    (output / "novelty_baseline.json").write_text("{}", encoding="utf-8")
    config = _target_configuration(args)
    commands = [build_training_command(args, config, output=output, episode=1)]
    for mode in ("base", "learned_frozen"):
        commands.append(build_flat_probe_command(args, config, output=output, mode=mode,
                                                seed=7, run_dir=output / mode))
    commands.append(build_sealed_evaluation_command(args, config, output=output,
                                                   terrain_seed=1009, run_dir=output / "sealed"))
    commands.append(build_development_control_command(args, config, output=output,
                                                     mode="initial_residual", run_dir=output / "control"))
    assert all(_value(command, "--residual-rate-limit") == "0.05" for command in commands)


def test_manifest_binds_action_semantics_and_rate_limit(tmp_path, monkeypatch):
    from scripts.reinforcement_learning.online_adaptation import run_rocky_road_multiepisode as module
    for name in ("_validate_input_lineage", "_frozen_benchmark_provenance", "_git_provenance"):
        monkeypatch.setattr(module, name, lambda *args: {})
    monkeypatch.setattr(module, "sha256_file", lambda path: "test-hash")
    args = _args(tmp_path, residual_rate_limit=0.05)
    manifest = _manifest(args, _target_configuration(args))
    assert manifest["critic_action_semantics"] == "requested_residual_before_safety_v1"
    assert manifest["replay_schema_version"] == 2
    args.residual_rate_limit = 0.03
    assert manifest != _manifest(args, _target_configuration(args))


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

    assert args.max_episodes == 100
    assert args.target_q_min is None and args.target_q_max is None
    assert args.initial_alpha == 0.02 and args.alpha_max == 0.05
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

    assert not stable_success_reached(state, window=10, min_success_rate=0.8, min_consecutive_successes=3)
    state["last_development_probe_episode"] = 10
    state["latest_development_validation"] = {"passed": True}

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


def test_development_validation_is_frozen_disjoint_and_uses_snapshot(tmp_path):
    from scripts.reinforcement_learning.online_adaptation.run_rocky_road_multiepisode import (
        build_development_validation_command, validation_seeds,
    )
    args = _args(tmp_path)
    snapshot = tmp_path / "episode-0020" / "experiment.pt"
    command = build_development_validation_command(
        args, _target_configuration(args), output=tmp_path, seed=103,
        run_dir=tmp_path / "validation", checkpoint=snapshot,
    )
    assert _value(command, "--experiment-checkpoint-in") == str(snapshot)
    assert _value(command, "--terrain-split") == "development"
    assert _value(command, "--mode") == "residual_frozen"
    assert _value(command, "--seed") == "7"
    assert "--evaluation-only" in command and "--stochastic-control" not in command
    assert set(validation_seeds(args)) == {103, 107, 109, 113}
    for invalid_seed in (101, 1009):
        with pytest.raises(ValueError, match="development seed"):
            build_development_validation_command(args, _target_configuration(args), output=tmp_path,
                seed=invalid_seed, run_dir=tmp_path / "invalid", checkpoint=snapshot)


def test_best_checkpoint_survives_later_regression_and_failed_flat_probe():
    from scripts.reinforcement_learning.online_adaptation.run_rocky_road_multiepisode import select_best_checkpoint
    best = {"episode": 20, "rank": [3, 0, -0.1, -0.1, -0.1], "flat_retention_passed": True}
    worse = {**best, "episode": 100, "rank": [2, -1, -0.2, -0.2, -0.2]}
    unsafe = {**best, "episode": 50, "rank": [4, 0, 0, 0, 0], "flat_retention_passed": False}
    assert select_best_checkpoint(best, worse) == best
    assert select_best_checkpoint(best, unsafe) == best
    assert select_best_checkpoint(best, {**best, "episode": 60}) == best
    assert select_best_checkpoint(None, unsafe) is None


def test_selected_checkpoint_is_not_latest_and_checksum_is_verified(tmp_path):
    from scripts.reinforcement_learning.online_adaptation.run_rocky_road_multiepisode import _selected_checkpoint
    from rl_training.online_adaptation.provenance import sha256_file
    snapshot = tmp_path / "episodes" / "episode-0020" / "experiment.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"best")
    digest = sha256_file(snapshot)
    snapshot.with_suffix(".pt.sha256").write_text(digest)
    (tmp_path / "experiment.pt").write_bytes(b"worse final")
    (tmp_path / "best_checkpoint.json").write_text(json.dumps({
        "episode": 20, "flat_retention_passed": True, "checkpoint_sha256": digest,
    }))
    assert _selected_checkpoint(tmp_path) == snapshot
    snapshot.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checksum"):
        _selected_checkpoint(tmp_path)


@pytest.mark.parametrize("mismatch", ["seed", "checkpoint", "stochastic", "updates"])
def test_validation_rejects_stale_or_exploratory_results(mismatch):
    from scripts.reinforcement_learning.online_adaptation.run_rocky_road_multiepisode import _validate_frozen_rollout
    summary = {"terrain_seed": 103, "mode": "residual_frozen", "actor_updates": 0,
               "critic_updates": 0, "parameter_changed": False, "experiment_checkpoint_in_sha256": "good"}
    config = {"arguments": {"stochastic_control": False, "evaluation_only": True, "seed": 7}}
    if mismatch == "seed": summary["terrain_seed"] = 107
    if mismatch == "checkpoint": summary["experiment_checkpoint_in_sha256"] = "old"
    if mismatch == "stochastic": config["arguments"]["stochastic_control"] = True
    if mismatch == "updates": summary["actor_updates"] = 1
    with pytest.raises(RuntimeError, match="matching deterministic"):
        _validate_frozen_rollout(summary, config, seed=103, checkpoint_sha256="good", simulation_seed=7)


def test_resume_completes_missing_development_probe_without_repeating_flat(tmp_path, monkeypatch):
    from scripts.reinforcement_learning.online_adaptation import run_rocky_road_multiepisode as module
    state = _initial_state()
    state.update(episodes_completed=10, last_flat_probe_episode=10, latest_flat_retention_passed=True)
    calls = []
    monkeypatch.setattr(module, "_run_flat_probe", lambda *a, **k: pytest.fail("already committed flat probe"))
    def record(*args, **kwargs):
        calls.append(kwargs["state"]["episodes_completed"])
        kwargs["state"]["last_development_probe_episode"] = 10
    monkeypatch.setattr(module, "_record_development_validation", record)
    args = _args(tmp_path)
    for _ in range(2):
        module._complete_current_probes(args, _target_configuration(args), output=tmp_path,
                                       state=state, base_summaries=[])
    assert calls == [10]


def test_failed_flat_probe_is_recorded_without_terminating_bounded_run(tmp_path, monkeypatch):
    from scripts.reinforcement_learning.online_adaptation import run_rocky_road_multiepisode as module

    state = _initial_state()
    state.update(episodes_completed=10, last_flat_probe_episode=0)
    probe = {"episode": 10, "passed": False}
    monkeypatch.setattr(module, "_run_flat_probe", lambda *args, **kwargs: probe)
    monkeypatch.setattr(module, "_record_development_validation", lambda *args, **kwargs: None)

    module._complete_current_probes(
        _args(tmp_path), _target_configuration(_args(tmp_path)),
        output=tmp_path, state=state, base_summaries=[],
    )

    assert state["status"] == "running"
    assert state["flat_retention_failures"] == 1
    assert state["latest_flat_retention_passed"] is False


def test_load_state_migrates_legacy_flat_retention_terminal_status(tmp_path):
    from scripts.reinforcement_learning.online_adaptation.run_rocky_road_multiepisode import _load_state

    state = _initial_state()
    state.pop("flat_retention_failures")
    state.update(
        status="flat_retention_failed",
        episodes_completed=30,
        flat_probes=[{"episode": 10, "passed": False}, {"episode": 20, "passed": True}],
    )
    (tmp_path / "learning_state.json").write_text(json.dumps(state), encoding="utf-8")

    migrated = _load_state(tmp_path)

    assert migrated["status"] == "running"
    assert migrated["flat_retention_failures"] == 1


def test_development_probe_aggregates_pairing_and_replaces_stale_cache(tmp_path, monkeypatch):
    from scripts.reinforcement_learning.online_adaptation import run_rocky_road_multiepisode as module
    from rl_training.online_adaptation.provenance import sha256_file

    args = _args(tmp_path, video=False)
    checkpoint = tmp_path / "episodes" / "episode-0010" / "experiment.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"snapshot")
    checkpoint.with_suffix(".pt.sha256").write_text(sha256_file(checkpoint))
    calls = []

    def run(command, run_dir, *, physical_gpu):
        seed = int(_value(command, "--terrain-seed"))
        mode = _value(command, "--mode")
        calls.append((mode, seed))
        summary = {
            "mode": mode, "terrain_seed": seed, "actor_updates": 0, "critic_updates": 0,
            "parameter_changed": False, "passed": mode == "residual_frozen" or seed in (103, 107),
            "fell": False, "safety_violation": False, "completion_remaining_m": 0.0,
            "second_half_velocity_error": 0.1, "second_half_tilt_rms": 0.1,
            "experiment_checkpoint_in_sha256": sha256_file(checkpoint) if mode == "residual_frozen" else None,
        }
        (run_dir / "episode_summary.json").write_text(json.dumps(summary))
        (run_dir / "config.json").write_text(json.dumps({"arguments": {
            "seed": 7, "stochastic_control": False, "evaluation_only": True,
        }}))
        return 0

    monkeypatch.setattr(module, "_run_logged", run)
    config = _target_configuration(args)
    report = module._run_development_validation(args, config, output=tmp_path, episode=10, flat_passed=True)
    assert report["passed"] is True
    assert report["base"]["successes"] == 2 and report["learned_frozen"]["successes"] == 4
    assert len(calls) == 8
    assert module._run_development_validation(args, config, output=tmp_path, episode=10, flat_passed=True) == report
    assert len(calls) == 8
    stale = tmp_path / "development_validation" / "episode-0010" / "seed-107" / "episode_summary.json"
    saved = json.loads(stale.read_text())
    saved["experiment_checkpoint_in_sha256"] = "old checkpoint"
    stale.write_text(json.dumps(saved))
    module._run_development_validation(args, config, output=tmp_path, episode=10, flat_passed=True)
    assert calls[-1] == ("residual_frozen", 107) and len(calls) == 9
    assert stale.parent.parent.joinpath("incomplete", "seed-107-attempt-01").is_dir()


def test_invalid_validation_metrics_cannot_select_a_winner():
    from scripts.reinforcement_learning.online_adaptation.run_rocky_road_multiepisode import _validation_rank
    with pytest.raises(RuntimeError, match="finite"):
        _validation_rank([{"passed": True, "fell": False, "second_half_velocity_error": float("nan"),
                           "second_half_tilt_rms": .1, "completion_remaining_m": 0}])


def test_synchronous_actor_budget_reaches_worker_without_relaxing_drift_guard(tmp_path):
    args = _args(tmp_path, update_schedule='control-step', actor_updates_per_transition=0.02)
    (tmp_path / 'novelty_baseline.json').write_text('{}', encoding='utf-8')
    command = build_training_command(args, _target_configuration(args), output=tmp_path, episode=1)
    assert _value(command, '--update-schedule') == 'control-step'
    assert float(_value(command, '--actor-updates-per-transition')) == 0.02
    assert float(_value(command, '--max-policy-residual-mse')) == 1e-5
    assert _value(command, '--terrain-feature-source') == 'proprio'


def test_development_pilot_can_skip_reused_sealed_evaluation():
    args = build_parser().parse_args(['--base-checkpoint', 'base.pt', '--base-manifest', 'base.json',
        '--initial-residual-checkpoint', 'initial.pt', '--flat-replay', 'flat.pt',
        '--novelty-baseline', 'novelty.json', '--frozen-benchmark-dir', 'benchmark',
        '--output-dir', 'out', '--no-sealed-evaluation', '--update-schedule', 'control-step'])
    assert not args.sealed_evaluation
    assert args.update_schedule == 'control-step'


def test_update_schedule_is_bound_to_run_manifest(tmp_path, monkeypatch):
    from scripts.reinforcement_learning.online_adaptation import run_rocky_road_multiepisode as runner
    args = _args(tmp_path)
    monkeypatch.setattr(runner, 'sha256_file', lambda path: 'fixed')
    monkeypatch.setattr(runner, '_validate_input_lineage', lambda args: {})
    monkeypatch.setattr(runner, '_frozen_benchmark_provenance', lambda args: {})
    monkeypatch.setattr(runner, '_git_provenance', lambda: {})
    before = _manifest(args, _target_configuration(args))
    assert 'update_schedule' not in before
    args.update_schedule = 'control-step'
    after = _manifest(args, _target_configuration(args))
    assert after['update_schedule'] == 'control-step' and before != after
