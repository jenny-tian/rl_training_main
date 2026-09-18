import json
from argparse import Namespace
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.reinforcement_learning.online_adaptation.run_lora_ppo_multiepisode import (
    _initial_state,
    _manifest,
    _prepare,
    build_parser,
    build_probe_command,
    build_training_command,
    apply_probe_result,
    stable_success_reached,
    update_state,
)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        base_checkpoint=str(tmp_path / "base.pt"),
        initial_checkpoint=None,
        initial_training_checkpoint=None,
        output_dir=str(tmp_path / "run"),
        terrain_seed=101,
        simulation_seed=7,
        training_seed_stride=1,
        physical_gpu=1,
        single_gpu_renderer=False,
        max_episodes=100,
        max_steps_per_episode=2000,
        probe_interval=5,
        probe_failure_policy="rollback",
        action_std=0.01,
        actor_lr=5e-5,
        critic_lr=5e-5,
        reference_anchor_coef=1.0,
        flat_anchor_observations=None,
        max_reference_mse=1e-4,
        stability_window=10,
        stability_min_success_rate=0.8,
        stability_min_consecutive_successes=3,
        video_probes=False,
        video_episodes=False,
    )


def test_parser_defaults_to_rollback_and_accepts_a_separate_training_candidate():
    parser = build_parser()
    defaults = parser.parse_args(["--base-checkpoint", "base.pt", "--output-dir", "run"])
    assert defaults.probe_failure_policy == "rollback"
    assert defaults.initial_training_checkpoint is None
    args = parser.parse_args([
        "--base-checkpoint", "base.pt", "--output-dir", "run",
        "--initial-checkpoint", "verified.pt", "--initial-training-checkpoint", "candidate.pt",
        "--probe-failure-policy", "continue_candidate",
    ])
    assert args.initial_checkpoint == "verified.pt"
    assert args.initial_training_checkpoint == "candidate.pt"
    assert args.probe_failure_policy == "continue_candidate"


@pytest.mark.parametrize("with_verified", [False, True])
def test_initial_training_candidate_does_not_receive_verified_labels(tmp_path, with_verified):
    args = _args(tmp_path)
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    args.initial_training_checkpoint = str(candidate)
    verified = tmp_path / "verified.pt"
    if with_verified:
        verified.write_bytes(b"verified")
        args.initial_checkpoint = str(verified)

    state = _initial_state(args)

    assert state["resume_checkpoint"] == str(candidate.resolve())
    assert state["latest_checkpoint"] is None
    assert state["last_training_success_checkpoint"] is None
    expected = str(verified.resolve()) if with_verified else None
    assert state["best_checkpoint"] == state["recovery_checkpoint"] == expected
    assert state["best_checkpoint_source"] == ("initial_verified_checkpoint" if with_verified else None)
    assert state["best_checkpoint_episode"] == state["recovery_checkpoint_episode"] == (0 if with_verified else None)


@pytest.mark.parametrize("change", ["candidate_path", "candidate_bytes", "candidate_removed", "verified_path", "policy"])
def test_prepare_records_checkpoint_hashes_and_rejects_resume_configuration_changes(tmp_path, change):
    args = _args(tmp_path)
    Path(args.base_checkpoint).write_bytes(b"base")
    verified, candidate = tmp_path / "verified.pt", tmp_path / "candidate.pt"
    verified.write_bytes(b"verified")
    candidate.write_bytes(b"candidate")
    args.initial_checkpoint, args.initial_training_checkpoint = str(verified), str(candidate)
    output, _ = _prepare(args)
    manifest_path = output / "run_manifest.json"
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    assert manifest["experiment"] == "fixed_development_lora_ppo_multiepisode_v8"
    assert manifest["learning_contract"] == "requested_ppo_absorbing_success_critic_context_v7"
    assert manifest["initial_checkpoint_sha256"] == sha256(b"verified").hexdigest()
    assert manifest["initial_training_checkpoint"] == str(candidate.resolve())
    assert manifest["initial_training_checkpoint_sha256"] == sha256(b"candidate").hexdigest()
    assert manifest["probe_failure_policy"] == "rollback"
    _prepare(args)  # An identical resume remains valid.

    if change == "candidate_path":
        other = tmp_path / "other_candidate.pt"
        other.write_bytes(b"candidate")
        args.initial_training_checkpoint = str(other)
    elif change == "candidate_bytes":
        candidate.write_bytes(b"changed candidate")
    elif change == "candidate_removed":
        args.initial_training_checkpoint = None
    elif change == "verified_path":
        other = tmp_path / "other_verified.pt"
        other.write_bytes(b"verified")
        args.initial_checkpoint = str(other)
    else:
        args.probe_failure_policy = "continue_candidate"

    with pytest.raises(ValueError, match="saved manifest"):
        _prepare(args)
    assert manifest_path.read_text() == manifest_text


def test_prepare_rejects_missing_initial_training_checkpoint_before_creating_run(tmp_path):
    args = _args(tmp_path)
    Path(args.base_checkpoint).write_bytes(b"base")
    args.initial_training_checkpoint = str(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="initial training checkpoint does not exist"):
        _prepare(args)
    assert not Path(args.output_dir).exists()


def test_legacy_v7_manifest_only_resumes_with_original_rollback_defaults(tmp_path):
    args = _args(tmp_path)
    Path(args.base_checkpoint).write_bytes(b"base")
    output = Path(args.output_dir)
    output.mkdir()
    legacy_manifest = _manifest(args)
    for key in ("probe_failure_policy", "initial_training_checkpoint", "initial_training_checkpoint_sha256"):
        del legacy_manifest[key]
    legacy_manifest["experiment"] = "fixed_development_lora_ppo_multiepisode_v7"
    legacy_manifest["candidate_retention"] = "retain_until_deterministic_probe_v1"
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(legacy_manifest))
    legacy_state = _initial_state(args)
    for key in ("probe_failure_policy", "probe_failures", "probe_retained_candidates"):
        del legacy_state[key]
    legacy_state["probe_rollbacks"] = 2
    (output / "learning_state.json").write_text(json.dumps(legacy_state))

    _, restored = _prepare(args)

    assert restored["probe_failure_policy"] == "rollback"
    assert restored["probe_failures"] == restored["probe_rollbacks"] == 2
    assert restored["probe_retained_candidates"] == 0
    assert json.loads(manifest_path.read_text()) == legacy_manifest
    args.probe_failure_policy = "continue_candidate"
    with pytest.raises(ValueError, match="saved manifest"):
        _prepare(args)
    args.probe_failure_policy = "rollback"
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    args.initial_training_checkpoint = str(candidate)
    with pytest.raises(ValueError, match="saved manifest"):
        _prepare(args)


def test_training_command_chains_checkpoint_and_stops_at_physical_boundary(tmp_path):
    args = _args(tmp_path)
    checkpoint = tmp_path / "previous.pt"
    command = build_training_command(args, tmp_path / "episode", checkpoint, episode=4)
    assert command[command.index("--checkpoint-in") + 1] == str(checkpoint)
    assert "--stop-after-termination" in command
    assert "--evaluation-only" not in command
    assert command[command.index("--seed") + 1] == "10"
    assert command[command.index("--action-std") + 1] == "0.01"
    assert command[command.index("--reference-anchor-coef") + 1] == "1.0"
    assert command[command.index("--max-reference-mse") + 1] == "0.0001"
    assert "--retain-terminal-learning" in command
    assert command[command.index("--trial-deadline-steps") + 1] == "2000"


def test_probe_is_deterministic_and_can_change_validation_seed(tmp_path):
    args = _args(tmp_path)
    command = build_probe_command(args, tmp_path / "probe", tmp_path / "latest.pt", terrain_seed=109)
    assert "--evaluation-only" in command
    assert command[command.index("--terrain-seed") + 1] == "109"
    assert "--stop-after-termination" not in command


def test_resumed_training_propagates_requested_rates_and_video(tmp_path):
    args = _args(tmp_path)
    args.actor_lr, args.critic_lr = 1e-5, 3e-4
    args.video_episodes = True
    command = build_training_command(args, tmp_path / "episode", tmp_path / "verified.pt")
    assert float(command[command.index("--actor-lr") + 1]) == 1e-5
    assert float(command[command.index("--critic-lr") + 1]) == 3e-4
    assert "--video" in command
    assert command[command.index("--video-length") + 1] == "2000"


def test_source_flat_anchor_is_forwarded_to_training_and_probe(tmp_path):
    args = _args(tmp_path)
    args.flat_anchor_observations = str(tmp_path / "source_anchor.pt")
    for command in (
        build_training_command(args, tmp_path / "episode", tmp_path / "previous.pt"),
        build_probe_command(args, tmp_path / "probe", tmp_path / "previous.pt"),
    ):
        assert command[command.index("--flat-anchor-observations") + 1] == args.flat_anchor_observations


def test_single_gpu_rendering_keeps_physics_on_the_requested_device(tmp_path):
    args = _args(tmp_path)
    args.single_gpu_renderer = True
    args.physical_gpu = 3
    command = build_training_command(args, tmp_path / "episode", None)
    assert command[command.index("--device") + 1] == "cuda:3"
    settings = next(value for value in command if value.startswith("--kit_args="))
    assert "--/renderer/activeGpu=3" in settings
    assert "--/renderer/multiGpu/enabled=false" in settings
    assert "--/renderer/multiGpu/autoEnable=false" in settings


def test_state_accumulates_updates_and_detects_stable_window(tmp_path):
    args = _args(tmp_path)
    state = _initial_state(args)
    for episode in range(1, 11):
        checkpoint = tmp_path / f"episode-{episode}.pt"
        checkpoint.write_bytes(str(episode).encode())
        success = episode >= 3
        summary = {
            "completion_qualified": success,
            "fell": False,
            "safety_violation": False,
            "reset_count": 0,
            "steps": 100,
            "updates": 2,
            "rollbacks": 0,
            "policy_version": episode,
        }
        state = update_state(state, episode=episode, summary=summary, checkpoint=checkpoint)
    assert state["successes"] == 8
    assert state["updates"] == 20
    assert stable_success_reached(state, probe_passed=True)
    assert not stable_success_reached(state, probe_passed=False)


def test_failed_probe_rolls_back_to_recovery_until_a_probe_is_verified(tmp_path):
    args = _args(tmp_path)
    state = _initial_state(args)
    recovery = tmp_path / "recovery.pt"
    recovery.write_bytes(b"recovery")
    state = update_state(
        state,
        episode=1,
        summary={
            "completion_qualified": True,
            "fell": False,
            "safety_violation": False,
            "reset_count": 0,
            "steps": 100,
            "updates": 1,
            "rollbacks": 0,
            "policy_version": 1,
        },
        checkpoint=recovery,
    )
    state["recovery_checkpoint"] = str(recovery.resolve())
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    state, selected = apply_probe_result(
        state, episode=5, checkpoint=candidate, probe_passed=False, training_success=False
    )
    assert selected == recovery
    assert state["probe_rollbacks"] == 1
    assert state["probe_failures"] == 1
    assert state["probe_retained_candidates"] == 0
    assert state["resume_checkpoint"] == str(recovery.resolve())
    assert state["latest_probe"]["action"] == "rollback"
    assert state["latest_probe"]["rollback_checkpoint"] == str(recovery.resolve())

    state, selected = apply_probe_result(
        state, episode=10, checkpoint=candidate, probe_passed=True, training_success=False
    )
    assert selected == candidate
    assert state["best_checkpoint"] == str(candidate.resolve())
    assert state["best_checkpoint_source"] == "deterministic_probe"
    assert state["recovery_checkpoint"] == str(candidate.resolve())
    assert state["latest_probe"]["action"] == "promote"
    assert state["latest_probe"]["rollback_checkpoint"] is None


@pytest.mark.parametrize("training_success", [False, True])
def test_failed_probe_can_retain_candidate_without_promoting_verified_recovery(tmp_path, training_success):
    args = _args(tmp_path)
    verified, candidate = tmp_path / "verified.pt", tmp_path / "candidate.pt"
    verified.write_bytes(b"verified")
    candidate.write_bytes(b"candidate")
    args.initial_checkpoint = str(verified)
    state = _initial_state(args)
    verified_fields = {key: value for key, value in state.items() if key.startswith(("best_checkpoint", "recovery_checkpoint"))}

    retained, selected = apply_probe_result(
        state, episode=5, checkpoint=candidate, probe_passed=False,
        training_success=training_success, probe_failure_policy="continue_candidate",
    )

    assert selected == candidate
    assert retained["resume_checkpoint"] == str(candidate.resolve())
    assert retained["probe_failure_policy"] == "continue_candidate"
    assert retained["probe_failures"] == retained["probe_retained_candidates"] == 1
    assert retained["probe_rollbacks"] == 0
    assert retained["latest_probe"]["action"] == "continue_candidate"
    assert retained["latest_probe"]["continuation_checkpoint"] == str(candidate.resolve())
    assert retained["latest_probe"]["rollback_checkpoint"] is None
    assert all(retained[key] == value for key, value in verified_fields.items())
    assert state["latest_probe"] is None and state["probe_failures"] == 0


@pytest.mark.parametrize("failure_policy", ["rollback", "continue_candidate"])
def test_passing_frozen_probe_promotes_candidate_under_either_failure_policy(tmp_path, failure_policy):
    args = _args(tmp_path)
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    args.initial_training_checkpoint = str(candidate)
    state = _initial_state(args)

    promoted, selected = apply_probe_result(
        state, episode=5, checkpoint=candidate, probe_passed=True,
        training_success=False, probe_failure_policy=failure_policy,
    )

    assert selected == candidate
    assert promoted["best_checkpoint"] == promoted["recovery_checkpoint"] == str(candidate.resolve())
    assert promoted["best_checkpoint_source"] == "deterministic_probe"
    assert promoted["probe_failure_policy"] == failure_policy
    assert promoted["best_checkpoint_episode"] == promoted["recovery_checkpoint_episode"] == 5
    assert promoted["latest_probe"]["action"] == "promote"
    assert promoted["latest_probe"]["continuation_checkpoint"] == str(candidate.resolve())
    assert promoted["latest_probe"]["rollback_checkpoint"] is None
    assert promoted["probe_failures"] == promoted["probe_rollbacks"] == promoted["probe_retained_candidates"] == 0


def test_failed_probe_without_verified_recovery_rolls_back_to_base_by_default(tmp_path):
    args = _args(tmp_path)
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    args.initial_training_checkpoint = str(candidate)
    state, selected = apply_probe_result(
        _initial_state(args), episode=5, checkpoint=candidate,
        probe_passed=False, training_success=False,
    )
    assert selected is None and state["resume_checkpoint"] is None
    assert state["best_checkpoint"] is None and state["recovery_checkpoint"] is None
    assert state["probe_failures"] == state["probe_rollbacks"] == 1
    assert state["latest_probe"]["action"] == "rollback"
    assert state["latest_probe"]["continuation_checkpoint"] is None


def test_failed_probe_does_not_count_a_rollback_when_selected_checkpoint_is_unchanged(tmp_path):
    args = _args(tmp_path)
    verified = tmp_path / "verified.pt"
    verified.write_bytes(b"verified")
    args.initial_checkpoint = str(verified)
    state, selected = apply_probe_result(
        _initial_state(args), episode=5, checkpoint=verified,
        probe_passed=False, training_success=False,
    )
    assert selected == verified
    assert state["probe_failures"] == 1
    assert state["probe_rollbacks"] == state["probe_retained_candidates"] == 0
    assert state["latest_probe"]["action"] == "unchanged"
    assert state["latest_probe"]["rollback_checkpoint"] is None


def test_training_success_stays_unverified_until_a_passing_frozen_probe(tmp_path):
    args = _args(tmp_path)
    state = _initial_state(args)
    first = tmp_path / "first.pt"
    first.write_bytes(b"first")
    state = update_state(
        state,
        episode=1,
        summary={
            "completion_qualified": True,
            "fell": False,
            "safety_violation": False,
            "reset_count": 0,
            "steps": 10,
            "updates": 1,
            "rollbacks": 0,
            "policy_version": 1,
        },
        checkpoint=first,
    )
    second = tmp_path / "second.pt"
    second.write_bytes(b"second")
    state = update_state(
        state,
        episode=2,
        summary={
            "completion_qualified": True,
            "fell": False,
            "safety_violation": False,
            "reset_count": 0,
            "steps": 10,
            "updates": 1,
            "rollbacks": 0,
            "policy_version": 2,
        },
        checkpoint=second,
    )
    assert state["recovery_checkpoint"] is None
    assert state["last_training_success_checkpoint"] == str(second.resolve())
    state, selected = apply_probe_result(
        state,
        episode=3,
        checkpoint=second,
        probe_passed=False,
        training_success=False,
        fallback_checkpoint=first,
    )
    assert selected == first
    assert state["resume_checkpoint"] == str(first.resolve())
    assert state["recovery_checkpoint"] is None
    assert state["best_checkpoint"] is None


def test_failed_episode_candidate_is_trained_until_next_probe(tmp_path, monkeypatch):
    import json
    from scripts.reinforcement_learning.online_adaptation import run_lora_ppo_multiepisode as runner

    args = _args(tmp_path)
    Path(args.base_checkpoint).write_bytes(b"base")
    initial = tmp_path / "verified.pt"
    initial.write_bytes(b"verified")
    args.initial_checkpoint = str(initial)
    args.max_episodes = args.probe_interval = 2
    calls = []

    def fake_run(command, run_dir):
        calls.append(command)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "latest.pt").write_bytes(str(run_dir).encode())
        (run_dir / "summary.json").write_text(json.dumps({
            "completion_qualified": False, "fell": True, "safety_violation": False,
            "reset_count": 1, "steps": 100, "max_x": 0.7, "updates": 1,
            "rollbacks": 0, "policy_version": len(calls),
            "stop_reason": "terminated", "command_verified_steps": 100,
        }))

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "build_parser", lambda: Namespace(parse_args=lambda: args))
    assert runner.main() == 0
    training = [c for c in calls if "--evaluation-only" not in c]
    assert training[1][training[1].index("--checkpoint-in") + 1] == str(
        Path(args.output_dir) / "episodes/episode-0001/latest.pt"
    )
    state = json.loads((Path(args.output_dir) / "learning_state.json").read_text())
    assert state["resume_checkpoint"] == str(initial.resolve())  # rejected by probe
    records = [json.loads(line) for line in
               (Path(args.output_dir) / "episodes.jsonl").read_text().splitlines()]
    assert all(row["stop_reason"] == "terminated" and row["command_verified_steps"] == 100
               for row in records)


@pytest.mark.parametrize("failure_policy", ["rollback", "continue_candidate"])
def test_failed_probe_selects_the_correct_next_training_branch_without_changing_verified_final_choice(
    tmp_path, monkeypatch, failure_policy,
):
    from scripts.reinforcement_learning.online_adaptation import run_lora_ppo_multiepisode as runner

    args = _args(tmp_path)
    Path(args.base_checkpoint).write_bytes(b"base")
    verified, initial_candidate = tmp_path / "verified.pt", tmp_path / "initial_candidate.pt"
    verified.write_bytes(b"verified")
    initial_candidate.write_bytes(b"unverified")
    args.initial_checkpoint = str(verified)
    args.initial_training_checkpoint = str(initial_candidate)
    args.probe_failure_policy = failure_policy
    args.max_episodes, args.probe_interval = 3, 1
    calls = []

    def fake_run(command, run_dir):
        calls.append((command, run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "latest.pt").write_bytes(str(run_dir).encode())
        (run_dir / "summary.json").write_text(json.dumps({
            "completion_qualified": False, "fell": True, "safety_violation": False,
            "reset_count": 1, "steps": 100, "max_x": 0.7, "updates": 1,
            "rollbacks": 0, "policy_version": len(calls), "stop_reason": "terminated",
        }))

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "build_parser", lambda: Namespace(parse_args=lambda: args))
    assert runner.main() == 0
    training = [(command, path) for command, path in calls if "--evaluation-only" not in command]
    assert training[0][0][training[0][0].index("--checkpoint-in") + 1] == str(initial_candidate.resolve())
    for index in (1, 2):
        expected = training[index - 1][1] / "latest.pt" if failure_policy == "continue_candidate" else verified
        command = training[index][0]
        assert command[command.index("--checkpoint-in") + 1] == str(expected.resolve())
    state = json.loads((Path(args.output_dir) / "learning_state.json").read_text())
    assert state["best_checkpoint"] == state["recovery_checkpoint"] == str(verified.resolve())
    assert state["best_checkpoint_source"] == "initial_verified_checkpoint"
    assert state["final_checkpoint"] == str(verified.resolve())
    assert state["status"] == "episode_cap_reached"
    assert state["stable_success_episode"] is None
    assert state["probe_failures"] == 3
    assert state["probe_rollbacks"] == (3 if failure_policy == "rollback" else 0)
    assert state["probe_retained_candidates"] == (3 if failure_policy == "continue_candidate" else 0)
    assert state["latest_probe"]["action"] == failure_policy
    expected_resume = training[-1][1] / "latest.pt" if failure_policy == "continue_candidate" else verified
    assert state["resume_checkpoint"] == state["latest_probe"]["continuation_checkpoint"] == str(expected_resume.resolve())
    assert state["latest_probe"]["rollback_checkpoint"] == (str(verified.resolve()) if failure_policy == "rollback" else None)
    validation = [command for command, path in calls if path.parent.name == "final_validation"]
    assert validation and all(command[command.index("--checkpoint-in") + 1] == str(verified.resolve()) for command in validation)
