from argparse import Namespace
from pathlib import Path

from scripts.reinforcement_learning.online_adaptation.run_lora_ppo_multiepisode import (
    _initial_state,
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
        output_dir=str(tmp_path / "run"),
        terrain_seed=101,
        simulation_seed=7,
        training_seed_stride=1,
        physical_gpu=1,
        max_episodes=100,
        max_steps_per_episode=2000,
        probe_interval=5,
        action_std=0.01,
        reference_anchor_coef=1.0,
        max_reference_mse=1e-4,
        stability_window=10,
        stability_min_success_rate=0.8,
        stability_min_consecutive_successes=3,
        video_probes=False,
    )


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


def test_probe_is_deterministic_and_can_change_validation_seed(tmp_path):
    args = _args(tmp_path)
    command = build_probe_command(args, tmp_path / "probe", tmp_path / "latest.pt", terrain_seed=109)
    assert "--evaluation-only" in command
    assert command[command.index("--terrain-seed") + 1] == "109"
    assert "--stop-after-termination" not in command


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


def test_failed_probe_rolls_back_to_first_success_until_a_probe_is_verified(tmp_path):
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
    assert state["resume_checkpoint"] == str(recovery.resolve())

    state, selected = apply_probe_result(
        state, episode=10, checkpoint=candidate, probe_passed=True, training_success=False
    )
    assert selected == candidate
    assert state["best_checkpoint"] == str(candidate.resolve())
    assert state["best_checkpoint_source"] == "deterministic_probe"
    assert state["recovery_checkpoint"] == str(candidate.resolve())


def test_each_successful_episode_becomes_the_next_failure_recovery_point(tmp_path):
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
