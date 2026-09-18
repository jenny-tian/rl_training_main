"""Run checkpoint-continuous multi-episode LoRA-PPO on one development layout."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol


TASK = "RockyRoad-Deeprobotics-Lite3-v0"
SCHEMA_VERSION = 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and in [0, 1]")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--initial-checkpoint",
        default=None,
        help="optional verified LoRA checkpoint used as the initial and recovery branch",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--terrain-seed", type=int, default=101)
    parser.add_argument("--simulation-seed", type=int, default=7)
    parser.add_argument("--training-seed-stride", type=_positive_int, default=1)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 3), default=1)
    parser.add_argument("--max-episodes", type=_positive_int, default=100)
    parser.add_argument("--max-steps-per-episode", type=_positive_int, default=2_000)
    parser.add_argument("--probe-interval", type=_positive_int, default=5)
    parser.add_argument("--action-std", type=_positive_float, default=0.01)
    parser.add_argument("--reference-anchor-coef", type=_nonnegative_float, default=1.0)
    parser.add_argument("--max-reference-mse", type=_positive_float, default=1e-4)
    parser.add_argument("--stability-window", type=_positive_int, default=10)
    parser.add_argument("--stability-min-success-rate", type=_unit_interval, default=0.8)
    parser.add_argument("--stability-min-consecutive-successes", type=_positive_int, default=3)
    parser.add_argument("--video-probes", action="store_true")
    return parser


def _common_command(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: Path | None,
    *,
    simulation_seed: int | None = None,
) -> list[str]:
    script = Path(__file__).with_name("train_lora_ppo.py")
    command = [
        "/home/tjx/anaconda3/envs/env_isaaclab/bin/python",
        "-u",
        str(script),
        "--base-checkpoint",
        str(Path(args.base_checkpoint).resolve()),
        "--task",
        TASK,
        "--terrain-stage",
        "rough_4cm",
        "--terrain-seed",
        str(args.terrain_seed),
        "--protocol-sha256",
        default_flat_online_protocol().sha256(),
        "--require-track-completion",
        "--seed",
        str(args.simulation_seed if simulation_seed is None else simulation_seed),
        "--forward-speed",
        "0.3",
        "--max-steps",
        str(args.max_steps_per_episode),
        "--rollout-steps",
        "256",
        "--output",
        str(run_dir),
        "--rank",
        "2",
        "--action-std",
        str(args.action_std),
        "--residual-rate-limit",
        "0.03",
        "--actor-lr",
        "0.00005",
        "--critic-lr",
        "0.00005",
        "--nonfoot-contact-rollback-force",
        "20",
        "--ppo-epochs",
        "2",
        "--minibatch-size",
        "64",
        "--clip-eps",
        "0.1",
        "--target-kl",
        "0.01",
        "--gamma",
        "0.99",
        "--gae-lambda",
        "0.95",
        "--entropy-coef",
        "0",
        "--value-coef",
        "0.5",
        "--reference-anchor-coef",
        str(args.reference_anchor_coef),
        "--max-reference-mse",
        str(args.max_reference_mse),
        "--terminal-failure-penalty",
        "-8",
        "--retain-terminal-learning",
        "--max-grad-norm",
        "1",
        "--headless",
        "--device",
        f"cuda:{args.physical_gpu}",
    ]
    if checkpoint is not None:
        command.extend(("--checkpoint-in", str(checkpoint)))
    return command


def build_training_command(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: Path | None,
    *,
    episode: int = 1,
) -> list[str]:
    simulation_seed = args.simulation_seed + (episode - 1) * args.training_seed_stride
    return [
        *_common_command(
            args,
            run_dir,
            checkpoint,
            simulation_seed=simulation_seed,
        ),
        "--stop-after-termination",
    ]


def build_probe_command(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: Path,
    *,
    terrain_seed: int | None = None,
) -> list[str]:
    command = _common_command(args, run_dir, checkpoint)
    command.extend(("--evaluation-only",))
    if terrain_seed is not None:
        command[command.index("--terrain-seed") + 1] = str(terrain_seed)
    if args.video_probes:
        command.extend(("--video", "--video-length", str(args.max_steps_per_episode)))
    return command


def _successful(summary: Mapping[str, object]) -> bool:
    return (
        summary.get("completion_qualified") is True
        and summary.get("fell") is False
        and summary.get("safety_violation") is False
        and summary.get("reset_count") == 0
    )


def update_state(
    state: Mapping[str, object], *, episode: int, summary: Mapping[str, object], checkpoint: Path
) -> dict[str, Any]:
    if episode != int(state["episodes_completed"]) + 1:
        raise ValueError("episode sequence is not contiguous")
    success = _successful(summary)
    recent = [bool(value) for value in state["recent_successes"]]
    recent.append(success)
    recent = recent[-int(state["stability_window"]):]
    falls = [bool(value) for value in state["recent_falls"]]
    falls.append(summary.get("fell") is True)
    falls = falls[-int(state["stability_window"]):]
    consecutive = int(state["consecutive_successes"]) + 1 if success else 0
    checkpoint_path = str(checkpoint.resolve())
    checkpoint_sha256 = sha256_file(checkpoint)
    recovery_checkpoint = state.get("recovery_checkpoint")
    recovery_checkpoint_sha256 = state.get("recovery_checkpoint_sha256")
    recovery_checkpoint_episode = state.get("recovery_checkpoint_episode")
    last_training_success_checkpoint = state.get("last_training_success_checkpoint")
    last_training_success_sha256 = state.get("last_training_success_sha256")
    last_training_success_episode = state.get("last_training_success_episode")
    if success:
        last_training_success_checkpoint = checkpoint_path
        last_training_success_sha256 = checkpoint_sha256
        last_training_success_episode = episode
    return {
        **state,
        "episodes_completed": episode,
        "interaction_steps": int(state["interaction_steps"]) + int(summary["steps"]),
        "successes": int(state["successes"]) + int(success),
        "falls": int(state["falls"]) + int(summary.get("fell") is True),
        "safety_violations": int(state["safety_violations"])
        + int(summary.get("safety_violation") is True),
        "updates": int(state["updates"]) + int(summary["updates"]),
        "accepted_updates": int(state["accepted_updates"])
        + int(summary.get("accepted_updates", summary["updates"])),
        "rejected_updates": int(state["rejected_updates"])
        + int(summary.get("rejected_updates", 0)),
        "rollbacks": int(state["rollbacks"]) + int(summary["rollbacks"]),
        "recent_successes": recent,
        "recent_falls": falls,
        "recent_success_rate": sum(recent) / len(recent),
        "consecutive_successes": consecutive,
        "max_consecutive_successes": max(int(state["max_consecutive_successes"]), consecutive),
        "first_success_episode": (
            episode if success and state["first_success_episode"] is None else state["first_success_episode"]
        ),
        "policy_version": int(summary["policy_version"]),
        "latest_checkpoint": checkpoint_path,
        "latest_checkpoint_sha256": checkpoint_sha256,
        "resume_checkpoint": checkpoint_path,
        "recovery_checkpoint": recovery_checkpoint,
        "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
        "recovery_checkpoint_episode": recovery_checkpoint_episode,
        "last_training_success_checkpoint": last_training_success_checkpoint,
        "last_training_success_sha256": last_training_success_sha256,
        "last_training_success_episode": last_training_success_episode,
    }


def apply_probe_result(
    state: Mapping[str, object],
    *,
    episode: int,
    checkpoint: Path,
    probe_passed: bool,
    training_success: bool,
    fallback_checkpoint: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Choose the next branch, preferring a deterministically verified checkpoint."""

    updated = dict(state)
    current_path = str(checkpoint.resolve())
    current_sha256 = sha256_file(checkpoint)
    if probe_passed:
        updated.update(
            {
                "best_checkpoint": current_path,
                "best_checkpoint_sha256": current_sha256,
                "best_checkpoint_episode": episode,
                "best_checkpoint_source": "deterministic_probe",
                "resume_checkpoint": current_path,
            }
        )
        updated.update(
            {
                "recovery_checkpoint": current_path,
                "recovery_checkpoint_sha256": current_sha256,
                "recovery_checkpoint_episode": episode,
            }
        )
        return updated, checkpoint

    fallback = (
        str(fallback_checkpoint.resolve())
        if fallback_checkpoint is not None
        else updated.get("recovery_checkpoint") or updated.get("best_checkpoint")
    )
    updated["probe_rollbacks"] = int(updated.get("probe_rollbacks", 0)) + 1
    updated["resume_checkpoint"] = fallback
    if fallback is not None:
        updated["recovery_checkpoint"] = fallback
        updated["recovery_checkpoint_sha256"] = sha256_file(Path(str(fallback)))
    return updated, Path(str(fallback)) if fallback is not None else None


def stable_success_reached(state: Mapping[str, object], *, probe_passed: bool) -> bool:
    recent = state["recent_successes"]
    return (
        len(recent) == int(state["stability_window"])
        and not any(state["recent_falls"])
        and float(state["recent_success_rate"]) >= float(state["stability_min_success_rate"])
        and int(state["consecutive_successes"])
        >= int(state["stability_min_consecutive_successes"])
        and int(state["accepted_updates"]) > 0
        and probe_passed
    )


def _initial_state(args: argparse.Namespace) -> dict[str, Any]:
    initial_checkpoint = (
        str(Path(args.initial_checkpoint).resolve()) if args.initial_checkpoint else None
    )
    initial_sha256 = sha256_file(args.initial_checkpoint) if args.initial_checkpoint else None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "episodes_completed": 0,
        "interaction_steps": 0,
        "successes": 0,
        "falls": 0,
        "safety_violations": 0,
        "updates": 0,
        "accepted_updates": 0,
        "rejected_updates": 0,
        "rollbacks": 0,
        "recent_successes": [],
        "recent_falls": [],
        "recent_success_rate": 0.0,
        "consecutive_successes": 0,
        "max_consecutive_successes": 0,
        "first_success_episode": None,
        "stable_success_episode": None,
        "policy_version": 0,
        "latest_checkpoint": None,
        "latest_checkpoint_sha256": None,
        "resume_checkpoint": initial_checkpoint,
        "recovery_checkpoint": initial_checkpoint,
        "recovery_checkpoint_sha256": initial_sha256,
        "recovery_checkpoint_episode": 0 if initial_checkpoint else None,
        "last_training_success_checkpoint": None,
        "last_training_success_sha256": None,
        "last_training_success_episode": None,
        "best_checkpoint": initial_checkpoint,
        "best_checkpoint_sha256": initial_sha256,
        "best_checkpoint_episode": 0 if initial_checkpoint else None,
        "best_checkpoint_source": "initial_verified_checkpoint" if initial_checkpoint else None,
        "probe_rollbacks": 0,
        "latest_probe": None,
        "stability_window": args.stability_window,
        "stability_min_success_rate": args.stability_min_success_rate,
        "stability_min_consecutive_successes": args.stability_min_consecutive_successes,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _run(command: list[str], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    with (run_dir / "runtime.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"LoRA-PPO worker failed with exit code {result.returncode}: {run_dir}")


def _record(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "fixed_development_lora_ppo_multiepisode_v4",
        "learning_contract": "boundary_samples_prereset_sticky_safety_v4",
        "candidate_retention": "retain_until_deterministic_probe_v1",
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "initial_checkpoint": (
            str(Path(args.initial_checkpoint).resolve()) if args.initial_checkpoint else None
        ),
        "initial_checkpoint_sha256": (
            sha256_file(args.initial_checkpoint) if args.initial_checkpoint else None
        ),
        "terrain_seed": args.terrain_seed,
        "simulation_seed": args.simulation_seed,
        "training_seed_stride": args.training_seed_stride,
        "max_episodes": args.max_episodes,
        "max_steps_per_episode": args.max_steps_per_episode,
        "probe_interval": args.probe_interval,
        "action_std": args.action_std,
        "reference_anchor_coef": args.reference_anchor_coef,
        "max_reference_mse": args.max_reference_mse,
        "reward_contract": "progress_stall_completion_lateral_v2",
        "terminal_failure_penalty": -8.0,
        "retain_terminal_learning": True,
        "protocol_sha256": default_flat_online_protocol().sha256(),
        "completion_qualification_version": default_flat_online_protocol().completion_qualification_version,
        "physical_gpu": args.physical_gpu,
    }


def _prepare(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.stability_min_consecutive_successes > args.stability_window:
        raise ValueError("consecutive success requirement exceeds stability window")
    if args.terrain_seed not in default_flat_online_protocol().development_seeds:
        raise ValueError("terrain seed must belong to the development split")
    if args.initial_checkpoint is not None and not Path(args.initial_checkpoint).is_file():
        raise ValueError("initial checkpoint does not exist")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and _load_json(manifest_path) != manifest:
        raise ValueError("resume configuration differs from the saved manifest")
    if not manifest_path.exists():
        write_json_atomic(manifest_path, manifest)
    state_path = output / "learning_state.json"
    state = _load_json(state_path) if state_path.exists() else _initial_state(args)
    # Only a deterministic probe can promote the recovery branch.  A failed
    # training episode still carries useful updates until the next probe.
    best_is_probe_verified = state.get("best_checkpoint_source") == "deterministic_probe"
    if state_path.exists() and best_is_probe_verified:
        state["recovery_checkpoint"] = state.get("best_checkpoint")
        state["recovery_checkpoint_sha256"] = state.get("best_checkpoint_sha256")
        state["recovery_checkpoint_episode"] = state.get("best_checkpoint_episode")
        state["resume_checkpoint"] = state.get("resume_checkpoint") or state.get("best_checkpoint")
    return output, state


def main() -> int:
    args = build_parser().parse_args()
    output, state = _prepare(args)
    records_path = output / "episodes.jsonl"
    resume_checkpoint = state.get("resume_checkpoint", state.get("latest_checkpoint"))
    checkpoint = Path(str(resume_checkpoint)) if resume_checkpoint else None
    for episode in range(int(state["episodes_completed"]) + 1, args.max_episodes + 1):
        recovery_before_episode = state.get("recovery_checkpoint")
        run_dir = output / "episodes" / f"episode-{episode:04d}"
        _run(build_training_command(args, run_dir, checkpoint, episode=episode), run_dir)
        summary = _load_json(run_dir / "summary.json")
        candidate_checkpoint = run_dir / "latest.pt"
        checkpoint = candidate_checkpoint
        state = update_state(
            state, episode=episode, summary=summary, checkpoint=candidate_checkpoint
        )
        episode_success = _successful(summary)
        record = {
            "episode": episode,
            "successful_completion": _successful(summary),
            "steps": summary["steps"],
            "max_x": summary["max_x"],
            "fell": summary["fell"],
            "safety_violation": summary["safety_violation"],
            "updates": summary["updates"],
            "accepted_updates": summary.get("accepted_updates", summary["updates"]),
            "rejected_updates": summary.get("rejected_updates", 0),
            "rollbacks": summary["rollbacks"],
            "policy_version": summary["policy_version"],
            "cumulative_success_rate": state["successes"] / episode,
            "recent_success_rate": state["recent_success_rate"],
            "checkpoint_sha256": state["latest_checkpoint_sha256"],
        }
        _record(records_path, record)
        probe_due = episode % args.probe_interval == 0 or episode == args.max_episodes
        if probe_due:
            probe_dir = output / "probes" / f"episode-{episode:04d}" / f"seed-{args.terrain_seed}"
            _run(build_probe_command(args, probe_dir, candidate_checkpoint), probe_dir)
            probe_summary = _load_json(probe_dir / "summary.json")
            state["latest_probe"] = {
                "episode": episode,
                "successful_completion": _successful(probe_summary),
                "checkpoint_sha256": sha256_file(candidate_checkpoint),
            }
            state, checkpoint = apply_probe_result(
                state,
                episode=episode,
                checkpoint=candidate_checkpoint,
                probe_passed=_successful(probe_summary),
                training_success=episode_success,
                fallback_checkpoint=(
                    Path(str(recovery_before_episode))
                    if recovery_before_episode
                    else None
                ),
            )
            state["latest_probe"]["rollback_checkpoint"] = (
                str(checkpoint.resolve()) if checkpoint is not None else None
            )
        write_json_atomic(output / "learning_state.json", state)
        if probe_due and stable_success_reached(
            state, probe_passed=state["latest_probe"]["successful_completion"] is True
        ):
            state["status"] = "stable_on_training_layout"
            state["stable_success_episode"] = episode
            break
    else:
        state["status"] = "episode_cap_reached"

    selected_checkpoint_value = (
        state.get("best_checkpoint")
        or state.get("recovery_checkpoint")
        or state.get("latest_checkpoint")
    )
    if selected_checkpoint_value is None:
        raise RuntimeError("training produced no checkpoint for final validation")
    checkpoint = Path(str(selected_checkpoint_value))
    state["final_checkpoint"] = str(checkpoint.resolve())
    state["final_checkpoint_sha256"] = sha256_file(checkpoint)
    validation = []
    for seed in default_flat_online_protocol().development_seeds:
        if seed == args.terrain_seed:
            continue
        run_dir = output / "final_validation" / f"seed-{seed}"
        _run(build_probe_command(args, run_dir, checkpoint, terrain_seed=seed), run_dir)
        summary = _load_json(run_dir / "summary.json")
        validation.append({"seed": seed, "successful_completion": _successful(summary)})
    state["final_validation"] = {
        "count": len(validation),
        "successes": sum(item["successful_completion"] for item in validation),
        "results": validation,
    }
    write_json_atomic(output / "learning_state.json", state)
    write_json_atomic(output / "summary.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
