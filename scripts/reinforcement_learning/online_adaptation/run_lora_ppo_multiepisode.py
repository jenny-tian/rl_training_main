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
from scripts.reinforcement_learning.online_adaptation.cli import (
    add_risk_gate_args,
    append_risk_gate_options,
    risk_gate_kwargs,
)


TASK = "RockyRoad-Deeprobotics-Lite3-v0"
SCHEMA_VERSION = 1
PROBE_FAILURE_POLICIES = ("rollback", "continue_candidate")
TRAINING_FAILURE_POLICIES = ("rollback", "continue_candidate")


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
    parser.add_argument(
        "--initial-training-checkpoint",
        default=None,
        help="optional unverified checkpoint used only to start the training branch",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--terrain-seed", type=int, default=101)
    parser.add_argument(
        "--train-seeds",
        default=None,
        help="comma-separated development terrain seeds rotated across training episodes",
    )
    parser.add_argument(
        "--validation-seeds",
        default=None,
        help="comma-separated held-out development seeds; sealed seeds are reserved for the final test stage",
    )
    parser.add_argument("--simulation-seed", type=int, default=7)
    parser.add_argument("--training-seed-stride", type=_positive_int, default=1)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 3), default=1)
    parser.add_argument("--single-gpu-renderer", action="store_true")
    parser.add_argument("--proprioceptive-context", action="store_true")
    parser.add_argument("--height-scan-adapter", action="store_true")
    parser.add_argument("--max-episodes", type=_positive_int, default=100)
    parser.add_argument("--max-steps-per-episode", type=_positive_int, default=2_000)
    parser.add_argument("--probe-interval", type=_positive_int, default=5)
    parser.add_argument(
        "--probe-failure-policy",
        choices=PROBE_FAILURE_POLICIES,
        default="rollback",
        help="rollback to verified recovery or continue training the unverified candidate after a failed probe",
    )
    parser.add_argument(
        "--training-failure-policy",
        choices=TRAINING_FAILURE_POLICIES,
        default="continue_candidate",
        help="rollback after an unsuccessful or unsafe training episode, or continue the candidate",
    )
    parser.add_argument("--action-std", type=_positive_float, default=0.01)
    parser.add_argument("--final-action-std", type=_positive_float, default=None,
                        help="geometrically decay exploration to this value over max-episodes")
    parser.add_argument("--actor-lr", type=_positive_float, default=5e-5)
    parser.add_argument("--critic-lr", type=_positive_float, default=5e-5)
    parser.add_argument("--reference-anchor-coef", type=_nonnegative_float, default=1.0)
    parser.add_argument("--flat-anchor-observations", default=None)
    parser.add_argument("--max-reference-mse", type=_positive_float, default=1e-4)
    parser.add_argument("--safety-projection-penalty", type=_nonnegative_float, default=0.0)
    parser.add_argument("--terrain-reference-checkpoint", default=None)
    parser.add_argument("--terrain-anchor-coef", type=_nonnegative_float, default=0.0)
    parser.add_argument("--max-terrain-anchor-mse", type=_positive_float, default=None)
    parser.add_argument("--stability-window", type=_positive_int, default=10)
    parser.add_argument("--stability-min-success-rate", type=_unit_interval, default=0.8)
    parser.add_argument("--stability-min-consecutive-successes", type=_positive_int, default=3)
    parser.add_argument("--video-probes", action="store_true")
    parser.add_argument("--video-episodes", action="store_true")
    add_risk_gate_args(parser)
    return parser


def _common_command(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: Path | None,
    *,
    simulation_seed: int | None = None,
    terrain_split: str = "development",
) -> list[str]:
    projection_penalty = float(getattr(args, "safety_projection_penalty", 0.0))
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
        "--terrain-split",
        terrain_split,
        "--protocol-sha256",
        default_flat_online_protocol().sha256(),
        "--require-track-completion",
        "--seed",
        str(args.simulation_seed if simulation_seed is None else simulation_seed),
        "--forward-speed",
        "0.3",
        "--max-steps",
        str(args.max_steps_per_episode),
        "--trial-deadline-steps",
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
        str(args.actor_lr),
        "--critic-lr",
        str(args.critic_lr),
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
        "--safety-projection-penalty",
        str(projection_penalty),
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
    if args.flat_anchor_observations is not None:
        command.extend(("--flat-anchor-observations", str(Path(args.flat_anchor_observations).resolve())))
    if args.proprioceptive_context:
        command.append("--proprioceptive-context")
    if getattr(args, "height_scan_adapter", False):
        command.append("--height-scan-adapter")
    if args.single_gpu_renderer:
        command.append(
            f"--kit_args=--/renderer/activeGpu={args.physical_gpu} "
            "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false "
            "--/renderer/multiGpu/maxGpuCount=1"
        )
    if hasattr(args, "risk_gate_enabled"):
        append_risk_gate_options(command, args)
    return command


def episode_action_std(args: argparse.Namespace, episode: int) -> float:
    """Schedule by absolute episode index so a resume never restarts decay.

    Distribution changes happen only between workers, before an empty rollout.
    Frozen probes still execute the checkpoint mean without any noise.
    """
    if not 1 <= episode <= args.max_episodes:
        raise ValueError("episode must be within the configured training budget")
    initial, final = args.action_std, args.final_action_std
    if final is None:
        return initial
    if final > initial:
        raise ValueError("final action std must not exceed initial action std")
    if args.max_episodes < 2 and final != initial:
        raise ValueError("noise decay requires at least two episodes")
    if episode == 1:
        return initial
    if episode == args.max_episodes:
        return final
    fraction = (episode - 1) / (args.max_episodes - 1)
    return math.exp((1 - fraction) * math.log(initial) + fraction * math.log(final))


def build_training_command(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: Path | None,
    *,
    episode: int = 1,
) -> list[str]:
    simulation_seed = args.simulation_seed + (episode - 1) * args.training_seed_stride
    command = [
        *_common_command(
            args,
            run_dir,
            checkpoint,
            simulation_seed=simulation_seed,
        ),
        "--stop-after-termination",
    ]
    command[command.index("--terrain-seed") + 1] = str(_episode_terrain_seed(args, episode))
    command[command.index("--action-std") + 1] = str(episode_action_std(args, episode))
    if args.terrain_reference_checkpoint:
        command.extend(("--terrain-reference-checkpoint", str(Path(args.terrain_reference_checkpoint).resolve()),
                        "--terrain-anchor-coef", str(args.terrain_anchor_coef)))
        if args.max_terrain_anchor_mse is not None:
            command.extend(("--max-terrain-anchor-mse", str(args.max_terrain_anchor_mse)))
    elif args.terrain_anchor_coef or args.max_terrain_anchor_mse is not None:
        raise ValueError("terrain anchor constraints require --terrain-reference-checkpoint")
    if args.video_episodes:
        command.extend(("--video", "--video-length", str(args.max_steps_per_episode)))
    return command


def _episode_terrain_seed(args: argparse.Namespace, episode: int) -> int:
    seeds = getattr(args, "train_seed_values", None)
    return int(seeds[(episode - 1) % len(seeds)]) if seeds else int(args.terrain_seed)


def _configure_train_seeds(args: argparse.Namespace) -> None:
    development = set(default_flat_online_protocol().development_seeds)
    requested = getattr(args, "train_seeds", None)
    if requested is None:
        args.train_seed_values = (int(args.terrain_seed),)
        return
    try:
        values = tuple(int(item.strip()) for item in requested.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--train-seeds must be a comma-separated integer list") from error
    if not values or any(value not in development for value in values):
        raise ValueError("all --train-seeds must belong to the development split")
    if len(set(values)) != len(values):
        raise ValueError("--train-seeds must not contain duplicates")
    args.train_seed_values = values


def _parse_seed_list(raw: str, *, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{option} must be a comma-separated integer list") from error
    if not values:
        raise ValueError(f"{option} must contain at least one seed")
    if len(set(values)) != len(values):
        raise ValueError(f"{option} must not contain duplicates")
    return values


def _configure_validation_seeds(args: argparse.Namespace) -> None:
    protocol = default_flat_online_protocol()
    if hasattr(args, "train_seed_values"):
        train_values = tuple(args.train_seed_values)
    elif getattr(args, "train_seeds", None) is not None:
        train_values = _parse_seed_list(args.train_seeds, option="--train-seeds")
    else:
        train_values = (int(args.terrain_seed),)
    requested = getattr(args, "validation_seeds", None)
    if requested is None:
        # A multi-seed training run must not validate on another training
        # terrain.  Legacy single-seed runs retain their historical
        # development validation unless an explicit split is supplied.
        values = (
            (109,)
            if getattr(args, "train_seeds", None) is not None
            and {101, 103, 107}.issubset(set(train_values))
            else tuple(seed for seed in protocol.development_seeds if seed not in train_values)
        )
    else:
        values = _parse_seed_list(requested, option="--validation-seeds")
    if not values:
        raise ValueError("training seeds leave no development seed for validation; select a subset or set validation explicitly")
    if any(seed not in protocol.development_seeds for seed in values):
        raise ValueError("--validation-seeds must belong to the development split; sealed seeds are final-test only")
    overlap = set(train_values) & set(values)
    if overlap:
        raise ValueError(
            "training and validation seeds must be disjoint; overlapping seeds: "
            + ",".join(str(seed) for seed in sorted(overlap))
        )
    args.validation_seed_values = values
    args.validation_split = "development"


def _terrain_split_for_seed(seed: int) -> str:
    protocol = default_flat_online_protocol()
    if seed in protocol.development_seeds:
        return "development"
    if seed in protocol.sealed_seeds:
        return "sealed"
    raise ValueError(f"unknown validation terrain seed: {seed}")


def build_probe_command(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: Path,
    *,
    terrain_seed: int | None = None,
    terrain_split: str = "development",
) -> list[str]:
    command = _common_command(args, run_dir, checkpoint, terrain_split=terrain_split)
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
    training_safety_violation: bool = False,
    fallback_checkpoint: Path | None = None,
    probe_failure_policy: str = "rollback",
    training_failure_policy: str = "continue_candidate",
) -> tuple[dict[str, Any], Path | None]:
    """Choose a training continuation; only a passing frozen probe promotes it."""

    if probe_failure_policy not in PROBE_FAILURE_POLICIES:
        raise ValueError("unknown probe failure policy")
    if training_failure_policy not in TRAINING_FAILURE_POLICIES:
        raise ValueError("unknown training failure policy")
    training_rejected = training_failure_policy == "rollback" and (
        not training_success or training_safety_violation
    )
    updated = dict(state)
    updated["probe_failure_policy"] = probe_failure_policy
    updated["probe_failures"] = int(updated.get("probe_failures", updated.get("probe_rollbacks", 0)))
    updated["probe_rollbacks"] = int(updated.get("probe_rollbacks", 0))
    updated["probe_retained_candidates"] = int(updated.get("probe_retained_candidates", 0))
    current_path = str(checkpoint.resolve())
    current_sha256 = sha256_file(checkpoint)
    if not isinstance(training_safety_violation, bool):
        raise ValueError("training_safety_violation must be a bool")
    # A deterministic probe can pass after the training rollout itself
    # triggered a safety violation. That checkpoint is not eligible for
    # recovery: promoting it would erase the safety evidence and allow the
    # next episode to continue from a policy that just violated the guard.
    promotion_allowed = probe_passed and not training_safety_violation and not training_rejected
    latest_probe = {
        "episode": episode,
        "successful_completion": probe_passed,
        "promotion_allowed": promotion_allowed,
        "training_rejected": training_rejected,
        "training_safety_violation": training_safety_violation,
        "checkpoint_sha256": current_sha256,
        "continuation_checkpoint": current_path,
        "rollback_checkpoint": None,
    }
    updated["latest_probe"] = latest_probe
    if promotion_allowed:
        latest_probe["action"] = "promote"
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

    updated["probe_failures"] += 1
    if probe_failure_policy == "continue_candidate" and not training_rejected:
        updated["probe_retained_candidates"] += 1
        updated["resume_checkpoint"] = current_path
        latest_probe["action"] = "continue_candidate"
        return updated, checkpoint

    fallback = (
        str(fallback_checkpoint.resolve())
        if fallback_checkpoint is not None
        else updated.get("recovery_checkpoint") or updated.get("best_checkpoint")
    )
    fallback = str(Path(str(fallback)).resolve()) if fallback is not None else None
    rolled_back = fallback != current_path
    updated["probe_rollbacks"] += int(rolled_back)
    updated["resume_checkpoint"] = fallback
    latest_probe.update(
        action="rollback" if rolled_back else "unchanged",
        continuation_checkpoint=fallback,
        rollback_checkpoint=fallback if rolled_back else None,
    )
    # A fallback selection is not new verification. In particular, an
    # explicitly supplied training checkpoint must not become recovery here.
    return updated, Path(str(fallback)) if fallback is not None else None


def apply_training_failure_policy(
    state: Mapping[str, Any],
    *,
    episode: int,
    candidate_checkpoint: Path,
    training_success: bool,
    training_safety_violation: bool,
    training_failure_policy: str = "continue_candidate",
) -> tuple[dict[str, Any], Path | None]:
    """Select continuation before the probe; only a frozen probe can promote.

    Rollback uses verified recovery, or the base policy when none exists.
    Latest candidate artifacts and failure evidence remain available for audit.
    """
    if training_failure_policy not in TRAINING_FAILURE_POLICIES:
        raise ValueError("unknown training failure policy")
    if not isinstance(training_success, bool) or not isinstance(training_safety_violation, bool):
        raise ValueError("training outcome flags must be bools")
    updated = dict(state)
    failed = not training_success or training_safety_violation
    updated["training_failure_policy"] = training_failure_policy
    updated["training_failures"] = int(updated.get("training_failures", 0)) + int(failed)
    candidate_path = str(candidate_checkpoint.resolve())
    rejected = failed and training_failure_policy == "rollback"
    continuation = updated.get("recovery_checkpoint") if rejected else candidate_path
    continuation = str(Path(continuation).resolve()) if continuation is not None else None
    rolled_back = rejected and continuation != candidate_path
    updated["training_rollbacks"] = int(updated.get("training_rollbacks", 0)) + int(rolled_back)
    updated["resume_checkpoint"] = continuation
    updated["latest_training_gate"] = {
        "episode": episode,
        "training_success": training_success,
        "training_safety_violation": training_safety_violation,
        "action": "rollback" if rolled_back else "continue_candidate",
        "candidate_checkpoint": candidate_path,
        "continuation_checkpoint": continuation,
        "rollback_checkpoint": continuation if rolled_back else None,
    }
    return updated, Path(continuation) if continuation is not None else None



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
    initial_training_checkpoint = (
        str(Path(args.initial_training_checkpoint).resolve()) if args.initial_training_checkpoint else None
    )
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
        "resume_checkpoint": initial_training_checkpoint or initial_checkpoint,
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
        "probe_failure_policy": args.probe_failure_policy,
        "training_failure_policy": getattr(args, "training_failure_policy", "continue_candidate"),
        "training_failures": 0,
        "training_rollbacks": 0,
        "latest_training_gate": None,
        "probe_failures": 0,
        "probe_rollbacks": 0,
        "probe_retained_candidates": 0,
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
    episode_action_std(args, 1)  # Validate before creating any experiment manifest.
    if not hasattr(args, "train_seed_values"):
        _configure_train_seeds(args)
    if not hasattr(args, "validation_seed_values"):
        _configure_validation_seeds(args)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "fixed_development_lora_ppo_multiepisode_v8",
        "learning_contract": "requested_ppo_absorbing_success_critic_context_v7",
        "sampling_contract": "requested_gaussian_through_fixed_safety_v1",
        "critic_observation_schema": "policy_plus_body_velocity_xy_elapsed_v1",
        "candidate_retention": "verified_recovery_with_optional_training_continuation_v2",
        "probe_failure_policy": args.probe_failure_policy,
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "initial_checkpoint": (
            str(Path(args.initial_checkpoint).resolve()) if args.initial_checkpoint else None
        ),
        "initial_checkpoint_sha256": (
            sha256_file(args.initial_checkpoint) if args.initial_checkpoint else None
        ),
        "initial_training_checkpoint": (
            str(Path(args.initial_training_checkpoint).resolve()) if args.initial_training_checkpoint else None
        ),
        "initial_training_checkpoint_sha256": (
            sha256_file(args.initial_training_checkpoint) if args.initial_training_checkpoint else None
        ),
        "terrain_seed": args.terrain_seed,
        "train_seeds": list(getattr(args, "train_seed_values", (args.terrain_seed,))),
        "validation_seeds": list(args.validation_seed_values),
        "validation_split": args.validation_split,
        "seed_partition_contract": "disjoint_train_validation_v1",
        "simulation_seed": args.simulation_seed,
        "training_seed_stride": args.training_seed_stride,
        "max_episodes": args.max_episodes,
        "max_steps_per_episode": args.max_steps_per_episode,
        "probe_interval": args.probe_interval,
        "action_std": args.action_std,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "reference_anchor_coef": args.reference_anchor_coef,
        "max_reference_mse": args.max_reference_mse,
        "reward_contract": "absorbing_success_explicit_task_deadline_projection_penalty_v5",
        "safety_projection_penalty": float(getattr(args, "safety_projection_penalty", 0.0)),
        "trial_deadline_steps": args.max_steps_per_episode,
        "terminal_failure_penalty": -8.0,
        "retain_terminal_learning": True,
        "protocol_sha256": default_flat_online_protocol().sha256(),
        "completion_qualification_version": default_flat_online_protocol().completion_qualification_version,
        "physical_gpu": args.physical_gpu,
        "video_episodes": args.video_episodes,
        "video_probes": args.video_probes,
    }
    if args.flat_anchor_observations is not None:
        source_anchor = Path(args.flat_anchor_observations).resolve()
        asset = source_anchor / "source_anchor.pt" if source_anchor.is_dir() else source_anchor
        manifest.update(
            experiment="fixed_development_lora_ppo_multiepisode_v9",
            reference_anchor_distribution="fixed_reconstructed_source_flat_v1",
            flat_anchor_observations=str(source_anchor),
            flat_anchor_observations_sha256=sha256_file(asset),
        )
    if getattr(args, "height_scan_adapter", False):
        manifest.update(
            experiment="fixed_development_lora_ppo_multiepisode_v11",
            actor_observation_schema="base45_heightscan10_gated_residual_v2",
            critic_observation_schema="policy45_heightscan10_velocity_xy_elapsed_v2",
            height_scan_adapter=True,
            height_scan_schema="height_scan_summary10_v2_lateral_rows_forward_columns",
            adaptation_method="risk_gated_latent_residual_steering_v1",
            latent_steering_status="zero_initialized_scan_gate; action_inversion_pending",
        )
    if args.single_gpu_renderer:
        manifest["renderer_device_policy"] = "single_physical_gpu_v1"
    if args.proprioceptive_context:
        from rl_training.online_adaptation.ppo_context_policy import CONTEXT_POLICY_OBSERVATION_SCHEMA

        manifest.update(
            experiment="fixed_development_lora_ppo_multiepisode_v10",
            learning_contract="requested_ppo_proprioceptive_context_v10",
            actor_observation_schema=CONTEXT_POLICY_OBSERVATION_SCHEMA,
            critic_observation_schema="policy45_velocity_xy_elapsed_odometry3_v1",
            proprioceptive_context=True,
            source_anchor_context="zero_initial_context_only",
        )
    if args.terrain_reference_checkpoint:
        manifest["terrain_anchor"] = {
            "checkpoint": str(Path(args.terrain_reference_checkpoint).resolve()),
            "sha256": sha256_file(args.terrain_reference_checkpoint),
            "coefficient": args.terrain_anchor_coef,
            "max_mse": args.max_terrain_anchor_mse,
            "contract": "fixed_terrain_teacher_on_valid_rollout_observations_v1",
        }
    elif args.terrain_anchor_coef or args.max_terrain_anchor_mse is not None:
        raise ValueError("terrain anchor constraints require --terrain-reference-checkpoint")
    if args.final_action_std is not None:
        manifest["exploration_schedule"] = {
            "kind": "geometric_by_absolute_episode_v1",
            "initial_std": args.action_std, "final_std": args.final_action_std,
            "episodes": args.max_episodes,
        }
    if getattr(args, "training_failure_policy", "continue_candidate") == "rollback":
        manifest["training_failure_policy"] = "rollback"
    if getattr(args, "risk_gate_enabled", False):
        manifest["risk_gate"] = {
            "semantics": "max_state_risk_residual_envelope_v1",
            **risk_gate_kwargs(args),
        }
    return manifest


def _manifest_matches(saved: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    if saved == expected:
        return True
    # The v7 supervisor had only rollback and no separate initial candidate.
    # Its existing manifest remains valid only with those same defaults.
    if expected["probe_failure_policy"] != "rollback" or expected["initial_training_checkpoint"] is not None:
        return False
    legacy = dict(expected)
    for key in ("probe_failure_policy", "initial_training_checkpoint", "initial_training_checkpoint_sha256"):
        legacy.pop(key)
    legacy["experiment"] = "fixed_development_lora_ppo_multiepisode_v7"
    legacy["candidate_retention"] = "retain_until_deterministic_probe_v1"
    return saved == legacy


def _prepare(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.stability_min_consecutive_successes > args.stability_window:
        raise ValueError("consecutive success requirement exceeds stability window")
    if args.terrain_seed not in default_flat_online_protocol().development_seeds:
        raise ValueError("terrain seed must belong to the development split")
    _configure_train_seeds(args)
    _configure_validation_seeds(args)
    if args.initial_checkpoint is not None and not Path(args.initial_checkpoint).is_file():
        raise ValueError("initial checkpoint does not exist")
    if args.initial_training_checkpoint is not None and not Path(args.initial_training_checkpoint).is_file():
        raise ValueError("initial training checkpoint does not exist")
    if args.probe_failure_policy not in PROBE_FAILURE_POLICIES:
        raise ValueError("unknown probe failure policy")
    training_failure_policy = getattr(args, "training_failure_policy", "continue_candidate")
    if training_failure_policy not in TRAINING_FAILURE_POLICIES:
        raise ValueError("unknown training failure policy")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and not _manifest_matches(_load_json(manifest_path), manifest):
        raise ValueError("resume configuration differs from the saved manifest")
    if not manifest_path.exists():
        write_json_atomic(manifest_path, manifest)
    state_path = output / "learning_state.json"
    state = _load_json(state_path) if state_path.exists() else _initial_state(args)
    state.setdefault("probe_failure_policy", args.probe_failure_policy)
    state.setdefault("training_failure_policy", getattr(args, "training_failure_policy", "continue_candidate"))
    state.setdefault("training_failures", 0)
    state.setdefault("training_rollbacks", 0)
    state.setdefault("latest_training_gate", None)
    state.setdefault("probe_failures", int(state.get("probe_rollbacks", 0)))
    state.setdefault("probe_retained_candidates", 0)
    # Only a deterministic probe can promote the recovery branch.
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
        state = update_state(
            state, episode=episode, summary=summary, checkpoint=candidate_checkpoint
        )
        episode_success = _successful(summary)
        state, checkpoint = apply_training_failure_policy(
            state, episode=episode, candidate_checkpoint=candidate_checkpoint,
            training_success=episode_success,
            training_safety_violation=bool(summary.get("safety_violation") is True),
            training_failure_policy=args.training_failure_policy,
        )
        record = {
            "episode": episode,
            "terrain_seed": _episode_terrain_seed(args, episode),
            "successful_completion": _successful(summary),
            "steps": summary["steps"],
            "max_x": summary["max_x"],
            "fell": summary["fell"],
            "safety_violation": summary["safety_violation"],
            "stop_reason": summary["stop_reason"],
            "task_failure_reason": summary.get("task_failure_reason"),
            "command_verified_steps": summary.get("command_verified_steps", 0),
            "updates": summary["updates"],
            "accepted_updates": summary.get("accepted_updates", summary["updates"]),
            "rejected_updates": summary.get("rejected_updates", 0),
            "rollbacks": summary["rollbacks"],
            "policy_version": summary["policy_version"],
            "cumulative_success_rate": state["successes"] / episode,
            "recent_success_rate": state["recent_success_rate"],
            "checkpoint_sha256": state["latest_checkpoint_sha256"],
        }
        probe_due = episode % args.probe_interval == 0 or episode == args.max_episodes
        if probe_due:
            probe_dir = output / "probes" / f"episode-{episode:04d}" / f"seed-{_episode_terrain_seed(args, episode)}"
            _run(build_probe_command(args, probe_dir, candidate_checkpoint), probe_dir)
            probe_summary = _load_json(probe_dir / "summary.json")
            state, checkpoint = apply_probe_result(
                state,
                episode=episode,
                checkpoint=candidate_checkpoint,
                probe_passed=_successful(probe_summary),
                training_success=episode_success,
                training_safety_violation=bool(summary.get("safety_violation") is True),
                fallback_checkpoint=(
                    Path(str(recovery_before_episode))
                    if recovery_before_episode
                    else None
                ),
                probe_failure_policy=args.probe_failure_policy,
                training_failure_policy=args.training_failure_policy,
            )
        record["training_gate"] = state["latest_training_gate"]
        record["continuation_checkpoint"] = state["resume_checkpoint"]
        _record(records_path, record)
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
    for seed in args.validation_seed_values:
        run_dir = output / "final_validation" / f"seed-{seed}"
        _run(
            build_probe_command(
                args,
                run_dir,
                checkpoint,
                terrain_seed=seed,
                terrain_split=_terrain_split_for_seed(seed),
            ),
            run_dir,
        )
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
