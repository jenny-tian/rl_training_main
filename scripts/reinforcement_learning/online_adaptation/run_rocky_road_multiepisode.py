"""Run auditable multi-episode online adaptation on one fixed RockyRoad layout.

Each physical episode starts from a fresh simulator state while the residual
actor, critics, optimizers, global critic replay, novelty gate, and stable
checkpoint are restored from the previous episode.  The short actor-only replay
is intentionally episode-local because its samples are tied to a drift reference
that closes at the physical boundary.  Training uses a development seed only;
the previously used held-out partition is re-evaluated with learning disabled
after training stops and is reported as reused, not as a blind confirmatory set.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from scripts.reinforcement_learning.online_adaptation.run_flat_online_curriculum import (
    FLAT_RETENTION_SEEDS,
    EpisodeRequest,
    TargetConfiguration,
    _install_latest_checkpoint,
    _load_json_object,
    build_retention_probe_command,
    build_worker_command,
    classify_worker_result,
    flat_retention_passes,
    protocol_for_configuration,
)


ROCKY_ROAD_TASK = "RockyRoad-Deeprobotics-Lite3-v0"
FLAT_TASK = "FlatOnline-Deeprobotics-Lite3-v0"
MULTI_GPU_OFF = (
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1"
)
STATE_SCHEMA_VERSION = 1
RUNNER_SCHEMA_VERSION = 1
DRIFT_FIELDS = (
    "policy_residual_mse",
    "policy_joint_abs_change_mean",
    "policy_joint_abs_change_max",
    "policy_parameter_l2",
)


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
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--initial-residual-checkpoint", required=True)
    parser.add_argument("--flat-replay", required=True)
    parser.add_argument("--novelty-baseline", required=True)
    parser.add_argument("--frozen-benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--terrain-seed", type=int, default=101)
    parser.add_argument("--simulation-seed", type=int, default=7)
    parser.add_argument("--terrain-stage", choices=("rough_4cm",), default="rough_4cm")
    parser.add_argument("--terrain-rough-amplitude", type=_positive_float, default=0.04)
    parser.add_argument("--max-episodes", type=_positive_int, default=100)
    parser.add_argument("--max-steps-per-episode", type=_positive_int, default=2_000)
    parser.add_argument("--forward-speed", type=_positive_float, default=0.3)
    parser.add_argument("--residual-gain", type=_unit_interval, default=0.10)
    parser.add_argument("--adapting-residual-gain", type=_unit_interval, default=0.25)
    parser.add_argument("--post-step-residual-gain", type=_unit_interval, default=0.25)
    parser.add_argument("--residual-rate-limit", type=_positive_float, default=0.05)
    parser.add_argument("--actor-updates-per-second", type=_positive_float, default=0.5)
    parser.add_argument("--critic-updates-per-transition", type=_nonnegative_float, default=0.5)
    parser.add_argument("--actor-updates-per-transition", type=_nonnegative_float, default=0.005)
    parser.add_argument("--control-timeout-ms", type=_positive_float, default=50.0)
    parser.add_argument("--target-q-min", type=_finite_float, default=None)
    parser.add_argument("--target-q-max", type=_finite_float, default=None)
    parser.add_argument("--initial-alpha", type=_positive_float, default=0.02)
    parser.add_argument("--alpha-max", type=_positive_float, default=0.05)
    parser.add_argument("--critic-huber-delta", type=_positive_float, default=1.0)
    parser.add_argument("--rough-swing-lift-weight", type=_positive_float, default=1.0)
    parser.add_argument("--rough-swing-lift-target", type=_positive_float, default=0.045)
    parser.add_argument("--deterministic-actor-weight", type=_unit_interval, default=0.25)
    parser.add_argument("--max-policy-residual-mse", type=_positive_float, default=1.0e-5)
    parser.add_argument("--flat-probe-interval", type=_positive_int, default=10)
    parser.add_argument("--stability-window", type=_positive_int, default=10)
    parser.add_argument(
        "--stability-min-success-rate", type=_unit_interval, default=0.8
    )
    parser.add_argument(
        "--stability-min-consecutive-successes", type=_positive_int, default=3
    )
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--video", action="store_true", default=True)
    parser.add_argument("--sealed-evaluation", action="store_true", default=True)
    return parser


def _target_configuration(args: argparse.Namespace) -> TargetConfiguration:
    return TargetConfiguration(
        target_semantics="timeout-aware",
        n_step=1,
        critic_replay_mode="dual",
        flat_fraction_initial=0.75,
        flat_fraction_ready=0.25,
        flat_anchor_weight=0.5,
        max_policy_residual_mse=args.max_policy_residual_mse,
        terrain_feature_source="proprio",
        terrain_min_valid_fraction=0.90,
        flat_critic_warmup_updates=1_000,
        max_episodes=args.max_episodes,
        max_total_steps=args.max_episodes * args.max_steps_per_episode,
        max_stage_episodes=args.max_episodes,
    )


def _worker_args(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
) -> Namespace:
    return Namespace(
        base_checkpoint=args.base_checkpoint,
        base_manifest=args.base_manifest,
        initial_residual_checkpoint=args.initial_residual_checkpoint,
        flat_replay=args.flat_replay,
        output_dir=str(output),
        forward_speed=args.forward_speed,
        adapting_residual_gain=args.adapting_residual_gain,
        post_step_residual_gain=args.post_step_residual_gain,
        residual_rate_limit=getattr(args, "residual_rate_limit", 0.03),
        post_step_stability_weight=1.0,
        post_step_progress_weight=2.0,
        stall_penalty_weight=2.0,
        completion_bonus=5.0,
        pre_step_clearance_weight=2.0,
        pre_step_clearance_target=0.045,
        pre_step_approach_distance=0.50,
        max_critic_grad_norm=3_000.0,
        critic_huber_delta=args.critic_huber_delta,
        max_policy_residual_mse=args.max_policy_residual_mse,
        target_q_min=args.target_q_min,
        target_q_max=args.target_q_max,
        deterministic_actor_weight=args.deterministic_actor_weight,
        initial_alpha=args.initial_alpha,
        alpha_max=args.alpha_max,
        rough_swing_lift_weight=args.rough_swing_lift_weight,
        rough_swing_lift_target=args.rough_swing_lift_target,
        actor_updates_per_second=args.actor_updates_per_second,
        critic_updates_per_transition=getattr(
            args, "critic_updates_per_transition", 0.5
        ),
        actor_updates_per_transition=getattr(
            args, "actor_updates_per_transition", 0.005
        ),
        adaptation_min_replay_size=128,
        stalled_actor_replay_stride=16,
        control_timeout_ms=args.control_timeout_ms,
        device=f"cuda:{args.physical_gpu}",
        headless=args.headless,
        video=args.video,
        target_semantics=configuration.target_semantics,
        n_step=configuration.n_step,
        critic_replay_mode=configuration.critic_replay_mode,
        flat_fraction_initial=configuration.flat_fraction_initial,
        flat_fraction_ready=configuration.flat_fraction_ready,
        flat_anchor_weight=configuration.flat_anchor_weight,
        terrain_feature_source=configuration.terrain_feature_source,
        terrain_min_valid_fraction=configuration.terrain_min_valid_fraction,
        flat_critic_warmup_updates=configuration.flat_critic_warmup_updates,
        max_episodes=configuration.max_episodes,
        max_total_steps=configuration.max_total_steps,
        max_stage_episodes=configuration.max_stage_episodes,
    )


def _append_runtime_options(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        (
            "--task",
            ROCKY_ROAD_TASK,
            "--terrain-split",
            "development",
            "--terrain-rough-amplitude",
            str(args.terrain_rough_amplitude),
            "--residual-frame-dim",
            "49",
            "--kit_args",
            MULTI_GPU_OFF,
        )
    )


def _replace_option(command: list[str], option: str, value: object) -> None:
    command[command.index(option) + 1] = str(value)


def build_training_command(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
    episode: int,
) -> list[str]:
    request = EpisodeRequest(
        stage=args.terrain_stage,
        seed=args.terrain_seed,
        episode=episode,
        gain=args.residual_gain,
        max_steps=args.max_steps_per_episode,
    )
    command = build_worker_command(
        _worker_args(args, configuration, output=output), request
    )
    _replace_option(command, "--seed", args.simulation_seed)
    _append_runtime_options(command, args)
    return command


def _protocol_sha256(
    args: argparse.Namespace, configuration: TargetConfiguration
) -> str:
    return protocol_for_configuration(
        configuration, target_stage=args.terrain_stage
    ).sha256()


def build_flat_probe_command(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
    mode: str,
    seed: int,
    run_dir: Path,
) -> list[str]:
    checkpoint = output / "experiment.pt"
    command = build_retention_probe_command(
        _worker_args(args, configuration, output=output),
        mode=mode,
        seed=seed,
        run_dir=run_dir,
        learned_checkpoint=checkpoint,
    )
    _replace_option(command, "--protocol-sha256", _protocol_sha256(args, configuration))
    _replace_option(
        command,
        "--residual-gain",
        0.0 if mode == "base" else args.post_step_residual_gain,
    )
    command.extend(("--terrain-stage", args.terrain_stage, "--task", FLAT_TASK))
    command.extend(("--kit_args", MULTI_GPU_OFF))
    return command


def build_sealed_evaluation_command(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
    terrain_seed: int,
    run_dir: Path,
    checkpoint: Path | None = None,
) -> list[str]:
    command = build_flat_probe_command(
        args,
        configuration,
        output=output,
        mode="learned_frozen",
        seed=terrain_seed,
        run_dir=run_dir,
    )
    command.remove("--flat-retention-probe")
    if checkpoint is not None:
        _replace_option(command, "--experiment-checkpoint-in", checkpoint)
    _replace_option(command, "--task", ROCKY_ROAD_TASK)
    _replace_option(command, "--seed", args.simulation_seed)
    _replace_option(command, "--max-steps", args.max_steps_per_episode)
    if "--video-length" in command:
        _replace_option(command, "--video-length", args.max_steps_per_episode)
    command.extend(
        (
            "--terrain-split",
            "sealed",
            "--terrain-rough-amplitude",
            str(args.terrain_rough_amplitude),
            "--evaluation-only",
            "--require-track-completion",
        )
    )
    return command


def validation_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    return tuple(seed for seed in default_flat_online_protocol().development_seeds if seed != args.terrain_seed)


def build_development_validation_command(args, configuration, *, output, seed, run_dir, checkpoint=None):
    if seed not in validation_seeds(args):
        raise ValueError("validation must use a development seed other than the training seed")
    command = build_flat_probe_command(
        args, configuration, output=output, mode="base" if checkpoint is None else "learned_frozen",
        seed=seed, run_dir=run_dir,
    )
    command.remove("--flat-retention-probe")
    _replace_option(command, "--task", ROCKY_ROAD_TASK)
    _replace_option(command, "--seed", args.simulation_seed)
    _replace_option(command, "--max-steps", args.max_steps_per_episode)
    if "--video-length" in command:
        _replace_option(command, "--video-length", args.max_steps_per_episode)
    if checkpoint is not None:
        _replace_option(command, "--experiment-checkpoint-in", checkpoint)
    command.extend(("--terrain-split", "development", "--terrain-rough-amplitude",
                    str(args.terrain_rough_amplitude), "--evaluation-only", "--require-track-completion"))
    return command


def build_development_control_command(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
    mode: str,
    run_dir: Path,
) -> list[str]:
    if mode not in {"base", "initial_residual"}:
        raise ValueError("development control mode must be base or initial_residual")
    worker_mode = "base" if mode == "base" else "learned_frozen"
    command = build_flat_probe_command(
        args,
        configuration,
        output=output,
        mode=worker_mode,
        seed=args.terrain_seed,
        run_dir=run_dir,
    )
    command.remove("--flat-retention-probe")
    _replace_option(command, "--task", ROCKY_ROAD_TASK)
    _replace_option(command, "--seed", args.simulation_seed)
    _replace_option(command, "--max-steps", args.max_steps_per_episode)
    _replace_option(
        command,
        "--residual-gain",
        0.0 if mode == "base" else args.residual_gain,
    )
    if "--video-length" in command:
        _replace_option(command, "--video-length", args.max_steps_per_episode)
    if mode == "initial_residual":
        checkpoint_option = command.index("--experiment-checkpoint-in")
        command[checkpoint_option] = "--residual-checkpoint"
        command[checkpoint_option + 1] = args.initial_residual_checkpoint
    command.extend(
        (
            "--terrain-split",
            "development",
            "--terrain-rough-amplitude",
            str(args.terrain_rough_amplitude),
            "--evaluation-only",
            "--require-track-completion",
        )
    )
    return command


def _initial_state() -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "running",
        "episodes_completed": 0,
        "interaction_steps": 0,
        "first_success_episode": None,
        "stable_success_episode": None,
        "base_control_passed": None,
        "initial_residual_control_passed": None,
        "successes": 0,
        "cumulative_success_rate": 0.0,
        "recent_successes": [],
        "recent_falls": [],
        "recent_success_rate": 0.0,
        "consecutive_successes": 0,
        "max_consecutive_successes": 0,
        "falls": 0,
        "safety_violations": 0,
        "rollbacks": 0,
        "performance_rollbacks": 0,
        "safety_rollbacks": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "latest_actor_version": 0,
        "latest_checkpoint_sha256": None,
        "max_policy_residual_mse": None,
        "last_policy_parameter_l2": None,
        "last_flat_probe_episode": 0,
        "latest_flat_retention_passed": None,
        "flat_probes": [],
        "flat_retention_failures": 0,
        "sealed_evaluation_complete": False,
        "last_development_probe_episode": 0,
        "latest_development_validation": None,
        "best_checkpoint": None,
    }


def _require_exact_int(mapping: Mapping[str, object], name: str) -> int:
    value = mapping.get(name)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"episode summary has invalid {name}")
    return value


def _require_bool(mapping: Mapping[str, object], name: str) -> bool:
    value = mapping.get(name)
    if type(value) is not bool:
        raise RuntimeError(f"episode summary has invalid {name}")
    return value


def _successful_completion(summary: Mapping[str, object]) -> bool:
    return (
        _require_bool(summary, "completion_reached")
        and _require_bool(summary, "passed")
        and not _require_bool(summary, "fell")
        and not _require_bool(summary, "safety_violation")
    )


def summarize_actor_drift(path: str | Path) -> dict[str, float | int | None]:
    records: list[dict[str, object]] = []
    actor_path = Path(path)
    if actor_path.is_file():
        for line_number, line in enumerate(
            actor_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"actor update line {line_number} is not a JSON object"
                )
            if value.get("accepted") is True:
                records.append(value)
    result: dict[str, float | int | None] = {
        "accepted_actor_update_records": len(records)
    }
    for name in DRIFT_FIELDS:
        values = []
        for record in records:
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"accepted actor update lacks numeric {name}")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(f"accepted actor update has non-finite {name}")
            values.append(numeric)
        result[f"max_{name}"] = max(values) if values else None
        result[f"last_{name}"] = values[-1] if values else None
    return result


def update_learning_state(
    state: Mapping[str, object],
    *,
    episode: int,
    summary: Mapping[str, object],
    drift: Mapping[str, object],
    rolling_window: int,
) -> dict[str, Any]:
    expected_episode = int(state["episodes_completed"]) + 1
    if episode != expected_episode:
        raise ValueError(
            f"episode sequence is not contiguous: expected {expected_episode}, got {episode}"
        )
    successful = _successful_completion(summary)
    recent = [bool(value) for value in state["recent_successes"]]
    recent.append(successful)
    recent = recent[-rolling_window:]
    recent_falls = [bool(value) for value in state["recent_falls"]]
    recent_falls.append(_require_bool(summary, "fell"))
    recent_falls = recent_falls[-rolling_window:]
    successes = int(state["successes"]) + int(successful)
    consecutive = int(state["consecutive_successes"]) + 1 if successful else 0
    first_success = state["first_success_episode"]
    if successful and first_success is None:
        first_success = episode
    episode_drift = drift.get("max_policy_residual_mse")
    maximum_drift = state.get("max_policy_residual_mse")
    if episode_drift is not None:
        episode_drift = float(episode_drift)
        maximum_drift = (
            episode_drift
            if maximum_drift is None
            else max(float(maximum_drift), episode_drift)
        )
    updated = dict(state)
    updated.update(
        {
            "episodes_completed": episode,
            "interaction_steps": int(state["interaction_steps"])
            + _require_exact_int(summary, "interaction_steps"),
            "first_success_episode": first_success,
            "successes": successes,
            "cumulative_success_rate": successes / episode,
            "recent_successes": recent,
            "recent_falls": recent_falls,
            "recent_success_rate": sum(recent) / len(recent),
            "consecutive_successes": consecutive,
            "max_consecutive_successes": max(
                int(state["max_consecutive_successes"]), consecutive
            ),
            "falls": int(state["falls"]) + int(_require_bool(summary, "fell")),
            "safety_violations": int(state["safety_violations"])
            + int(_require_bool(summary, "safety_violation")),
            "rollbacks": int(state["rollbacks"])
            + _require_exact_int(summary, "rollbacks"),
            "performance_rollbacks": int(state["performance_rollbacks"])
            + _require_exact_int(summary, "performance_rollbacks"),
            "safety_rollbacks": int(state["safety_rollbacks"])
            + _require_exact_int(summary, "safety_rollbacks"),
            "actor_updates": int(state["actor_updates"])
            + _require_exact_int(summary, "actor_updates"),
            "critic_updates": int(state["critic_updates"])
            + _require_exact_int(summary, "critic_updates"),
            "latest_actor_version": _require_exact_int(summary, "actor_version"),
            "latest_checkpoint_sha256": summary.get(
                "experiment_checkpoint_out_sha256"
            ),
            "max_policy_residual_mse": maximum_drift,
            "last_policy_parameter_l2": drift.get("last_policy_parameter_l2"),
        }
    )
    return updated


def stable_success_reached(
    state: Mapping[str, object],
    *,
    window: int,
    min_success_rate: float,
    min_consecutive_successes: int,
) -> bool:
    recent = state.get("recent_successes")
    return (
        isinstance(recent, list)
        and len(recent) == window
        and isinstance(state.get("recent_falls"), list)
        and len(state["recent_falls"]) == window
        and not any(state["recent_falls"])
        and float(state.get("recent_success_rate", 0.0)) >= min_success_rate
        and int(state.get("consecutive_successes", 0)) >= min_consecutive_successes
        and int(state.get("actor_updates", 0)) > 0
        and state.get("base_control_passed") is False
        and state.get("initial_residual_control_passed") is False
        and state.get("latest_flat_retention_passed") is True
        and state.get("last_flat_probe_episode") == state.get("episodes_completed")
        and state.get("last_development_probe_episode") == state.get("episodes_completed")
        and isinstance(state.get("latest_development_validation"), dict)
        and state["latest_development_validation"].get("passed") is True
    )


def _copy_novelty_baseline(source: Path, destination: Path) -> None:
    expected = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != expected:
            raise ValueError("saved novelty baseline differs from the requested source")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(args: argparse.Namespace, configuration: TargetConfiguration) -> dict[str, Any]:
    protocol = protocol_for_configuration(configuration, target_stage=args.terrain_stage)
    inputs = {
        name: str(Path(getattr(args, name)).resolve())
        for name in (
            "base_checkpoint",
            "base_manifest",
            "initial_residual_checkpoint",
            "flat_replay",
            "novelty_baseline",
        )
    }
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "experiment": "fixed_development_rocky_road_multiepisode_v1",
        "task": ROCKY_ROAD_TASK,
        "terrain_split": "development",
        "terrain_stage": args.terrain_stage,
        "terrain_seed": args.terrain_seed,
        "simulation_seed": args.simulation_seed,
        "terrain_rough_amplitude": args.terrain_rough_amplitude,
        "terrain_width_m": 4.0,
        "flat_start_length_m": 3.0,
        "transition_length_m": 0.5,
        "inputs": inputs,
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "input_lineage": _validate_input_lineage(args),
        "frozen_benchmark": _frozen_benchmark_provenance(args),
        "code": _git_provenance(),
        "target_configuration": asdict(configuration),
        "forward_speed": args.forward_speed,
        "residual_gain": args.residual_gain,
        "adapting_residual_gain": args.adapting_residual_gain,
        "post_step_residual_gain": args.post_step_residual_gain,
        "residual_rate_limit": getattr(args, "residual_rate_limit", 0.03),
        "critic_action_semantics": "requested_residual_before_safety_v1",
        "replay_schema_version": 2,
        "completion_qualification_version": protocol.completion_qualification_version,
        "lateral_margin_m": protocol.lateral_margin_m,
        "required_swing_lift_m": protocol.required_swing_lift_m,
        "min_challenge_observation_steps": protocol.min_challenge_observation_steps,
        "min_challenge_swing_steps": protocol.min_challenge_swing_steps,
        "min_stable_completion_steps": protocol.min_stable_completion_steps,
        "max_completion_velocity_error": protocol.max_completion_velocity_error,
        "max_completion_tilt_rms": protocol.max_completion_tilt_rms,
        "max_completion_foot_slip": protocol.max_completion_foot_slip,
        "actor_updates_per_second": args.actor_updates_per_second,
        "critic_updates_per_transition": getattr(
            args, "critic_updates_per_transition", 0.5
        ),
        "actor_updates_per_transition": getattr(
            args, "actor_updates_per_transition", 0.005
        ),
        "control_timeout_ms": args.control_timeout_ms,
        "target_q_min": args.target_q_min,
        "target_q_max": args.target_q_max,
        "deterministic_actor_weight": args.deterministic_actor_weight,
        "initial_alpha": args.initial_alpha,
        "alpha_max": args.alpha_max,
        "critic_huber_delta": args.critic_huber_delta,
        "rough_swing_lift_weight": args.rough_swing_lift_weight,
        "rough_swing_lift_target": args.rough_swing_lift_target,
        "max_policy_residual_mse": args.max_policy_residual_mse,
        "policy_drift_interpretation": (
            "local_to_each_novelty_adaptation_episode_not_clean_initializer"
        ),
        "max_steps_per_episode": args.max_steps_per_episode,
        "flat_probe_interval": args.flat_probe_interval,
        "flat_retention_seeds": list(FLAT_RETENTION_SEEDS),
        "development_validation_seeds": list(validation_seeds(args)),
        "checkpoint_selection": "frozen_development_success_falls_remaining_v1",
        "stability_window": args.stability_window,
        "stability_min_success_rate": args.stability_min_success_rate,
        "stability_min_consecutive_successes": args.stability_min_consecutive_successes,
        "video": args.video,
        "physical_gpu": args.physical_gpu,
        "logical_device": f"cuda:{args.physical_gpu}",
        "visible_physical_gpus": list(range(args.physical_gpu + 1)),
        "sealed_evaluation": args.sealed_evaluation,
        "sealed_seeds": list(default_flat_online_protocol().sealed_seeds),
        "evaluation_interpretation": "reused_heldout_benchmark_not_blind_confirmatory",
        "protocol_sha256": _protocol_sha256(args, configuration),
        "protocol_note": (
            "terrain amplitude is bound by this manifest because the legacy protocol "
            "hash names rough_4cm but does not encode CLI amplitude overrides"
        ),
    }


def _validate_input_lineage(args: argparse.Namespace) -> dict[str, Any]:
    base_checkpoint = Path(args.base_checkpoint).resolve()
    base_manifest_path = Path(args.base_manifest).resolve()
    base_manifest = _load_json_object(base_manifest_path)
    files = base_manifest.get("files")
    if not isinstance(files, dict) or str(base_checkpoint) not in files:
        raise ValueError("base manifest does not bind the requested PPO checkpoint")
    base_record = files[str(base_checkpoint)]
    if not isinstance(base_record, dict):
        raise ValueError("base checkpoint manifest record is invalid")
    if (
        base_record.get("sha256") != sha256_file(base_checkpoint)
        or base_record.get("bytes") != base_checkpoint.stat().st_size
    ):
        raise ValueError("base checkpoint differs from its manifest")

    flat_replay = Path(args.flat_replay).resolve()
    flat_manifest_path = flat_replay.with_name("flat_replay_manifest.json")
    if not flat_manifest_path.is_file():
        raise FileNotFoundError(flat_manifest_path)
    flat_manifest = _load_json_object(flat_manifest_path)
    if flat_manifest.get("frozen") is not True:
        raise ValueError("flat replay manifest does not mark the replay frozen")
    if flat_manifest.get("transition_count") != 100_000:
        raise ValueError("flat replay does not contain the formal 100000 transitions")
    if flat_manifest.get("replay_sha256") != sha256_file(flat_replay):
        raise ValueError("flat replay differs from its manifest")
    if flat_manifest.get("baseline_manifest_sha256") != sha256_file(
        base_manifest_path
    ):
        raise ValueError("flat replay lineage differs from the base manifest")

    initializer = Path(args.initial_residual_checkpoint).resolve()
    initializer_manifest_path = initializer.with_name("initializer.json")
    if not initializer_manifest_path.is_file():
        raise FileNotFoundError(initializer_manifest_path)
    initializer_manifest = _load_json_object(initializer_manifest_path)
    if initializer_manifest.get("updater_sha256") != sha256_file(initializer):
        raise ValueError("residual initializer differs from its manifest")
    if initializer_manifest.get("flat_replay_sha256") != sha256_file(flat_replay):
        raise ValueError("residual initializer lineage differs from flat replay")
    if (
        initializer_manifest.get("critic_action_semantics") != "requested_residual_before_safety_v1"
        or type(initializer_manifest.get("updater_schema_version")) is not int
        or initializer_manifest["updater_schema_version"] != 1
        or type(initializer_manifest.get("replay_schema_version")) is not int
        or initializer_manifest["replay_schema_version"] != 2
    ):
        raise ValueError("residual initializer has incompatible action semantics/schema; regenerate it")
    if (
        initializer_manifest.get("frame_dim") != 49
        or initializer_manifest.get("actor_source") != "fresh_zero_mean_actor"
    ):
        raise ValueError("residual initializer is not the strict clean 49D artifact")
    temperature = initializer_manifest.get("sac_config", {})
    if any(temperature.get(name) != getattr(args, name) for name in ("initial_alpha", "alpha_max")):
        raise ValueError("initializer temperature configuration differs; regenerate the clean initializer")
    return {
        "base_manifest_schema_version": base_manifest.get("schema_version"),
        "flat_replay_manifest": str(flat_manifest_path),
        "flat_replay_manifest_sha256": sha256_file(flat_manifest_path),
        "initializer_manifest": str(initializer_manifest_path),
        "initializer_manifest_sha256": sha256_file(initializer_manifest_path),
    }


def _frozen_benchmark_run_dir(args: argparse.Namespace, terrain_seed: int) -> Path:
    return Path(args.frozen_benchmark_dir) / (
        f"base_amp4cm_sealed_seed{terrain_seed}_v2"
    )


def _load_frozen_benchmark(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for terrain_seed in default_flat_online_protocol().sealed_seeds:
        run_dir = _frozen_benchmark_run_dir(args, terrain_seed)
        summary = _require_rollout_outputs(run_dir, video=True)
        config = _load_json_object(run_dir / "config.json")
        arguments = config.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"frozen benchmark config is invalid: {run_dir}")
        expected = {
            "seed": args.simulation_seed,
            "terrain_seed": terrain_seed,
            "terrain_rough_amplitude": args.terrain_rough_amplitude,
            "task": ROCKY_ROAD_TASK,
            "mode": "base",
        }
        if any(arguments.get(name) != value for name, value in expected.items()):
            raise ValueError(f"frozen benchmark configuration differs: {run_dir}")
        if summary.get("terrain_seed") != terrain_seed or type(
            summary.get("passed")
        ) is not bool:
            raise ValueError(f"frozen benchmark summary is invalid: {run_dir}")
        records[terrain_seed] = summary
    return records


def _frozen_benchmark_provenance(args: argparse.Namespace) -> dict[str, Any]:
    records = _load_frozen_benchmark(args)
    return {
        "path": str(Path(args.frozen_benchmark_dir).resolve()),
        "interpretation": "previously_observed_paired_frozen_control",
        "summary_sha256": {
            str(seed): sha256_file(
                _frozen_benchmark_run_dir(args, seed) / "episode_summary.json"
            )
            for seed in records
        },
        "video_sha256": {
            str(seed): sha256_file(
                _frozen_benchmark_run_dir(args, seed)
                / "videos"
                / "rl-video-step-0.mp4"
            )
            for seed in records
        },
    }


def _git_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode("ascii").strip()
    tracked_diff = subprocess.run(
        ("git", "diff", "--binary", "--", "."),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return {
        "git_commit": revision,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_worktree_clean": not tracked_diff,
    }


def _prepare_output(
    args: argparse.Namespace, configuration: TargetConfiguration
) -> Path:
    for name in (
        "base_checkpoint",
        "base_manifest",
        "initial_residual_checkpoint",
        "flat_replay",
        "novelty_baseline",
    ):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError(path)
    if not Path(args.frozen_benchmark_dir).is_dir():
        raise FileNotFoundError(args.frozen_benchmark_dir)
    protocol = default_flat_online_protocol()
    if args.terrain_seed not in protocol.development_seeds:
        raise ValueError("--terrain-seed must be a development seed")
    if args.stability_min_consecutive_successes > args.stability_window:
        raise ValueError("stable consecutive-success requirement exceeds the window")
    if args.target_q_min is not None and args.target_q_max is not None and args.target_q_min >= args.target_q_max:
        raise ValueError("--target-q-min must be less than --target-q-max")
    if args.initial_alpha > args.alpha_max:
        raise ValueError("initial-alpha must not exceed alpha-max")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested = _manifest(args, configuration)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if _load_json_object(manifest_path) != requested:
            raise ValueError(
                "resume configuration differs from run manifest; preserve the existing "
                "run and use a new output directory after code or control changes"
            )
    else:
        write_json_atomic(manifest_path, requested)
    _copy_novelty_baseline(Path(args.novelty_baseline), output / "novelty_baseline.json")
    return output


def _load_state(output: Path) -> dict[str, Any]:
    path = output / "learning_state.json"
    if not path.exists():
        return _initial_state()
    state = _load_json_object(path)
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("learning state schema version mismatch")
    if type(state.get("episodes_completed")) is not int or state["episodes_completed"] < 0:
        raise ValueError("learning state has invalid episode count")
    # Older supervisors treated one failed flat-retention probe as a terminal
    # run status. That status is no longer terminal: retention is a checkpoint
    # selection gate, while the bounded RockyRoad loop continues learning.
    # Migrate it on resume so an interrupted legacy run can reach its episode
    # cap and produce a later eligible checkpoint.
    if state.get("status") == "flat_retention_failed":
        state["status"] = "running"
        if type(state.get("flat_retention_failures")) is not int:
            probes = state.get("flat_probes", [])
            state["flat_retention_failures"] = sum(
                isinstance(probe, dict) and probe.get("passed") is False
                for probe in probes
            )
    if type(state.get("flat_retention_failures", 0)) is not int or state.get("flat_retention_failures", 0) < 0:
        raise ValueError("learning state has invalid flat-retention failure count")
    return state


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    if path.is_file():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matching = [item for item in existing if item.get("episode") == record.get("episode")]
        if matching:
            if matching[-1] != dict(record):
                raise RuntimeError("existing episode record differs from recovered result")
            return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def worker_environment(physical_gpu: int) -> dict[str, str]:
    if physical_gpu not in {0, 1}:
        raise ValueError("physical GPU must be 0 or 1")
    environment = dict(os.environ)
    # AppLauncher forwards the numeric CUDA ordinal to Vulkan as activeGpu.
    # Keep lower ordinals visible so physical GPU 1 remains ordinal 1, while
    # always hiding the user-reserved physical GPUs 2 and 3.
    visible = ",".join(str(index) for index in range(physical_gpu + 1))
    environment["CUDA_VISIBLE_DEVICES"] = visible
    environment["NVIDIA_VISIBLE_DEVICES"] = visible
    environment.pop("DISPLAY", None)
    return environment


@contextmanager
def gpu_run_lock(physical_gpu: int):
    path = Path("/tmp") / f"lite3-rocky-multiepisode-gpu-{physical_gpu}.lock"
    stream = path.open("w", encoding="ascii")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another RockyRoad supervisor holds physical GPU {physical_gpu}"
            ) from error
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _run_logged(command: list[str], run_dir: Path, *, physical_gpu: int) -> int:
    with (run_dir / "runtime.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            check=False,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=worker_environment(physical_gpu),
        )
    return result.returncode


def _archive_incomplete(run_dir: Path) -> None:
    archive_root = run_dir.parent / "incomplete"
    archive_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        destination = archive_root / f"{run_dir.name}-attempt-{attempt:02d}"
        if not destination.exists():
            run_dir.rename(destination)
            return
        attempt += 1


def _require_rollout_outputs(run_dir: Path, *, video: bool) -> dict[str, Any]:
    summary_path = run_dir / "episode_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"worker produced no episode summary: {run_dir}")
    video_path = run_dir / "videos" / "rl-video-step-0.mp4"
    if video and (not video_path.is_file() or video_path.stat().st_size <= 0):
        raise RuntimeError(f"worker produced no nonempty video: {run_dir}")
    return _load_json_object(summary_path)


def _mean(summaries: list[Mapping[str, object]], name: str) -> float:
    values = [summary.get(name) for summary in summaries]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise RuntimeError(f"rollout summaries lack numeric {name}")
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"rollout summaries lack finite {name}")
    return sum(float(value) for value in values) / len(values)


def _summarize_rollouts(summaries: list[Mapping[str, object]]) -> dict[str, Any]:
    return {
        "count": len(summaries),
        "successes": sum(summary.get("passed") is True for summary in summaries),
        "falls": sum(summary.get("fell") is True for summary in summaries),
        "mean_velocity_error": _mean(summaries, "second_half_velocity_error"),
        "mean_tilt_rms": _mean(summaries, "second_half_tilt_rms"),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= positive total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def _run_flat_mode(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
    root: Path,
    mode: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    digest = None if mode == "base" else sha256_file(output / "experiment.pt")
    for seed in FLAT_RETENTION_SEEDS:
        run_dir = root / f"seed-{seed}"
        summary_path = run_dir / "episode_summary.json"
        if summary_path.is_file():
            try:
                cached = _require_rollout_outputs(run_dir, video=args.video)
                _validate_frozen_rollout(cached, _load_json_object(run_dir / "config.json"),
                                        seed=seed, checkpoint_sha256=digest, simulation_seed=seed,
                                        evaluation_only=False)
                summaries.append(cached)
                continue
            except (RuntimeError, ValueError, OSError):
                _archive_incomplete(run_dir)
        if run_dir.exists():
            _archive_incomplete(run_dir)
        run_dir.mkdir(parents=True)
        command = build_flat_probe_command(
            args,
            configuration,
            output=output,
            mode=mode,
            seed=seed,
            run_dir=run_dir,
        )
        returncode = _run_logged(command, run_dir, physical_gpu=args.physical_gpu)
        if returncode not in {0, 1}:
            raise RuntimeError(f"flat {mode} probe failed for seed {seed}")
        summary = _require_rollout_outputs(run_dir, video=args.video)
        _validate_frozen_rollout(summary, _load_json_object(run_dir / "config.json"),
                                seed=seed, checkpoint_sha256=digest, simulation_seed=seed,
                                evaluation_only=False)
        summaries.append(summary)
    return summaries


def _ensure_flat_baseline(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
) -> list[dict[str, Any]]:
    root = output / "flat_retention" / "base"
    summaries = _run_flat_mode(
        args, configuration, output=output, root=root, mode="base"
    )
    summary_path = root / "summary.json"
    aggregate = _summarize_rollouts(summaries)
    if summary_path.exists():
        if _load_json_object(summary_path) != aggregate:
            raise RuntimeError("saved flat baseline summary is inconsistent")
    else:
        write_json_atomic(summary_path, aggregate)
    return summaries


def _run_flat_probe(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
    episode: int,
    base_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    root = output / "flat_retention" / f"episode-{episode:04d}"
    aggregate_path = root / "summary.json"
    if aggregate_path.is_file():
        cached = _load_json_object(aggregate_path)
        if cached.get("checkpoint_sha256") != sha256_file(output / "experiment.pt"):
            raise RuntimeError("flat retention result belongs to a different checkpoint")
        return cached
    learned = _run_flat_mode(
        args, configuration, output=output, root=root, mode="learned_frozen"
    )
    report = {
        "episode": episode,
        "checkpoint_sha256": sha256_file(output / "experiment.pt"),
        "passed": flat_retention_passes(base_summaries, learned),
        "base": _summarize_rollouts(base_summaries),
        "learned_frozen": _summarize_rollouts(learned),
    }
    write_json_atomic(aggregate_path, report)
    return report


def _validate_frozen_rollout(summary, config, *, seed, checkpoint_sha256, simulation_seed, evaluation_only=True):
    expected_mode = "base" if checkpoint_sha256 is None else "residual_frozen"
    arguments = config.get("arguments", {})
    if (
        summary.get("terrain_seed") != seed
        or summary.get("mode") != expected_mode
        or summary.get("actor_updates") != 0
        or summary.get("critic_updates") != 0
        or summary.get("parameter_changed") is not False
        or arguments.get("stochastic_control") is not False
        or arguments.get("evaluation_only") is not evaluation_only
        or arguments.get("seed") != simulation_seed
        or (checkpoint_sha256 is not None and summary.get("experiment_checkpoint_in_sha256") != checkpoint_sha256)
    ):
        raise RuntimeError("validation is not a matching deterministic frozen rollout")
    for name in ("passed", "fell", "safety_violation"):
        if type(summary.get(name)) is not bool:
            raise RuntimeError(f"validation lacks boolean {name}")
    if summary["passed"] and (summary["fell"] or summary["safety_violation"]):
        raise RuntimeError("validation cannot pass after a fall/safety violation")


def _validation_rank(summaries):
    aggregate = _summarize_rollouts(summaries)
    return [aggregate["successes"], -aggregate["falls"],
            -_mean(summaries, "completion_remaining_m"),
            -aggregate["mean_velocity_error"], -aggregate["mean_tilt_rms"]]


def select_best_checkpoint(previous, candidate):
    # Never replace a tie with a later checkpoint. A failed flat probe cannot
    # win, even when its rough-terrain success count is higher.
    if not candidate["flat_retention_passed"]:
        return previous
    if previous is None or candidate["rank"] > previous["rank"]:
        return candidate
    return previous


def _run_development_validation(args, configuration, *, output, episode, flat_passed):
    checkpoint = output / "episodes" / f"episode-{episode:04d}" / "experiment.pt"
    digest = sha256_file(checkpoint)
    if checkpoint.with_suffix(".pt.sha256").read_text(encoding="ascii").strip() != digest:
        raise RuntimeError("development checkpoint checksum mismatch")
    groups = {}
    for mode in ("base", "learned_frozen"):
        root = output / "development_validation" / ("base" if mode == "base" else f"episode-{episode:04d}")
        source = None if mode == "base" else checkpoint
        source_digest = None if mode == "base" else digest
        summaries = []
        for seed in validation_seeds(args):
            run_dir = root / f"seed-{seed}"
            summary = None
            if (run_dir / "episode_summary.json").is_file():
                try:
                    summary = _require_rollout_outputs(run_dir, video=args.video)
                    _validate_frozen_rollout(summary, _load_json_object(run_dir / "config.json"),
                                            seed=seed, checkpoint_sha256=source_digest,
                                            simulation_seed=args.simulation_seed)
                except (RuntimeError, ValueError, OSError):
                    summary = None
            if summary is None:
                if run_dir.exists():
                    _archive_incomplete(run_dir)
                run_dir.mkdir(parents=True)
                command = build_development_validation_command(
                    args, configuration, output=output, seed=seed, run_dir=run_dir, checkpoint=source
                )
                returncode = _run_logged(command, run_dir, physical_gpu=args.physical_gpu)
                if returncode not in {0, 1}:
                    raise RuntimeError(f"development validation worker failed for seed {seed}")
                summary = _require_rollout_outputs(run_dir, video=args.video)
                _validate_frozen_rollout(summary, _load_json_object(run_dir / "config.json"),
                                        seed=seed, checkpoint_sha256=source_digest,
                                        simulation_seed=args.simulation_seed)
            summaries.append(summary)
        groups[mode] = summaries
    base = _summarize_rollouts(groups["base"])
    learned = _summarize_rollouts(groups["learned_frozen"])
    report = {
        "episode": episode, "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": digest,
        "seeds": list(validation_seeds(args)), "learning_enabled": False,
        "flat_retention_passed": flat_passed,
        "rank": _validation_rank(groups["learned_frozen"]),
        "base": base, "learned_frozen": learned,
        "passed": (flat_passed and learned["falls"] == 0
                   and learned["successes"] / learned["count"] >= args.stability_min_success_rate
                   and learned["successes"] > base["successes"]),
    }
    write_json_atomic(output / "development_validation" / f"episode-{episode:04d}" / "summary.json", report)
    return report


def _record_development_validation(args, configuration, *, output, state):
    episode = int(state["episodes_completed"])
    report = _run_development_validation(
        args, configuration, output=output, episode=episode,
        flat_passed=state["latest_flat_retention_passed"] is True,
    )
    state["last_development_probe_episode"] = episode
    state["latest_development_validation"] = report
    state["best_checkpoint"] = select_best_checkpoint(state["best_checkpoint"], report)
    if state["best_checkpoint"] is not None:
        write_json_atomic(output / "best_checkpoint.json", state["best_checkpoint"])
    write_json_atomic(output / "learning_state.json", state)


def _selected_checkpoint(output):
    selected = _load_json_object(output / "best_checkpoint.json")
    checkpoint = output / "episodes" / f"episode-{int(selected['episode']):04d}" / "experiment.pt"
    digest = sha256_file(checkpoint)
    if (selected.get("flat_retention_passed") is not True
        or digest != selected.get("checkpoint_sha256")
        or checkpoint.with_suffix(".pt.sha256").read_text(encoding="ascii").strip() != digest):
        raise RuntimeError("selected checkpoint provenance/checksum mismatch")
    return checkpoint


def _complete_current_probes(args, configuration, *, output, state, base_summaries):
    episode = int(state["episodes_completed"])
    if state["last_flat_probe_episode"] != episode:
        probe = _run_flat_probe(args, configuration, output=output, episode=episode,
                                base_summaries=base_summaries)
        state["last_flat_probe_episode"] = episode
        state["latest_flat_retention_passed"] = probe["passed"]
        state["flat_probes"] = [*state["flat_probes"], probe]
        if probe["passed"] is not True:
            # A retention probe is a noisy gate, not a terminal condition for
            # the whole bounded experiment. Keep learning so later snapshots
            # can recover, while best-checkpoint selection still excludes this
            # candidate. The run ends at the episode cap or stable success.
            state["flat_retention_failures"] = int(state.get("flat_retention_failures", 0)) + 1
        write_json_atomic(output / "learning_state.json", state)
    if state["last_development_probe_episode"] != episode:
        _record_development_validation(args, configuration, output=output, state=state)


def _episode_record(
    *,
    episode: int,
    summary: Mapping[str, object],
    state: Mapping[str, object],
    drift: Mapping[str, object],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "episode": episode,
        "terrain_seed": summary.get("terrain_seed"),
        "successful_completion": _successful_completion(summary),
        "passed": summary.get("passed"),
        "fell": summary.get("fell"),
        "safety_violation": summary.get("safety_violation"),
        "interaction_steps": summary.get("interaction_steps"),
        "cumulative_interaction_steps": state["interaction_steps"],
        "cumulative_success_rate": state["cumulative_success_rate"],
        "recent_success_rate": state["recent_success_rate"],
        "consecutive_successes": state["consecutive_successes"],
        "actor_updates": summary.get("actor_updates"),
        "critic_updates": summary.get("critic_updates"),
        "actor_version": summary.get("actor_version"),
        "rollbacks": summary.get("rollbacks"),
        "performance_rollbacks": summary.get("performance_rollbacks"),
        "safety_rollbacks": summary.get("safety_rollbacks"),
        "max_x": summary.get("max_x"),
        "completion_remaining_m": summary.get("completion_remaining_m"),
        "second_half_velocity_error": summary.get("second_half_velocity_error"),
        "second_half_tilt_rms": summary.get("second_half_tilt_rms"),
        "termination_reasons": summary.get("termination_reasons", []),
        "checkpoint_sha256": summary.get("experiment_checkpoint_out_sha256"),
        "video": str(run_dir / "videos" / "rl-video-step-0.mp4"),
        "drift": dict(drift),
    }


def _write_progress_summary(
    args: argparse.Namespace, output: Path, state: Mapping[str, object]
) -> None:
    sealed_summary_path = output / "sealed_evaluation" / "summary.json"
    write_json_atomic(
        output / "summary.json",
        {
            "schema_version": 1,
            "status": state["status"],
            "terrain_seed": args.terrain_seed,
            "episodes_completed": state["episodes_completed"],
            "first_success_episode": state["first_success_episode"],
            "stable_success_episode": state["stable_success_episode"],
            "base_control_passed": state["base_control_passed"],
            "initial_residual_control_passed": state[
                "initial_residual_control_passed"
            ],
            "successes": state["successes"],
            "cumulative_success_rate": state["cumulative_success_rate"],
            "recent_success_rate": state["recent_success_rate"],
            "consecutive_successes": state["consecutive_successes"],
            "max_consecutive_successes": state["max_consecutive_successes"],
            "falls": state["falls"],
            "safety_violations": state["safety_violations"],
            "actor_updates": state["actor_updates"],
            "critic_updates": state["critic_updates"],
            "rollbacks": state["rollbacks"],
            "performance_rollbacks": state["performance_rollbacks"],
            "safety_rollbacks": state["safety_rollbacks"],
            "max_policy_residual_mse": state["max_policy_residual_mse"],
            "policy_drift_interpretation": (
                "local_to_each_novelty_adaptation_episode_not_clean_initializer"
            ),
            "last_policy_parameter_l2": state["last_policy_parameter_l2"],
            "latest_flat_retention_passed": state[
                "latest_flat_retention_passed"
            ],
            "flat_retention_failures": state.get("flat_retention_failures", 0),
            "flat_probes": state["flat_probes"],
            "latest_development_validation": state["latest_development_validation"],
            "best_checkpoint": state["best_checkpoint"],
            "stability_criterion": {
                "window": args.stability_window,
                "minimum_success_rate": args.stability_min_success_rate,
                "minimum_consecutive_successes": args.stability_min_consecutive_successes,
                "requires_current_flat_retention": True,
                "requires_frozen_development_improvement": True,
            },
            "sealed_evaluation_complete": state["sealed_evaluation_complete"],
            "sealed_evaluation": (
                _load_json_object(sealed_summary_path)
                if sealed_summary_path.is_file()
                else None
            ),
        },
    )


def _ensure_development_controls(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
) -> dict[str, Any]:
    report_path = output / "development_controls" / "summary.json"
    if report_path.is_file():
        return _load_json_object(report_path)
    controls: dict[str, dict[str, Any]] = {}
    for mode in ("base", "initial_residual"):
        run_dir = output / "development_controls" / mode
        if (run_dir / "episode_summary.json").is_file():
            try:
                summary = _require_rollout_outputs(run_dir, video=args.video)
            except RuntimeError:
                _archive_incomplete(run_dir)
                summary = {}
        else:
            summary = {}
            if run_dir.exists():
                _archive_incomplete(run_dir)
        if not summary:
            run_dir.mkdir(parents=True)
            command = build_development_control_command(
                args,
                configuration,
                output=output,
                mode=mode,
                run_dir=run_dir,
            )
            returncode = _run_logged(
                command, run_dir, physical_gpu=args.physical_gpu
            )
            if returncode not in {0, 1}:
                raise RuntimeError(f"development {mode} control failed")
            summary = _require_rollout_outputs(run_dir, video=args.video)
        controls[mode] = {
            "passed": summary.get("passed"),
            "fell": summary.get("fell"),
            "safety_violation": summary.get("safety_violation"),
            "interaction_steps": summary.get("interaction_steps"),
            "max_x": summary.get("max_x"),
            "completion_remaining_m": summary.get("completion_remaining_m"),
            "second_half_velocity_error": summary.get(
                "second_half_velocity_error"
            ),
            "second_half_tilt_rms": summary.get("second_half_tilt_rms"),
            "video": str(run_dir / "videos" / "rl-video-step-0.mp4"),
        }
    report = {
        "schema_version": 1,
        "terrain_seed": args.terrain_seed,
        "base": controls["base"],
        "initial_residual": controls["initial_residual"],
        "learning_improvement_eligible": (
            controls["base"]["passed"] is False
            and controls["initial_residual"]["passed"] is False
        ),
    }
    write_json_atomic(report_path, report)
    return report


def _run_sealed_evaluation(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    output: Path,
) -> dict[str, Any]:
    root = output / "sealed_evaluation"
    report_path = root / "summary.json"
    checkpoint = _selected_checkpoint(output)
    checkpoint_sha256 = sha256_file(checkpoint)
    if report_path.is_file():
        cached = _load_json_object(report_path)
        if cached.get("checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError("sealed evaluation belongs to a different selected checkpoint")
        return cached
    frozen = _load_frozen_benchmark(args)
    summaries: list[dict[str, Any]] = []
    for terrain_seed in default_flat_online_protocol().sealed_seeds:
        run_dir = root / f"seed-{terrain_seed}"
        if (run_dir / "episode_summary.json").is_file():
            try:
                summaries.append(_require_rollout_outputs(run_dir, video=args.video))
                continue
            except RuntimeError:
                _archive_incomplete(run_dir)
        if run_dir.exists():
            _archive_incomplete(run_dir)
        run_dir.mkdir(parents=True)
        command = build_sealed_evaluation_command(
            args,
            configuration,
            output=output,
            terrain_seed=terrain_seed,
            run_dir=run_dir,
            checkpoint=checkpoint,
        )
        returncode = _run_logged(command, run_dir, physical_gpu=args.physical_gpu)
        if returncode not in {0, 1}:
            raise RuntimeError(f"sealed evaluation failed for seed {terrain_seed}")
        summaries.append(_require_rollout_outputs(run_dir, video=args.video))
        print(
            f"sealed seed {terrain_seed}: passed={summaries[-1].get('passed')}",
            flush=True,
        )
    paired: list[dict[str, Any]] = []
    for learned in summaries:
        terrain_seed = learned.get("terrain_seed")
        if type(terrain_seed) is not int or terrain_seed not in frozen:
            raise RuntimeError("learned held-out summary has an unexpected terrain seed")
        if (
            learned.get("mode") != "residual_frozen"
            or learned.get("actor_updates") != 0
            or learned.get("critic_updates") != 0
            or learned.get("parameter_changed") is not False
            or learned.get("experiment_checkpoint_in_sha256") != checkpoint_sha256
        ):
            raise RuntimeError("held-out evaluation was not a frozen checkpoint rollout")
        base = frozen[terrain_seed]
        paired.append(
            {
                "terrain_seed": terrain_seed,
                "frozen_passed": base["passed"],
                "learned_passed": learned["passed"],
                "frozen_velocity_error": base["second_half_velocity_error"],
                "learned_velocity_error": learned["second_half_velocity_error"],
                "frozen_tilt_rms": base["second_half_tilt_rms"],
                "learned_tilt_rms": learned["second_half_tilt_rms"],
            }
        )
    frozen_successes = sum(item["frozen_passed"] is True for item in paired)
    learned_successes = sum(item["learned_passed"] is True for item in paired)
    report = {
        "schema_version": 1,
        "learning_enabled": False,
        "terrain_split": "sealed",
        "interpretation": "reused_heldout_benchmark_not_blind_confirmatory",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_selection": "frozen_development_validation",
        "seeds": list(default_flat_online_protocol().sealed_seeds),
        "per_seed": [
            {
                "terrain_seed": summary.get("terrain_seed"),
                "passed": summary.get("passed"),
                "fell": summary.get("fell"),
                "safety_violation": summary.get("safety_violation"),
                "interaction_steps": summary.get("interaction_steps"),
                "max_x": summary.get("max_x"),
                "completion_remaining_m": summary.get("completion_remaining_m"),
                "second_half_velocity_error": summary.get(
                    "second_half_velocity_error"
                ),
                "second_half_tilt_rms": summary.get("second_half_tilt_rms"),
            }
            for summary in summaries
        ],
        "paired_with_frozen": paired,
        "frozen_successes": frozen_successes,
        "frozen_success_rate": frozen_successes / len(paired),
        "frozen_success_rate_wilson95": _wilson_interval(
            frozen_successes, len(paired)
        ),
        "learned_success_rate_wilson95": _wilson_interval(
            learned_successes, len(paired)
        ),
        "rescued_seeds": [
            item["terrain_seed"]
            for item in paired
            if item["frozen_passed"] is False and item["learned_passed"] is True
        ],
        "regressed_seeds": [
            item["terrain_seed"]
            for item in paired
            if item["frozen_passed"] is True and item["learned_passed"] is False
        ],
        **_summarize_rollouts(summaries),
    }
    report["success_rate"] = report["successes"] / report["count"]
    write_json_atomic(report_path, report)
    return report


def run_experiment(args: argparse.Namespace) -> int:
    configuration = _target_configuration(args)
    output = _prepare_output(args, configuration)
    base_summaries = _ensure_flat_baseline(
        args, configuration, output=output
    )
    controls = _ensure_development_controls(
        args, configuration, output=output
    )
    state = _load_state(output)
    control_values = {
        "base_control_passed": controls["base"].get("passed"),
        "initial_residual_control_passed": controls["initial_residual"].get(
            "passed"
        ),
    }
    if any(type(value) is not bool for value in control_values.values()):
        raise RuntimeError("development controls did not emit boolean pass evidence")
    for name, value in control_values.items():
        if state[name] is not None and state[name] != value:
            raise RuntimeError("saved learning state differs from development controls")
        state[name] = value
    if (
        state["episodes_completed"] == 0
        and controls["learning_improvement_eligible"] is not True
    ):
        state["status"] = "control_already_passes"
        write_json_atomic(output / "learning_state.json", state)
        _write_progress_summary(args, output, state)
        return 0
    write_json_atomic(output / "learning_state.json", state)
    if state["episodes_completed"]:
        committed = (
            output
            / "episodes"
            / f"episode-{int(state['episodes_completed']):04d}"
        )
        if not (committed / "experiment.pt").is_file():
            raise RuntimeError("last committed episode checkpoint is unavailable")
        _install_latest_checkpoint(committed, output)
        expected = state.get("latest_checkpoint_sha256")
        actual = sha256_file(output / "experiment.pt")
        if expected is not None and expected != actual:
            raise RuntimeError("learning state checkpoint hash mismatch")

    pending_episode = int(state["episodes_completed"])
    if (
        pending_episode > 0
        and (pending_episode % args.flat_probe_interval == 0
             or pending_episode >= args.max_episodes or state["status"] != "running")
    ):
        _complete_current_probes(args, configuration, output=output, state=state,
                                 base_summaries=base_summaries)
        if state["status"] == "running" and stable_success_reached(
            state,
            window=args.stability_window,
            min_success_rate=args.stability_min_success_rate,
            min_consecutive_successes=args.stability_min_consecutive_successes,
        ):
            state["status"] = "stable_success"
            state["stable_success_episode"] = pending_episode
        write_json_atomic(output / "learning_state.json", state)
        _write_progress_summary(args, output, state)

    while (
        state["status"] == "running"
        and state["episodes_completed"] < args.max_episodes
    ):
        episode = int(state["episodes_completed"]) + 1
        run_dir = output / "episodes" / f"episode-{episode:04d}"
        checkpoint_path = run_dir / "experiment.pt"
        checkpoint_sidecar = run_dir / "experiment.pt.sha256"
        recoverable = (
            (run_dir / "episode_summary.json").is_file()
            and checkpoint_path.is_file()
            and checkpoint_sidecar.is_file()
            and (
                not args.video
                or (run_dir / "videos" / "rl-video-step-0.mp4").is_file()
            )
        )
        if recoverable:
            recovered_summary = _load_json_object(run_dir / "episode_summary.json")
            returncode = 1 if recovered_summary.get("fell") is True else 0
        else:
            if run_dir.exists():
                _archive_incomplete(run_dir)
            run_dir.mkdir(parents=True)
            command = build_training_command(
                args, configuration, output=output, episode=episode
            )
            returncode = _run_logged(
                command, run_dir, physical_gpu=args.physical_gpu
            )
        request = EpisodeRequest(
            stage=args.terrain_stage,
            seed=args.terrain_seed,
            episode=episode,
            gain=args.residual_gain,
            max_steps=args.max_steps_per_episode,
        )
        outcome = classify_worker_result(
            run_dir,
            returncode,
            request=request,
            expected_protocol_sha256=_protocol_sha256(args, configuration),
            protocol=protocol_for_configuration(
                configuration, target_stage=args.terrain_stage
            ),
        )
        summary = _require_rollout_outputs(run_dir, video=args.video)
        if outcome.passed is not summary.get("passed"):
            raise RuntimeError("classified outcome differs from worker summary")
        drift = summarize_actor_drift(run_dir / "actor_updates.jsonl")
        _install_latest_checkpoint(run_dir, output)
        state = update_learning_state(
            state,
            episode=episode,
            summary=summary,
            drift=drift,
            rolling_window=args.stability_window,
        )
        _append_jsonl(
            output / "episodes.jsonl",
            _episode_record(
                episode=episode,
                summary=summary,
                state=state,
                drift=drift,
                run_dir=run_dir,
            ),
        )
        write_json_atomic(output / "learning_state.json", state)

        if episode % args.flat_probe_interval == 0:
            _complete_current_probes(args, configuration, output=output, state=state,
                                     base_summaries=base_summaries)

        if state["status"] == "running":
            if stable_success_reached(
                state,
                window=args.stability_window,
                min_success_rate=args.stability_min_success_rate,
                min_consecutive_successes=args.stability_min_consecutive_successes,
            ):
                state["status"] = "stable_success"
                state["stable_success_episode"] = episode
            elif episode >= args.max_episodes:
                state["status"] = "episode_cap_reached"
        write_json_atomic(output / "learning_state.json", state)
        _write_progress_summary(args, output, state)
        print(
            "episode "
            f"{episode}/{args.max_episodes}: passed={summary['passed']} "
            f"cumulative={state['successes']}/{episode} "
            f"recent={state['recent_success_rate']:.3f} "
            f"actor_updates={summary['actor_updates']} "
            f"rollbacks={summary['rollbacks']}",
            flush=True,
        )

    if state["episodes_completed"]:
        _complete_current_probes(args, configuration, output=output, state=state,
                                 base_summaries=base_summaries)

    if state["status"] in {"running", "episode_cap_reached"}:
        if stable_success_reached(
            state,
            window=args.stability_window,
            min_success_rate=args.stability_min_success_rate,
            min_consecutive_successes=args.stability_min_consecutive_successes,
        ):
            state["status"] = "stable_success"
            state["stable_success_episode"] = state["episodes_completed"]
        elif state["episodes_completed"] >= args.max_episodes:
            state["status"] = "episode_cap_reached"
        write_json_atomic(output / "learning_state.json", state)
    _write_progress_summary(args, output, state)

    if (
        args.sealed_evaluation
        and state["status"] in {"stable_success", "episode_cap_reached", "flat_retention_failed"}
        and state["best_checkpoint"] is not None
        and not state["sealed_evaluation_complete"]
    ):
        _run_sealed_evaluation(args, configuration, output=output)
        state["sealed_evaluation_complete"] = True
        write_json_atomic(output / "learning_state.json", state)
        _write_progress_summary(args, output, state)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    with gpu_run_lock(args.physical_gpu):
        return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
