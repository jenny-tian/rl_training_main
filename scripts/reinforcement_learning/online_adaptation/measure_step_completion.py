"""Measure how many online-adaptation episodes a residual policy needs to clear a step.

This is an observation experiment, not a sealed acceptance run.  The episode
cap bounds compute time but does not turn an incomplete run into a claim that
the method failed an arbitrary 100-episode acceptance threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from scripts.reinforcement_learning.online_adaptation.run_flat_online_curriculum import (
    NOVELTY_BASELINE_FILENAME,
    NOVELTY_BASELINE_SEED,
    EpisodeRequest,
    TargetConfiguration,
    _install_latest_checkpoint,
    _load_json_object,
    build_novelty_baseline_probe_command,
    build_worker_command,
    classify_worker_result,
    protocol_for_configuration,
)


CONTACT_TERMINATION_PROTOCOL = "sustained_nonfoot_contact_5n_3steps_v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _successful_completion(
    *,
    completion_reached: object,
    passed: object,
    fell: object,
    safety_violation: object,
) -> bool:
    evidence = {
        "completion_reached": completion_reached,
        "passed": passed,
        "fell": fell,
        "safety_violation": safety_violation,
    }
    for name, value in evidence.items():
        if type(value) is not bool:
            raise RuntimeError(f"worker did not emit boolean {name} evidence")
    return completion_reached and passed and not fell and not safety_violation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--initial-residual-checkpoint", required=True)
    parser.add_argument("--flat-replay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--terrain-stage", choices=("step_2cm", "step_4cm", "step_6cm"), default="step_2cm")
    parser.add_argument("--max-episodes", type=_positive_int, default=500)
    parser.add_argument("--max-steps-per-episode", type=_positive_int, default=2_000)
    parser.add_argument("--forward-speed", type=_positive_float, default=0.3)
    parser.add_argument("--residual-gain", type=_unit_interval, default=0.10)
    parser.add_argument("--adapting-residual-gain", type=_unit_interval, default=0.10)
    parser.add_argument("--post-step-residual-gain", type=_unit_interval, default=0.10)
    parser.add_argument("--post-step-stability-weight", type=_finite_float, default=1.0)
    parser.add_argument("--post-step-progress-weight", type=_positive_float, default=2.0)
    parser.add_argument("--stall-penalty-weight", type=_positive_float, default=2.0)
    parser.add_argument("--completion-bonus", type=_positive_float, default=5.0)
    parser.add_argument("--pre-step-clearance-weight", type=_positive_float, default=2.0)
    parser.add_argument("--pre-step-clearance-target", type=_positive_float, default=0.045)
    parser.add_argument("--pre-step-approach-distance", type=_positive_float, default=0.50)
    parser.add_argument("--max-critic-grad-norm", type=_positive_float, default=3_000.0)
    parser.add_argument("--critic-huber-delta", type=_positive_float, default=None)
    parser.add_argument("--max-policy-residual-mse", type=_positive_float, default=None)
    parser.add_argument("--target-q-min", type=_finite_float, default=-1_000.0)
    parser.add_argument("--target-q-max", type=_finite_float, default=500.0)
    parser.add_argument("--deterministic-actor-weight", type=_unit_interval, default=0.25)
    parser.add_argument("--actor-updates-per-second", type=_positive_float, default=0.5)
    parser.add_argument("--adaptation-min-replay-size", type=_positive_int, default=128)
    parser.add_argument("--stalled-actor-replay-stride", type=_positive_int, default=16)
    parser.add_argument(
        "--control-timeout-ms",
        type=_positive_float,
        default=50.0,
        help="simulation validation deadline; deployment keeps the stricter 20 ms default",
    )
    parser.add_argument("--comparison-seed", type=int, default=101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _target_configuration(args: argparse.Namespace) -> TargetConfiguration:
    max_total_steps = args.max_episodes * args.max_steps_per_episode
    return TargetConfiguration(
        target_semantics="timeout-aware",
        n_step=3,
        critic_replay_mode="dual",
        flat_fraction_initial=0.75,
        flat_fraction_ready=0.25,
        flat_anchor_weight=0.5,
        terrain_feature_source="geometry",
        terrain_min_valid_fraction=0.90,
        flat_critic_warmup_updates=1_000,
        max_episodes=args.max_episodes,
        max_total_steps=max_total_steps,
        max_stage_episodes=args.max_episodes,
    )


def _worker_args(args: argparse.Namespace, configuration: TargetConfiguration) -> Namespace:
    return Namespace(
        base_checkpoint=args.base_checkpoint,
        base_manifest=args.base_manifest,
        flat_replay=args.flat_replay,
        output_dir=args.output_dir,
        initial_residual_checkpoint=args.initial_residual_checkpoint,
        forward_speed=args.forward_speed,
        adapting_residual_gain=args.adapting_residual_gain,
        post_step_residual_gain=args.post_step_residual_gain,
        post_step_stability_weight=args.post_step_stability_weight,
        post_step_progress_weight=args.post_step_progress_weight,
        stall_penalty_weight=args.stall_penalty_weight,
        completion_bonus=args.completion_bonus,
        pre_step_clearance_weight=args.pre_step_clearance_weight,
        pre_step_clearance_target=args.pre_step_clearance_target,
        pre_step_approach_distance=args.pre_step_approach_distance,
        max_critic_grad_norm=args.max_critic_grad_norm,
        critic_huber_delta=getattr(args, "critic_huber_delta", None),
        target_q_min=getattr(args, "target_q_min", -1_000.0),
        target_q_max=getattr(args, "target_q_max", 500.0),
        deterministic_actor_weight=getattr(args, "deterministic_actor_weight", 0.25),
        actor_updates_per_second=getattr(args, "actor_updates_per_second", 0.5),
        adaptation_min_replay_size=getattr(args, "adaptation_min_replay_size", 128),
        stalled_actor_replay_stride=getattr(args, "stalled_actor_replay_stride", 16),
        control_timeout_ms=getattr(args, "control_timeout_ms", 50.0),
        device=args.device,
        headless=args.headless,
        video=args.video,
        target_semantics=configuration.target_semantics,
        n_step=configuration.n_step,
        critic_replay_mode=configuration.critic_replay_mode,
        flat_fraction_initial=configuration.flat_fraction_initial,
        flat_fraction_ready=configuration.flat_fraction_ready,
        flat_anchor_weight=configuration.flat_anchor_weight,
        max_policy_residual_mse=args.max_policy_residual_mse,
        terrain_feature_source=configuration.terrain_feature_source,
        terrain_min_valid_fraction=configuration.terrain_min_valid_fraction,
        flat_critic_warmup_updates=configuration.flat_critic_warmup_updates,
        max_episodes=configuration.max_episodes,
        max_total_steps=configuration.max_total_steps,
        max_stage_episodes=configuration.max_stage_episodes,
    )


def _prepare_output(args: argparse.Namespace, configuration: TargetConfiguration) -> Path:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for path in (
        args.base_checkpoint,
        args.base_manifest,
        args.initial_residual_checkpoint,
        args.flat_replay,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    manifest_path = output / "run_manifest.json"
    requested = {
        "schema_version": 1,
        "measurement_mode": "first_safe_step_completion_without_acceptance_deadline_v3",
        "fall_handling": "preserve_updater_reset_episode_v1",
        "contact_termination": CONTACT_TERMINATION_PROTOCOL,
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "base_manifest": str(Path(args.base_manifest).resolve()),
        "initial_residual_checkpoint": str(Path(args.initial_residual_checkpoint).resolve()),
        "initial_residual_checkpoint_sha256": sha256_file(args.initial_residual_checkpoint),
        "flat_replay": str(Path(args.flat_replay).resolve()),
        "flat_replay_sha256": sha256_file(args.flat_replay),
        "terrain_stage": args.terrain_stage,
        "target_configuration": asdict(configuration),
        "residual_gain": args.residual_gain,
        "adapting_residual_gain": args.adapting_residual_gain,
        "post_step_residual_gain": args.post_step_residual_gain,
        "post_step_stability_weight": args.post_step_stability_weight,
        "post_step_progress_weight": args.post_step_progress_weight,
        "stall_penalty_weight": args.stall_penalty_weight,
        "completion_bonus": args.completion_bonus,
        "pre_step_clearance_weight": args.pre_step_clearance_weight,
        "pre_step_clearance_target": args.pre_step_clearance_target,
        "pre_step_approach_distance": args.pre_step_approach_distance,
        "max_critic_grad_norm": args.max_critic_grad_norm,
        "critic_huber_delta": args.critic_huber_delta,
        "max_policy_residual_mse": args.max_policy_residual_mse,
        "target_q_min": args.target_q_min,
        "target_q_max": args.target_q_max,
        "deterministic_actor_weight": args.deterministic_actor_weight,
        "actor_updates_per_second": args.actor_updates_per_second,
        "adaptation_min_replay_size": args.adaptation_min_replay_size,
        "stalled_actor_replay_stride": args.stalled_actor_replay_stride,
        "control_timeout_ms": args.control_timeout_ms,
        "forward_speed": args.forward_speed,
        "comparison_seed": args.comparison_seed,
        "video": args.video,
    }
    if manifest_path.exists():
        existing = _load_json_object(manifest_path)
        if existing != requested:
            raise ValueError("existing measurement output has different provenance or configuration")
    else:
        write_json_atomic(manifest_path, requested)
    return output


def _ensure_novelty_baseline(worker_args: Namespace, output: Path) -> None:
    baseline = output / NOVELTY_BASELINE_FILENAME
    if baseline.is_file():
        return
    baseline_dir = output / "novelty_baseline" / f"seed-{NOVELTY_BASELINE_SEED}"
    if baseline_dir.exists():
        raise RuntimeError("incomplete novelty baseline directory; choose a new output directory")
    baseline_dir.mkdir(parents=True)
    command = build_novelty_baseline_probe_command(
        worker_args, run_dir=baseline_dir, baseline_out=baseline
    )
    if worker_args.video:
        command.extend(("--video", "--video-length", "384"))
    result = subprocess.run(command, check=False)
    summary_path = baseline_dir / "episode_summary.json"
    if result.returncode != 0 or not baseline.is_file() or not summary_path.is_file():
        raise RuntimeError("novelty baseline probe did not complete")
    summary = _load_json_object(summary_path)
    if summary.get("fell") is not False or summary.get("safety_violation") is not False:
        baseline.unlink(missing_ok=True)
        raise RuntimeError("novelty baseline probe was unsafe")


def _comparison_command(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    *,
    mode: str,
    run_dir: Path,
) -> list[str]:
    if mode not in {"base", "residual_frozen"}:
        raise ValueError("comparison mode must be base or residual_frozen")
    protocol = protocol_for_configuration(configuration, target_stage=args.terrain_stage)
    command = [
        sys.executable,
        str(Path(__file__).with_name("play_online_adaptation.py")),
        "--base-checkpoint",
        args.base_checkpoint,
        "--mode",
        mode,
        "--run-dir",
        str(run_dir),
        "--terrain-stage",
        args.terrain_stage,
        "--terrain-seed",
        str(args.comparison_seed),
        "--seed",
        str(args.comparison_seed),
        "--max-steps",
        str(args.max_steps_per_episode),
        "--forward-speed",
        str(args.forward_speed),
        "--residual-gain",
        str(0.0 if mode == "base" else args.residual_gain),
        "--control-timeout-ms",
        str(args.control_timeout_ms),
        "--target-semantics",
        configuration.target_semantics,
        "--n-step",
        str(configuration.n_step),
        "--critic-replay-mode",
        configuration.critic_replay_mode,
        "--flat-fraction-initial",
        str(configuration.flat_fraction_initial),
        "--flat-fraction-ready",
        str(configuration.flat_fraction_ready),
        "--flat-anchor-weight",
        str(configuration.flat_anchor_weight),
        "--flat-critic-warmup-updates",
        str(configuration.flat_critic_warmup_updates),
        "--protocol-max-episodes",
        str(configuration.max_episodes),
        "--protocol-max-total-steps",
        str(configuration.max_total_steps),
        "--protocol-max-stage-episodes",
        str(configuration.max_stage_episodes),
        "--protocol-sha256",
        protocol.sha256(),
        "--device",
        args.device,
        "--require-track-completion",
    ]
    if mode == "residual_frozen":
        command.extend(("--residual-checkpoint", args.initial_residual_checkpoint))
    if args.headless:
        command.append("--headless")
    if args.video:
        command.extend(("--video", "--video-length", str(args.max_steps_per_episode)))
    return command


def _ensure_same_seed_controls(
    args: argparse.Namespace,
    configuration: TargetConfiguration,
    output: Path,
) -> dict[str, Any]:
    summary_path = output / "same_seed_controls.json"
    if summary_path.is_file():
        return _load_json_object(summary_path)
    controls: dict[str, Any] = {
        "schema_version": 1,
        "seed": args.comparison_seed,
        "forward_speed": args.forward_speed,
        "residual_gain": args.residual_gain,
    }
    for mode in ("base", "residual_frozen"):
        run_dir = output / "controls" / mode / f"seed-{args.comparison_seed}"
        if run_dir.exists():
            raise RuntimeError(f"incomplete comparison directory: {run_dir}")
        run_dir.mkdir(parents=True)
        result = subprocess.run(
            _comparison_command(args, configuration, mode=mode, run_dir=run_dir),
            check=False,
        )
        episode_summary = run_dir / "episode_summary.json"
        if result.returncode not in {0, 1} or not episode_summary.is_file():
            raise RuntimeError(f"{mode} comparison did not complete")
        summary = _load_json_object(episode_summary)
        controls[mode] = {
            "completion_reached": summary.get("completion_reached"),
            "interaction_steps": summary.get("interaction_steps"),
            "max_x": summary.get("max_x"),
            "fell": summary.get("fell"),
            "safety_violation": summary.get("safety_violation"),
            "second_half_velocity_error": summary.get("second_half_velocity_error"),
            "second_half_tilt_rms": summary.get("second_half_tilt_rms"),
            "video": str(run_dir / "videos" / "rl-video-step-0.mp4"),
        }
    write_json_atomic(summary_path, controls)
    return controls


def _state_path(output: Path) -> Path:
    return output / "measurement_state.json"


def _load_state(output: Path) -> dict[str, Any]:
    path = _state_path(output)
    if not path.exists():
        return {"episodes_completed": 0, "interaction_steps": 0, "first_completion_episode": None}
    state = _load_json_object(path)
    for name in ("episodes_completed", "interaction_steps"):
        if type(state.get(name)) is not int or state[name] < 0:
            raise ValueError(f"measurement state has invalid {name}")
    if state.get("first_completion_episode") is not None and (
        type(state["first_completion_episode"]) is not int or state["first_completion_episode"] <= 0
    ):
        raise ValueError("measurement state has invalid first_completion_episode")
    return state


def _write_state(output: Path, state: dict[str, Any]) -> None:
    write_json_atomic(_state_path(output), state)


def _append_record(output: Path, record: dict[str, Any]) -> None:
    with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def run_measurement(args: argparse.Namespace) -> int:
    configuration = _target_configuration(args)
    protocol = protocol_for_configuration(configuration, target_stage=args.terrain_stage)
    output = _prepare_output(args, configuration)
    worker_args = _worker_args(args, configuration)
    _ensure_novelty_baseline(worker_args, output)
    controls = _ensure_same_seed_controls(args, configuration, output)
    state = _load_state(output)
    completed = state["episodes_completed"]
    if state["first_completion_episode"] is not None:
        return 0

    development_seeds = default_flat_online_protocol().development_seeds
    if args.comparison_seed not in development_seeds:
        raise ValueError("--comparison-seed must be a development seed")
    seeds = (args.comparison_seed,) + tuple(
        seed for seed in development_seeds if seed != args.comparison_seed
    )
    while completed < args.max_episodes:
        episode = completed + 1
        directory = output / "episodes" / f"episode-{episode:04d}"
        if directory.exists():
            raise RuntimeError(f"episode directory already exists without state: {directory}")
        directory.mkdir(parents=True)
        request = EpisodeRequest(
            stage=args.terrain_stage,
            seed=seeds[(episode - 1) % len(seeds)],
            episode=episode,
            gain=args.residual_gain,
            max_steps=args.max_steps_per_episode,
        )
        result = subprocess.run(build_worker_command(worker_args, request), check=False)
        outcome = classify_worker_result(
            directory,
            result.returncode,
            request=request,
            expected_protocol_sha256=protocol.sha256(),
        )
        summary = _load_json_object(directory / "episode_summary.json")
        _install_latest_checkpoint(directory, output)
        completion_reached = summary.get("completion_reached")
        successful_completion = _successful_completion(
            completion_reached=completion_reached,
            passed=outcome.passed,
            fell=outcome.fell,
            safety_violation=summary.get("safety_violation"),
        )
        interaction_steps = int(summary["interaction_steps"])
        completed = episode
        state = {
            "episodes_completed": completed,
            "interaction_steps": state["interaction_steps"] + interaction_steps,
            "first_completion_episode": episode if successful_completion else None,
        }
        _append_record(
            output,
            {
                "episode": episode,
                "seed": request.seed,
                "interaction_steps": interaction_steps,
                "cumulative_interaction_steps": state["interaction_steps"],
                "completion_reached": completion_reached,
                "successful_completion": successful_completion,
                "passed": outcome.passed,
                "fell": outcome.fell,
                "cost": outcome.cost,
                "max_x": summary["max_x"],
                "completion_target_x": summary["completion_target_x"],
                "safety_violation": summary["safety_violation"],
                "termination_reasons": summary.get("termination_reasons", []),
                "terminal_nonfoot_contact_force_n": summary.get(
                    "terminal_nonfoot_contact_force_n", 0.0
                ),
                "terminal_nonfoot_contact_body": summary.get(
                    "terminal_nonfoot_contact_body"
                ),
                "contact_termination": summary.get(
                    "contact_termination", CONTACT_TERMINATION_PROTOCOL
                ),
                "actor_updates": summary["actor_updates"],
                "critic_updates": summary["critic_updates"],
                "video": str(directory / "videos" / "rl-video-step-0.mp4"),
            },
        )
        _write_state(output, state)
        if successful_completion:
            break

    first_completion = state["first_completion_episode"]
    report = {
        "schema_version": 1,
        "measurement_mode": "first_safe_step_completion_without_acceptance_deadline_v3",
        "fall_handling": "preserve_updater_reset_episode_v1",
        "contact_termination": CONTACT_TERMINATION_PROTOCOL,
        "status": (
            "safe_completion_observed"
            if first_completion is not None
            else "observation_cap_reached"
        ),
        "first_completion_episode": first_completion,
        "episodes_completed": state["episodes_completed"],
        "interaction_steps": state["interaction_steps"],
        "observation_cap_episodes": args.max_episodes,
        "forward_speed": args.forward_speed,
        "residual_gain": args.residual_gain,
        "adapting_residual_gain": args.adapting_residual_gain,
        "post_step_residual_gain": args.post_step_residual_gain,
        "comparison_seed": args.comparison_seed,
        "same_seed_controls": controls,
        "note": (
            "The cap bounds compute only; it is not a 100-episode acceptance requirement."
        ),
    }
    write_json_atomic(output / "summary.json", report)
    return 0


def main() -> int:
    return run_measurement(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
