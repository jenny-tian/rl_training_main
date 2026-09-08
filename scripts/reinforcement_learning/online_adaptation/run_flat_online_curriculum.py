from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from argparse import Namespace
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from rl_training.online_adaptation.curriculum import (
    CurriculumCoordinator,
    EpisodeOutcome,
    ResidualGainScheduler,
)
from rl_training.online_adaptation.provenance import write_json_atomic
from rl_training.online_adaptation.terrain_protocol import (
    FlatOnlineProtocol,
    default_flat_online_protocol,
)
from rl_training.online_adaptation.track_completion import (
    FULL_TRACK_COMPLETION_MODE,
    PROGRESS_COST_VERSION,
)
from scripts.reinforcement_learning.online_adaptation.cli import (
    build_flat_online_curriculum_parser,
)


@dataclass(frozen=True)
class EpisodeRequest:
    stage: str
    seed: int
    episode: int
    gain: float
    max_steps: int = 2_000


@dataclass(frozen=True)
class TargetConfiguration:
    target_semantics: str = "timeout-aware"
    n_step: int = 3
    critic_replay_mode: str = "dual"
    flat_fraction_initial: float = 0.75
    flat_fraction_ready: float = 0.25
    flat_anchor_weight: float = 0.5
    max_policy_residual_mse: float | None = None
    terrain_feature_source: str = "geometry"
    terrain_min_valid_fraction: float = 0.90
    flat_critic_warmup_updates: int = 1_000
    max_episodes: int = 100
    max_total_steps: int = 200_000
    max_stage_episodes: int = 20

    def __post_init__(self) -> None:
        if self.target_semantics not in {"legacy-done", "timeout-aware"}:
            raise ValueError("target_semantics is invalid")
        if self.n_step not in {1, 3, 5}:
            raise ValueError("n_step must be one of 1, 3, 5")
        if self.critic_replay_mode not in {"online-only", "dual"}:
            raise ValueError("critic_replay_mode is invalid")
        for name in ("flat_fraction_initial", "flat_fraction_ready"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            isinstance(self.flat_anchor_weight, bool)
            or not isinstance(self.flat_anchor_weight, (int, float))
            or not math.isfinite(float(self.flat_anchor_weight))
            or self.flat_anchor_weight < 0
        ):
            raise ValueError("flat_anchor_weight must be nonnegative")
        if self.max_policy_residual_mse is not None and (
            isinstance(self.max_policy_residual_mse, bool)
            or not isinstance(self.max_policy_residual_mse, (int, float))
            or not math.isfinite(float(self.max_policy_residual_mse))
            or self.max_policy_residual_mse < 0
        ):
            raise ValueError("max_policy_residual_mse must be finite and nonnegative")
        if self.terrain_feature_source not in {"proprio", "geometry", "height_scan"}:
            raise ValueError("terrain_feature_source is invalid")
        if (
            isinstance(self.terrain_min_valid_fraction, bool)
            or not isinstance(self.terrain_min_valid_fraction, (int, float))
            or not math.isfinite(float(self.terrain_min_valid_fraction))
            or not 0.0 < float(self.terrain_min_valid_fraction) <= 1.0
        ):
            raise ValueError("terrain_min_valid_fraction must be finite and in (0, 1]")
        for name in (
            "flat_critic_warmup_updates", "max_episodes", "max_total_steps",
            "max_stage_episodes",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "flat_critic_warmup_updates" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


FLAT_RETENTION_SEEDS = (7, 11, 19, 23, 31)
FLAT_RETENTION_STEPS = 1_000
SAFE_DEPLOYMENT_RESIDUAL_GAIN = 0.25
NOVELTY_BASELINE_SEED = 7
NOVELTY_BASELINE_FILENAME = "novelty_baseline.json"
NOVELTY_BASELINE_STEPS = 384


def resolve_target_configuration(
    configuration_path: str | Path | None,
    direct_values: Mapping[str, object],
) -> TargetConfiguration:
    file_values: dict[str, object] = {}
    if configuration_path is not None:
        loaded = json.loads(Path(configuration_path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("target configuration must be a JSON object")
        if isinstance(loaded.get("target_configuration"), dict):
            loaded = loaded["target_configuration"]
        file_values = dict(loaded)
    allowed = {field.name for field in fields(TargetConfiguration)}
    unknown = set(file_values) - allowed
    if unknown:
        raise ValueError(f"unknown target configuration fields: {sorted(unknown)}")
    merged = dict(file_values)
    for name, value in direct_values.items():
        if value is None or name not in allowed:
            continue
        if name in merged and merged[name] != value:
            raise ValueError(f"conflicting duplicate value for {name}")
        merged[name] = value
    return TargetConfiguration(**merged)


def protocol_for_configuration(
    configuration: TargetConfiguration,
    *,
    target_stage: str | None = None,
) -> FlatOnlineProtocol:
    default = default_flat_online_protocol()
    if configuration.max_stage_episodes >= configuration.max_episodes:
        # A single-stage run is the explicit protocol for experiments that
        # allocate the full 100-episode budget to the target terrain.
        if target_stage is None:
            selected_stage = default.stages[0]
        else:
            try:
                selected_stage = next(
                    stage for stage in default.stages if stage.name == target_stage
                )
            except StopIteration as error:
                raise ValueError(f"unknown target stage: {target_stage}") from error
        stages = (
            replace(selected_stage, max_episodes=configuration.max_stage_episodes),
        )
    else:
        stages = tuple(
            replace(stage, max_episodes=configuration.max_stage_episodes)
            for stage in default.stages
        )
    return replace(
        default,
        stages=stages,
        max_total_episodes=configuration.max_episodes,
        max_total_steps=configuration.max_total_steps,
    )


def worker_step_budget(
    *,
    total_episodes: int,
    total_steps: int,
    max_episodes: int = 100,
    max_total_steps: int = 200_000,
    max_steps_per_episode: int = 2_000,
) -> int:
    if total_episodes >= max_episodes:
        raise RuntimeError("episode budget exhausted")
    remaining = max_total_steps - total_steps
    if remaining <= 0:
        raise RuntimeError("step budget exhausted")
    return min(max_steps_per_episode, remaining)


def episode_directory(output_dir: str | Path, episode: int) -> Path:
    return Path(output_dir) / "episodes" / f"episode-{episode:04d}"


def _option(command: list[str], name: str, value: object) -> None:
    command.extend((name, str(value)))


def build_worker_command(args: Namespace, request: EpisodeRequest) -> list[str]:
    run_dir = episode_directory(args.output_dir, request.episode)
    checkpoint_out = run_dir / "experiment.pt"
    command = [sys.executable, str(Path(__file__).with_name("play_online_adaptation.py"))]
    options = [
        ("--base-checkpoint", args.base_checkpoint),
        ("--mode", "online"),
        ("--run-dir", run_dir),
        ("--flat-replay", args.flat_replay),
        ("--terrain-stage", request.stage),
        ("--terrain-seed", request.seed),
        ("--seed", request.seed),
        ("--max-steps", request.max_steps),
        ("--forward-speed", getattr(args, "forward_speed", 0.5)),
        ("--residual-gain", request.gain),
        ("--adapting-residual-gain", getattr(args, "adapting_residual_gain", 0.5)),
        ("--target-semantics", args.target_semantics),
        ("--n-step", args.n_step),
        ("--critic-replay-mode", args.critic_replay_mode),
        ("--flat-fraction-initial", args.flat_fraction_initial),
        ("--flat-fraction-ready", args.flat_fraction_ready),
        ("--flat-anchor-weight", args.flat_anchor_weight),
        ("--flat-critic-warmup-updates", args.flat_critic_warmup_updates),
        (
            "--actor-updates-per-second",
            getattr(args, "actor_updates_per_second", 2.0),
        ),
        (
            "--adaptation-min-replay-size",
            getattr(args, "adaptation_min_replay_size", 128),
        ),
        (
            "--stalled-actor-replay-stride",
            getattr(args, "stalled_actor_replay_stride", 16),
        ),
        ("--control-timeout-ms", getattr(args, "control_timeout_ms", 20.0)),
        ("--max-critic-grad-norm", getattr(args, "max_critic_grad_norm", 3_000.0)),
        ("--critic-huber-delta", getattr(args, "critic_huber_delta", None)),
        ("--target-q-min", getattr(args, "target_q_min", None)),
        ("--target-q-max", getattr(args, "target_q_max", None)),
        (
            "--deterministic-actor-weight",
            getattr(args, "deterministic_actor_weight", 0.0),
        ),
        (
            "--post-step-residual-gain",
            getattr(args, "post_step_residual_gain", SAFE_DEPLOYMENT_RESIDUAL_GAIN),
        ),
        ("--post-step-stability-weight", getattr(args, "post_step_stability_weight", 3.0)),
        ("--post-step-progress-weight", getattr(args, "post_step_progress_weight", 2.0)),
        ("--stall-penalty-weight", getattr(args, "stall_penalty_weight", 2.0)),
        ("--completion-bonus", getattr(args, "completion_bonus", 5.0)),
        ("--post-step-recovery-steps", 25),
        ("--pre-step-clearance-weight", getattr(args, "pre_step_clearance_weight", 4.0)),
        ("--pre-step-clearance-target", getattr(args, "pre_step_clearance_target", 0.045)),
        ("--pre-step-foot-forward-weight", 2.0),
        ("--pre-step-foot-forward-target", 0.04),
        ("--pre-step-approach-distance", getattr(args, "pre_step_approach_distance", 0.50)),
        ("--protocol-max-episodes", args.max_episodes),
        ("--protocol-max-total-steps", args.max_total_steps),
        ("--protocol-max-stage-episodes", args.max_stage_episodes),
        ("--protocol-sha256", protocol_for_configuration(TargetConfiguration(
            target_semantics=args.target_semantics,
            n_step=args.n_step,
            critic_replay_mode=args.critic_replay_mode,
            flat_fraction_initial=args.flat_fraction_initial,
            flat_fraction_ready=args.flat_fraction_ready,
            flat_anchor_weight=args.flat_anchor_weight,
            flat_critic_warmup_updates=args.flat_critic_warmup_updates,
            max_episodes=args.max_episodes,
            max_total_steps=args.max_total_steps,
            max_stage_episodes=args.max_stage_episodes,
        ), target_stage=request.stage).sha256()),
        ("--experiment-checkpoint-out", checkpoint_out),
        ("--device", args.device),
    ]
    if args.max_policy_residual_mse is not None:
        options.append(("--max-policy-residual-mse", args.max_policy_residual_mse))
    terrain_source = getattr(args, "terrain_feature_source", "geometry")
    if terrain_source == "proprio":
        options.append(("--terrain-feature-source", terrain_source))
    if terrain_source == "height_scan":
        options.extend(
            (
                ("--terrain-feature-source", terrain_source),
                ("--terrain-min-valid-fraction", getattr(args, "terrain_min_valid_fraction", 0.90)),
            )
        )
    for name, value in options:
        if value is not None:
            _option(command, name, value)
    command.append("--require-track-completion")
    latest = Path(args.output_dir) / "experiment.pt"
    if latest.is_file():
        _option(command, "--experiment-checkpoint-in", latest)
    else:
        initial_residual = getattr(args, "initial_residual_checkpoint", None)
        if initial_residual is not None:
            _option(command, "--residual-checkpoint", initial_residual)
        baseline = Path(args.output_dir) / NOVELTY_BASELINE_FILENAME
        if not baseline.is_file():
            raise RuntimeError("sealed novelty baseline is unavailable")
        _option(command, "--novelty-baseline-in", baseline)
        _option(command, "--novelty-baseline-sha256", _sha256(baseline))
    if args.headless:
        command.append("--headless")
    if getattr(args, "video", False):
        command.append("--video")
        _option(command, "--video-length", request.max_steps)
    return command


def build_retention_probe_command(
    args: Namespace,
    *,
    mode: str,
    seed: int,
    run_dir: str | Path,
    learned_checkpoint: str | Path,
) -> list[str]:
    if mode not in {"base", "learned_frozen"}:
        raise ValueError("retention probe mode must be base or learned_frozen")
    worker_mode = "base" if mode == "base" else "residual_frozen"
    command = [sys.executable, str(Path(__file__).with_name("play_online_adaptation.py"))]
    options = [
        ("--base-checkpoint", args.base_checkpoint),
        ("--mode", worker_mode),
        ("--run-dir", run_dir),
        ("--terrain-seed", seed),
        ("--seed", seed),
        ("--max-steps", FLAT_RETENTION_STEPS),
        ("--forward-speed", getattr(args, "forward_speed", 0.5)),
        ("--control-timeout-ms", getattr(args, "control_timeout_ms", 20.0)),
        ("--residual-gain", 0.0 if mode == "base" else SAFE_DEPLOYMENT_RESIDUAL_GAIN),
        ("--target-semantics", args.target_semantics),
        ("--n-step", args.n_step),
        ("--critic-replay-mode", args.critic_replay_mode),
        ("--flat-fraction-initial", args.flat_fraction_initial),
        ("--flat-fraction-ready", args.flat_fraction_ready),
        ("--flat-anchor-weight", args.flat_anchor_weight),
        ("--flat-critic-warmup-updates", args.flat_critic_warmup_updates),
        ("--protocol-max-episodes", args.max_episodes),
        ("--protocol-max-total-steps", args.max_total_steps),
        ("--protocol-max-stage-episodes", args.max_stage_episodes),
        ("--protocol-sha256", protocol_for_configuration(TargetConfiguration(
            target_semantics=args.target_semantics,
            n_step=args.n_step,
            critic_replay_mode=args.critic_replay_mode,
            flat_fraction_initial=args.flat_fraction_initial,
            flat_fraction_ready=args.flat_fraction_ready,
            flat_anchor_weight=args.flat_anchor_weight,
            flat_critic_warmup_updates=args.flat_critic_warmup_updates,
            max_episodes=args.max_episodes,
            max_total_steps=args.max_total_steps,
            max_stage_episodes=args.max_stage_episodes,
        )).sha256()),
        ("--device", args.device),
    ]
    if args.max_policy_residual_mse is not None:
        options.append(("--max-policy-residual-mse", args.max_policy_residual_mse))
    terrain_source = getattr(args, "terrain_feature_source", "geometry")
    if terrain_source == "proprio":
        options.append(("--terrain-feature-source", terrain_source))
    if terrain_source == "height_scan":
        options.extend(
            (
                ("--terrain-feature-source", terrain_source),
                ("--terrain-min-valid-fraction", getattr(args, "terrain_min_valid_fraction", 0.90)),
            )
        )
    for name, value in options:
        _option(command, name, value)
    if mode == "learned_frozen":
        _option(command, "--experiment-checkpoint-in", learned_checkpoint)
        # Geometry/scan-trained residual actors carry ten terrain features;
        # a proprio-only actor keeps the frozen 49-dimensional interface.
        _option(command, "--residual-frame-dim", 49 if terrain_source == "proprio" else 59)
    command.append("--flat-retention-probe")
    if args.headless:
        command.append("--headless")
    if getattr(args, "video", False):
        command.append("--video")
        _option(command, "--video-length", FLAT_RETENTION_STEPS)
    return command


def build_novelty_baseline_probe_command(
    args: Namespace, *, run_dir: str | Path, baseline_out: str | Path
) -> list[str]:
    command = build_retention_probe_command(
        args,
        mode="base",
        seed=NOVELTY_BASELINE_SEED,
        run_dir=run_dir,
        learned_checkpoint=Path(args.output_dir) / "unused.pt",
    )
    command[command.index("--max-steps") + 1] = str(NOVELTY_BASELINE_STEPS)
    _option(command, "--novelty-baseline-out", baseline_out)
    return command


def flat_retention_passes(
    base_summaries: list[Mapping[str, object]],
    learned_summaries: list[Mapping[str, object]],
) -> bool:
    if len(base_summaries) != len(FLAT_RETENTION_SEEDS) or len(learned_summaries) != len(
        FLAT_RETENTION_SEEDS
    ):
        return False

    def mean(summaries: list[Mapping[str, object]], name: str) -> float:
        values = [summary.get(name) for summary in summaries]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"retention summaries lack numeric {name}")
        return sum(float(value) for value in values) / len(values)

    try:
        if any(
            not flat_retention_summary_is_disabled(item)
            for item in base_summaries + learned_summaries
        ):
            return False
        if any(
            type(item.get("passed")) is not bool or type(item.get("fell")) is not bool
            for item in base_summaries + learned_summaries
        ):
            return False
        base_completion = sum(bool(item.get("passed")) for item in base_summaries) / len(
            base_summaries
        )
        learned_completion = sum(
            bool(item.get("passed")) for item in learned_summaries
        ) / len(learned_summaries)
        base_velocity = mean(base_summaries, "second_half_velocity_error")
        learned_velocity = mean(learned_summaries, "second_half_velocity_error")
        base_tilt = mean(base_summaries, "second_half_tilt_rms")
        learned_tilt = mean(learned_summaries, "second_half_tilt_rms")
        base_contacts = sum(float(item.get("contacts", 0)) for item in base_summaries)
        learned_contacts = sum(float(item.get("contacts", 0)) for item in learned_summaries)
        base_falls = sum(bool(item.get("fell")) for item in base_summaries)
        learned_falls = sum(bool(item.get("fell")) for item in learned_summaries)
    except (TypeError, ValueError):
        return False
    return (
        learned_completion + 0.05 >= base_completion
        and learned_velocity <= 1.05 * base_velocity
        and learned_tilt <= 1.05 * base_tilt
        and learned_falls <= base_falls
        and learned_contacts <= base_contacts
    )


def flat_retention_summary_is_disabled(summary: Mapping[str, object]) -> bool:
    return (
        summary.get("completion_mode") == "disabled"
        and summary.get("completion_reached") is False
        and summary.get("completion_target_x") is None
        and summary.get("completion_remaining_m") is None
    )


def run_flat_retention_probe(
    args: Namespace,
    *,
    window: int,
    learned_checkpoint: Path,
) -> tuple[bool, int]:
    root = Path(args.output_dir) / "evaluations" / f"window-{window:04d}"
    base_summaries: list[Mapping[str, object]] = []
    learned_summaries: list[Mapping[str, object]] = []
    evaluation_steps = 0
    for mode, summaries in (
        ("base", base_summaries),
        ("learned_frozen", learned_summaries),
    ):
        for seed in FLAT_RETENTION_SEEDS:
            run_dir = root / mode / f"seed-{seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            command = build_retention_probe_command(
                args,
                mode=mode,
                seed=seed,
                run_dir=run_dir,
                learned_checkpoint=learned_checkpoint,
            )
            result = subprocess.run(command, check=False)
            summary_path = run_dir / "episode_summary.json"
            if result.returncode not in {0, 1} or not summary_path.is_file():
                return False, evaluation_steps
            summary = _load_json_object(summary_path)
            steps = summary.get("interaction_steps")
            if type(steps) is not int or steps <= 0:
                return False, evaluation_steps
            evaluation_steps += steps
            summaries.append(summary)
    write_json_atomic(
        root / "retention_summary.json",
        {
            "base": list(base_summaries),
            "learned_frozen": list(learned_summaries),
            "passed": flat_retention_passes(base_summaries, learned_summaries),
            "evaluation_steps": evaluation_steps,
        },
    )
    return flat_retention_passes(base_summaries, learned_summaries), evaluation_steps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def classify_worker_result(
    episode_dir: str | Path,
    returncode: int,
    *,
    request: EpisodeRequest | None = None,
    expected_protocol_sha256: str | None = None,
) -> EpisodeOutcome:
    directory = Path(episode_dir)
    summary_path = directory / "episode_summary.json"
    if summary_path.is_file():
        if returncode not in {0, 1}:
            raise RuntimeError("worker return code conflicts with its episode summary")
        summary = _load_json_object(summary_path)
        checkpoint = directory / "experiment.pt"
        sidecar = directory / "experiment.pt.sha256"
        if not checkpoint.is_file() or not sidecar.is_file():
            raise RuntimeError("worker did not produce a complete checkpoint")
        expected = sidecar.read_text(encoding="ascii").strip()
        if not expected or _sha256(checkpoint) != expected:
            raise RuntimeError("worker checkpoint checksum mismatch")
        if summary.get("experiment_checkpoint_out_sha256") != expected:
            raise RuntimeError("worker summary checkpoint hash mismatch")
        if (
            expected_protocol_sha256 is not None
            and summary.get("protocol_sha256") != expected_protocol_sha256
        ):
            raise RuntimeError("worker summary protocol hash mismatch")
        if request is not None and (
            summary.get("terrain_stage") != request.stage
            or summary.get("terrain_seed") != request.seed
        ):
            raise RuntimeError("worker summary geometry differs from request")
        steps = summary.get("interaction_steps")
        passed = summary.get("passed")
        fell = summary.get("fell")
        safety_violation = summary.get("safety_violation")
        if (
            type(steps) is not int
            or steps <= 0
            or type(passed) is not bool
            or type(fell) is not bool
            or type(safety_violation) is not bool
        ):
            raise RuntimeError("worker summary schema is invalid")
        if request is not None:
            validate_track_completion_summary(summary)
        cost_value = summary.get("cost")
        if cost_value is None:
            if request is not None:
                raise RuntimeError("formal worker summary is missing finite cost")
            cost_value = summary.get("second_half_velocity_error")
        if (
            isinstance(cost_value, bool)
            or not isinstance(cost_value, (int, float))
            or not math.isfinite(float(cost_value))
            or float(cost_value) < 0.0
        ):
            raise RuntimeError("worker summary cost is invalid")
        cost = float(cost_value)
        return EpisodeOutcome(passed=passed, fell=fell, steps=steps, cost=cost)
    failure_path = directory / "failure.json"
    if returncode != 0 and failure_path.is_file():
        failure = _load_json_object(failure_path)
        if all(isinstance(failure.get(name), str) for name in ("error_type", "message", "traceback")):
            return EpisodeOutcome(passed=False, fell=True, steps=1, cost=1.0e12)
    raise RuntimeError("worker produced no valid summary or failure record")


def validate_track_completion_summary(summary: Mapping[str, object]) -> None:
    if summary.get("completion_mode") != FULL_TRACK_COMPLETION_MODE:
        raise RuntimeError("formal worker completion mode is invalid")
    if summary.get("progress_cost_version") != PROGRESS_COST_VERSION:
        raise RuntimeError("formal worker progress cost version is invalid")
    if type(summary.get("fell")) is not bool or type(summary.get("safety_violation")) is not bool:
        raise RuntimeError("formal worker safety fields must be booleans")
    fields = (
        "initial_x",
        "max_x",
        "completion_target_x",
        "completion_remaining_m",
        "cost",
    )
    values: dict[str, float] = {}
    for name in fields:
        value = summary.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"formal worker summary lacks numeric {name}")
        numeric = float(value)
        if not math.isfinite(numeric) or (
            name in {"completion_remaining_m", "cost"} and numeric < 0.0
        ):
            raise RuntimeError(f"formal worker summary has invalid {name}")
        values[name] = numeric
    initial_x = values["initial_x"]
    max_x = values["max_x"]
    target_x = values["completion_target_x"]
    remaining = values["completion_remaining_m"]
    cost = values["cost"]
    if not max_x >= initial_x:
        raise RuntimeError("formal worker max_x moved behind initial_x")
    target_distance = target_x - initial_x
    if not math.isfinite(target_distance) or target_distance <= 0.0:
        raise RuntimeError("formal worker target distance is invalid")
    expected_remaining = max(0.0, target_x - max_x)
    if not math.isclose(remaining, expected_remaining, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise RuntimeError("formal worker completion remaining distance is inconsistent")
    reached = summary.get("completion_reached")
    if type(reached) is not bool or reached != (max_x >= target_x):
        raise RuntimeError("formal worker completion evidence is inconsistent")
    passed = summary.get("passed")
    fell = summary.get("fell")
    safety_violation = summary.get("safety_violation")
    expected_passed = reached and not fell and not safety_violation
    if type(passed) is not bool or passed != expected_passed:
        raise RuntimeError("formal worker passed flag is inconsistent with safety/completion")
    expected_cost = remaining + (target_distance if fell or safety_violation else 0.0)
    if not math.isclose(cost, expected_cost, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise RuntimeError("formal worker cost is inconsistent with completion evidence")
    if passed and cost != 0.0:
        raise RuntimeError("formal worker pass must have zero cost")
    if not reached and cost <= 0.0:
        raise RuntimeError("formal worker incomplete episode must have positive cost")


def _coordinator_json_state(coordinator: CurriculumCoordinator) -> dict[str, object]:
    state = coordinator.state_dict()
    state["outcomes"] = [asdict(outcome) for outcome in state["outcomes"]]
    return state


def write_curriculum_state(
    path: str | Path,
    coordinator: CurriculumCoordinator,
    *,
    evaluation_steps: int,
    gain_scheduler: ResidualGainScheduler | None = None,
) -> None:
    scheduler = gain_scheduler or ResidualGainScheduler()
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "protocol_sha256": coordinator.protocol.sha256(),
            "coordinator": _coordinator_json_state(coordinator),
            "gain_scheduler": scheduler.state_dict(),
            "evaluation_steps": evaluation_steps,
        },
    )


def load_curriculum_state(
    path: str | Path,
    protocol: FlatOnlineProtocol,
    gain_scheduler: ResidualGainScheduler | None = None,
) -> tuple[CurriculumCoordinator, int]:
    payload = _load_json_object(Path(path))
    if payload.get("schema_version") != 1:
        raise ValueError("curriculum state schema version mismatch")
    if payload.get("protocol_sha256") != protocol.sha256():
        raise ValueError("curriculum protocol differs from saved state")
    raw_state = payload.get("coordinator")
    if not isinstance(raw_state, dict):
        raise ValueError("curriculum coordinator state is invalid")
    state = dict(raw_state)
    outcomes = state.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("curriculum outcome window is invalid")
    state["outcomes"] = tuple(EpisodeOutcome(**item) for item in outcomes)
    coordinator = CurriculumCoordinator(protocol)
    coordinator.load_state_dict(state)
    scheduler = gain_scheduler or ResidualGainScheduler()
    scheduler.load_state_dict(payload.get("gain_scheduler"))
    evaluation_steps = payload.get("evaluation_steps")
    if type(evaluation_steps) is not int or evaluation_steps < 0:
        raise ValueError("evaluation_steps must be a nonnegative int")
    return coordinator, evaluation_steps


def _install_latest_checkpoint(episode_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("experiment.pt", "experiment.pt.sha256"):
        source = episode_dir / name
        descriptor, temporary_name = tempfile.mkstemp(dir=output_dir, prefix=f".{name}.")
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, output_dir / name)
        finally:
            temporary.unlink(missing_ok=True)


def run_curriculum(args: Namespace, configuration: TargetConfiguration) -> int:
    protocol = protocol_for_configuration(configuration)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "curriculum_state.json"
    scheduler = ResidualGainScheduler()
    if state_path.exists():
        coordinator, evaluation_steps = load_curriculum_state(
            state_path, protocol, scheduler
        )
        manifest = _load_json_object(output_dir / "run_manifest.json")
        if (
            manifest.get("protocol_sha256") != protocol.sha256()
            or manifest.get("target_configuration") != asdict(configuration)
        ):
            raise ValueError("resume configuration differs from run manifest")
        if coordinator.state_dict()["total_episodes"] > 0:
            latest = output_dir / "experiment.pt"
            sidecar = output_dir / "experiment.pt.sha256"
            if (
                not latest.is_file()
                or not sidecar.is_file()
                or sidecar.read_text(encoding="ascii").strip() != _sha256(latest)
            ):
                raise RuntimeError("latest complete learner checkpoint is unavailable")
        baseline_path = output_dir / NOVELTY_BASELINE_FILENAME
        expected_baseline_sha256 = manifest.get("novelty_baseline_sha256")
        if (
            not baseline_path.is_file()
            or not isinstance(expected_baseline_sha256, str)
            or _sha256(baseline_path) != expected_baseline_sha256
        ):
            raise RuntimeError("sealed novelty baseline is unavailable or modified")
    else:
        coordinator = CurriculumCoordinator(protocol)
        evaluation_steps = 0
        scheduler.open_for_adaptation()
        baseline_path = output_dir / NOVELTY_BASELINE_FILENAME
        baseline_run_dir = output_dir / "novelty_baseline" / f"seed-{NOVELTY_BASELINE_SEED}"
        baseline_run_dir.mkdir(parents=True, exist_ok=False)
        baseline_result = subprocess.run(
            build_novelty_baseline_probe_command(
                args,
                run_dir=baseline_run_dir,
                baseline_out=baseline_path,
            ),
            check=False,
        )
        baseline_summary_path = baseline_run_dir / "episode_summary.json"
        if (
            baseline_result.returncode != 0
            or not baseline_summary_path.is_file()
            or not baseline_path.is_file()
        ):
            raise RuntimeError("strict-flat novelty baseline probe failed")
        baseline_summary = _load_json_object(baseline_summary_path)
        if baseline_summary.get("fell") is not False:
            raise RuntimeError("strict-flat novelty baseline probe fell")
        baseline_payload = _load_json_object(baseline_path)
        source = baseline_payload.get("source")
        if not isinstance(source, dict) or source.get("seed") != NOVELTY_BASELINE_SEED:
            raise RuntimeError("strict-flat novelty baseline provenance is invalid")
        baseline_sha256 = _sha256(baseline_path)
        write_json_atomic(
            output_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "protocol_sha256": protocol.sha256(),
                "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                "base_manifest": str(Path(args.base_manifest).resolve()),
                "flat_replay": str(Path(args.flat_replay).resolve()),
                "novelty_baseline": str(baseline_path.resolve()),
                "novelty_baseline_sha256": baseline_sha256,
                "target_configuration": asdict(configuration),
                "device": args.device,
                "headless": args.headless,
            },
        )
    while coordinator.stop_reason is None:
        state = coordinator.state_dict()
        max_steps = worker_step_budget(
            total_episodes=state["total_episodes"],
            total_steps=state["total_steps"],
            max_episodes=configuration.max_episodes,
            max_total_steps=configuration.max_total_steps,
            max_steps_per_episode=protocol.max_steps_per_episode,
        )
        episode = state["total_episodes"] + 1
        request = EpisodeRequest(
            stage=coordinator.stage.name,
            seed=coordinator.next_development_seed(),
            episode=episode,
            gain=scheduler.gain,
            max_steps=max_steps,
        )
        directory = episode_directory(output_dir, episode)
        directory.mkdir(parents=True, exist_ok=False)
        result = subprocess.run(build_worker_command(args, request), check=False)
        outcome = classify_worker_result(
            directory,
            result.returncode,
            request=request,
            expected_protocol_sha256=protocol.sha256(),
        )
        if (state["stage_episodes"] + 1) % coordinator.WINDOW_SIZE == 0:
            checkpoint = directory / "experiment.pt"
            if checkpoint.is_file():
                retention_passed, probe_steps = run_flat_retention_probe(
                    args,
                    window=episode // coordinator.WINDOW_SIZE,
                    learned_checkpoint=checkpoint,
                )
            else:
                retention_passed, probe_steps = False, 0
            evaluation_steps += probe_steps
            best_cost = state["best_cost"]
            outcome = replace(
                outcome,
                flat_retention_passed=retention_passed,
                cost=(
                    outcome.cost
                    if retention_passed or best_cost is None
                    else max(outcome.cost, float(best_cost))
                ),
            )
        coordinator.record_episode(outcome)
        scheduler.record_episode(
            safe=outcome.passed and not outcome.fell,
            rolled_back=False,
            performance_degraded=not outcome.flat_retention_passed,
        )
        if (directory / "experiment.pt").is_file():
            _install_latest_checkpoint(directory, output_dir)
        write_curriculum_state(
            state_path,
            coordinator,
            evaluation_steps=evaluation_steps,
            gain_scheduler=scheduler,
        )
    return 0


def main() -> int:
    args = build_flat_online_curriculum_parser().parse_args()
    direct = {
        field.name: getattr(args, field.name)
        for field in fields(TargetConfiguration)
    }
    configuration = resolve_target_configuration(args.configuration, direct)
    for field in fields(TargetConfiguration):
        setattr(args, field.name, getattr(configuration, field.name))
    return run_curriculum(args, configuration)


if __name__ == "__main__":
    raise SystemExit(main())
