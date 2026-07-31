"""Collect immutable strict-plane transitions from a frozen Lite3 PPO."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterator, TextIO

import torch

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.provenance import (
    sha256_file,
    snapshot_file,
    write_json_atomic,
)
from rl_training.online_adaptation.replay_buffer import ReplayBuffer
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from rl_training.online_adaptation.types import TransitionBatch
from scripts.reinforcement_learning.online_adaptation.cli import (
    FORMAL_FLAT_REPLAY_TRANSITIONS,
    STRICT_FLAT_BASELINE_TASK,
    build_collect_flat_replay_parser,
)


REPLAY_FILENAME = "flat_replay.pt"
MANIFEST_FILENAME = "flat_replay_manifest.json"
METRICS_FILENAME = "metrics.jsonl"
STAGING_PREFIX = ".flat_replay.stage-"
READY_FILENAME = "READY.json"
FLAT_REPLAY_MANIFEST_KEYS = frozenset(
    {
        "baseline_manifest_sha256",
        "protocol_sha256",
        "task",
        "seed",
        "transition_count",
        "replay_sha256",
        "falls",
        "safety_invalid_rows",
        "frozen",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_READY_KEYS = frozenset({"schema_version", "replay_sha256", "manifest_sha256"})


@dataclass(frozen=True)
class ValidatedCollectionInputs:
    base_checkpoint: Path
    base_manifest: Path
    output_dir: Path
    baseline_manifest_sha256: str
    checkpoint_bytes: bytes


@dataclass(frozen=True)
class FlatRowResult:
    valid_rows: int
    falls: int
    truncated_rows: int
    safety_invalid_rows: int


def _require_exact_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be an exact positive integer")
    return value


def _artifact_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / REPLAY_FILENAME,
        output_dir / MANIFEST_FILENAME,
        output_dir / METRICS_FILENAME,
    )


def _load_and_verify_baseline(checkpoint: Path, manifest_path: Path) -> tuple[str, bytes]:
    manifest_snapshot = snapshot_file(manifest_path, capture_bytes=True)
    assert manifest_snapshot.content is not None
    try:
        manifest = json.loads(manifest_snapshot.content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("base manifest must contain valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("base manifest must contain a JSON object")
    if manifest.get("task") != STRICT_FLAT_BASELINE_TASK:
        raise ValueError(f"base manifest task must be {STRICT_FLAT_BASELINE_TASK!r}")
    if manifest.get("terrain_assertion") != "plane-only":
        raise ValueError("base manifest must assert plane-only terrain")

    file_records = manifest.get("files")
    checkpoint_key = str(checkpoint.resolve())
    if not isinstance(file_records, dict) or checkpoint_key not in file_records:
        raise ValueError("base checkpoint is not recorded by the baseline manifest")
    checkpoint_record = file_records[checkpoint_key]
    if not isinstance(checkpoint_record, dict):
        raise ValueError("base checkpoint manifest record must be an object")

    checkpoint_snapshot = snapshot_file(checkpoint, capture_bytes=True)
    assert checkpoint_snapshot.content is not None
    if (
        checkpoint_record.get("sha256") != checkpoint_snapshot.sha256
        or checkpoint_record.get("bytes") != checkpoint_snapshot.bytes
    ):
        raise ValueError("base checkpoint does not match the baseline manifest")
    return manifest_snapshot.sha256, checkpoint_snapshot.content


def validate_collect_args(args: Any) -> ValidatedCollectionInputs:
    if getattr(args, "task", None) != STRICT_FLAT_BASELINE_TASK:
        raise ValueError(f"collection task must be exactly {STRICT_FLAT_BASELINE_TASK!r}")
    _require_exact_positive_int(getattr(args, "seed", None), "seed")
    transition_count = _require_exact_positive_int(
        getattr(args, "transitions", None), "transitions"
    )
    if transition_count < FORMAL_FLAT_REPLAY_TRANSITIONS:
        raise ValueError(
            f"formal collection requires at least {FORMAL_FLAT_REPLAY_TRANSITIONS} transitions"
        )
    if not isinstance(getattr(args, "device", None), str) or not args.device.strip():
        raise ValueError("device must be a nonempty string")
    if type(getattr(args, "headless", None)) is not bool:
        raise ValueError("headless must be a bool")

    path_values = {}
    for name in ("base_checkpoint", "base_manifest", "output_dir"):
        value = getattr(args, name, None)
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            raise ValueError(f"{name} must be a nonempty path")
        path_values[name] = Path(value)

    checkpoint = path_values["base_checkpoint"]
    baseline_manifest = path_values["base_manifest"]
    output_dir = path_values["output_dir"]
    for name, path in (
        ("base checkpoint", checkpoint),
        ("base manifest", baseline_manifest),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is not a file: {path}")
    if checkpoint.resolve() == baseline_manifest.resolve():
        raise ValueError("base checkpoint and base manifest must be distinct files")

    if output_dir.is_symlink():
        raise ValueError("output_dir must not be a symbolic link")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {output_dir}")
    if output_dir.exists():
        recover_incomplete_publication(output_dir)
        existing = sorted(output_dir.iterdir(), key=lambda path: path.name)
        if existing:
            raise FileExistsError(f"output_dir must be empty; found {existing[0].name}")
    resolved_inputs = {checkpoint.resolve(), baseline_manifest.resolve()}
    if any(path.resolve() in resolved_inputs for path in _artifact_paths(output_dir)):
        raise ValueError("output artifacts must be distinct from provenance inputs")

    baseline_hash, checkpoint_bytes = _load_and_verify_baseline(checkpoint, baseline_manifest)
    return ValidatedCollectionInputs(
        base_checkpoint=checkpoint.resolve(),
        base_manifest=baseline_manifest.resolve(),
        output_dir=output_dir,
        baseline_manifest_sha256=baseline_hash,
        checkpoint_bytes=checkpoint_bytes,
    )


@contextmanager
def materialize_verified_checkpoint(
    validated: ValidatedCollectionInputs,
) -> Iterator[Path]:
    descriptor, temporary_name = tempfile.mkstemp(prefix="lite3-flat-checkpoint-", suffix=".pt")
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(validated.checkpoint_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        yield temporary_path
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def validate_flat_replay_manifest(manifest: Mapping[str, object]) -> None:
    if not isinstance(manifest, Mapping) or set(manifest) != FLAT_REPLAY_MANIFEST_KEYS:
        raise ValueError("flat replay manifest must contain the exact formal key set")
    for name in ("baseline_manifest_sha256", "replay_sha256"):
        value = manifest[name]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if manifest["protocol_sha256"] != default_flat_online_protocol().sha256():
        raise ValueError("protocol_sha256 must match the default protocol")
    if manifest["task"] != STRICT_FLAT_BASELINE_TASK:
        raise ValueError(f"task must be exactly {STRICT_FLAT_BASELINE_TASK!r}")
    _require_exact_positive_int(manifest["seed"], "seed")
    transition_count = _require_exact_positive_int(
        manifest["transition_count"], "transition_count"
    )
    if transition_count < FORMAL_FLAT_REPLAY_TRANSITIONS:
        raise ValueError(
            f"formal manifest requires at least {FORMAL_FLAT_REPLAY_TRANSITIONS} transitions"
        )
    if type(manifest["falls"]) is not int or manifest["falls"] != 0:
        raise ValueError("formal manifest requires falls == 0")
    if (
        type(manifest["safety_invalid_rows"]) is not int
        or manifest["safety_invalid_rows"] != 0
    ):
        raise ValueError("formal manifest requires safety_invalid_rows == 0")
    if manifest["frozen"] is not True:
        raise ValueError("formal manifest requires frozen is True")


def deterministic_base_action(
    base_policy: Callable[[torch.Tensor], torch.Tensor],
    observation: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        action = base_policy(observation)
    if not isinstance(action, torch.Tensor) or action.ndim != 2:
        raise RuntimeError("base PPO must return a batched action tensor")
    if not bool(torch.isfinite(action).all().item()):
        raise RuntimeError("base PPO returned a non-finite action")
    return clip_lite3_policy_action(action).detach().clone()


def build_one_step_flat_transition(
    *,
    history: torch.Tensor,
    current_frame: torch.Tensor,
    base_action: torch.Tensor,
    reward: torch.Tensor,
    next_history: torch.Tensor,
    next_frame: torch.Tensor,
    next_base_action: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    safety_mask: torch.Tensor,
    curriculum_stage_id: torch.Tensor,
    episode_id: torch.Tensor,
    discount: float,
) -> TransitionBatch:
    if isinstance(discount, bool) or not isinstance(discount, (int, float)):
        raise ValueError("discount must be a real scalar between 0 and 1")
    discount_value = float(discount)
    if not 0.0 <= discount_value <= 1.0:
        raise ValueError("discount must be a real scalar between 0 and 1")
    return TransitionBatch(
        history=history,
        current_frame=current_frame,
        base_action=base_action,
        residual_action=torch.zeros_like(base_action),
        reward=reward,
        next_history=next_history,
        next_frame=next_frame,
        next_base_action=next_base_action,
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=bootstrap_mask,
        discount=torch.full_like(reward, discount_value),
        safety_mask=safety_mask,
        curriculum_stage_id=curriculum_stage_id,
        episode_id=episode_id,
    )


def safety_diagnostics(decision: Any, tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    """Return JSON-safe evidence explaining why a row failed the safety mask."""
    return {
        "safety_paused_rows": int(decision.paused.bool().sum().item()),
        "safety_projected_rows": int(decision.projected.bool().sum().item()),
        "safety_reasons": list(decision.reasons),
        "nonfinite_fields": [
            name for name, value in tensors.items() if not bool(torch.isfinite(value).all().item())
        ],
    }


def store_flat_transition(
    replay: ReplayBuffer,
    transition: TransitionBatch,
) -> FlatRowResult:
    safety_mask = transition.safety_mask.bool()
    terminated = transition.terminated.bool()
    truncated = transition.truncated.bool()
    valid_mask = safety_mask & ~terminated & ~truncated
    transition_values = {field.name: getattr(transition, field.name) for field in fields(transition)}
    replay.add_batch(**transition_values, valid_mask=valid_mask)
    return FlatRowResult(
        valid_rows=int(valid_mask.sum().item()),
        falls=int(terminated.sum().item()),
        truncated_rows=int(truncated.sum().item()),
        safety_invalid_rows=int((~safety_mask).sum().item()),
    )


def open_metrics_stream(output_dir: str | Path) -> TextIO:
    directory = Path(output_dir)
    if directory.is_symlink():
        raise ValueError("output_dir must not be a symbolic link")
    directory.mkdir(parents=True, exist_ok=True)
    return (directory / METRICS_FILENAME).open("x", encoding="utf-8", newline="\n")


def write_metric(stream: TextIO, record: Mapping[str, object]) -> None:
    stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return payload


def _regular_entry_state(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "absent"
    if stat.S_ISREG(mode) and not stat.S_ISLNK(mode):
        return "regular"
    return "invalid"


def _validate_ready_stage(stage: Path) -> tuple[dict[str, object], str]:
    expected_names = {REPLAY_FILENAME, MANIFEST_FILENAME, READY_FILENAME}
    if stage.is_symlink() or not stage.is_dir():
        raise RuntimeError(f"ambiguous publication stage: {stage.name}")
    entries = {path.name: path for path in stage.iterdir()}
    if set(entries) != expected_names or any(
        path.is_symlink() or not path.is_file() for path in entries.values()
    ):
        raise RuntimeError(f"incomplete or ambiguous publication stage: {stage.name}")

    ready = _read_json_object(entries[READY_FILENAME], "publication READY journal")
    if set(ready) != _READY_KEYS or ready.get("schema_version") != 1:
        raise RuntimeError("publication READY journal has an invalid schema")
    for name in ("replay_sha256", "manifest_sha256"):
        value = ready.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise RuntimeError(f"publication READY journal has invalid {name}")

    replay_hash = sha256_file(entries[REPLAY_FILENAME])
    manifest_hash = sha256_file(entries[MANIFEST_FILENAME])
    if replay_hash != ready["replay_sha256"] or manifest_hash != ready["manifest_sha256"]:
        raise RuntimeError("publication stage bytes do not match the READY journal")
    manifest = _read_json_object(entries[MANIFEST_FILENAME], "staged flat replay manifest")
    try:
        validate_flat_replay_manifest(manifest)
    except ValueError as error:
        raise RuntimeError("staged flat replay manifest is invalid") from error
    if manifest["replay_sha256"] != replay_hash:
        raise RuntimeError("staged replay does not match its manifest")
    return manifest, manifest_hash


def _validate_final_pair(replay_path: Path, manifest_path: Path) -> None:
    for path in (replay_path, manifest_path):
        if _regular_entry_state(path) != "regular":
            raise RuntimeError(f"final artifact must be a regular non-symlink file: {path.name}")
    manifest = _read_json_object(manifest_path, "final flat replay manifest")
    try:
        validate_flat_replay_manifest(manifest)
    except ValueError as error:
        raise RuntimeError("final flat replay manifest is invalid") from error
    if sha256_file(replay_path) != manifest["replay_sha256"]:
        raise RuntimeError("final replay does not match its manifest")


def _remove_stage(stage: Path, output_dir: Path) -> None:
    shutil.rmtree(stage)
    _fsync_directory(output_dir)


def recover_incomplete_publication(output_dir: str | Path) -> bool:
    directory = Path(output_dir)
    if not directory.exists():
        return False
    stages = sorted(directory.glob(f"{STAGING_PREFIX}*"), key=lambda path: path.name)
    if not stages:
        return False
    if len(stages) != 1:
        raise RuntimeError("ambiguous publication state: multiple staging directories")

    stage = stages[0]
    _, staged_manifest_hash = _validate_ready_stage(stage)
    staged_replay = stage / REPLAY_FILENAME
    staged_manifest = stage / MANIFEST_FILENAME
    final_replay = directory / REPLAY_FILENAME
    final_manifest = directory / MANIFEST_FILENAME

    replay_state = _regular_entry_state(final_replay)
    manifest_state = _regular_entry_state(final_manifest)
    if "invalid" in (replay_state, manifest_state):
        raise RuntimeError("ambiguous publication state: final artifacts must be regular non-symlink files")
    if manifest_state == "regular" and replay_state == "absent":
        raise RuntimeError("ambiguous publication state: manifest exists without replay")
    if replay_state == "regular" and sha256_file(final_replay) != sha256_file(staged_replay):
        raise RuntimeError("ambiguous publication state: final replay differs from staged replay")
    if manifest_state == "regular" and sha256_file(final_manifest) != staged_manifest_hash:
        raise RuntimeError("ambiguous publication state: final manifest differs from staged manifest")

    if replay_state == "absent":
        os.link(staged_replay, final_replay)
        _fsync_directory(directory)
    if manifest_state == "absent":
        os.link(staged_manifest, final_manifest)
        _fsync_directory(directory)
    _validate_final_pair(final_replay, final_manifest)
    _remove_stage(stage, directory)
    return True


def _cleanup_failed_publication(
    *,
    stage: Path,
    output_dir: Path,
    replay_linked: bool,
    manifest_linked: bool,
) -> None:
    if manifest_linked:
        (output_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    if replay_linked:
        (output_dir / REPLAY_FILENAME).unlink(missing_ok=True)
    if stage.exists():
        shutil.rmtree(stage)
    _fsync_directory(output_dir)


def publish_flat_replay(
    *,
    replay: ReplayBuffer,
    output_dir: str | Path,
    baseline_manifest_sha256: str,
    task: str,
    seed: int,
    transition_count: int,
    falls: int,
    safety_invalid_rows: int,
) -> dict[str, object]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    recover_incomplete_publication(directory)
    replay_path = directory / REPLAY_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    for path in (replay_path, manifest_path):
        if _regular_entry_state(path) != "absent":
            raise FileExistsError(f"immutable formal artifact already exists: {path.name}")
    if not replay.frozen:
        raise ValueError("flat replay must be frozen before publication")
    if len(replay) != transition_count:
        raise ValueError("transition_count must match the serialized replay size")

    stage = Path(tempfile.mkdtemp(dir=directory, prefix=STAGING_PREFIX))
    stage_replay = stage / REPLAY_FILENAME
    stage_manifest = stage / MANIFEST_FILENAME
    stage_ready = stage / READY_FILENAME
    replay_linked = False
    manifest_linked = False
    try:
        with stage_replay.open("xb") as stream:
            torch.save(replay.state_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        replay_hash = sha256_file(stage_replay)
        manifest = {
            "baseline_manifest_sha256": baseline_manifest_sha256,
            "protocol_sha256": default_flat_online_protocol().sha256(),
            "task": task,
            "seed": seed,
            "transition_count": transition_count,
            "replay_sha256": replay_hash,
            "falls": falls,
            "safety_invalid_rows": safety_invalid_rows,
            "frozen": replay.frozen,
        }
        validate_flat_replay_manifest(manifest)
        write_json_atomic(stage_manifest, manifest)
        _fsync_file(stage_manifest)
        manifest_hash = sha256_file(stage_manifest)
        write_json_atomic(
            stage_ready,
            {
                "schema_version": 1,
                "replay_sha256": replay_hash,
                "manifest_sha256": manifest_hash,
            },
        )
        _fsync_file(stage_ready)
        _fsync_directory(stage)
        _validate_ready_stage(stage)
        _fsync_directory(directory)

        os.link(stage_replay, replay_path)
        replay_linked = True
        _fsync_directory(directory)
        os.link(stage_manifest, manifest_path)
        manifest_linked = True
        _fsync_directory(directory)
        _validate_final_pair(replay_path, manifest_path)
        _remove_stage(stage, directory)
    except Exception:
        _cleanup_failed_publication(
            stage=stage,
            output_dir=directory,
            replay_linked=replay_linked,
            manifest_linked=manifest_linked,
        )
        raise
    return manifest


def _run_collection(args: Any, validated: ValidatedCollectionInputs) -> int:
    metrics_stream = open_metrics_stream(validated.output_dir)
    simulation_app = None
    wrapped_env = None
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        import gymnasium as gym

        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        import rl_training.tasks  # noqa: F401
        from rl_training.online_adaptation.history import HistoryBuffer
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
        from rl_training.online_adaptation.rsl_rl_compat import (
            freeze_rsl_rl_models,
            prepare_rsl_rl_agent_cfg,
            split_rsl_rl_endings,
        )
        from rl_training.online_adaptation.sac import SACConfig
        from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
        from rl_training.online_adaptation.terrain_guards import assert_plane_only_terrain

        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg = prepare_rsl_rl_agent_cfg(agent_cfg)
        assert_plane_only_terrain(env_cfg.scene.terrain)
        env = gym.make(args.task, cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        with materialize_verified_checkpoint(validated) as private_checkpoint:
            runner.load(str(private_checkpoint))
        base_policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
        freeze_rsl_rl_models(runner)

        device = torch.device(wrapped_env.unwrapped.device)
        adapter = Lite3IsaacLabAdapter(wrapped_env.unwrapped)
        history = HistoryBuffer(num_envs=1, history_length=25, frame_dim=49, device=device)
        replay = ReplayBuffer(
            capacity=args.transitions,
            history_length=25,
            frame_dim=49,
            action_dim=12,
            storage_dtype=torch.float16,
        )
        action_limits = torch.tensor([0.24, 0.20, 0.20] * 4, device=device)
        safety = SafetyFilter(SafetyLimits(residual_limits=action_limits))
        reward_weights = RewardWeights()
        discount = SACConfig().gamma

        observation, _ = wrapped_env.reset()
        previous_action = torch.zeros(1, 12, device=device)
        previous_residual = torch.zeros_like(previous_action)
        episode_id = torch.zeros(1, 1, dtype=torch.int64, device=device)
        falls = 0
        safety_invalid_rows = 0
        truncated_rows = 0
        attempted_rows = 0

        while len(replay) < args.transitions:
            attempted_rows += 1
            frame = adapter.frame(previous_action)
            history.append(frame)
            current_history = history.snapshot()
            base_action = deterministic_base_action(base_policy, observation)
            timeout = torch.zeros(1, 1, dtype=torch.bool, device=device)
            safety_state = adapter.safety_state(timeout)
            decision = safety.project(
                base_action,
                torch.zeros_like(base_action),
                safety_state,
                force_zero_residual=True,
            )
            with torch.inference_mode():
                next_observation, _, done, extras = wrapped_env.step(decision.action)
            next_base_action = deterministic_base_action(base_policy, next_observation)
            next_frame = adapter.frame(decision.action)
            signals = adapter.reward_signals()
            reward = compute_online_reward(
                signals,
                torch.zeros_like(base_action),
                previous_residual,
                reward_weights,
            )
            terminated, truncated, bootstrap_mask = split_rsl_rl_endings(done, extras)
            done_mask = terminated | truncated
            next_history = history.preview(next_frame, reset_mask=done_mask.squeeze(-1))

            diagnostic_tensors = {
                "current_history": current_history,
                "frame": frame,
                "base_action": base_action,
                "final_action": decision.action,
                "reward": reward,
                "next_history": next_history,
                "next_frame": next_frame,
                "next_base_action": next_base_action,
            }
            finite_rows = torch.stack(
                [torch.isfinite(value).reshape(1, -1).all(dim=1) for value in diagnostic_tensors.values()]
            ).all(dim=0, keepdim=True).T
            safety_mask = (~decision.paused.bool()) & (~decision.projected.bool()) & finite_rows
            transition = build_one_step_flat_transition(
                history=current_history,
                current_frame=frame,
                base_action=base_action,
                reward=reward,
                next_history=next_history,
                next_frame=next_frame,
                next_base_action=next_base_action,
                terminated=terminated,
                truncated=truncated,
                bootstrap_mask=bootstrap_mask,
                safety_mask=safety_mask.float(),
                curriculum_stage_id=torch.zeros_like(episode_id),
                episode_id=episode_id,
                discount=discount,
            )
            stored = store_flat_transition(replay, transition)
            safety_invalid_rows += stored.safety_invalid_rows
            falls += stored.falls
            truncated_rows += stored.truncated_rows
            diagnostic_record = safety_diagnostics(decision, diagnostic_tensors)
            diagnostic_record.update(
                {
                    "base_height_min": float(safety_state.base_height.amin().item()),
                    "joint_velocity_max_abs": float(safety_state.joint_vel.abs().amax().item()),
                    "projected_gravity_xy_norm_max": float(
                        safety_state.projected_gravity_xy.norm(dim=-1).amax().item()
                    ),
                    "nonfoot_contact_rows": int(safety_state.nonfoot_contact.bool().sum().item()),
                    "action_correction_max_abs": float(
                        (decision.action - base_action).abs().amax().item()
                    ),
                }
            )
            if not bool(safety_mask.all().item()):
                diagnostic_record["invalid_row_detail"] = {
                    "joint_names": list(adapter.robot.joint_names),
                    "base_action": base_action[0].detach().cpu().tolist(),
                    "final_action": decision.action[0].detach().cpu().tolist(),
                    "default_joint_pos": safety_state.default_joint_pos[0].detach().cpu().tolist(),
                    "joint_lower": safety_state.joint_lower[0].detach().cpu().tolist(),
                    "joint_upper": safety_state.joint_upper[0].detach().cpu().tolist(),
                    "action_scale": safety_state.action_scale[0].detach().cpu().tolist(),
                }
            write_metric(
                metrics_stream,
                {
                    "attempted_rows": attempted_rows,
                    "falls": falls,
                    "safety_invalid_rows": safety_invalid_rows,
                    "truncated_rows": truncated_rows,
                    "transition_count": len(replay),
                    **diagnostic_record,
                },
            )

            reset_ids = torch.nonzero(done_mask.squeeze(-1), as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                history.reset(reset_ids)
            episode_id = episode_id + done_mask.to(dtype=torch.int64)
            previous_action = torch.where(done_mask, torch.zeros_like(decision.action), decision.action)
            previous_residual.zero_()
            observation = next_observation

            if falls or safety_invalid_rows:
                raise RuntimeError(
                    "formal flat collection invalid: "
                    f"falls={falls}, safety_invalid_rows={safety_invalid_rows}"
                )

        replay.freeze()
        publish_flat_replay(
            replay=replay,
            output_dir=validated.output_dir,
            baseline_manifest_sha256=validated.baseline_manifest_sha256,
            task=args.task,
            seed=args.seed,
            transition_count=len(replay),
            falls=falls,
            safety_invalid_rows=safety_invalid_rows,
        )
        return 0
    finally:
        metrics_stream.close()
        if wrapped_env is not None:
            wrapped_env.close()
        if simulation_app is not None:
            simulation_app.close()


def main(argv: list[str] | None = None) -> int:
    args = build_collect_flat_replay_parser().parse_args(argv)
    validated = validate_collect_args(args)
    return _run_collection(args, validated)


if __name__ == "__main__":
    raise SystemExit(main())
