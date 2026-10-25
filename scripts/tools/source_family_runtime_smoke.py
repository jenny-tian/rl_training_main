"""Bounded Source Terrain runtime diagnostics, never a qualification scan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/rl_training"))

from rl_training.source_scan.manifest import load_manifest
from scripts.tools.runtime_progress import RuntimeProgressJournal
from scripts.reinforcement_learning.online_adaptation.run_source_qualification_scan import (
    DEFAULT_CHECKPOINT, _geometry_audit_for_cell, _make_wrapped_env, _run_episode,
    _seeded_wrapper_reset, _terrain_height_at_local_xy,
    _sha256, _tensor_sha256,
)

MANIFEST = ROOT / "docs/lite3-source-scan-preparation-v0.2-2026-10-22/source_qualification_scan_manifest_v0.2_preparation.json"
MANIFEST_SHA256 = "0fce8c53e2e2c8537c8b048b2c5f718ebf74211bf2197e1da99eb3cecf6c1cb1"
CELL_IDS = (
    "slope_L1_uphill", "slope_L1_downhill",
    "multi_step_stairs_L1_ascending", "multi_step_stairs_L1_descending",
    "multi_step_stairs_L2_ascending",
)
MAX_STEPS = 1600
HEARTBEAT_STEPS = 50


class RuntimeInterrupted(Exception):
    """Raised after a handled process signal has been journaled."""


def _json_safe_value(value):
    """Convert small manager configuration values to JSON-safe primitives."""

    if isinstance(value, slice):
        return {
            "start": value.start,
            "stop": value.stop,
            "step": value.step,
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _termination_diagnostic(unwrapped):
    """Return configured and actually triggered termination information.

    ``TerminationManager.active_terms`` is the list of configured terms.  The
    per-step values from ``get_term`` are required to identify the term(s) that
    fired before Isaac Lab performs its automatic reset.
    """

    manager = getattr(unwrapped, "termination_manager", None)
    configured_terms = list(getattr(manager, "active_terms", ())) if manager is not None else []
    term_values = {}
    for name in configured_terms:
        try:
            value = manager.get_term(name)
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[0]
            if hasattr(value, "item"):
                value = value.item()
            term_values[str(name)] = bool(value)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            # Keep the configured term visible even if an older manager does
            # not expose its current per-term value.
            term_values[str(name)] = None
    triggered_terms = [name for name, value in term_values.items() if value is True]

    term_configurations = {}
    get_term_cfg = getattr(manager, "get_term_cfg", None) if manager is not None else None
    if callable(get_term_cfg):
        for name in configured_terms:
            try:
                term_cfg = get_term_cfg(name)
            except (AttributeError, KeyError, ValueError):
                continue
            params = getattr(term_cfg, "params", {}) or {}
            configuration = {
                "time_out": getattr(term_cfg, "time_out", None),
            }
            if name == "illegal_contact":
                sensor_cfg = params.get("sensor_cfg")
                configuration.update({
                    "sensor_name": getattr(sensor_cfg, "name", None),
                    "sensor_body_names": _json_safe_value(
                        getattr(sensor_cfg, "body_names", None)
                    ),
                    "sensor_body_ids": _json_safe_value(
                        getattr(sensor_cfg, "body_ids", None)
                    ),
                    "threshold_n": _json_safe_value(params.get("threshold")),
                    "consecutive_steps": _json_safe_value(
                        params.get("consecutive_steps", 1)
                    ),
                })
            term_configurations[str(name)] = configuration

    return {
        "configured_termination_terms": [str(name) for name in configured_terms],
        "triggered_termination_terms": triggered_terms,
        "termination_term_values": term_values,
        "termination_term_configurations": term_configurations,
    }


def _phase_sin_cos(phase_angle: float) -> list[float]:
    """Return scalar phase trigonometry without requiring a torch tensor."""

    if not isinstance(phase_angle, (int, float)) or not math.isfinite(float(phase_angle)):
        raise ValueError("phase_angle must be a finite scalar")
    angle = float(phase_angle)
    return [math.sin(angle), math.cos(angle)]


def select_cell(cell_id: str):
    if cell_id not in CELL_IDS:
        raise ValueError("cell is not in the bounded runtime diagnostic allowlist")
    if _sha256(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("preparation manifest changed")
    manifest = load_manifest(MANIFEST)
    if manifest["status"] != "draft_not_for_run" or manifest["execution_authorized"] is not False:
        raise ValueError("diagnostic requires the non-executable preparation draft")
    if manifest["split"] != "source_qualification_scan":
        raise ValueError("Source-only manifest required")
    for record in manifest["execution_contract_draft"]["artifacts"].values():
        if "path" in record and _sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"registered artifact changed: {record['path']}")
    checkpoint = Path(manifest["base_checkpoint"]["path"])
    if checkpoint.resolve() != DEFAULT_CHECKPOINT.resolve() or _sha256(checkpoint) != manifest["base_checkpoint"]["sha256"]:
        raise ValueError("frozen checkpoint differs from registration")
    cells = [cell for cell in manifest["seed_plan"]["cells"] if cell["cell_id"] == cell_id]
    if len(cells) != 1:
        raise ValueError("expected exactly one registered cell")
    return manifest, cells[0]


def resolve_registered_seed_pair(
    cell: dict, geometry_seed: int | None, initial_state_seed: int | None
) -> tuple[int, int]:
    """Resolve one seed pair, defaulting to the first registered pair."""

    if (geometry_seed is None) != (initial_state_seed is None):
        raise ValueError("geometry and initial-state seeds must be supplied together")
    geometry_lo, geometry_hi = cell["geometry_seed_range"]
    state_lo, state_hi = cell["initial_state_seed_range"]
    resolved_geometry = geometry_lo if geometry_seed is None else geometry_seed
    resolved_state = state_lo if initial_state_seed is None else initial_state_seed
    if not geometry_lo <= resolved_geometry <= geometry_hi:
        raise ValueError(
            f"geometry seed must be in the registered range [{geometry_lo}, {geometry_hi}]"
        )
    if not state_lo <= resolved_state <= state_hi:
        raise ValueError(
            f"initial-state seed must be in the registered range [{state_lo}, {state_hi}]"
        )
    if resolved_geometry - geometry_lo != resolved_state - state_lo:
        raise ValueError("geometry and initial-state seeds must preserve the registered pair index")
    return int(resolved_geometry), int(resolved_state)


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True, choices=CELL_IDS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--heartbeat-steps", type=int, default=HEARTBEAT_STEPS,
        help="persist progress every N control steps (1-200)",
    )
    parser.add_argument("--geometry-seed", type=int, help="registered geometry seed for this cell")
    parser.add_argument("--initial-state-seed", type=int, help="registered initial-state seed for this cell")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.headless:
        parser.error("this diagnostic requires --headless")
    if not 1 <= args.heartbeat_steps <= 200:
        parser.error("--heartbeat-steps must be in [1, 200]")
    manifest, cell = select_cell(args.cell_id)
    try:
        geometry_seed, state_seed = resolve_registered_seed_pair(
            cell, args.geometry_seed, args.initial_state_seed
        )
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    checkpoint = Path(manifest["base_checkpoint"]["path"])
    source_files = [Path(__file__).resolve(), ROOT / "scripts/tools/runtime_progress.py",
                    ROOT / "scripts/reinforcement_learning/online_adaptation/run_source_qualification_scan.py",
                    ROOT / "source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py",
                    ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_isaac.py",
                    ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/events.py",
                    ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py"]
    source_hashes = {str(path): _sha256(path) for path in source_files}
    started = datetime.now(timezone.utc).isoformat()
    phase = "prepared"
    episode_progress = {"step": 0, "max_steps": MAX_STEPS}
    journal = RuntimeProgressJournal(output, {
        "schema_version": "lite3_source_family_runtime_progress_v1",
        "status": "started_not_completed",
        "started_at_utc": started,
        "cell_id": args.cell_id,
        "geometry_seed": geometry_seed,
        "initial_state_seed": state_seed,
        "manifest_sha256": MANIFEST_SHA256,
        "checkpoint_sha256": manifest["base_checkpoint"]["sha256"],
        "source_sha256": source_hashes,
        "headless": True,
        "max_steps": MAX_STEPS,
        "heartbeat_steps": args.heartbeat_steps,
        "process_id": os.getpid(),
        "device": args.device,
        "policy_updated": False,
        "sealed_accessed": False,
        "qualification_claim": False,
        "phase": phase,
        "episode_progress": episode_progress,
    })
    old_signal_handlers = {}

    def handle_signal(signum, _frame):
        nonlocal phase
        phase = "interrupted_by_signal"
        if not journal.finalized:
            journal.update(
                status="interrupted_by_signal",
                phase=phase,
                signal_number=int(signum),
                signal_name=signal.Signals(signum).name,
                episode_progress=dict(episode_progress),
            )
        raise RuntimeInterrupted(f"received {signal.Signals(signum).name}")

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        old_signal_handlers[signal_number] = signal.signal(signal_number, handle_signal)

    app = None
    try:
        phase = "launching_isaac"
        journal.update(phase=phase)
        app = AppLauncher(args).app
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from rl_training.online_adaptation.rsl_rl_compat import freeze_rsl_rl_models, load_frozen_rsl_rl_checkpoint

        phase = "building_environment"
        journal.update(phase=phase, geometry_seed=geometry_seed, initial_state_seed=state_seed)
        env, cfg = _make_wrapped_env(cell, geometry_seed, state_seed, device=args.device)
        phase = "environment_ready"
        journal.update(phase=phase)
        try:
            runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device=args.device)
            load_frozen_rsl_rl_checkpoint(runner, str(checkpoint))
            modules = freeze_rsl_rl_models(runner)
            before = [_tensor_sha256(module) for module in modules]
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            phase = "policy_loaded_and_frozen"
            journal.update(phase=phase, frozen_module_sha256_before=before)
            terminal_contact_diagnostic = {}
            contact_event_diagnostics = {
                "first_nonfoot_contact": None,
                "peak_nonfoot_contact": None,
                "sample_count_with_nonfoot_contact": 0,
            }
            geometry_audit = _geometry_audit_for_cell(cell, geometry_seed)
            geometry = cell["resolved_parameters"]["geometry"]
            geometry_for_metrics = {**geometry, "family": cell["terrain_family"]}

            def capture_terminal_contact(_env_ids, terminal_step):
                """Copy contact geometry before the manager performs auto-reset."""
                nonlocal phase
                phase = "capturing_terminal_state"
                journal.update(
                    phase=phase,
                    terminal_control_step=int(terminal_step),
                    episode_progress=dict(episode_progress),
                )
                if terminal_contact_diagnostic:
                    return
                robot = env.unwrapped.scene["robot"]
                sensor = env.unwrapped.scene.sensors["contact_forces"]
                positions = robot.data.body_pos_w[0].detach().cpu()
                body_velocities = getattr(robot.data, "body_lin_vel_w", None)
                if isinstance(body_velocities, torch.Tensor):
                    body_velocities = body_velocities[0].detach().cpu()
                robot_body_names = [str(name) for name in getattr(robot, "body_names", ())]
                sensor_body_names = [str(name) for name in sensor.body_names]
                robot_index_by_name = {name: index for index, name in enumerate(robot_body_names)}
                history = sensor.data.net_forces_w_history
                current = sensor.data.net_forces_w[0]
                if history is None:
                    history = current.unsqueeze(0)
                else:
                    history = history[0]
                history_oldest_to_newest = history.flip(0)
                peak_force = torch.linalg.vector_norm(history, dim=-1).amax(dim=0).cpu()
                current_force = torch.linalg.vector_norm(current, dim=-1).cpu()
                origin = env.unwrapped.scene.terrain.env_origins[0].detach().cpu()
                bodies = []
                feet = []
                for sensor_index, body_name in enumerate(sensor_body_names):
                    if body_name not in robot_index_by_name:
                        continue
                    body_index = robot_index_by_name[body_name]
                    position = [float(value) for value in positions[body_index].tolist()]
                    local_x = position[0] - float(origin[0]) + float(geometry["flat_length_m"]) / 2.0
                    local_y = position[1] - float(origin[1]) + float(geometry["size"][1]) / 2.0
                    surface = _terrain_height_at_local_xy(
                        local_x, local_y, geometry_for_metrics, geometry_audit
                    )
                    entry = {
                        "body_index": body_index,
                        "sensor_body_index": sensor_index,
                        "body_name": body_name,
                        "position_w_m": position,
                        "local_xy_m": [local_x, local_y],
                        "surface_z_m": surface,
                        "vertical_clearance_m": None if surface is None else position[2] - surface,
                        "current_force_n": float(current_force[sensor_index].item()),
                        "history_peak_force_n": float(peak_force[sensor_index].item()),
                        "current_force_vector_w_n": [
                            float(value) for value in current[sensor_index].detach().cpu().tolist()
                        ],
                        "history_force_vectors_w_n_oldest_to_newest": [
                            [float(value) for value in vector]
                            for vector in history_oldest_to_newest[:, sensor_index].detach().cpu().tolist()
                        ],
                        "history_contact_over_1n_oldest_to_newest": [
                            bool(value)
                            for value in (
                                torch.linalg.vector_norm(
                                    history_oldest_to_newest[:, sensor_index], dim=-1
                                ) > 1.0
                            ).tolist()
                        ],
                    }
                    if isinstance(body_velocities, torch.Tensor):
                        entry["linear_velocity_w_mps"] = [
                            float(value) for value in body_velocities[body_index].tolist()
                        ]
                    if body_name.endswith("_FOOT"):
                        entry["current_contact"] = bool(
                            current_force[sensor_index].item() > 1.0
                        )
                        feet.append(entry)
                    else:
                        bodies.append(entry)
                episode_length = getattr(env.unwrapped, "episode_length_buf", None)
                episode_length_value = None
                if isinstance(episode_length, torch.Tensor) and episode_length.numel():
                    episode_length_value = int(episode_length.reshape(-1)[0].item())
                cycle_time_s = 0.425
                phase_step = terminal_step if episode_length_value is None else episode_length_value
                phase_time_s = phase_step * float(env.unwrapped.step_dt)
                phase_turns = phase_time_s / cycle_time_s
                phase_angle = 2.0 * torch.pi * phase_turns
                phase_offsets = {"FL_FOOT": 0.0, "FR_FOOT": 1.0, "HL_FOOT": 1.0, "HR_FOOT": 0.0}
                reference_phase_s = {
                    foot_name: float((2.0 * phase_time_s / cycle_time_s + offset) % 2.0)
                    for foot_name, offset in phase_offsets.items()
                }
                joint_pos = getattr(robot.data, "joint_pos", None)
                joint_vel = getattr(robot.data, "joint_vel", None)
                default_joint_pos = getattr(robot.data, "default_joint_pos", None)
                def _joint_row(value):
                    if not isinstance(value, torch.Tensor):
                        return None
                    return [float(item) for item in value[0].detach().cpu().tolist()]
                joint_names = [str(name) for name in getattr(robot, "joint_names", ())]
                joint_positions = _joint_row(joint_pos)
                joint_velocities = _joint_row(joint_vel)
                joint_defaults = _joint_row(default_joint_pos)
                joint_state_by_name = None
                if (joint_positions is not None and joint_velocities is not None
                        and joint_defaults is not None
                        and len(joint_names) == len(joint_positions)
                        == len(joint_velocities) == len(joint_defaults)):
                    joint_state_by_name = {
                        name: {
                            "position_rad": joint_positions[index],
                            "velocity_rad_s": joint_velocities[index],
                            "default_position_rad": joint_defaults[index],
                            "position_error_rad": joint_positions[index] - joint_defaults[index],
                        }
                        for index, name in enumerate(joint_names)
                    }
                sensor_cfg = getattr(sensor, "cfg", None)
                termination_diagnostic = _termination_diagnostic(env.unwrapped)
                terminal_contact_diagnostic.update({
                    "terminal_step": terminal_step,
                    "episode_length_buf_before_reset": episode_length_value,
                    "gait_phase": {
                        "cycle_time_s": cycle_time_s,
                        "phase_time_s": phase_time_s,
                        "phase_turns": phase_turns,
                        "sin_cos": _phase_sin_cos(phase_angle),
                        "reference_phase_s_by_foot_0_to_2": reference_phase_s,
                        "phase_source": "episode_length_buf_before_reset_or_observer_step_using_configured_foot_trajectory_cycle_and_offsets",
                    },
                    "root_position_w_m": [float(value) for value in robot.data.root_pos_w[0].detach().cpu().tolist()],
                    "root_quaternion_w": [float(value) for value in robot.data.root_quat_w[0].detach().cpu().tolist()],
                    "root_linear_velocity_w_mps": [float(value) for value in robot.data.root_lin_vel_w[0].detach().cpu().tolist()],
                    "root_angular_velocity_b_rps": [float(value) for value in robot.data.root_ang_vel_b[0].detach().cpu().tolist()],
                    "robot_body_names": robot_body_names,
                    "contact_sensor_body_names": sensor_body_names,
                    "nonfoot_bodies": bodies,
                    "foot_bodies": feet,
                    "joint_names": joint_names,
                    "joint_state_by_name": joint_state_by_name,
                    "contact_threshold_n": 1.0,
                    "contact_history_shape": list(history.shape),
                    "contact_history_order": "oldest_to_newest",
                    "contact_history_update_period_s": (
                        None if sensor_cfg is None or getattr(sensor_cfg, "update_period", None) is None
                        else float(sensor_cfg.update_period)
                    ),
                    **termination_diagnostic,
                })
                journal.update(
                    phase="terminal_state_captured",
                    terminal_diagnostic_summary={
                        "terminal_step": int(terminal_step),
                        "configured_termination_terms": terminal_contact_diagnostic[
                            "configured_termination_terms"
                        ],
                        "triggered_termination_terms": terminal_contact_diagnostic[
                            "triggered_termination_terms"
                        ],
                        "nonfoot_body_names_with_force": [
                            item["body_name"] for item in bodies if item["history_peak_force_n"] > 1.0
                        ],
                        "nonfoot_force_peak_n": max(
                            (item["history_peak_force_n"] for item in bodies), default=0.0
                        ),
                    },
                )

            def capture_contact_event(_step, sample):
                if not isinstance(sample, dict) or sample.get("invalid"):
                    return
                force = float(sample.get("sample_nonfoot_contact_peak_force_n", 0.0))
                snapshot = sample.get("contact_event_snapshot")
                if force <= 1.0 or not isinstance(snapshot, dict):
                    return
                contact_event_diagnostics["sample_count_with_nonfoot_contact"] += 1
                event = {"control_step": int(_step), **snapshot}
                if contact_event_diagnostics["first_nonfoot_contact"] is None:
                    contact_event_diagnostics["first_nonfoot_contact"] = event
                peak = contact_event_diagnostics["peak_nonfoot_contact"]
                if peak is None or force > peak["contact_force_peak_n_in_sensor_history"]:
                    contact_event_diagnostics["peak_nonfoot_contact"] = event

            def observe_sample(_step, sample):
                nonlocal phase
                capture_contact_event(_step, sample)
                episode_progress["step"] = int(_step)
                episode_progress["sample_valid"] = not bool(
                    isinstance(sample, dict) and sample.get("invalid")
                )
                if isinstance(sample, dict) and not sample.get("invalid"):
                    episode_progress.update({
                        "forward_progress_m": sample.get("forward_progress_m"),
                        "lateral_offset_m": sample.get("lateral_offset_m"),
                        "root_position_w_m": sample.get("root_position_w_m"),
                        "velocity_error_mps": sample.get("velocity_error_mps"),
                        "max_projected_gravity_xy": sample.get("max_projected_gravity_xy"),
                        "nonfoot_contact": sample.get("nonfoot_contact"),
                        "sample_nonfoot_contact_peak_force_n": sample.get(
                            "sample_nonfoot_contact_peak_force_n"
                        ),
                    })
                phase = "running_episode"
                if (
                    _step % args.heartbeat_steps == 0
                    or bool(terminal_contact_diagnostic)
                    or (
                        contact_event_diagnostics["first_nonfoot_contact"] is not None
                        and contact_event_diagnostics["first_nonfoot_contact"]["control_step"] == _step
                    )
                ):
                    journal.update(
                        phase=phase,
                        episode_progress=dict(episode_progress),
                        nonfoot_contact_event_count=contact_event_diagnostics[
                            "sample_count_with_nonfoot_contact"
                        ],
                        first_nonfoot_contact_step=(
                            None if contact_event_diagnostics["first_nonfoot_contact"] is None
                            else contact_event_diagnostics["first_nonfoot_contact"]["control_step"]
                        ),
                        peak_nonfoot_contact=contact_event_diagnostics[
                            "peak_nonfoot_contact"
                        ],
                    )

            phase = "running_episode"
            journal.update(phase=phase, episode_progress=dict(episode_progress))
            record = _run_episode(env, policy, cell, geometry_seed, state_seed,
                                  episode_index=1, max_steps=MAX_STEPS, bounded_smoke=True,
                                  pre_reset_observer=capture_terminal_contact,
                                  sample_observer=observe_sample)
            phase = "episode_completed"
            episode_summary = {
                key: record.get(key) for key in (
                    "steps", "completed", "termination_reason", "forward_progress_m",
                    "max_forward_progress_m", "nonfoot_contact_count",
                    "nonfoot_contact_max_force_n", "nonfoot_contact_max_force_body",
                    "safety_filter_projection_fraction", "invalid_state_count",
                )
            }
            journal.update(
                phase=phase,
                episode_progress={**episode_progress, "step": record["steps"]},
                episode_summary=episode_summary,
            )
            root = env.unwrapped.scene["robot"].data.root_pos_w
            hits = env.unwrapped.scene.sensors["height_scanner_base"].data.ray_hits_w
            height_diagnostic = {
                "post_episode_root_position_w_m": root.cpu().tolist(),
                "ray_hits_shape": list(hits.shape),
                "ray_hits_w_m": [
                    [[float(value) if torch.isfinite(value) else None for value in hit]
                     for hit in row] for row in hits.cpu()
                ],
                "all_hits_finite": bool(torch.isfinite(hits).all()),
                "max_hit_horizontal_distance_m": float(torch.linalg.vector_norm(
                    hits[:, :, :2] - root[:, None, :2], dim=-1
                ).nan_to_num(posinf=1e6).max()),
            }
            phase = "explicit_reset_check"
            journal.update(phase=phase, episode_summary=episode_summary)
            observation, _ = _seeded_wrapper_reset(env, state_seed)
            reset_root = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().tolist()
            observations = list(observation.values()) if hasattr(observation, "values") else [observation]
            reset_check = {
                "root_position_w_m": reset_root,
                "finite_observation": all(bool(torch.isfinite(value).all()) for value in observations),
                "episode_length_zero": bool((env.unwrapped.episode_length_buf == 0).all()),
                "root_xy_matches_origin": bool(torch.allclose(
                    env.unwrapped.scene["robot"].data.root_pos_w[:, :2],
                    env.unwrapped.scene.terrain.env_origins[:, :2], atol=1e-6)),
            }
            after = [_tensor_sha256(module) for module in modules]
            if before != after:
                raise RuntimeError("frozen model state changed")
            if not all(reset_check[key] for key in ("finite_observation", "episode_length_zero", "root_xy_matches_origin")):
                raise RuntimeError(f"explicit reset check failed: {reset_check}")
        finally:
            env.close()
        if _sha256(checkpoint) != manifest["base_checkpoint"]["sha256"]:
            raise RuntimeError("checkpoint bytes changed")
        if any(_sha256(Path(path)) != digest for path, digest in source_hashes.items()):
            raise RuntimeError("runtime source changed during diagnostic")
        report = {
            "schema_version": "lite3_source_family_runtime_smoke_v2",
            "status": "bounded_runtime_diagnostic_completed",
            "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": MANIFEST_SHA256, "checkpoint_sha256": manifest["base_checkpoint"]["sha256"],
            "source_sha256": source_hashes, "cell": cell,
            "headless": True, "max_steps": MAX_STEPS,
            "heartbeat_steps": args.heartbeat_steps,
            "termination_probe_injected": False,
            "policy_updated": False, "sealed_accessed": False, "qualification_claim": False,
            "frozen_module_sha256_before": before, "frozen_module_sha256_after": after,
            "explicit_reset_check": reset_check, "record": record,
            "height_diagnostic_before_explicit_reset": height_diagnostic,
            "terminal_contact_diagnostic": terminal_contact_diagnostic,
            "contact_event_diagnostics": contact_event_diagnostics,
        }
        journal.finalize(report)
        print(json.dumps({"output": str(output), "steps": record["steps"],
                          "termination_reason": record["termination_reason"]}), flush=True)
    except BaseException as error:
        if not journal.finalized:
            status = (
                "interrupted_by_signal" if isinstance(error, RuntimeInterrupted)
                else "failed_with_exception"
            )
            try:
                journal.update(
                    status=status,
                    phase=phase,
                    failure={
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    episode_progress=dict(episode_progress),
                )
            except BaseException as journal_error:
                print(
                    f"failed to persist diagnostic exception to {output}: {journal_error!r}",
                    file=sys.stderr,
                    flush=True,
                )
        print(
            json.dumps({
                "output": str(output), "status": journal.payload.get("status"),
                "phase": phase, "error": repr(error),
            }),
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        if app is not None:
            app.close()
        for signal_number, old_handler in old_signal_handlers.items():
            signal.signal(signal_number, old_handler)


if __name__ == "__main__":
    main()
