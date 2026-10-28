"""Execute a pre-approved, read-only Lite3 Source Terrain scan.

The complete manifest must pass the v0.2 execution preflight before Isaac Sim
is launched.  This runner never updates or saves a policy.  A separate bounded
bridge smoke is provided by ``scripts/tools/source_terrain_policy_safety_smoke.py``
and cannot be used as an execution approval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/rl_training"))

from rl_training.source_scan.manifest import validate_manifest_file  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT
    / "logs/rsl_rl/deeprobotics_lite3_flat_adapted/"
      "2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt"
)
TASK = "SourceTerrain-Deeprobotics-Lite3-v0"
CONTROL_DT = 0.02
MAX_STEPS = 2000
STABLE_HOLD_STEPS = 50
FOOT_SLIP_SPEED_THRESHOLD_MPS = 0.35
STAIR_FRONT_COLLISION_FORCE_THRESHOLD_N = 20.0
STAIR_RISER_PROXIMITY_M = 0.08
STAIR_STUCK_DWELL_STEPS = 25
STAIR_STUCK_PROGRESS_TOLERANCE_M = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _build_env_cfg(
    cell: dict,
    geometry_seed: int,
    initial_state_seed: int,
    *,
    device: str,
    termination_probe_steps: int | None = None,
):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import rl_training.tasks  # noqa: F401
    from rl_training.source_scan.terrain_config import source_subterrain_from_cell

    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.seed = initial_state_seed
    env_cfg.scene.num_envs = 1
    # Safety queries occur after physics stepping; avoid a control-period-old
    # base patch when substep rewards have already consumed the scanner.
    env_cfg.scene.height_scanner_base.update_period = env_cfg.sim.dt
    generator = env_cfg.scene.terrain.terrain_generator
    generator.seed = geometry_seed
    generator.curriculum = True
    generator.size = tuple(cell["resolved_parameters"]["geometry"]["size"])
    generator.num_rows = 1
    generator.num_cols = 1
    generator.difficulty_range = (0.0, 0.0)
    generator.use_cache = False
    generator.sub_terrains = {
        cell["cell_id"]: source_subterrain_from_cell(cell, seed=geometry_seed)
    }
    # The scan has a registered 40 s / 2000 control-step cap implemented by
    # the runner.  Disable manager timeouts so each ending is classified here.
    env_cfg.terminations.time_out = None
    env_cfg.terminations.terrain_out_of_bounds = None
    _configure_forward_command(env_cfg.commands.base_velocity, 0.3)
    if termination_probe_steps is not None:
        if type(termination_probe_steps) is not int or not 1 <= termination_probe_steps <= 100:
            raise ValueError("termination_probe_steps must be an integer in [1, 100]")
        # The probe deliberately replaces the normal Source termination set
        # with a short manager timeout. It is a wiring test only and must not
        # be interpreted as a policy failure or qualification result.
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from rl_training.tasks.manager_based.locomotion.velocity import mdp

        env_cfg.episode_length_s = termination_probe_steps * CONTROL_DT
        env_cfg.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)
        env_cfg.terminations.illegal_contact = None
        env_cfg.terminations.bad_orientation_2 = None
    return env_cfg


def _configure_forward_command(command: Any, speed_mps: float) -> None:
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    # The command class has independent standing/lateral/yaw override shares.
    command.rel_zero_vel_envs = 0.0
    command.rel_only_lin_y_envs = 0.0
    command.rel_only_lin_x_envs = 1.0
    command.rel_only_ang_z_envs = 0.0
    command.ranges.lin_vel_x = (speed_mps, speed_mps)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)


def _seeded_wrapper_reset(wrapped_env: Any, seed: int):
    """Seed the Isaac Lab manager env before its no-argument RSL reset."""
    wrapped_env.seed(seed)
    return wrapped_env.reset()


def _make_wrapped_env(
    cell: dict,
    geometry_seed: int,
    initial_state_seed: int,
    *,
    device: str,
    termination_probe_steps: int | None = None,
):
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils import load_cfg_from_registry
    from rl_training.online_adaptation.rsl_rl_compat import prepare_rsl_rl_agent_cfg

    env_cfg = _build_env_cfg(
        cell,
        geometry_seed,
        initial_state_seed,
        device=device,
        termination_probe_steps=termination_probe_steps,
    )
    agent_cfg = prepare_rsl_rl_agent_cfg(
        load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    )
    env = gym.make(TASK, cfg=env_cfg)
    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions), agent_cfg


def _new_safety_filter(device: Any):
    import torch

    from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits

    action_limits = torch.tensor([0.24, 0.20, 0.20] * 4, device=device)
    return SafetyFilter(SafetyLimits(residual_limits=action_limits))


def _runtime_sample(
    adapter: Any,
    unwrapped: Any,
    origins: Any,
    *,
    foot_slip_threshold_mps: float = FOOT_SLIP_SPEED_THRESHOLD_MPS,
) -> dict[str, Any]:
    """Copy metrics while the requested (possibly terminal) state is current."""
    import torch

    from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES

    signals = adapter.reward_signals()
    state = adapter.safety_state(
        torch.zeros((1, 1), dtype=torch.bool, device=unwrapped.device)
    )
    root = unwrapped.scene["robot"].data.root_pos_w
    root_quat = unwrapped.scene["robot"].data.root_quat_w
    gravity = unwrapped.scene["robot"].data.projected_gravity_b
    foot_velocity_xy = adapter.robot.data.body_lin_vel_w[:, adapter.robot_foot_ids, :2]
    foot_horizontal_speed = torch.linalg.vector_norm(foot_velocity_xy, dim=-1)
    contacting_feet = signals.foot_contact > 0.0
    contacting_foot_count = int(contacting_feet[0].sum().item())
    slipping_contacting_feet = contacting_feet & (foot_horizontal_speed > foot_slip_threshold_mps)
    slipping_contact_count = int(slipping_contacting_feet[0].sum().item())
    contact_horizontal_speed_sum = float(
        (foot_horizontal_speed * contacting_feet.float()).sum().item()
    )
    foot_heights = signals.foot_height[0]
    foot_positions_w = getattr(adapter.robot.data, "body_pos_w", None)
    if isinstance(foot_positions_w, torch.Tensor):
        foot_positions_w = foot_positions_w[0, adapter.robot_foot_ids]
    else:
        foot_positions_w = None
    contact_sensor = getattr(adapter, "contact_sensor", None)
    foot_force_n = None
    foot_horizontal_force_n = None
    if contact_sensor is not None:
        net_forces = getattr(contact_sensor.data, "net_forces_w", None)
        if isinstance(net_forces, torch.Tensor):
            selected_foot_forces = net_forces[0, adapter.reward_foot_ids]
            foot_force_n = torch.linalg.vector_norm(selected_foot_forces, dim=-1)
            foot_horizontal_force_n = torch.linalg.vector_norm(selected_foot_forces[:, :2], dim=-1)
    contact_peak = adapter.observe_nonfoot_contact()
    diagnostics = adapter.contact_diagnostics()
    tensors = (
        root,
        root_quat,
        gravity,
        signals.base_lin_vel_xy,
        signals.command_lin_vel_xy,
        signals.foot_slip,
        signals.foot_contact,
        foot_horizontal_speed,
        foot_heights,
        signals.nonfoot_contact,
        state.projected_gravity_xy,
        state.base_height,
        state.joint_vel,
    )
    optional_tensors = tuple(
        value for value in (foot_positions_w, foot_force_n, foot_horizontal_force_n)
        if isinstance(value, torch.Tensor)
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in optional_tensors):
        return {"invalid": True}
    if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
        return {"invalid": True}
    qw, qx, qy, qz = root_quat[0].unbind()
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    contact_event_snapshot = None
    contact_peak_force_n = float(contact_peak["max_nonfoot_contact_force_n"])
    contact_peak_body = contact_peak["max_nonfoot_contact_body"]
    if contact_peak_body is not None and contact_peak_force_n > adapter.contact_threshold:
        robot = adapter.robot
        robot_body_names = [str(name) for name in getattr(robot, "body_names", ())]
        robot_index_by_name = {name: index for index, name in enumerate(robot_body_names)}
        sensor_body_names = [str(name) for name in adapter.contact_sensor.body_names]
        sensor_index_by_name = {name: index for index, name in enumerate(sensor_body_names)}
        body_index = robot_index_by_name.get(str(contact_peak_body))
        sensor_index = sensor_index_by_name.get(str(contact_peak_body))
        body_position = None
        body_velocity = None
        if body_index is not None:
            body_position = [
                float(value) for value in robot.data.body_pos_w[0, body_index].detach().cpu().tolist()
            ]
            body_velocity = [
                float(value) for value in robot.data.body_lin_vel_w[0, body_index].detach().cpu().tolist()
            ]
        contact_force_history = None
        if sensor_index is not None:
            raw_history = getattr(adapter.contact_sensor.data, "net_forces_w_history", None)
            if isinstance(raw_history, torch.Tensor):
                contact_force_history = [
                    [float(value) for value in vector]
                    for vector in raw_history[0, :, sensor_index].flip(0).detach().cpu().tolist()
                ]
        episode_step = int(unwrapped.episode_length_buf[0].item())
        phase_time_s = episode_step * float(unwrapped.step_dt)
        cycle_time_s = 0.425
        phase_offsets = (0.0, 1.0, 1.0, 0.0)
        phase_values = [
            float((2.0 * phase_time_s / cycle_time_s + offset) % 2.0)
            for offset in phase_offsets
        ]
        joint_names = [str(name) for name in LITE3_POLICY_JOINT_NAMES]
        joint_positions = [float(value) for value in state.joint_pos[0].detach().cpu().tolist()]
        joint_velocities = [float(value) for value in state.joint_vel[0].detach().cpu().tolist()]
        default_joint_positions = [
            float(value) for value in state.default_joint_pos[0].detach().cpu().tolist()
        ]
        root_position = [float(value) for value in root[0].detach().cpu().tolist()]
        root_orientation = [
            float(value) for value in root_quat[0].detach().cpu().tolist()
        ]
        contact_event_snapshot = {
            "episode_step": episode_step,
            "episode_time_s": phase_time_s,
            "contact_body_name": str(contact_peak_body),
            "contact_force_peak_n_in_sensor_history": contact_peak_force_n,
            "contact_threshold_n": float(adapter.contact_threshold),
            "contact_sensor_history_force_vectors_w_n_oldest_to_newest": contact_force_history,
            "contact_sensor_history_update_period_s": (
                None if getattr(adapter.contact_sensor.cfg, "update_period", None) is None
                else float(adapter.contact_sensor.cfg.update_period)
            ),
            "contact_body_position_w_m_at_control_sample_end": body_position,
            "contact_body_linear_velocity_w_mps_at_control_sample_end": body_velocity,
            "root_position_w_m": root_position,
            "root_quaternion_w": root_orientation,
            "root_angular_velocity_b_rps": [
                float(value) for value in robot.data.root_ang_vel_b[0].detach().cpu().tolist()
            ],
            "projected_gravity_b": [float(value) for value in gravity[0].detach().cpu().tolist()],
            "gait_phase": {
                "cycle_time_s": cycle_time_s,
                "phase_0_to_2_by_foot": {
                    str(name): phase_values[index]
                    for index, name in enumerate(adapter.robot_foot_names)
                },
                "phase_source": "episode_length_buf and Lite3 phase_foot_trajectory_exp configured cycle/offsets; reference clock, not measured foot phase",
            },
            "feet": [
                {
                    "body_name": str(name),
                    "position_w_m": (
                        [float(value) for value in foot_positions_w[index].tolist()]
                        if isinstance(foot_positions_w, torch.Tensor) else None
                    ),
                    "linear_velocity_w_mps": [
                        float(value) for value in robot.data.body_lin_vel_w[
                            0, adapter.robot_foot_ids[index]
                        ].detach().cpu().tolist()
                    ],
                    "contact": bool(contacting_feet[0, index].item()),
                    "contact_force_n": (
                        float(foot_force_n[index].item())
                        if isinstance(foot_force_n, torch.Tensor) else None
                    ),
                }
                for index, name in enumerate(adapter.robot_foot_names)
            ],
            "policy_joint_names": joint_names,
            "policy_joint_position_rad": joint_positions,
            "policy_joint_velocity_rad_s": joint_velocities,
            "policy_joint_default_position_rad": default_joint_positions,
            "policy_joint_position_error_rad": [
                position - default
                for position, default in zip(joint_positions, default_joint_positions, strict=True)
            ],
            "joint_state_semantics": "state at end of control sample; contact force peak lies within the contact sensor history, whose pose samples are not exposed by ContactSensor",
        }
    return {
        "root_position_w_m": [float(value) for value in root[0].tolist()],
        "root_yaw_rad": float(yaw.item()),
        "root_lin_vel_b_mps": [float(value) for value in signals.base_lin_vel_xy[0].tolist()],
        "command_lin_vel_b_mps": [float(value) for value in signals.command_lin_vel_xy[0].tolist()],
        "base_lin_vel_b_mps": [float(value) for value in signals.base_lin_vel_xy[0].tolist()],
        "root_ang_vel_b_rps": [
            float(value) for value in unwrapped.scene["robot"].data.root_ang_vel_b[0].tolist()
        ],
        "forward_progress_m": float((root[0, 0] - origins[0, 0]).item()),
        "lateral_offset_m": float((root[0, 1] - origins[0, 1]).item()),
        "velocity_error_mps": float(
            torch.linalg.vector_norm(
                signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1
            )[0].item()
        ),
        "roll_rad": float(torch.asin(gravity[0, 1].clamp(-1.0, 1.0)).item()),
        "pitch_rad": float(torch.asin(gravity[0, 0].clamp(-1.0, 1.0)).item()),
        "foot_slip_mps": float(signals.foot_slip[0].item()),
        "foot_slip_speed_threshold_mps": foot_slip_threshold_mps,
        "foot_contact_count": contacting_foot_count,
        "foot_slipping_contact_count": slipping_contact_count,
        "foot_contact_horizontal_speed_sum_mps": contact_horizontal_speed_sum,
        "foot_horizontal_speed_mps": [float(value) for value in foot_horizontal_speed[0].tolist()],
        "foot_contact": [bool(value) for value in contacting_feet[0].tolist()],
        "foot_heights_m": [float(value) for value in foot_heights.tolist()],
        "foot_positions_w_m": (
            [[float(value) for value in row.tolist()] for row in foot_positions_w]
            if isinstance(foot_positions_w, torch.Tensor) else None
        ),
        "foot_force_n": (
            [float(value) for value in foot_force_n.tolist()]
            if isinstance(foot_force_n, torch.Tensor) else None
        ),
        "foot_horizontal_force_n": (
            [float(value) for value in foot_horizontal_force_n.tolist()]
            if isinstance(foot_horizontal_force_n, torch.Tensor) else None
        ),
        "foot_body_names": list(getattr(adapter, "robot_foot_names", ())) or None,
        "nonfoot_contact": bool(signals.nonfoot_contact[0].item() > 0.0),
        "max_nonfoot_contact_force_n": float(
            diagnostics["episode_max_nonfoot_contact_force_n"]
        ),
        "max_nonfoot_contact_body": diagnostics["episode_max_nonfoot_contact_body"],
        "sample_nonfoot_contact_peak_force_n": contact_peak_force_n,
        "sample_nonfoot_contact_peak_body": contact_peak_body,
        "contact_event_snapshot": contact_event_snapshot,
        "max_projected_gravity_xy": float(
            torch.linalg.vector_norm(state.projected_gravity_xy, dim=-1)[0].item()
        ),
        "base_height_m": float(state.base_height[0, 0].item()),
        "max_joint_speed_rad_s": float(state.joint_vel.abs().max().item()),
        "nonfoot_contact_threshold_n": float(adapter.contact_threshold),
    }


def _termination_reason_before_reset(unwrapped: Any) -> tuple[str, list[str]]:
    manager = unwrapped.termination_manager
    active = [
        name for name in manager.active_terms if bool(manager.get_term(name)[0].item())
    ]
    for term_name, reason in (
        ("bad_orientation_2", "fall"),
        ("illegal_contact", "body_contact_termination"),
        ("terrain_out_of_bounds", "lateral_out_of_bounds"),
        ("time_out", "timeout"),
    ):
        if term_name in active:
            return reason, active
    if bool(manager.time_outs[0].item()):
        return "timeout", active
    return "environment_termination", active


def _geometry_audit_for_cell(cell: dict, geometry_seed: int) -> dict[str, Any] | None:
    """Recreate the exact seeded geometry metadata used by the importer.

    The importer intentionally returns only meshes to Isaac Lab.  This helper
    calls the same deterministic geometry path before the episode loop so the
    report can distinguish audited mesh facts from candidate manifest values.
    """

    try:
        from rl_training.source_scan.terrain_config import source_subterrain_from_cell
        from rl_training.tasks.manager_based.locomotion.velocity.terrains.source_terrain_isaac import (
            generate_source_terrain_geometry,
        )

        cfg = source_subterrain_from_cell(cell, seed=geometry_seed)
        geometry = generate_source_terrain_geometry(0.0, cfg)
    except (ImportError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    regions = [
        {
            "region_id": region.region_id,
            "friction_coefficient_candidate": region.friction_coefficient_candidate,
            "bounds_xy_m": list(region.bounds_xy_m),
        }
        for region in geometry.friction_regions
    ]
    return {
        "metadata": dict(geometry.metadata),
        "friction_regions": regions,
        "geometry_object": geometry,
    }


def _summary(values: list[float], unit: str) -> dict[str, Any] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "rms": (sum(value * value for value in values) / len(values)) ** 0.5,
        "peak": max(abs(value) for value in values),
        "minimum": min(values),
        "maximum": max(values),
        "sample_count": len(values),
        "unit": unit,
    }


def _stair_step_intervals(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    """Return each tread interval in reset-relative progress coordinates."""

    origin_x = float(geometry["flat_length_m"]) / 2.0
    entry = float(geometry["flat_length_m"]) + float(geometry["transition_length_m"])
    depth = float(geometry["tread_depth_m"])
    count = int(geometry["step_count"])
    return [
        (entry + index * depth - origin_x, entry + (index + 1) * depth - origin_x)
        for index in range(count)
    ]


def _stair_riser_progresses(
    geometry: dict[str, Any], geometry_audit: dict[str, Any] | None
) -> list[float]:
    origin_x = float(geometry["flat_length_m"]) / 2.0
    metadata = geometry_audit.get("metadata", {}) if geometry_audit else {}
    steps = metadata.get("steps") if isinstance(metadata, dict) else None
    if isinstance(steps, list) and all(isinstance(item, dict) for item in steps):
        return [float(item["riser_x_m"]) - origin_x for item in steps]
    entry = float(geometry["flat_length_m"]) + float(geometry["transition_length_m"])
    depth = float(geometry["tread_depth_m"])
    return [entry + index * depth - origin_x for index in range(int(geometry["step_count"]))]


def _stair_progress_metrics(
    progress: list[float], geometry: dict[str, Any]
) -> tuple[int, list[float]]:
    intervals = _stair_step_intervals(geometry)
    max_progress = max(progress, default=0.0)
    completed = sum(1 for _start, end in intervals if max_progress >= end)
    per_step = [
        max(0.0, min(max_progress, end) - start) if max_progress >= start else 0.0
        for start, end in intervals
    ]
    return completed, per_step


def _stair_stuck_count(
    progress: list[float], base_velocity_x: list[float], geometry: dict[str, Any]
) -> int:
    intervals = _stair_step_intervals(geometry)
    count = 0
    paired = list(zip(progress, base_velocity_x, strict=False))
    for start, end in intervals:
        stuck = False
        for offset in range(max(0, len(paired) - STAIR_STUCK_DWELL_STEPS + 1)):
            window = paired[offset : offset + STAIR_STUCK_DWELL_STEPS]
            positions = [value for value, _velocity in window]
            velocities = [velocity for _value, velocity in window]
            if (
                all(start <= value <= end for value in positions)
                and all(abs(velocity) <= 0.05 for velocity in velocities)
                and max(positions) - min(positions) <= STAIR_STUCK_PROGRESS_TOLERANCE_M
            ):
                stuck = True
                break
        count += int(stuck)
    return count


def _stair_front_collision_flags(
    sample: dict[str, Any], origins: Any, geometry: dict[str, Any],
    geometry_audit: dict[str, Any] | None,
) -> list[bool] | None:
    positions = sample.get("foot_positions_w_m")
    forces = sample.get("foot_horizontal_force_n")
    names = sample.get("foot_body_names")
    if (
        not isinstance(positions, list)
        or not isinstance(forces, list)
        or not isinstance(names, list)
        or len(positions) != len(forces)
        or len(positions) != len(names)
    ):
        return None
    front_names = {"FL_FOOT", "FR_FOOT"}
    front = [index for index, name in enumerate(names) if str(name) in front_names]
    if len(front) != 2:
        return None
    risers = _stair_riser_progresses(geometry, geometry_audit)
    origin_x = float(origins[0, 0].item())
    flags = [False] * len(positions)
    for index in front:
        foot_progress = float(positions[index][0]) - origin_x
        near_riser = any(abs(foot_progress - riser) <= STAIR_RISER_PROXIMITY_M for riser in risers)
        flags[index] = near_riser and float(forces[index]) >= STAIR_FRONT_COLLISION_FORCE_THRESHOLD_N
    return flags


def _terrain_height_at_local_xy(
    x_m: float, y_m: float, geometry: dict[str, Any],
    geometry_audit: dict[str, Any] | None,
) -> float | None:
    import numpy as np

    if geometry_audit is None:
        return None
    metadata = geometry_audit.get("metadata", {})
    family = geometry.get("family")
    if family in {"flat", "friction_variation"}:
        return 0.0
    if family == "multi_step_stairs":
        steps = metadata.get("steps") if isinstance(metadata, dict) else None
        if not isinstance(steps, list):
            return None
        level = float(metadata.get("initial_surface_height_m", 0.0))
        for step in steps:
            if x_m < float(step["riser_x_m"]):
                return level
            level = float(step["height_after_m"])
            if x_m < float(step["tread_end_x_m"]):
                return level
        return level

    geometry_object = geometry_audit.get("geometry_object")
    meshes = getattr(geometry_object, "meshes", None)
    if not meshes:
        return None
    vertices = np.asarray(meshes[0].vertices, dtype=np.float64)
    mesh_x = np.unique(vertices[:, 0])
    mesh_y = np.unique(vertices[:, 1])
    rows, columns = len(mesh_y), len(mesh_x)
    if rows < 2 or columns < 2 or rows * columns > len(vertices):
        return None
    top = vertices[: rows * columns]
    # The source heightfield builder emits top vertices in [y, x] order,
    # followed by the bottom prism. Guard the assumption before interpolation.
    if not np.allclose(top[:, 0], np.tile(mesh_x, rows)) or not np.allclose(
        top[:, 1], np.repeat(mesh_y, columns)
    ):
        return None
    heights = top[:, 2].reshape(rows, columns)
    if not (mesh_x[0] <= x_m <= mesh_x[-1] and mesh_y[0] <= y_m <= mesh_y[-1]):
        return None
    ix = min(max(int(np.searchsorted(mesh_x, x_m, side="right") - 1), 0), columns - 2)
    iy = min(max(int(np.searchsorted(mesh_y, y_m, side="right") - 1), 0), rows - 2)
    dx = (x_m - mesh_x[ix]) / (mesh_x[ix + 1] - mesh_x[ix])
    dy = (y_m - mesh_y[iy]) / (mesh_y[iy + 1] - mesh_y[iy])
    z00, z10 = heights[iy, ix], heights[iy, ix + 1]
    z01, z11 = heights[iy + 1, ix], heights[iy + 1, ix + 1]
    if dx + dy <= 1.0:
        return float(z00 + dx * (z10 - z00) + dy * (z01 - z00))
    return float((1.0 - dy) * z10 + (dx + dy - 1.0) * z11 + (1.0 - dx) * z01)


def _foot_ground_heights(
    sample: dict[str, Any], origins: Any, geometry: dict[str, Any],
    geometry_audit: dict[str, Any] | None,
) -> list[float | None] | None:
    positions = sample.get("foot_positions_w_m")
    if not isinstance(positions, list):
        return None
    flat_center = float(geometry["flat_length_m"]) / 2.0
    track_width = float(geometry["size"][1])
    result = []
    for position in positions:
        local_x = float(position[0]) - float(origins[0, 0].item()) + flat_center
        local_y = float(position[1]) - float(origins[0, 1].item()) + track_width / 2.0
        result.append(_terrain_height_at_local_xy(local_x, local_y, geometry, geometry_audit))
    return result


def _friction_region_for_foot(
    position_w: list[float], origins: Any, geometry: dict[str, Any],
    regions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(position_w) < 2:
        return None
    local_x = float(position_w[0]) - float(origins[0, 0].item()) + float(geometry["flat_length_m"]) / 2.0
    local_y = float(position_w[1]) - float(origins[0, 1].item()) + float(geometry["size"][1]) / 2.0
    for region in regions:
        x0, x1, y0, y1 = (float(value) for value in region["bounds_xy_m"])
        if x0 <= local_x <= x1 and y0 <= local_y <= y1:
            return region
    return None


def _update_friction_observations(
    observations: dict[str, dict[str, Any]], sample: dict[str, Any], origins: Any,
    geometry: dict[str, Any], geometry_audit: dict[str, Any] | None,
) -> bool:
    regions = geometry_audit.get("friction_regions", []) if geometry_audit else []
    positions = sample.get("foot_positions_w_m")
    contacts = sample.get("foot_contact")
    speeds = sample.get("foot_horizontal_speed_mps")
    if not regions or not isinstance(positions, list) or not isinstance(contacts, list) or not isinstance(speeds, list):
        return False
    for index, (position, contact, speed) in enumerate(zip(positions, contacts, speeds, strict=False)):
        if not contact:
            continue
        region = _friction_region_for_foot(position, origins, geometry, regions)
        if region is None:
            continue
        key = str(region["region_id"])
        foot_names = sample.get("foot_body_names")
        foot_name = (
            str(foot_names[index])
            if isinstance(foot_names, list) and index < len(foot_names)
            else f"foot_{index}"
        )
        entry = observations.setdefault(
            key,
            {
                "region_id": int(region["region_id"]),
                "candidate_friction_coefficient": float(region["friction_coefficient_candidate"]),
                "contact_samples": 0,
                "slipping_contact_samples": 0,
                "horizontal_speed_sum_mps": 0.0,
                "feet": {},
            },
        )
        entry["contact_samples"] += 1
        entry["slipping_contact_samples"] += int(float(speed) > FOOT_SLIP_SPEED_THRESHOLD_MPS)
        entry["horizontal_speed_sum_mps"] += float(speed)
        foot_entry = entry["feet"].setdefault(
            foot_name,
            {"foot_index": index, "contact_samples": 0, "slipping_contact_samples": 0,
             "horizontal_speed_sum_mps": 0.0},
        )
        foot_entry["contact_samples"] += 1
        foot_entry["slipping_contact_samples"] += int(float(speed) > FOOT_SLIP_SPEED_THRESHOLD_MPS)
        foot_entry["horizontal_speed_sum_mps"] += float(speed)
    return True


def _finalize_friction_observations(observations: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not observations:
        return None
    result: dict[str, Any] = {}
    for key in sorted(observations, key=int):
        entry = observations[key]
        contacts = entry["contact_samples"]
        result[key] = {
            "region_id": entry["region_id"],
            "candidate_friction_coefficient": entry["candidate_friction_coefficient"],
            "contact_samples": contacts,
            "slipping_contact_samples": entry["slipping_contact_samples"],
            "slip_contact_fraction": (
                entry["slipping_contact_samples"] / contacts if contacts else None
            ),
            "mean_horizontal_speed_mps": entry["horizontal_speed_sum_mps"] / max(contacts, 1),
            "by_foot": {
                foot_name: {
                    "foot_index": foot["foot_index"],
                    "contact_samples": foot["contact_samples"],
                    "slipping_contact_samples": foot["slipping_contact_samples"],
                    "slip_contact_fraction": (
                        foot["slipping_contact_samples"] / foot["contact_samples"]
                        if foot["contact_samples"] else None
                    ),
                    "mean_horizontal_speed_mps": (
                        foot["horizontal_speed_sum_mps"] / max(foot["contact_samples"], 1)
                    ),
                }
                for foot_name, foot in sorted(entry["feet"].items())
            },
            "slip_threshold_mps": FOOT_SLIP_SPEED_THRESHOLD_MPS,
        }
    return {
        "regions": result,
        "region_assignment": "foot_world_xy mapped to generated region bounds",
        "effective_contact_friction_measured": False,
    }


def _run_episode(wrapped_env: Any, policy: Any, cell: dict, geometry_seed: int,
                 initial_state_seed: int, episode_index: int, *,
                 max_steps: int = MAX_STEPS, bounded_smoke: bool = False,
                 termination_capture_test: bool = False,
                 pre_reset_observer: Any | None = None,
                 sample_observer: Any | None = None) -> dict[str, Any]:
    import torch

    from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
    from rl_training.source_scan.bridge import FrozenPolicySafetyBridge

    unwrapped = wrapped_env.unwrapped
    device = torch.device(unwrapped.device)
    adapter = Lite3IsaacLabAdapter(unwrapped, safety_height_source="base_scan")
    safety = _new_safety_filter(device)
    bridge = FrozenPolicySafetyBridge(policy, adapter, safety)
    # RslRlVecEnvWrapper.reset() has no seed argument; seed before resetting.
    observation, _ = _seeded_wrapper_reset(wrapped_env, initial_state_seed)
    adapter.reset_contact_diagnostics()
    origins = unwrapped.scene.terrain.env_origins
    if not isinstance(origins, torch.Tensor) or origins.shape != (1, 3):
        raise RuntimeError("Source Terrain importer did not expose one terrain origin")
    terrain_size = cell["resolved_parameters"]["geometry"]["size"]
    track = cell["resolved_parameters"]["track"]
    geometry = cell["resolved_parameters"]["geometry"]
    geometry_for_metrics = {**geometry, "family": cell["terrain_family"]}
    geometry_audit = _geometry_audit_for_cell(cell, geometry_seed)
    goal_progress = track["completion_x_m"] - track["start_x_m"]
    lateral_limit = terrain_size[1] / 2.0 - track["lateral_margin_m"]
    control_dt = CONTROL_DT
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_STEPS:
        raise ValueError(f"max_steps must be an integer in [1, {MAX_STEPS}]")
    if termination_capture_test and not bounded_smoke:
        raise ValueError("termination_capture_test is available only in a bounded smoke")
    stable_hold_steps = STABLE_HOLD_STEPS
    hold_count = 0
    completed = False
    termination_reason = "timeout"
    velocity_error: list[float] = []
    base_velocity_x: list[float] = []
    command_velocity_x: list[float] = []
    roll: list[float] = []
    pitch: list[float] = []
    lateral: list[float] = []
    progress: list[float] = []
    foot_contact_steps = 0
    foot_contact_count = 0
    foot_contact_speed_sum = 0.0
    foot_slipping_contact_count = 0
    previous_foot_contact: list[bool] | None = None
    footfall_ground_heights: list[float] = []
    swing_clearances: list[float] = []
    contact_sole_offsets: list[list[float]] = [[] for _ in range(4)]
    nonfoot_contact_threshold_n = float(adapter.contact_threshold)
    nonfoot_contact_steps = 0
    nonfoot_contact_count = 0
    previous_nonfoot_contact = False
    projected_steps = 0
    projection_reasons: dict[str, int] = {}
    invalid_state_count = 0
    safety_stopped = False
    action_compute_latency: list[float] = []
    action_compute_execute_latency: list[float] = []
    action_trace: list[dict[str, Any]] = []
    root_position_history: list[list[float]] = []
    root_yaw_history: list[float] = []
    max_nonfoot_contact_force_n = 0.0
    max_nonfoot_contact_body = None
    min_base_height = float("inf")
    max_tilt = 0.0
    max_joint_speed = 0.0
    stair_front_collision_count: int | None = 0 if cell["terrain_family"] == "multi_step_stairs" else None
    stair_collision_measurement_available = cell["terrain_family"] != "multi_step_stairs"
    previous_stair_collision_flags: list[bool] | None = None
    friction_observations: dict[str, dict[str, Any]] = {}
    friction_measurement_available = cell["terrain_family"] != "friction_variation"
    steps = 0
    active_step = False
    terminal_snapshot: dict[str, Any] = {}
    post_reset_root_position: list[float] | None = None

    def capture_terminal_state(env_ids) -> None:
        if not active_step:
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)
        if ids.tolist() != [0]:
            raise RuntimeError(f"single-env scan received unexpected pre-reset ids: {ids.tolist()}")
        reason, active_terms = _termination_reason_before_reset(unwrapped)
        terminal_snapshot.update(
            sample=_runtime_sample(adapter, unwrapped, origins),
            termination_reason=reason,
            active_termination_terms=active_terms,
        )
        if pre_reset_observer is not None:
            pre_reset_observer(env_ids, steps + 1)

    callback_adder = getattr(unwrapped, "add_pre_reset_callback", None)
    if not callable(callback_adder):
        raise RuntimeError("Source scan environment must support pre-reset state capture")
    # Lite3IsaacLabAdapter registered its contact snapshot callback first.
    callback_adder(capture_terminal_state)

    for _ in range(max_steps):
        timed_out = torch.zeros((1, 1), dtype=torch.bool, device=device)
        state = adapter.safety_state(timed_out)
        tilt = torch.linalg.vector_norm(state.projected_gravity_xy, dim=-1)
        current_height = state.base_height[:, 0]
        current_lateral = unwrapped.scene["robot"].data.root_pos_w[:, 1] - origins[:, 1]
        current_progress = unwrapped.scene["robot"].data.root_pos_w[:, 0] - origins[:, 0]
        max_tilt = max(max_tilt, float(tilt.max().item()))
        min_base_height = min(min_base_height, float(current_height.min().item()))
        max_joint_speed = max(max_joint_speed, float(state.joint_vel.abs().max().item()))
        progress_value = float(current_progress[0].item())
        lateral_value = float(current_lateral[0].item())
        if abs(lateral_value) > lateral_limit:
            termination_reason = "lateral_out_of_bounds"
            progress.append(progress_value)
            lateral.append(lateral_value)
            break
        if not all(torch.isfinite(value).all().item() for value in (
            state.projected_gravity_xy, state.base_height, state.joint_pos, state.joint_vel,
            state.joint_lower, state.joint_upper, state.default_joint_pos, state.action_scale,
        )):
            invalid_state_count += 1
            termination_reason = "safety_termination"
            break
        action_started = perf_counter()
        try:
            trace = bridge.step(observation, timed_out)
        except FloatingPointError:
            invalid_state_count += 1
            termination_reason = "safety_termination"
            break
        if bool(trace.decision.paused.any().item()):
            safety_stopped = True
            termination_reason = "safety_termination"
            progress.append(progress_value)
            lateral.append(lateral_value)
            action_trace.append({
                "step": steps + 1,
                "a_policy_raw": trace.raw_policy_action[0].cpu().tolist(),
                "a_base": trace.base_action[0].cpu().tolist(),
                "requested_delta": [0.0] * 12,
                "a_requested": trace.requested_action[0].cpu().tolist(),
                "a_filter_output_not_executed": trace.executed_action[0].cpu().tolist(),
                "filter_state_before": {
                    name: value[0].cpu().tolist()
                    for name, value in trace.filter_state_before.items()
                },
                "projected": bool(trace.decision.projected[0].item()),
                "projection_reasons": list(trace.decision.reasons),
                "safety_paused": True,
                "policy_compute_latency_ms": trace.action_latency_ms,
                "compute_and_execute_latency_ms": None,
            })
            for reason in trace.decision.reasons:
                projection_reasons[reason] = projection_reasons.get(reason, 0) + 1
            break

        action_compute_latency.append(trace.action_latency_ms)
        projected = bool(trace.decision.projected.any().item())
        projected_steps += int(projected)
        for reason in trace.decision.reasons:
            projection_reasons[reason] = projection_reasons.get(reason, 0) + 1
        terminal_snapshot.clear()
        active_step = True
        try:
            observation, _, done, _extras = wrapped_env.step(trace.executed_action)
        finally:
            active_step = False
        end_to_end_latency = (perf_counter() - action_started) * 1000.0
        action_compute_execute_latency.append(end_to_end_latency)
        steps += 1
        done_bool = bool(done.reshape(-1)[0].item())
        if done_bool:
            if "sample" not in terminal_snapshot:
                raise RuntimeError("environment ended without a pre-reset terminal snapshot")
            post_reset_root_position = [
                float(value)
                for value in unwrapped.scene["robot"].data.root_pos_w[0].tolist()
            ]
            if termination_capture_test and "time_out" not in terminal_snapshot.get(
                "active_termination_terms", []
            ):
                raise RuntimeError("termination capture test ended without its injected manager timeout")
            sample = terminal_snapshot["sample"]
            termination_reason = terminal_snapshot["termination_reason"]
        else:
            sample = _runtime_sample(adapter, unwrapped, origins)
        if sample_observer is not None:
            sample_observer(steps, sample)
        if sample.get("invalid"):
            invalid_state_count += 1
            termination_reason = "safety_termination"
            safety_stopped = True
            break
        progress_value = sample["forward_progress_m"]
        lateral_value = sample["lateral_offset_m"]
        root_position_history.append(sample["root_position_w_m"])
        root_yaw_history.append(sample["root_yaw_rad"])
        progress.append(progress_value)
        lateral.append(lateral_value)
        velocity_error.append(sample["velocity_error_mps"])
        base_velocity_x.append(sample["base_lin_vel_b_mps"][0])
        command_velocity_x.append(sample["command_lin_vel_b_mps"][0])
        roll.append(sample["roll_rad"])
        pitch.append(sample["pitch_rad"])
        foot_contact_steps += int(sample["foot_contact_count"] > 0)
        foot_contact_count += sample["foot_contact_count"]
        foot_slipping_contact_count += sample["foot_slipping_contact_count"]
        foot_contact_speed_sum += sample["foot_contact_horizontal_speed_sum_mps"]
        current_foot_contact = sample.get("foot_contact")
        current_foot_heights = sample.get("foot_heights_m")
        new_contact_indices: list[int] = []
        if isinstance(current_foot_contact, list) and isinstance(current_foot_heights, list):
            if previous_foot_contact is None:
                previous_foot_contact = [False] * len(current_foot_contact)
            new_contact_indices = [
                index for index, contact in enumerate(current_foot_contact)
                if contact and not previous_foot_contact[index]
            ]
            previous_foot_contact = [bool(value) for value in current_foot_contact]
        ground_heights = _foot_ground_heights(
            sample, origins, geometry_for_metrics, geometry_audit
        )
        if (
            isinstance(current_foot_contact, list)
            and isinstance(current_foot_heights, list)
            and isinstance(ground_heights, list)
            and len(current_foot_contact) == len(current_foot_heights) == len(ground_heights)
            and all(value is not None for value in ground_heights)
        ):
            for index in new_contact_indices:
                ground_height = ground_heights[index]
                if ground_height is not None:
                    footfall_ground_heights.append(float(ground_height))
            for index, (contact, foot_height, ground_height) in enumerate(
                zip(current_foot_contact, current_foot_heights, ground_heights, strict=True)
            ):
                if index >= len(contact_sole_offsets) or ground_height is None:
                    continue
                height_above_surface = float(foot_height) - float(ground_height)
                if contact:
                    contact_sole_offsets[index].append(height_above_surface)
                elif contact_sole_offsets[index]:
                    offsets = sorted(contact_sole_offsets[index])
                    sole_offset = offsets[len(offsets) // 2]
                    swing_clearances.append(height_above_surface - sole_offset)
        if cell["terrain_family"] == "multi_step_stairs":
            collision_flags = _stair_front_collision_flags(
                sample, origins, geometry, geometry_audit
            )
            if collision_flags is None:
                stair_collision_measurement_available = False
            else:
                stair_collision_measurement_available = True
                if previous_stair_collision_flags is not None:
                    stair_front_collision_count += sum(
                        int(current and not previous)
                        for current, previous in zip(
                            collision_flags, previous_stair_collision_flags, strict=False
                        )
                    )
                previous_stair_collision_flags = collision_flags
        if cell["terrain_family"] == "friction_variation":
            friction_measurement_available = _update_friction_observations(
                friction_observations, sample, origins, geometry, geometry_audit
            )
        nonfoot_contact_threshold_n = sample["nonfoot_contact_threshold_n"]
        nonfoot = sample["nonfoot_contact"]
        nonfoot_contact_count += int(nonfoot and not previous_nonfoot_contact)
        nonfoot_contact_steps += int(nonfoot)
        previous_nonfoot_contact = nonfoot
        max_nonfoot_contact_force_n = max(
            max_nonfoot_contact_force_n, sample["max_nonfoot_contact_force_n"]
        )
        if sample["max_nonfoot_contact_force_n"] >= max_nonfoot_contact_force_n:
            max_nonfoot_contact_body = sample["max_nonfoot_contact_body"]
        max_tilt = max(max_tilt, sample["max_projected_gravity_xy"])
        min_base_height = min(min_base_height, sample["base_height_m"])
        max_joint_speed = max(max_joint_speed, sample["max_joint_speed_rad_s"])
        action_trace.append({
            "step": steps,
            "a_policy_raw": trace.raw_policy_action[0].cpu().tolist(),
            "a_base": trace.base_action[0].cpu().tolist(),
            "requested_delta": [0.0] * 12,
            "a_requested": trace.requested_action[0].cpu().tolist(),
            "a_executed_command": trace.executed_action[0].cpu().tolist(),
            "filter_state_before": {
                name: value[0].cpu().tolist()
                for name, value in trace.filter_state_before.items()
            },
            "projected": projected,
            "projection_reasons": list(trace.decision.reasons),
            "safety_paused": False,
            "policy_compute_latency_ms": trace.action_latency_ms,
            "compute_and_execute_latency_ms": end_to_end_latency,
        })

        if done_bool:
            break
        if abs(lateral_value) > lateral_limit:
            termination_reason = "lateral_out_of_bounds"
            break

        stable = (
            sample["max_projected_gravity_xy"] <= safety.limits.max_projected_gravity_xy
            and sample["base_height_m"] >= safety.limits.min_base_height
            and sample["max_joint_speed_rad_s"] <= safety.limits.max_joint_velocity
            and not nonfoot
        )
        hold_count = hold_count + 1 if progress_value >= goal_progress and stable else 0
        if hold_count >= stable_hold_steps:
            completed = True
            termination_reason = "completion"
            break

    else:
        termination_reason = "smoke_step_limit" if bounded_smoke else "timeout"

    final_progress = float(progress[-1]) if progress else float(current_progress[0].item())
    max_progress = max(progress, default=0.0)
    adjacent_footfall_differences = [
        abs(current - previous)
        for previous, current in zip(footfall_ground_heights, footfall_ground_heights[1:], strict=False)
    ]
    foot_slip_fraction = (
        foot_slipping_contact_count / foot_contact_count if foot_contact_count else None
    )
    foot_slip_status = "measured" if foot_contact_count else "unavailable_no_contact_samples"
    finalized_friction_observations = _finalize_friction_observations(friction_observations)
    return {
        "cell_id": cell["cell_id"],
        "terrain_family": cell["terrain_family"],
        "level": cell["level"],
        "direction_or_layout": cell["direction_or_layout"],
        "geometry_seed": geometry_seed,
        "initial_state_seed": initial_state_seed,
        "episode_index": episode_index,
        "completed": completed,
        "termination_reason": termination_reason,
        "episode_type": (
            "termination_capture_test"
            if termination_capture_test
            else "bounded_policy_smoke" if bounded_smoke else "source_qualification_scan"
        ),
        "termination_capture_test_injected": termination_capture_test,
        "steps": steps,
        "forward_progress_m": final_progress,
        "max_forward_progress_m": max_progress,
        "terrain_origin_w_m": [float(value) for value in origins[0].tolist()],
        "root_position_w_start_m": root_position_history[0] if root_position_history else None,
        "root_position_w_end_m": root_position_history[-1] if root_position_history else None,
        "post_reset_root_position_w_m": post_reset_root_position,
        "pre_reset_capture_observed": "sample" in terminal_snapshot,
        "pre_reset_state_differs_from_post_reset": (
            max(
                abs(float(pre) - float(post))
                for pre, post in zip(
                    terminal_snapshot["sample"]["root_position_w_m"],
                    post_reset_root_position,
                    strict=True,
                )
            ) > 1.0e-6
            if post_reset_root_position is not None and "sample" in terminal_snapshot
            else None
        ),
        "root_yaw_start_rad": root_yaw_history[0] if root_yaw_history else None,
        "root_yaw_end_rad": root_yaw_history[-1] if root_yaw_history else None,
        "mean_velocity_error_mps": sum(velocity_error) / max(len(velocity_error), 1),
        "rms_velocity_error_mps": (
            sum(value * value for value in velocity_error) / max(len(velocity_error), 1)
        ) ** 0.5,
        "mean_base_velocity_x_mps": sum(base_velocity_x) / max(len(base_velocity_x), 1),
        "mean_command_velocity_x_mps": sum(command_velocity_x) / max(len(command_velocity_x), 1),
        "roll_mean_rad": sum(roll) / max(len(roll), 1),
        "roll_rms_rad": (sum(value * value for value in roll) / max(len(roll), 1)) ** 0.5,
        "roll_peak_rad": max((abs(value) for value in roll), default=0.0),
        "pitch_mean_rad": sum(pitch) / max(len(pitch), 1),
        "pitch_rms_rad": (sum(value * value for value in pitch) / max(len(pitch), 1)) ** 0.5,
        "pitch_peak_rad": max((abs(value) for value in pitch), default=0.0),
        "lateral_offset_mean_m": sum(abs(value) for value in lateral) / max(len(lateral), 1),
        "lateral_offset_peak_m": max((abs(value) for value in lateral), default=0.0),
        "nonfoot_contact_count": nonfoot_contact_count,
        "nonfoot_contact_duration_s": nonfoot_contact_steps * control_dt,
        "nonfoot_contact_max_force_n": max_nonfoot_contact_force_n,
        "nonfoot_contact_max_force_body": max_nonfoot_contact_body,
        "foot_slip_mean_mps": foot_contact_speed_sum / max(foot_contact_count, 1),
        "foot_slip_contact_fraction": foot_slip_fraction,
        "foot_slip_contact_fraction_status": foot_slip_status,
        "foot_slip_speed_threshold_mps": FOOT_SLIP_SPEED_THRESHOLD_MPS,
        "foot_slip_contact_fraction_numerator": foot_slipping_contact_count,
        "foot_slip_contact_fraction_denominator": foot_contact_count,
        "foot_slip_contact_fraction_semantics": (
            "contacting-foot samples whose world-horizontal foot-speed proxy exceeds "
            f"{FOOT_SLIP_SPEED_THRESHOLD_MPS:.2f} m/s divided by all contacting-foot samples"
        ),
        "foot_contact_step_fraction": foot_contact_steps / max(steps, 1),
        "foot_contact_steps": foot_contact_steps,
        "contacting_foot_samples": foot_contact_count,
        "nonfoot_contact_detection_threshold_n": nonfoot_contact_threshold_n,
        "safety_filter_projection_fraction": projected_steps / max(steps, 1),
        "safety_filter_projection_reasons": projection_reasons,
        "safety_filter_rejection_or_emergency_count": int(safety_stopped),
        "action_compute_latency_ms_mean": sum(action_compute_latency) / max(len(action_compute_latency), 1),
        "action_compute_and_execute_latency_ms_mean": (
            sum(action_compute_execute_latency) / max(len(action_compute_execute_latency), 1)
        ),
        "action_compute_and_execute_latency_ms_peak": max(action_compute_execute_latency, default=0.0),
        "invalid_state_count": invalid_state_count,
        "max_projected_gravity_xy": max_tilt,
        "min_base_height_m": min_base_height if min_base_height != float("inf") else None,
        "base_height_semantics": "vertical root clearance above highest of nine base-scan hits; invalid scan yields zero and safety stop",
        "max_joint_speed_rad_s": max_joint_speed,
        "termination_terms": terminal_snapshot.get("active_termination_terms", []),
        "action_trace": action_trace,
        "family_metrics": _family_metrics(
            cell,
            progress,
            base_velocity_x=base_velocity_x,
            roll=roll,
            pitch=pitch,
            geometry_audit=geometry_audit,
            adjacent_footfall_differences=adjacent_footfall_differences,
            swing_clearances=swing_clearances,
            stair_front_collision_count=(
                stair_front_collision_count if stair_collision_measurement_available else None
            ),
            friction_observations=(
                finalized_friction_observations if friction_measurement_available else None
            ),
        ),
    }


def _family_metrics(
    cell: dict,
    progress: list[float],
    *,
    base_velocity_x: list[float] | None = None,
    roll: list[float] | None = None,
    pitch: list[float] | None = None,
    geometry_audit: dict[str, Any] | None = None,
    adjacent_footfall_differences: list[float] | None = None,
    swing_clearances: list[float] | None = None,
    stair_front_collision_count: int | None = None,
    friction_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report family metrics with explicit measurement and unavailable states."""

    family = cell["terrain_family"]
    geometry = cell["resolved_parameters"]["geometry"]
    track = cell["resolved_parameters"]["track"]
    result: dict[str, Any] = {}
    unavailable: list[str] = []
    if family in {"slope", "slope_plus_rough"}:
        transition_start = float(geometry["flat_length_m"]) / 2.0
        transition_length = float(geometry["transition_length_m"])
        exit_flat_start = float(geometry["size"][0]) - float(geometry["exit_flat_length_m"])
        exit_transition_start = exit_flat_start - transition_length
        reset_x = float(track["start_x_m"])
        result.update(
            direction=cell["direction_or_layout"],
            slope_angle_deg=geometry["slope_angle_deg"],
            pitch_response=_summary(pitch or [], "rad"),
            roll_response=_summary(roll or [], "rad"),
            transition_progress_m=[
                max(0.0, min(float(value) - transition_start, transition_length))
                for value in progress
            ],
            exit_transition_progress_m=[
                max(0.0, min(float(value) - (exit_transition_start - reset_x), transition_length))
                for value in progress
            ],
            terminal_flat_progress_m=[
                max(0.0, min(float(value) - (exit_flat_start - reset_x),
                             float(geometry["exit_flat_length_m"])))
                for value in progress
            ],
            transition_progress_semantics=(
                "reset-relative root progress clipped to entry transition; separate exit-transition "
                "and terminal-flat progress arrays use their registered local-x intervals"
            ),
        )
        if result["pitch_response"] is None:
            unavailable.append("pitch_response")
        if result["roll_response"] is None:
            unavailable.append("roll_response")

    if family == "multi_step_stairs":
        completed_index, per_step_progress = _stair_progress_metrics(progress, geometry)
        velocity = base_velocity_x or []
        swing = _summary(swing_clearances or [], "m")
        result.update(
            direction=cell["direction_or_layout"],
            step_count=geometry["step_count"],
            completed_step_index=completed_index,
            completed_step_index_semantics="number of registered treads whose exit boundary was crossed; 0 means none",
            per_step_progress_m=per_step_progress,
            per_step_progress_semantics="maximum reset-relative progress inside each registered tread interval, clipped to tread length",
            front_foot_collision_count=stair_front_collision_count,
            front_foot_collision_threshold_n=STAIR_FRONT_COLLISION_FORCE_THRESHOLD_N,
            front_foot_collision_semantics="rising edge of horizontal front-foot contact force near a registered riser",
            stuck_or_catch_step_count=_stair_stuck_count(progress, velocity, geometry),
            stuck_or_catch_semantics=(
                f"step interval with at least {STAIR_STUCK_DWELL_STEPS} low-speed samples "
                f"within {STAIR_STUCK_PROGRESS_TOLERANCE_M:.2f} m"
            ),
            swing_foot_clearance_m=swing,
            swing_foot_clearance_semantics="minimum/summary foot height above the generated surface for non-contact samples",
        )
        if stair_front_collision_count is None:
            unavailable.append("front_foot_collision_count")
        if swing is None:
            unavailable.append("swing_foot_clearance_m")

    if family == "friction_variation":
        geometry_audit_metadata = geometry_audit.get("metadata", {}) if geometry_audit else {}
        result.update(
            region_layout=cell["direction_or_layout"],
            material_parameters={
                "candidate_coefficient_range": geometry["friction_range"],
                "region_count": geometry["friction_region_count"],
                "effective_contact_friction_readback": False,
                "effective_contact_friction_status": "unavailable_without_contact_material_readback",
                "importer_material_binding_status": geometry_audit_metadata.get(
                    "material_binding_status", "unknown"
                ),
            },
            actual_contact_friction=None,
            slip_by_foot_and_region=friction_observations,
            slip_by_foot_and_region_semantics=(
                "contacting-foot samples assigned by generated region bounds; candidate coefficient is not treated as measured contact friction"
            ),
        )
        unavailable.append("actual_contact_friction")
        if friction_observations is None:
            unavailable.append("slip_by_foot_and_region")

    if family in {"multi_scale_rough", "slope_plus_rough"}:
        metadata = geometry_audit.get("metadata", {}) if geometry_audit else {}
        audited = None
        scales = None
        if metadata:
            audited = {
                "window_m": metadata.get("rough_audit_window_m"),
                "max_m": metadata.get("sampled_local_peak_valley_max_m"),
                "p50_m": metadata.get("sampled_local_peak_valley_p50_m"),
                "p95_m": metadata.get("sampled_local_peak_valley_p95_m"),
                "semantics": metadata.get("local_audit_semantics"),
            }
            scales = {
                "registered_scales_m": metadata.get("rough_spatial_scales_m"),
                "band_rms_before_scaling": metadata.get("rough_band_rms_before_scaling"),
                "semantics": "seeded geometry bands before requested peak-valley scaling",
            }
        adjacent = _summary(adjacent_footfall_differences or [], "m")
        result.update(
            audited_local_peak_valley_m=audited,
            spatial_scale_statistics=scales,
            adjacent_footfall_height_difference_m=adjacent,
            geometry_candidate_peak_valley_m=geometry["rough_peak_valley_m"],
            geometry_candidate_spatial_scales_m=geometry["spatial_scales_m"],
        )
        if audited is None:
            unavailable.append("audited_local_peak_valley_m")
        if scales is None:
            unavailable.append("spatial_scale_statistics")
        if adjacent is None:
            unavailable.append("adjacent_footfall_height_difference_m")

    if unavailable:
        result["unavailable_metrics"] = unavailable
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error("--output is required")
    try:
        preflight = validate_manifest_file(args.manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not preflight.execution_ready:
        parser.error("manifest preflight did not produce an execution-ready report")
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    started_at = _utc_now()
    try:
        import gymnasium as gym
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from rsl_rl.runners import OnPolicyRunner

        import rl_training.tasks  # noqa: F401
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.rsl_rl_compat import (
            freeze_rsl_rl_models,
            load_frozen_rsl_rl_checkpoint,
        )

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        checkpoint = Path(manifest["base_checkpoint"]["path"])
        if checkpoint.resolve() != DEFAULT_CHECKPOINT.resolve():
            raise RuntimeError("runner is restricted to the registered model_1998.pt checkpoint")

        records: list[dict[str, Any]] = []
        policy = None
        actor = None
        actor_digest_before = None
        runner = None
        for cell in manifest["seed_plan"]["cells"]:
            geometry_lo, geometry_hi = cell["geometry_seed_range"]
            initial_lo, initial_hi = cell["initial_state_seed_range"]
            geometry_seeds = list(range(geometry_lo, geometry_hi + 1))
            initial_seeds = list(range(initial_lo, initial_hi + 1))
            if len(geometry_seeds) != len(initial_seeds) or len(geometry_seeds) != cell["episodes"]:
                raise RuntimeError(f"invalid paired seed list for {cell['cell_id']}")
            for episode_index, (geometry_seed, initial_seed) in enumerate(
                zip(geometry_seeds, initial_seeds, strict=True), start=1
            ):
                wrapped_env, agent_cfg = _make_wrapped_env(
                    cell, geometry_seed, initial_seed, device=args.device
                )
                try:
                    if runner is None:
                        runner = OnPolicyRunner(
                            wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device
                        )
                        load_frozen_rsl_rl_checkpoint(runner, str(checkpoint))
                        modules = freeze_rsl_rl_models(runner)
                        actor = getattr(runner.alg, "actor", None) or getattr(runner.alg, "policy", None)
                        if actor is None:
                            actor = modules[0]
                        actor_digest_before = _tensor_sha256(actor)
                        policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
                    record = _run_episode(
                        wrapped_env,
                        policy,
                        cell,
                        geometry_seed,
                        initial_seed,
                        episode_index,
                    )
                    records.append(record)
                finally:
                    wrapped_env.close()
            print(f"[source-scan] completed {cell['cell_id']}", flush=True)

        if actor is None or actor_digest_before is None:
            raise RuntimeError("no scan policy was initialized")
        actor_digest_after = _tensor_sha256(actor)
        if actor_digest_after != actor_digest_before:
            raise RuntimeError("frozen policy parameters changed during the read-only scan")
        checkpoint_digest = _sha256(checkpoint)
        if checkpoint_digest != manifest["base_checkpoint"]["sha256"]:
            raise RuntimeError("base checkpoint changed while scan was running")
        unavailable_family_metrics = {
            record["cell_id"]: record["family_metrics"]["unavailable_metrics"]
            for record in records
            if record["family_metrics"].get("unavailable_metrics")
        }
        unavailable_common_metrics = sorted({
            "foot_slip_contact_fraction"
            for record in records
            if record.get("foot_slip_contact_fraction_status") != "measured"
        })
        report = {
            "schema_version": "lite3_source_scan_results_v1",
            "manifest_id": manifest["manifest_id"],
            "split": manifest["split"],
            "status": "complete_read_only_scan_diagnostic_only",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "checkpoint_sha256": checkpoint_digest,
            "actor_state_sha256_before": actor_digest_before,
            "actor_state_sha256_after": actor_digest_after,
            "policy_updated": False,
            "sealed_accessed": False,
            "metric_completeness": {
                "complete": False,
                "missing_common_metrics": [],
                "unavailable_common_metrics": unavailable_common_metrics,
                "unavailable_family_metrics": unavailable_family_metrics,
                "qualification_claim": False,
            },
            "cell_count": len(manifest["seed_plan"]["cells"]),
            "episode_count": len(records),
            "records": records,
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        if temporary.exists():
            raise RuntimeError(f"refusing to overwrite existing temporary output: {temporary}")
        temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_path)
        print(json.dumps({"output": str(output_path), "episodes": len(records)}, indent=2))
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
