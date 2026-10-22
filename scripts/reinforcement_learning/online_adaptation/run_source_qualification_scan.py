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


def _build_env_cfg(cell: dict, geometry_seed: int, initial_state_seed: int, *, device: str):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import rl_training.tasks  # noqa: F401
    from rl_training.source_scan.terrain_config import source_subterrain_from_cell

    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.seed = initial_state_seed
    env_cfg.scene.num_envs = 1
    generator = env_cfg.scene.terrain.terrain_generator
    generator.seed = geometry_seed
    generator.curriculum = True
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


def _make_wrapped_env(cell: dict, geometry_seed: int, initial_state_seed: int, *, device: str):
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils import load_cfg_from_registry
    from rl_training.online_adaptation.rsl_rl_compat import prepare_rsl_rl_agent_cfg

    env_cfg = _build_env_cfg(cell, geometry_seed, initial_state_seed, device=device)
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


def _runtime_sample(adapter: Any, unwrapped: Any, origins: Any) -> dict[str, Any]:
    """Copy metrics while the requested (possibly terminal) state is current."""
    import torch

    signals = adapter.reward_signals()
    state = adapter.safety_state(
        torch.zeros((1, 1), dtype=torch.bool, device=unwrapped.device)
    )
    root = unwrapped.scene["robot"].data.root_pos_w
    root_quat = unwrapped.scene["robot"].data.root_quat_w
    gravity = unwrapped.scene["robot"].data.projected_gravity_b
    foot_velocity_xy = adapter.robot.data.body_lin_vel_w[:, adapter.robot_foot_ids, :2]
    foot_tangential_speed = torch.linalg.vector_norm(foot_velocity_xy, dim=-1)
    contacting_feet = signals.foot_contact > 0.0
    contacting_foot_count = int(contacting_feet[0].sum().item())
    contact_tangential_speed_sum = float(
        (foot_tangential_speed * contacting_feet.float()).sum().item()
    )
    adapter.observe_nonfoot_contact()
    diagnostics = adapter.contact_diagnostics()
    tensors = (
        root,
        root_quat,
        gravity,
        signals.base_lin_vel_xy,
        signals.command_lin_vel_xy,
        signals.foot_slip,
        signals.foot_contact,
        foot_tangential_speed,
        signals.nonfoot_contact,
        state.projected_gravity_xy,
        state.base_height,
        state.joint_vel,
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
        return {"invalid": True}
    qw, qx, qy, qz = root_quat[0].unbind()
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
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
        "foot_contact_count": contacting_foot_count,
        "foot_contact_tangential_speed_sum_mps": contact_tangential_speed_sum,
        "nonfoot_contact": bool(signals.nonfoot_contact[0].item() > 0.0),
        "max_nonfoot_contact_force_n": float(
            diagnostics["episode_max_nonfoot_contact_force_n"]
        ),
        "max_nonfoot_contact_body": diagnostics["episode_max_nonfoot_contact_body"],
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


def _run_episode(wrapped_env: Any, policy: Any, cell: dict, geometry_seed: int,
                 initial_state_seed: int, episode_index: int, *,
                 max_steps: int = MAX_STEPS, bounded_smoke: bool = False) -> dict[str, Any]:
    import torch

    from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
    from rl_training.source_scan.bridge import FrozenPolicySafetyBridge

    unwrapped = wrapped_env.unwrapped
    device = torch.device(unwrapped.device)
    adapter = Lite3IsaacLabAdapter(unwrapped)
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
    goal_progress = track["completion_x_m"] - track["start_x_m"]
    lateral_limit = terrain_size[1] / 2.0 - track["lateral_margin_m"]
    control_dt = CONTROL_DT
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_STEPS:
        raise ValueError(f"max_steps must be an integer in [1, {MAX_STEPS}]")
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
    steps = 0
    active_step = False
    terminal_snapshot: dict[str, Any] = {}

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
            sample = terminal_snapshot["sample"]
            termination_reason = terminal_snapshot["termination_reason"]
        else:
            sample = _runtime_sample(adapter, unwrapped, origins)
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
        foot_contact_speed_sum += sample["foot_contact_tangential_speed_sum_mps"]
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
        "steps": steps,
        "forward_progress_m": final_progress,
        "max_forward_progress_m": max_progress,
        "terrain_origin_w_m": [float(value) for value in origins[0].tolist()],
        "root_position_w_start_m": root_position_history[0] if root_position_history else None,
        "root_position_w_end_m": root_position_history[-1] if root_position_history else None,
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
        "foot_slip_contact_fraction": None,
        "foot_slip_contact_fraction_status": "unavailable_without_registered_slip_speed_threshold",
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
        "max_joint_speed_rad_s": max_joint_speed,
        "termination_terms": terminal_snapshot.get("active_termination_terms", []),
        "action_trace": action_trace,
        "family_metrics": _family_metrics(cell, progress),
    }


def _family_metrics(cell: dict, progress: list[float]) -> dict[str, Any]:
    """Report terrain descriptors and identify unmeasured contract fields."""
    family = cell["terrain_family"]
    geometry = cell["resolved_parameters"]["geometry"]
    result: dict[str, Any] = {}
    if family in {"slope", "slope_plus_rough"}:
        result.update(
            direction=cell["direction_or_layout"],
            slope_angle_deg=geometry["slope_angle_deg"],
            transition_progress_m=[
                max(0.0, min(float(value) - geometry["flat_length_m"], geometry["transition_length_m"]))
                for value in progress
            ],
            pitch_response_available=True,
            roll_response_available=True,
        )
    if family == "multi_step_stairs":
        result.update(
            direction=cell["direction_or_layout"],
            step_count=geometry["step_count"],
            completed_step_index=None,
            per_step_progress_m=None,
            front_foot_collision_count=None,
            stuck_or_catch_step_count=None,
            swing_foot_clearance_m=None,
            unavailable_metrics=[
                "completed_step_index", "per_step_progress_m", "front_foot_collision_count",
                "stuck_or_catch_step_count", "swing_foot_clearance_m",
            ],
        )
    if family == "friction_variation":
        result.update(
            region_layout=cell["direction_or_layout"],
            material_parameters={
                "candidate_coefficient_range": geometry["friction_range"],
                "region_count": geometry["friction_region_count"],
            },
            actual_contact_friction=None,
            slip_by_foot_and_region=None,
            unavailable_metrics=["actual_contact_friction", "slip_by_foot_and_region"],
        )
    if family in {"multi_scale_rough", "slope_plus_rough"}:
        result.update(
            audited_local_peak_valley_m=None,
            spatial_scale_statistics=None,
            adjacent_footfall_height_difference_m=None,
            geometry_candidate_peak_valley_m=geometry["rough_peak_valley_m"],
            geometry_candidate_spatial_scales_m=geometry["spatial_scales_m"],
            unavailable_metrics=[
                "audited_local_peak_valley_m", "spatial_scale_statistics",
                "adjacent_footfall_height_difference_m",
            ],
        )
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
                "missing_common_metrics": [
                    "foot_slip_contact_fraction_requires_a_registered_slip_speed_threshold",
                ],
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
