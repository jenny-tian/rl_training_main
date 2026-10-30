"""Recompute the archived rough cross-slope-left baseline on CPU; never launch simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/rl_training")]

import numpy as np
import torch

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.source_scan.rigid_body_reference import BODY_APIS, validate_body_reference_state
from rl_training.source_scan.tile_bounds_diagnostics import validate_tile_bounds_trace
from scripts.tools.source_policy_prequalification import episode_passed, select_case, verify_hashes
from scripts.tools.source_parallel_terrain_startup_supervisor import find_collision_cooking_failure, find_fatal_exception


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def rows(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines()]


def review():
    manifest, receipt, outcome, result = (read(name) for name in (
        "manifest.json", "execution_receipt.json", "outcome.json", "result.json"))
    runtime, after = read("runtime_readback.json"), read("gpu_after.json")
    record, samples, bounds, gpu = result["record"], rows("samples.jsonl"), rows("tile_bounds.jsonl"), rows("gpu_samples.jsonl")
    require(receipt["supervisor_exit_code"] == 1 and receipt["attempt_count"] == 1, "execution receipt")
    require(manifest["revision"] == receipt["frozen_revision"] == "b9fbed3703818103a487e05322beaa3201ddc04b", "frozen revision")
    for name, digest in receipt["archive_raw_sha256"].items():
        require(sha(HERE / name) == digest, "raw archive changed: " + name)
    verify_hashes(manifest["input_sha256"])
    preparation_path = ROOT / receipt["preparation_path"]
    require(sha(preparation_path) == manifest["preparation_sha256"], "preparation hash")
    prepared = json.loads(preparation_path.read_text())
    require(sha(Path(prepared["checkpoint"])) == prepared["checkpoint_sha256"], "checkpoint hash")
    require(prepared["checkpoint_sha256"] == "87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c", "frozen checkpoint")
    require(manifest["checkpoint"] == prepared["checkpoint"], "checkpoint provenance")
    case = select_case(prepared, "slope_plus_rough_L1_cross_slope_left")
    require(manifest["case_id"] == result["case_id"] == record["cell_id"] == case["case_id"], "case provenance")
    require(manifest["cell"] == case["cell"] and manifest["episode_count"] == 1, "single case")
    expected_geometry = {key: case[key] for key in (
        "bank_row", "bank_column", "subterrain", "geometry_difficulty", "mesh_fingerprint")}
    require(manifest["bank_geometry"] == result["bank_geometry"] == expected_geometry, "bank provenance")
    require((record["geometry_seed"], record["initial_state_seed"], record["episode_index"]) == (10001, 20260, 1), "registered seeds")
    require((manifest["geometry_seed"], manifest["initial_state_seed"]) == (10001, 20260), "manifest seeds")
    require(manifest["num_envs"] == 1 and manifest["gpu_index"] == 0 and manifest["max_control_steps"] == 2000, "single-episode budget")
    require(manifest["startup_timeout_s"] == 120 and manifest["total_timeout_s"] == 240, "time budgets")
    require(manifest["command_b"] == [0.3, 0., 0.], "manifest command")
    require(runtime["geometry_seed"] == 10001 and runtime["geometry_difficulty_range"] == [case["geometry_difficulty"]] * 2, "geometry realization")
    require(runtime["tile_guard"] == dict(size=[12., 4.], flat_length_m=3., collision_radius_m=0.27, numerical_margin_m=0.001), "guard thresholds")
    require(runtime["contact_threshold_n"] == 5. and runtime["contact_consecutive_steps"] == 3
            and runtime["max_projected_gravity_xy"] == 0.65, "safety terms")
    require(runtime["num_envs"] == 1 and runtime["step_dt"] == 0.02 and runtime["base_scan_update_period"] == 0.005, "timing")
    require(runtime["body_names"] == manifest["expected_body_names"] and runtime["guard_body_ids"] == list(range(17)), "body mapping")
    require(set(runtime["termination_terms"]) == {"source_terrain_tile_bounds", "illegal_contact", "bad_orientation_2"}, "termination set")
    require(outcome["status"] == "failed" and outcome["errors"] == ["episode_or_actor_failed"], "supervisor acceptance")
    execution = outcome["execution"]
    require(execution["exit_code"] == 0 and execution["elapsed_s"] < 240, "worker status/budget")
    require(execution["startup_timeout_s"] == 120 and execution["total_timeout_s"] == 240
            and execution["terminate_grace_s"] == 10., "supervision limits")
    for key in ("stop_reason", "failure_marker", "collision_cooking_failure", "fatal_exception"):
        require(execution[key] is None, key)
    require(result["status"] == "failed" and not episode_passed(record), "episode acceptance")
    require(result["policy_updated"] is False and result["ppo_initialized"] is False, "frozen actor scope")
    require(result["g3_qualified"] is outcome["g3_qualified"] is False, "G3 not qualified")
    require(result["actor_sha256_before"] == result["actor_sha256_after"] == "aeeeb55466a7f7369e7bc041708201c0a965779e6240a7be8751fe75c1b9a0d4", "actor hash")
    require(record["completed"] is False and record["termination_reason"] == "source_terrain_tile_bounds"
            and record["termination_terms"] == ["source_terrain_tile_bounds"], "failure classification")
    require(record["termination_capture_test_injected"] is False and record["stable_hold_steps_required"] == 50, "completion protocol")
    for key in ("invalid_state_count", "nonfoot_contact_count", "safety_filter_rejection_or_emergency_count"):
        require(record[key] == 0, key)
    require(record["safety_filter_projection_fraction"] == 0., "action projection")
    ids = list(range(1, record["steps"] + 1))
    require([row["step"] for row in samples] == [row["step"] for row in record["action_trace"]] == ids, "step coverage")
    actions = record["action_trace"]
    raw = torch.tensor([row["a_policy_raw"] for row in actions], dtype=torch.float32)
    base = torch.tensor([row["a_base"] for row in actions], dtype=torch.float32)
    require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(base).all()), "nonfinite actions")
    require(torch.equal(clip_lite3_policy_action(raw), base), "action clipping")
    for row in actions:
        require(row["a_base"] == row["a_requested"] == row["a_executed_command"], "action changed after clipping")
        require(row["requested_delta"] == [0.] * 12 and row["safety_paused"] is row["projected"] is False, "action safety state")
    hold, first_goal = 0, None
    for row in samples:
        s = row["sample"]
        require(not s.get("invalid", False) and s["nonfoot_contact"] is False, "sample event")
        require(abs(s["lateral_offset_m"]) <= 1.8, "root lateral envelope")
        stable = s["base_height_m"] >= 0.2 and s["max_projected_gravity_xy"] <= 0.65 and s["max_joint_speed_rad_s"] <= 30.
        require(stable, "unsafe sample")
        if s["forward_progress_m"] >= 9.0 and first_goal is None:
            first_goal = row["step"]
        hold = hold + 1 if stable and s["forward_progress_m"] >= 9.0 else 0
        if row["step"] != record["steps"]:
            require(hold < 50, "continued after completion")
    require(hold == record["stable_hold_steps_observed"] == 0 and first_goal is None, "hold reconstruction")
    values = [row["sample"] for row in samples]
    metrics = {
        "forward_progress_m": values[-1]["forward_progress_m"],
        "max_forward_progress_m": max(s["forward_progress_m"] for s in values),
        "min_base_height_m": min(s["base_height_m"] for s in values),
        "max_projected_gravity_xy": max(s["max_projected_gravity_xy"] for s in values),
        "max_joint_speed_rad_s": max(s["max_joint_speed_rad_s"] for s in values),
        "lateral_offset_peak_m": max(abs(s["lateral_offset_m"]) for s in values),
        "mean_velocity_error_mps": float(np.mean([s["velocity_error_mps"] for s in values])),
        "rms_velocity_error_mps": float(np.sqrt(np.mean([s["velocity_error_mps"] ** 2 for s in values]))),
        "mean_base_velocity_x_mps": float(np.mean([s["base_lin_vel_b_mps"][0] for s in values])),
        "mean_command_velocity_x_mps": float(np.mean([s["command_lin_vel_b_mps"][0] for s in values])),
    }
    for key, value in metrics.items():
        require(math.isfinite(value) and math.isclose(value, record[key], abs_tol=1e-6), "sample/summary mismatch: " + key)
    kinematics = [row["kinematics"] for row in samples]
    require([row["kinematics_phase"] for row in samples] == ["post_control_step"] * (len(samples) - 1) + ["pre_reset"], "sampling phase")
    require(record["pre_reset_capture_observed"] is True and record["pre_reset_state_differs_from_post_reset"] is True, "terminal reset snapshot")
    require(record["post_reset_root_position_w_m"] == [-4.5, 0., 0.375], "reset root position")
    reset_difference = float(np.abs(np.asarray(record["root_position_w_end_m"]) - record["post_reset_root_position_w_m"]).max())
    require(reset_difference > 1e-6, "terminal and reset state separation")
    require(record["root_position_w_end_m"] == values[-1]["root_position_w_m"], "terminal position")
    require(record["root_yaw_end_rad"] == values[-1]["root_yaw_rad"], "terminal yaw")
    q = np.asarray([k["root_quat_w_wxyz"] for k in kinematics])
    body_v = np.asarray([k["root_lin_vel_b_mps"] for k in kinematics])
    world_v = np.asarray([k["root_lin_vel_w_mps"] for k in kinematics])
    gravity = np.asarray([k["projected_gravity_b"] for k in kinematics])
    commands = np.asarray([k["command_b_vx_vy_wz"] for k in kinematics])
    require(all(np.isfinite(x).all() for x in (q, body_v, world_v, gravity, commands)), "nonfinite kinematics")
    def rotate(quaternion, vectors):
        t = 2. * np.cross(quaternion[:, 1:], vectors)
        return vectors + quaternion[:, :1] * t + np.cross(quaternion[:, 1:], t)
    q_error = float(np.abs(np.linalg.norm(q, axis=1) - 1.).max())
    v_error = float(np.abs(rotate(q, body_v) - world_v).max())
    inverse = q.copy()
    inverse[:, 1:] *= -1
    g_error = float(np.abs(rotate(inverse, np.broadcast_to([0., 0., -1.], gravity.shape)) - gravity).max())
    require(max(q_error, v_error, g_error) < 1e-5, "world/body kinematics")
    require(np.allclose(commands, [0.3, 0., 0.], atol=1e-6, rtol=0), "command changed")
    require(np.allclose(body_v[:, :2], [s["base_lin_vel_b_mps"] for s in values], atol=1e-6, rtol=0), "sample body velocity")
    require(np.allclose(np.linalg.norm(body_v[:, :2] - commands[:, :2], axis=1),
            [s["velocity_error_mps"] for s in values], atol=1e-6, rtol=0), "XY velocity errors")
    require(np.allclose(np.linalg.norm(gravity[:, :2], axis=1),
            [s["max_projected_gravity_xy"] for s in values], atol=1e-6, rtol=0), "sample gravity norm")
    for s in values:
        speeds = np.asarray(s["foot_horizontal_speed_mps"])
        touching = np.asarray(s["foot_contact"], dtype=bool)
        require(speeds.shape == touching.shape == (4,) and np.isfinite(speeds).all(), "foot sample")
        require(s["foot_slip_speed_threshold_mps"] == 0.35, "slip threshold")
        require(int(touching.sum()) == s["foot_contact_count"], "contact sample count")
        require(int((touching & (speeds > 0.35)).sum()) == s["foot_slipping_contact_count"], "slip sample count")
        require(math.isclose(float(speeds[touching].sum()), s["foot_contact_horizontal_speed_sum_mps"], abs_tol=1e-6), "contact speed sum")
    contact = sum(s["foot_contact_count"] for s in values)
    slip = sum(s["foot_slipping_contact_count"] for s in values)
    require(contact == record["foot_slip_contact_fraction_denominator"] and slip == record["foot_slip_contact_fraction_numerator"], "slip counters")
    require(math.isclose(slip / contact, record["foot_slip_contact_fraction"], abs_tol=1e-12), "slip fraction")
    diagnostic = validate_tile_bounds_trace(HERE / "tile_bounds.jsonl", record, manifest["expected_body_names"], require_link_reference=True)
    require(diagnostic == read("tile_bounds_review.json") and diagnostic["reference_conversion"]["status"] == "verified", "predicate/reference review")
    expected_hit = {"branch": "prediction", "body_id": 6, "body_name": "FR_THIGH", "axis": "y"}
    require(diagnostic["terminal_geometric_hits"] == [expected_hit] and diagnostic["terminal_boundary_flag"] is True, "terminal boundary")
    terminal_bounds = bounds[-1]
    require(terminal_bounds["outside_now"] == [False] and terminal_bounds["outside_prediction"] == [True], "terminal branch")
    require(terminal_bounds["valid_indices"] == [True] and terminal_bounds["invalid_state"] == [False], "terminal predicate inputs")
    lower_y = terminal_bounds["safe_lower_xy_w"][0][1]
    body_position = terminal_bounds["body_positions_w"][0][6]
    body_velocity = terminal_bounds["body_linear_velocities_w"][0][6]
    predicted = terminal_bounds["predicted_positions_xy_w"][0][6]
    trigger = {
        **expected_hit, "boundary_side": "lower", "link_position_w_m": body_position,
        "link_velocity_w_mps": body_velocity,
        "com_velocity_w_mps": terminal_bounds["body_com_linear_velocities_w"][0][6],
        "com_offset_link_m": terminal_bounds["body_com_offsets_link_m"][0][6],
        "link_quaternion_wxyz": terminal_bounds["body_link_quats_w_wxyz"][0][6],
        "world_angular_velocity_rad_s": terminal_bounds["body_angular_velocities_w"][0][6],
        "implemented_predicted_xy_w_m": predicted, "safe_lower_y_w_m": lower_y,
        "current_link_y_clearance_m": body_position[1] - lower_y,
        "implemented_prediction_y_excess_m": lower_y - predicted[1],
        "point_consistent_geometric_prediction_verified": True,
    }
    require(trigger["current_link_y_clearance_m"] > 0 and trigger["implemented_prediction_y_excess_m"] > 0, "trigger distances")
    for row, body in zip(samples, bounds, strict=True):
        root = row["kinematics"]["root_reference_state"]
        require(validate_body_reference_state(root)["status"] == "verified", "root reference")
        require(root["body_positions_w"][0][0] == row["sample"]["root_position_w_m"], "root sample position")
        require(root["body_com_linear_velocities_w"][0][0] == row["kinematics"]["root_lin_vel_w_mps"], "legacy COM velocity")
        require(root["body_link_quats_w_wxyz"][0][0] == row["kinematics"]["root_quat_w_wxyz"], "root quaternion")
        for field in BODY_APIS:
            require(np.array_equal(root[field], np.asarray(body[field])[:, :1]), "root/predicate disagreement: " + field)
        require(root["body_com_offsets_link_m"] == [[[0., 0., 0.]]], "actual root COM")
    for label in ("body", "root"):
        require(validate_body_reference_state(runtime[label + "_reference_state"]) == runtime[label + "_reference_review"], "startup reference")
    require(manifest["gpu_admission"] == dict(minimum_available_mib=12288, allow_existing_compute_processes=True,
            memory_basis="memory.total - memory.used"), "memory admission")
    before_gpu = next(line for line in manifest["gpu_before"].splitlines() if line.startswith("0, ")).split(", ")
    after_gpu = next(line for line in after["gpu"].splitlines() if line.startswith("0, ")).split(", ")
    require(before_gpu[1] == after_gpu[1] == "GPU-044ddc24-86ef-a67d-7e14-9358300bc3dc", "GPU UUID")
    def occupants(snapshot):
        return sorted(int(line.split(", ")[1]) for line in snapshot.splitlines() if line.startswith(before_gpu[1] + ", "))
    processes = read("processes_after.json")
    require(processes["matching_runtime_processes"] == [] and processes["own_unreadable_pids"] == [], "runtime process cleanup")
    console = (HERE / "console.log").read_text()
    require(find_collision_cooking_failure(console) is None and find_fatal_exception(console) is None, "fatal console marker")
    require("Traceback (most recent call last)" not in console and "[Error]" not in console and "[ERROR]" not in console, "console error")
    family = record["family_metrics"]
    geometry = case["cell"]["resolved_parameters"]["geometry"]
    require(family["direction"] == "cross_slope_left" and family["slope_angle_deg"] == geometry["slope_angle_deg"] == 5., "slope registration")
    posture = {}
    for label in ("roll", "pitch"):
        a = np.asarray([s[label + "_rad"] for s in values])
        summary = dict(mean=float(a.mean()), rms=float(np.sqrt(np.mean(a*a))), peak=float(np.abs(a).max()),
                       minimum=float(a.min()), maximum=float(a.max()), sample_count=len(a), unit="rad")
        for key, value in summary.items():
            actual = family[label + "_response"][key]
            require(math.isclose(value, actual, abs_tol=1e-8) if isinstance(value, float) else value == actual, "posture summary " + label + key)
        for suffix in ("mean", "rms", "peak"):
            require(math.isclose(summary[suffix], record[label + "_" + suffix + "_rad"], abs_tol=1e-8), "posture record")
        posture[label] = summary
    expected_roll = np.arcsin(np.clip(gravity[:,1], -1, 1))
    expected_pitch = np.arcsin(np.clip(gravity[:,0], -1, 1))
    require(np.allclose(expected_roll, [s["roll_rad"] for s in values], atol=1e-6, rtol=0), "roll gravity proxy")
    require(np.allclose(expected_pitch, [s["pitch_rad"] for s in values], atol=1e-6, rtol=0), "pitch gravity proxy")
    progress = np.asarray([s["forward_progress_m"] for s in values])
    transitions = {}
    for key, start, length in (("transition_progress_m", 1.5, 0.5), ("exit_transition_progress_m", 8., 0.5), ("terminal_flat_progress_m", 8.5, 2.)):
        rebuilt = np.clip(progress-start, 0, length)
        require(np.allclose(rebuilt, family[key], atol=1e-10, rtol=0), "transition array " + key)
        transitions[key] = dict(last_m=float(rebuilt[-1]), max_m=float(rebuilt.max()))
    require(family["geometry_candidate_peak_valley_m"] == geometry["rough_peak_valley_m"] == 0.01, "rough parameter")
    require(family["geometry_candidate_spatial_scales_m"] == geometry["spatial_scales_m"] == [0.2, 0.4], "spatial scales")
    slope_review = dict(posture_reconstructed=posture, posture_semantics="roll=asin(projected_gravity_y), pitch=asin(projected_gravity_x); these are gravity-based proxies, not standard Euler angles", transition_progress_reconstructed=transitions,
        geometry_metadata_from_runtime=family["audited_local_peak_valley_m"],
        spatial_scale_statistics_from_runtime=family["spatial_scale_statistics"],
        adjacent_footfall_height_difference_from_runtime=family["adjacent_footfall_height_difference_m"],
        verification_scope="Posture and transition arrays independently rebuilt; geometry/footfall metadata retained from runtime, not independently regenerated by this reviewer. Exact registered mesh equivalence and source hashes were checked before execution.")
    return dict(
        slope_rough_review=slope_review,
        schema="lite3_rough_cross_slope_left_l1_post_run_review_v1", review_result="single_condition_safety_failed_complete_reference_evidence",
        condition_passed=False, revision=manifest["revision"], input_hashes_verified=len(manifest["input_sha256"]),
        raw_files_verified=len(receipt["archive_raw_sha256"]), control_steps=len(samples), simulated_control_time_s=len(samples)*0.02,
        first_goal_step=first_goal, reconstructed_final_hold_steps=hold, metrics_reconstructed_from_samples=metrics,
        root_yaw_end_degrees=math.degrees(record["root_yaw_end_rad"]), action_trace_rows=len(actions),
        lateral_offset_signed_m=dict(minimum=min(s["lateral_offset_m"] for s in values),
            maximum=max(s["lateral_offset_m"] for s in values), end=values[-1]["lateral_offset_m"]),
        mean_body_velocity_y_mps=float(body_v[:, 1].mean()), mean_world_velocity_y_mps=float(world_v[:, 1].mean()),
        terminal_trigger=trigger, terminal_state=dict(pre_reset_capture_observed=True,
            root_position_w_m=record["root_position_w_end_m"], post_reset_root_position_w_m=record["post_reset_root_position_w_m"],
            max_pre_post_reset_difference_m=reset_difference),
        action_projection_count=0, safety_rejection_count=0, invalid_state_count=0, nonfoot_contact_count=0,
        predicate_diagnostic_review=diagnostic, root_reference_evaluations_verified=len(samples), actual_root_com_offset_m=[0., 0., 0.],
        kinematic_errors=dict(quaternion_norm=q_error, body_to_world_velocity_mps=v_error, projected_gravity=g_error),
        foot_slip_fraction=slip/contact, foot_slip_numerator=slip, foot_slip_denominator=contact,
        actor_sha256=result["actor_sha256_after"], policy_updated=False, ppo_initialized=False, g3_qualified=False, g4_qualified=False,
        supervisor_exit_code=receipt["supervisor_exit_code"], worker_exit_code=execution["exit_code"], execution_elapsed_s=execution["elapsed_s"],
        sampled_gpu_peak_whole_device_mib=max(row["memory_used_mib"] for row in gpu), gpu_sample_count=len(gpu),
        gpu0_before_mib=int(before_gpu[3].split()[0]), gpu0_after_mib=int(after_gpu[3].split()[0]),
        existing_gpu0_pids_before=occupants(manifest["processes_before"]), existing_gpu0_pids_after=occupants(after["processes"]),
        residual_worker_processes=0, console=dict(error_markers=0, tracebacks=0,
            unresolved_visual_reference_warnings=console.count("Unresolved reference prim path"),
            cuda_device_skip_warnings=console.count("Skipping NVIDIA GPU due CUDA being in bad state")),
        limitations=["This is one fixed rough left-high/right-low cross-slope instance and one seed, not a family success rate or learned gait.",
                     "Safety acceptance has no tight actual-speed tracking threshold: observed mean forward speed exceeds the 0.3 m/s command.",
                     "The actor is unchanged; this failure measures existing transfer capability, not a Source training change.",
                     "The terminal pre-reset snapshot is verified; this one reset does not qualify shared-bank or general sensor/reset timing.",
                     "Linear link-origin prediction crossing the safety envelope is not proof of a physical body crossing the terrain edge or its actual next-step position.",
                     "Downslope drift direction alone does not establish gravity, foot slipping or policy asymmetry as its sole cause.",
                     "Foot-slip contact fraction counts contacting-foot horizontal-speed proxies above 0.35 m/s; it is not a measured slip distance.",
                     "Whole-GPU resource samples include other users; timing and memory are not exclusive-task measurements."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = review()
    if args.write:
        with (HERE / "post_run_review.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
