"""Recheck this archived run on CPU; --write creates its separate review record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/rl_training"))

import torch
import numpy as np

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from scripts.tools.source_policy_prequalification import episode_passed, verify_hashes, select_case
from scripts.tools.source_parallel_terrain_startup_supervisor import (
    find_collision_cooking_failure, find_fatal_exception,
)
from rl_training.source_scan.tile_bounds_diagnostics import validate_tile_bounds_trace
from rl_training.source_scan.rigid_body_reference import BODY_APIS, validate_body_reference_state


def require(condition, description):
    if not condition:
        raise ValueError(description)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite archived value")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def read_json(name):
    value = json.loads((HERE / name).read_text())
    finite_tree(value)
    return value


def read_jsonl(name):
    rows = [json.loads(line) for line in (HERE / name).read_text().splitlines()]
    finite_tree(rows)
    return rows


def review():
    manifest = read_json("manifest.json")
    receipt = read_json("execution_receipt.json")
    outcome = read_json("outcome.json")
    result = read_json("result.json")
    runtime = read_json("runtime_readback.json")
    record = result["record"]
    samples = read_jsonl("samples.jsonl")
    gpu = read_jsonl("gpu_samples.jsonl")
    after = read_json("gpu_after.json")
    require(receipt["supervisor_exit_code"] == 1 and receipt["attempt_count"] == 1, "execution receipt")
    require(manifest["revision"] == receipt["frozen_revision"], "revision changed")
    for name, digest in receipt["archive_raw_sha256"].items():
        require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, f"raw archive changed: {name}")
    verify_hashes(manifest["input_sha256"])
    preparation = ROOT / "docs/lite3-source-uphill-memory-admission-preparation-20261029/preparation.json"
    require(hashlib.sha256(preparation.read_bytes()).hexdigest() == manifest["preparation_sha256"], "preparation hash")
    prepared = json.loads(preparation.read_text())
    case = select_case(prepared, "slope_L1_uphill")
    require(manifest["revision"] == "84819d753bb32a0d6c719015e7aafe1265d2512a", "frozen runtime revision")
    require(manifest["manifest_kind"] == "immutable_run_spec", "manifest kind")
    require(not {"status", "simulation_run", "cases"}.intersection(manifest), "preparation metadata leaked into execution spec")
    require(manifest["cell"] == case["cell"] and manifest["episode_count"] == 1, "single case spec")
    require(result["case_id"] == manifest["case_id"] == case["case_id"], "case provenance")
    expected_geometry = {key: case[key] for key in (
        "bank_row", "bank_column", "subterrain", "geometry_difficulty", "mesh_fingerprint")}
    require(result["bank_geometry"] == manifest["bank_geometry"] == expected_geometry, "bank realization provenance")
    require(runtime["geometry_seed"] == case["geometry_seed"], "runtime geometry seed")
    require(runtime["geometry_difficulty_range"] == [case["geometry_difficulty"]] * 2, "runtime fixed difficulty")
    require(runtime["contact_threshold_n"] == 5.0 and runtime["contact_consecutive_steps"] == 3, "contact thresholds")
    require(runtime["max_projected_gravity_xy"] == 0.65, "tilt threshold")
    require(runtime["tile_guard"] == dict(size=[12.0, 4.0], flat_length_m=3.0,
            collision_radius_m=0.27, numerical_margin_m=0.001), "tile guard")
    require(outcome["status"] == "failed" and outcome["errors"] == ["episode_or_actor_failed"], "supervisor outcome")
    execution = outcome["execution"]
    require(execution["exit_code"] == 0 and execution["elapsed_s"] < 240, "execution status/budget")
    for key in ("stop_reason", "failure_marker", "collision_cooking_failure", "fatal_exception"):
        require(execution[key] is None, key)
    require(result["status"] == "failed" and not episode_passed(record), "episode acceptance")
    require(result["policy_updated"] is False and result["ppo_initialized"] is False, "frozen policy scope")
    require(result["g3_qualified"] is False and outcome["g3_qualified"] is False, "G3 must remain unqualified")
    require(result["actor_sha256_before"] == result["actor_sha256_after"], "actor changed")
    require(record["completed"] is False and record["termination_reason"] == "source_terrain_tile_bounds", "unexpected failure classification")
    require(record["termination_terms"] == ["source_terrain_tile_bounds"], "unexpected termination term")
    require(record["termination_capture_test_injected"] is False and record["stable_hold_steps_required"] == 50, "failure protocol changed")
    require(type(record["steps"]) is int and 1 <= record["steps"] <= 2000, "control-step budget")
    for key in ("invalid_state_count", "nonfoot_contact_count", "safety_filter_rejection_or_emergency_count"):
        require(type(record[key]) is int and record[key] == 0, f"unexpected additional event: {key}")
    require(record["safety_filter_projection_fraction"] == 0.0, "unexpected projection fraction")
    require(record["cell_id"] == "slope_L1_uphill", "unexpected cell")
    require((record["geometry_seed"], record["initial_state_seed"], record["episode_index"]) == (10001, 20060, 1), "seeds/episode")
    require(runtime["body_names"] == manifest["expected_body_names"] and runtime["guard_body_ids"] == list(range(17)), "body mapping")
    require(runtime["num_envs"] == 1 and runtime["step_dt"] == 0.02 and runtime["base_scan_update_period"] == 0.005, "runtime timing")
    require(set(runtime["termination_terms"]) == {"source_terrain_tile_bounds", "illegal_contact", "bad_orientation_2"}, "termination set")
    expected_ids = list(range(1, record["steps"] + 1))
    require([row["step"] for row in samples] == expected_ids, "missing/reordered samples")
    trace = record["action_trace"]
    require([row["step"] for row in trace] == expected_ids, "missing/reordered actions")
    raw = torch.tensor([row["a_policy_raw"] for row in trace], dtype=torch.float32)
    base = torch.tensor([row["a_base"] for row in trace], dtype=torch.float32)
    require(torch.equal(clip_lite3_policy_action(raw), base), "action clipping mismatch")
    for row in trace:
        require(row["a_base"] == row["a_requested"] == row["a_executed_command"], "unexpected action projection")
        require(row["requested_delta"] == [0.0] * 12 and row["safety_paused"] is False and row["projected"] is False, "action safety state")
    hold = 0
    first_goal_step = None
    for row in samples:
        sample = row["sample"]
        require(not sample.get("invalid", False), "invalid sample")
        command = sample["command_lin_vel_b_mps"]
        require(len(command) == 2 and abs(command[0] - 0.3) <= 1e-6 and command[1] == 0.0, "command changed")
        require(sample["nonfoot_contact"] is False, "nonfoot contact")
        require(abs(sample["lateral_offset_m"]) <= 1.8, "root lateral limit")
        stable = (sample["base_height_m"] >= 0.2 and sample["max_projected_gravity_xy"] <= 0.65
                  and sample["max_joint_speed_rad_s"] <= 30.0)
        require(stable, "unsafe sample")
        if sample["forward_progress_m"] >= 9.0 and first_goal_step is None:
            first_goal_step = row["step"]
        hold = hold + 1 if sample["forward_progress_m"] >= 9.0 and stable else 0
        if row["step"] != record["steps"]:
            require(hold < 50, "continued after completion")
    require(hold == record["stable_hold_steps_observed"] == 0 and first_goal_step is None, "hold reconstruction")
    values = [row["sample"] for row in samples]
    comparisons = {
        "forward_progress_m": values[-1]["forward_progress_m"],
        "min_base_height_m": min(s["base_height_m"] for s in values),
        "max_projected_gravity_xy": max(s["max_projected_gravity_xy"] for s in values),
        "max_joint_speed_rad_s": max(s["max_joint_speed_rad_s"] for s in values),
        "lateral_offset_peak_m": max(abs(s["lateral_offset_m"]) for s in values),
        "mean_velocity_error_mps": sum(s["velocity_error_mps"] for s in values) / len(values),
    }
    for key, value in comparisons.items():
        require(math.isclose(value, record[key], abs_tol=1e-6), f"sample/summary mismatch: {key}")
    # Independent checks of the newly logged full body/world kinematics.
    require([row["kinematics_phase"] for row in samples] == ["post_control_step"] * (len(samples) - 1) + ["pre_reset"], "unexpected sampling phase")
    kinematics = [row["kinematics"] for row in samples]
    q = np.asarray([k["root_quat_w_wxyz"] for k in kinematics])
    body_v = np.asarray([k["root_lin_vel_b_mps"] for k in kinematics])
    world_v = np.asarray([k["root_lin_vel_w_mps"] for k in kinematics])
    gravity = np.asarray([k["projected_gravity_b"] for k in kinematics])
    commands = np.asarray([k["command_b_vx_vy_wz"] for k in kinematics])
    require(q.shape == (len(samples), 4) and body_v.shape == world_v.shape == gravity.shape == commands.shape == (len(samples), 3), "kinematic shape")
    require(np.isfinite(q).all() and np.isfinite(body_v).all() and np.isfinite(world_v).all()
            and np.isfinite(gravity).all() and np.isfinite(commands).all(), "nonfinite kinematics")
    quat_norm_error = float(np.abs(np.linalg.norm(q, axis=1) - 1.0).max())
    def rotate(quaternion, vectors):
        t = 2.0 * np.cross(quaternion[:, 1:], vectors)
        return vectors + quaternion[:, :1] * t + np.cross(quaternion[:, 1:], t)
    velocity_error = float(np.abs(rotate(q, body_v) - world_v).max())
    inverse = q.copy()
    inverse[:, 1:] *= -1
    expected_gravity = rotate(inverse, np.broadcast_to([0.0, 0.0, -1.0], gravity.shape))
    gravity_error = float(np.abs(expected_gravity - gravity).max())
    require(max(quat_norm_error, velocity_error, gravity_error) < 1e-5, "body/world kinematics mismatch")
    require(np.allclose(body_v[:, :2], [s["base_lin_vel_b_mps"] for s in values], atol=1e-6, rtol=0), "sample body velocity mismatch")
    require(np.allclose(commands, [0.3, 0.0, 0.0], atol=1e-6, rtol=0), "full command changed")
    contact_samples = sum(s["foot_contact_count"] for s in values)
    slipping_samples = sum(s["foot_slipping_contact_count"] for s in values)
    require(contact_samples == record["foot_slip_contact_fraction_denominator"], "slip denominator")
    require(slipping_samples == record["foot_slip_contact_fraction_numerator"], "slip numerator")
    require(math.isclose(slipping_samples / contact_samples, record["foot_slip_contact_fraction"], abs_tol=1e-12), "slip fraction")
    require(record["pre_reset_capture_observed"] is True and record["pre_reset_state_differs_from_post_reset"] is True, "terminal reset snapshot missing")
    terminal_position = np.asarray(values[-1]["root_position_w_m"])
    reset_position = np.asarray(record["post_reset_root_position_w_m"])
    require(np.array_equal(terminal_position, record["root_position_w_end_m"]), "terminal summary mixed with reset state")
    reset_difference = float(np.abs(terminal_position - reset_position).max())
    require(reset_difference > 1e-6, "terminal and reset root states not distinguished")
    guard = runtime["tile_guard"]
    origin = np.asarray(record["terrain_origin_w_m"][:2])
    radius_margin = guard["collision_radius_m"] + guard["numerical_margin_m"]
    lower = origin + [-guard["flat_length_m"] / 2, -guard["size"][1] / 2] + radius_margin
    upper = origin + [guard["size"][0] - guard["flat_length_m"] / 2, guard["size"][1] / 2] - radius_margin
    logged_centers = np.vstack([terminal_position, values[-1]["foot_positions_w_m"]])[:, :2]
    logged_centers_inside = bool(((logged_centers >= lower) & (logged_centers <= upper)).all())
    # Root and four feet are only a subset; exact attribution below uses
    # the full snapshot captured at the predicate evaluation.
    terminal_review = {
        "step": record["steps"], "termination_terms": record["termination_terms"],
        "root_position_w_m": terminal_position.tolist(), "post_reset_root_position_w_m": reset_position.tolist(),
        "max_pre_post_reset_root_difference_m": reset_difference,
        "root_yaw_rad": values[-1]["root_yaw_rad"],
        "root_yaw_degrees": math.degrees(values[-1]["root_yaw_rad"]),
        "body_velocity_mps": body_v[-1].tolist(), "world_velocity_mps": world_v[-1].tolist(),
        "safe_body_center_lower_xy_w_m": lower.tolist(), "safe_body_center_upper_xy_w_m": upper.tolist(),
        "logged_root_and_four_feet_inside_instantaneous_safe_box": logged_centers_inside,
        "max_logged_foot_lateral_center_m": float(np.max(np.asarray(values[-1]["foot_positions_w_m"])[:, 1] - origin[1])),
        "exact_triggering_body_and_now_vs_prediction_branch": "recorded_in_tile_bounds_jsonl",
    }
    diagnostic = validate_tile_bounds_trace(HERE / "tile_bounds.jsonl", record, manifest["expected_body_names"], require_link_reference=True)
    require(diagnostic["reference_conversion"]["status"] == "verified", "link reference conversion")
    require(diagnostic == read_json("tile_bounds_review.json"), "worker diagnostic review changed")
    expected_hit = {"branch": "prediction", "body_id": 5, "body_name": "FL_THIGH", "axis": "y"}
    require(diagnostic["terminal_geometric_hits"] == [expected_hit], "unexpected body/branch")
    bounds = read_jsonl("tile_bounds.jsonl")
    terminal_bounds = bounds[-1]
    require(terminal_bounds["outside_now"] == [False] and terminal_bounds["outside_prediction"] == [True], "terminal branch")
    require(terminal_bounds["valid_indices"] == [True] and terminal_bounds["invalid_state"] == [False], "invalid terminal inputs")
    upper_y = terminal_bounds["safe_upper_xy_w"][0][1]
    body_position = terminal_bounds["body_positions_w"][0][5]
    body_velocity = terminal_bounds["body_linear_velocities_w"][0][5]
    predicted = terminal_bounds["predicted_positions_xy_w"][0][5]
    trigger = {
        **expected_hit, "link_position_w_m": body_position,
        "link_velocity_w_mps": body_velocity,
        "com_velocity_w_mps": terminal_bounds["body_com_linear_velocities_w"][0][5],
        "com_offset_link_m": terminal_bounds["body_com_offsets_link_m"][0][5],
        "link_quaternion_wxyz": terminal_bounds["body_link_quats_w_wxyz"][0][5],
        "world_angular_velocity_rad_s": terminal_bounds["body_angular_velocities_w"][0][5],
        "implemented_predicted_xy_w_m": predicted,
        "safe_upper_y_w_m": upper_y,
        "current_link_y_clearance_m": upper_y - body_position[1],
        "implemented_prediction_y_excess_m": predicted[1] - upper_y,
        "point_consistent_geometric_prediction_verified": True,
    }
    require(trigger["current_link_y_clearance_m"] > 0 and trigger["implemented_prediction_y_excess_m"] > 0, "trigger distances")
    previous = ROOT / "docs/lite3-source-uphill-boundary-diagnostic-recheck-20261029"
    previous_samples = [json.loads(line) for line in (previous / "samples.jsonl").read_text().splitlines()]
    previous_bounds = [json.loads(line) for line in (previous / "tile_bounds.jsonl").read_text().splitlines()]
    previous_record = json.loads((previous / "result.json").read_text())["record"]
    require(len(previous_samples) == len(samples) == len(previous_bounds) == len(bounds), "paired step counts")
    action_fields = ("a_policy_raw", "a_base", "a_requested", "a_executed_command")
    first_action_difference = next((i for i, (old, new) in enumerate(zip(previous_record["action_trace"], trace, strict=True), 1)
        if any(old[k] != new[k] for k in action_fields)), None)
    first_sample_difference = next((i for i, (old, new) in enumerate(zip(previous_samples, samples, strict=True), 1)
        if old["sample"] != new["sample"]), None)
    first_legacy_kinematics_difference = next((i for i, (old, new) in enumerate(zip(previous_samples, samples, strict=True), 1)
        if any(old["kinematics"][k] != new["kinematics"][k] for k in old["kinematics"])), None)
    first_termination_difference = next((i for i, (old, new) in enumerate(zip(previous_bounds, bounds, strict=True), 1)
        if old["terminated"] != new["terminated"]), None)
    comparison = dict(first_action_value_difference_step=first_action_difference,
                      first_sample_value_difference_step=first_sample_difference,
                      first_legacy_kinematics_difference_step=first_legacy_kinematics_difference,
                      first_termination_flag_difference_step=first_termination_difference,
                      samples_byte_identical=(HERE / "samples.jsonl").read_bytes() == (previous / "samples.jsonl").read_bytes(),
                      scope="paired same-seed trajectories; current sample file adds explicit root fields, timings are excluded from action-value comparison")
    require(all(value is None for value in (first_action_difference, first_sample_difference,
            first_legacy_kinematics_difference, first_termination_difference)), "unexpected trajectory difference")
    require(all(old["body_positions_w"] == new["body_positions_w"]
                and old["body_linear_velocities_w"] == new["body_com_linear_velocities_w"]
                for old, new in zip(previous_bounds, bounds, strict=True)), "old link positions/COM velocities differ")
    old_predictions = np.asarray([r["predicted_positions_xy_w"] for r in previous_bounds])
    new_predictions = np.asarray([r["predicted_positions_xy_w"] for r in bounds])
    comparison["max_abs_prediction_coordinate_change_m"] = float(np.abs(new_predictions-old_predictions).max())
    comparison["evaluations_with_changed_prediction_coordinates"] = int(np.any(new_predictions != old_predictions, axis=(1, 2, 3)).sum())
    comparison["evaluations_with_changed_geometric_branch_flags"] = sum(
        any(old[k] != new[k] for k in ("outside_now_by_axis", "outside_prediction_by_axis"))
        for old, new in zip(previous_bounds, bounds, strict=True))
    comparison["terminal_old_prediction_y_m"] = previous_bounds[-1]["predicted_positions_xy_w"][0][5][1]
    comparison["terminal_new_prediction_y_m"] = predicted[1]
    comparison["terminal_prediction_shift_m"] = predicted[1] - comparison["terminal_old_prediction_y_m"]
    # The old logs are not rewritten. New paired observations now supply the
    # previously missing orientations/angular velocities for this repeated path.
    root_reviews = []
    for index, (row, body) in enumerate(zip(samples, bounds, strict=True), 1):
        root = row["kinematics"]["root_reference_state"]
        checked = validate_body_reference_state(root)
        require(checked["status"] == "verified", "root reference state")
        root_reviews.append(checked)
        require(root["body_positions_w"][0][0] == row["sample"]["root_position_w_m"], "root reference/sample position")
        require(root["body_com_linear_velocities_w"][0][0] == row["kinematics"]["root_lin_vel_w_mps"], "legacy root COM semantics")
        require(root["body_link_quats_w_wxyz"][0][0] == row["kinematics"]["root_quat_w_wxyz"], "root quaternion semantics")
        for field in BODY_APIS:
            require(np.array_equal(root[field], np.asarray(body[field])[:, :1]), "root/predicate mismatch: " + field)
        require(root["body_com_offsets_link_m"] == [[[0., 0., 0.]]], "actual root COM offset")
    reference_audit = json.loads((previous / "reference_point_audit.json").read_text())
    expected_thigh_com = reference_audit["static_body_properties"]["FL_THIGH"]["center_of_mass_in_link_m"]
    require(all(row["body_com_offsets_link_m"][0][5] == expected_thigh_com for row in bounds), "actual thigh COM differs from USD audit")
    for label in ("body", "root"):
        require(validate_body_reference_state(runtime[label + "_reference_state"]) == runtime[label + "_reference_review"], "startup reference review")
    root_positions = np.asarray([k["root_reference_state"]["body_positions_w"][0][0] for k in kinematics])
    root_link_velocities = np.asarray([k["root_reference_state"]["body_linear_velocities_w"][0][0] for k in kinematics])
    displacement = float(root_positions[-1, 1]-root_positions[0, 1])
    integral = float(((root_link_velocities[:-1, 1]+root_link_velocities[1:, 1])*0.01).sum())
    root_reference_review = dict(evaluations_verified=len(root_reviews), actual_com_offset_m=[0., 0., 0.],
        link_position_and_velocity_same_point=True, net_lateral_displacement_m=displacement,
        world_link_velocity_trapezoid_integral_m=integral, residual_m=displacement-integral,
        residual_scope="50 Hz trapezoid between first and last sample; substep/solver components not isolated")
    require(manifest["gpu_admission"] == dict(minimum_available_mib=12288, allow_existing_compute_processes=True,
            memory_basis="memory.total - memory.used"), "GPU-sharing admission")
    gpu0 = next(line for line in after["gpu"].splitlines() if line.startswith("0, ")).split(", ")
    before_gpu0 = next(line for line in manifest["gpu_before"].splitlines() if line.startswith("0, ")).split(", ")
    require(gpu0[1] == before_gpu0[1] == "GPU-044ddc24-86ef-a67d-7e14-9358300bc3dc", "GPU UUID changed")
    def gpu0_pids(snapshot):
        return sorted(int(line.split(", ")[1]) for line in snapshot.splitlines() if line.startswith(gpu0[1] + ", "))
    before_pids, after_pids = gpu0_pids(manifest["processes_before"]), gpu0_pids(after["processes"])
    require(before_pids == after_pids and len(before_pids) == 4, "shared GPU occupant change")
    require(read_json("processes_after.json")["matching_runtime_processes"] == [], "worker still running")
    require(read_json("processes_after.json")["own_unreadable_pids"] == [], "own process inspection incomplete")
    console = (HERE / "console.log").read_text()
    require(find_collision_cooking_failure(console) is None and find_fatal_exception(console) is None, "fatal log marker")
    require("Traceback (most recent call last)" not in console and "[Error]" not in console and "[ERROR]" not in console, "console error")
    return {
        "schema": "lite3_uphill_link_reference_recheck_post_run_review_v1",
        "review_result": "failed_condition_complete_link_reference_evidence_verified", "condition_passed": False,
        "revision": manifest["revision"], "input_hashes_verified": len(manifest["input_sha256"]),
        "raw_files_verified": len(receipt["archive_raw_sha256"]),
        "policy_updated": False, "ppo_initialized": False, "g3_qualified": False, "g4_qualified": False,
        "control_steps": len(samples), "simulated_control_time_s": len(samples) * 0.02,
        "action_trace_rows": len(trace), "first_goal_step": first_goal_step, "reconstructed_final_hold_steps": hold,
        "metrics_reconstructed_from_samples": comparisons,
        "invalid_states": 0, "nonfoot_contact_events": 0, "safety_rejections": 0, "projections": 0,
        "kinematics": {"rows_verified": len(kinematics), "max_quaternion_norm_error": quat_norm_error,
            "max_body_to_world_velocity_error_mps": velocity_error, "max_projected_gravity_error": gravity_error,
            "full_command_verified": True, "pre_reset_rows": 1},
        "bank_geometry": manifest["bank_geometry"],
        "supervisor_exit_code": receipt["supervisor_exit_code"],
        "worker_exit_code": execution["exit_code"],
        "terminal_state_review": terminal_review,
        "predicate_diagnostic_review": diagnostic,
        "exact_implemented_trigger": trigger,
        "root_reference_review": root_reference_review,
        "previous_run_comparison": comparison,
        "reference_point_alignment": "verified_on_all_runtime_snapshots",
        "root_yaw_end_rad": record["root_yaw_end_rad"],
        "foot_slip_fraction": record["foot_slip_contact_fraction"],
        "foot_slip_numerator": slipping_samples, "foot_slip_denominator": contact_samples,
        "actor_sha256": result["actor_sha256_after"],
        "execution_elapsed_s": execution["elapsed_s"], "sampled_gpu_peak_whole_device_mib": max(row["memory_used_mib"] for row in gpu),
        "gpu0_before_mib": int(before_gpu0[3].split()[0]),
        "existing_gpu0_pids_before": before_pids, "existing_gpu0_pids_after": after_pids,
        "gpu_sample_count": len(gpu), "gpu0_after_mib": int(gpu0[3].split()[0]), "residual_worker_processes": 0,
        "console": {"error_markers": 0, "tracebacks": 0,
                    "unresolved_visual_reference_warnings": console.count("Unresolved reference prim path"),
                    "cuda_device_skip_warnings": console.count("Skipping NVIDIA GPU due CUDA being in bad state")},
        "limitations": [
            "The condition failed; successful evidence validation does not make the episode a safety pass.",
            "One pinned 5-degree uphill L1 geometry and one state seed; no general success-rate estimate.",
            "The unchanged flat actor was tested; no Source training or policy update occurred.",
            "All 17 link-origin predictions and COM conversions were verified. A one-step linear extrapolation is not proof of future continuous physical edge crossing.",
            "GPU0 was shared; whole-device memory and runtime timing cannot be attributed solely to this episode or compared directly with exclusive-device runs.",
            "Kinematic drift and the recorded termination term do not isolate policy, contact dynamics or initialization causes.",
            "The last sample preserves pre-reset root/velocity state; it does not qualify all reset or motion-isolation behavior.",
            "Worker exit code zero is not a success criterion: result and supervisor acceptance correctly record failure.",
            "Foot-slip fractions are thresholded speed proxies; visual-reference and device-enumeration warnings remain in the raw log.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = review()
    if args.write:
        with (HERE / "post_run_review.json").open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
