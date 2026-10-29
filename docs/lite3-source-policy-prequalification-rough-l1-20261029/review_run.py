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
    require(receipt["supervisor_exit_code"] == 0 and receipt["attempt_count"] == 1, "execution receipt")
    require(manifest["revision"] == receipt["frozen_revision"], "revision changed")
    for name, digest in receipt["archive_raw_sha256"].items():
        require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, f"raw archive changed: {name}")
    verify_hashes(manifest["input_sha256"])
    preparation = ROOT / "docs/lite3-source-followup-preparation-20261029/preparation.json"
    require(hashlib.sha256(preparation.read_bytes()).hexdigest() == manifest["preparation_sha256"], "preparation hash")
    prepared = json.loads(preparation.read_text())
    case = select_case(prepared, "multi_scale_rough_L1_forward_lane")
    require(manifest["revision"] == "75589cb07f72dfe6758c070963b2d9708561a632", "frozen runtime revision")
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
    require(outcome["status"] == "passed" and outcome["errors"] == [], "supervisor outcome")
    execution = outcome["execution"]
    require(execution["exit_code"] == 0 and execution["elapsed_s"] < 240, "execution status/budget")
    for key in ("stop_reason", "failure_marker", "collision_cooking_failure", "fatal_exception"):
        require(execution[key] is None, key)
    require(result["status"] == "single_condition_safety_passed" and episode_passed(record), "episode acceptance")
    require(result["policy_updated"] is False and result["ppo_initialized"] is False, "frozen policy scope")
    require(result["g3_qualified"] is False and outcome["g3_qualified"] is False, "G3 must remain unqualified")
    require(result["actor_sha256_before"] == result["actor_sha256_after"], "actor changed")
    require(record["cell_id"] == "multi_scale_rough_L1_forward_lane", "unexpected cell")
    require((record["geometry_seed"], record["initial_state_seed"], record["episode_index"]) == (10001, 20030, 1), "seeds/episode")
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
    require(hold == record["stable_hold_steps_observed"] == 50, "hold reconstruction")
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
    require(all(row["kinematics_phase"] == "post_control_step" for row in samples), "unexpected sampling phase")
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
    require(record["pre_reset_capture_observed"] is False and record["post_reset_root_position_w_m"] is None, "unexpected reset")
    gpu0 = next(line for line in after["gpu"].splitlines() if line.startswith("0, ")).split(", ")
    require(gpu0[1] == "GPU-044ddc24-86ef-a67d-7e14-9358300bc3dc", "GPU UUID changed")
    require(not any(line.startswith(gpu0[1]) for line in after["processes"].splitlines()), "GPU0 has compute process after exit")
    require(read_json("processes_after.json")["matching_runtime_processes"] == [], "worker still running")
    console = (HERE / "console.log").read_text()
    require(find_collision_cooking_failure(console) is None and find_fatal_exception(console) is None, "fatal log marker")
    require("Traceback (most recent call last)" not in console and "[Error]" not in console and "[ERROR]" not in console, "console error")
    return {
        "schema": "lite3_rough_l1_policy_prequalification_post_run_review_v1",
        "review_result": "single_condition_safety_passed_with_scope_limits",
        "revision": manifest["revision"], "input_hashes_verified": len(manifest["input_sha256"]),
        "raw_files_verified": len(receipt["archive_raw_sha256"]),
        "policy_updated": False, "ppo_initialized": False, "g3_qualified": False, "g4_qualified": False,
        "control_steps": len(samples), "simulated_control_time_s": len(samples) * 0.02,
        "action_trace_rows": len(trace), "first_goal_step": first_goal_step, "reconstructed_final_hold_steps": hold,
        "metrics_reconstructed_from_samples": comparisons,
        "invalid_states": 0, "nonfoot_contact_events": 0, "safety_rejections": 0, "projections": 0,
        "kinematics": {"rows_verified": len(kinematics), "max_quaternion_norm_error": quat_norm_error,
            "max_body_to_world_velocity_error_mps": velocity_error, "max_projected_gravity_error": gravity_error,
            "full_command_verified": True, "pre_reset_rows": 0},
        "bank_geometry": manifest["bank_geometry"],
        "root_yaw_end_rad": record["root_yaw_end_rad"],
        "foot_slip_fraction": record["foot_slip_contact_fraction"],
        "foot_slip_numerator": slipping_samples, "foot_slip_denominator": contact_samples,
        "actor_sha256": result["actor_sha256_after"],
        "execution_elapsed_s": execution["elapsed_s"], "sampled_gpu_peak_mib": max(row["memory_used_mib"] for row in gpu),
        "gpu_sample_count": len(gpu), "gpu0_after_mib": int(gpu0[3].split()[0]), "residual_worker_processes": 0,
        "console": {"error_markers": 0, "tracebacks": 0,
                    "unresolved_visual_reference_warnings": console.count("Unresolved reference prim path"),
                    "cuda_device_skip_warnings": console.count("Skipping NVIDIA GPU due CUDA being in bad state")},
        "limitations": [
            "One pinned multi-scale rough L1 geometry and one state seed; no general success-rate or six-family qualification.",
            "This is transfer by the unchanged flat actor, not a learned Source skill or a training result.",
            "Lateral drift and yaw deviation remain motion-quality observations; no straight-line quality gate was added retrospectively.",
            "Completion caused no manager termination or automatic reset; terminal kinematic capture is not exercised here.",
            "No pairwise physics-substep contact or full rigid-body trace; shared-bank motion and full G3/G4 remain unqualified.",
            "Foot-slip fractions are thresholded world-horizontal foot-speed proxies, not a direct friction measurement.",
            "Device-enumeration and foot visual-reference warnings remain in the raw log.",
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
