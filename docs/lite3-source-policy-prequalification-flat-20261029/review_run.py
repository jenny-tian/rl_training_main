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

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from scripts.tools.source_policy_prequalification import episode_passed, verify_hashes
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
    preparation = ROOT / "docs/lite3-source-g3-offline-preparation-20261029/preparation.json"
    require(hashlib.sha256(preparation.read_bytes()).hexdigest() == manifest["preparation_sha256"], "preparation hash")
    require(outcome["status"] == "passed" and outcome["errors"] == [], "supervisor outcome")
    execution = outcome["execution"]
    require(execution["exit_code"] == 0 and execution["elapsed_s"] < 240, "execution status/budget")
    for key in ("stop_reason", "failure_marker", "collision_cooking_failure", "fatal_exception"):
        require(execution[key] is None, key)
    require(result["status"] == "single_condition_safety_passed" and episode_passed(record), "episode acceptance")
    require(result["policy_updated"] is False and result["ppo_initialized"] is False, "frozen policy scope")
    require(result["g3_qualified"] is False and outcome["g3_qualified"] is False, "G3 must remain unqualified")
    require(result["actor_sha256_before"] == result["actor_sha256_after"], "actor changed")
    require(record["cell_id"] == "flat_L1_forward_flat", "unexpected cell")
    require((record["geometry_seed"], record["initial_state_seed"], record["episode_index"]) == (10000, 20000, 1), "seeds/episode")
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
    require(record["pre_reset_capture_observed"] is False and record["post_reset_root_position_w_m"] is None, "unexpected reset")
    gpu0 = next(line for line in after["gpu"].splitlines() if line.startswith("0, ")).split(", ")
    require(gpu0[1] == "GPU-044ddc24-86ef-a67d-7e14-9358300bc3dc", "GPU UUID changed")
    require(not any(line.startswith(gpu0[1]) for line in after["processes"].splitlines()), "GPU0 has compute process after exit")
    require(receipt["post_run_process_check"]["exit_code"] == 1, "worker still running")
    console = (HERE / "console.log").read_text()
    require(find_collision_cooking_failure(console) is None and find_fatal_exception(console) is None, "fatal log marker")
    require("Traceback (most recent call last)" not in console and "[Error]" not in console and "[ERROR]" not in console, "console error")
    return {
        "schema": "lite3_flat_policy_prequalification_post_run_review_v1",
        "review_result": "single_condition_safety_passed_with_scope_limits",
        "revision": manifest["revision"], "input_hashes_verified": len(manifest["input_sha256"]),
        "raw_files_verified": len(receipt["archive_raw_sha256"]),
        "policy_updated": False, "ppo_initialized": False, "g3_qualified": False, "g4_qualified": False,
        "control_steps": len(samples), "simulated_control_time_s": len(samples) * 0.02,
        "action_trace_rows": len(trace), "first_goal_step": first_goal_step, "reconstructed_final_hold_steps": hold,
        "metrics_reconstructed_from_samples": comparisons,
        "invalid_states": 0, "nonfoot_contact_events": 0, "safety_rejections": 0, "projections": 0,
        "actor_sha256": result["actor_sha256_after"],
        "execution_elapsed_s": execution["elapsed_s"], "sampled_gpu_peak_mib": max(row["memory_used_mib"] for row in gpu),
        "gpu_sample_count": len(gpu), "gpu0_after_mib": int(gpu0[3].split()[0]), "residual_worker_processes": 0,
        "console": {"error_markers": 0, "tracebacks": 0,
                    "unresolved_visual_reference_warnings": console.count("Unresolved reference prim path"),
                    "cuda_device_skip_warnings": console.count("Skipping NVIDIA GPU due CUDA being in bad state")},
        "limitations": [
            "One Flat L1 condition only; does not qualify other terrains, shared-bank motion, full G3/G4 or PPO.",
            "Peak lateral drift is 0.71997 m and final yaw is -0.28623 rad. These did not violate the registered safety gate; straight-line tracking quality has not been qualified.",
            "Completion caused no manager termination or automatic reset; this episode does not exercise pre-reset capture or post-reset physics.",
            "No physics-substep pairwise contact or full rigid-body dynamics trace was collected, so motion isolation is not established.",
            "CUDA visibility/device enumeration and foot visual-reference warnings remain in the raw log; their general resolution is not established.",
            "manifest.json inherits status/simulation_run and cell episode-range fields from preparation. Actual authorization is recorded in execution_receipt.json; result/outcome and this review describe the executed single episode. Raw manifest was not rewritten.",
            "The updated base-scanner configuration still needs current-revision shared-bank startup/reset validation.",
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
