"""One separately approved Flat L1 safety episode; this does not run G3.

The supervisor freezes hashes, checks GPU0, and enforces 120/240 s timeouts.
The worker uses a deterministic actor without constructing a PPO runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/rl_training"))

from scripts.tools.source_parallel_terrain_startup_supervisor import (
    _capture, _gpu_preflight, _sha256, run_monitored_process,
)

PREPARATION = ROOT / "docs/lite3-source-g3-offline-preparation-20261029/preparation.json"


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n")


def verify_hashes(hashes):
    if not hashes:
        raise ValueError("empty frozen file inventory")
    for name, expected in hashes.items():
        if _sha256(ROOT / name) != expected:
            raise ValueError(f"frozen input changed: {name}")


def episode_passed(record):
    """Only a complete safe episode qualifies this one frozen condition."""
    try:
        if (record["completed"] is not True or record["termination_reason"] != "completion"
                or record["termination_terms"] != []
                or record["termination_capture_test_injected"] is not False
                or record["stable_hold_steps_required"] != 50 or record["stable_hold_steps_observed"] != 50
                or type(record["steps"]) is not int or not 50 <= record["steps"] <= 2000):
            return False
        for key in ("invalid_state_count", "safety_filter_rejection_or_emergency_count", "nonfoot_contact_count"):
            if type(record[key]) is not int or record[key] != 0:
                return False
        for key, lower, upper in (
            ("forward_progress_m", 9.0, 10.5),
            ("min_base_height_m", 0.20, math.inf),
            ("max_projected_gravity_xy", 0.0, 0.65),
            ("max_joint_speed_rad_s", 0.0, 30.0),
            ("mean_command_velocity_x_mps", 0.3 - 1e-6, 0.3 + 1e-6),
        ):
            value = record[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                return False
        trace = record["action_trace"]
        if not isinstance(trace, list) or len(trace) != record["steps"]:
            return False
        for index, row in enumerate(trace, 1):
            if row["step"] != index or row["safety_paused"] is not False:
                return False
            for name in ("a_policy_raw", "a_base", "a_requested", "a_executed_command"):
                if len(row[name]) != 12 or not all(math.isfinite(x) for x in row[name]):
                    return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def runtime_readback(env, expected_body_names):
    body_names = list(env.scene["robot"].body_names)
    manager = env.termination_manager
    expected_terms = {"source_terrain_tile_bounds", "illegal_contact", "bad_orientation_2"}
    if set(manager.active_terms) != expected_terms:
        raise ValueError("runtime termination terms differ from the frozen set")
    params = manager.get_term_cfg("source_terrain_tile_bounds").params
    guard = params["asset_cfg"]
    ids = list(range(len(body_names)))[guard.body_ids] if isinstance(guard.body_ids, slice) else list(guard.body_ids)
    if body_names != expected_body_names or sorted(ids) != list(range(17)):
        raise ValueError("runtime guard does not cover the pinned 17-body mapping")
    if env.num_envs != 1 or not math.isclose(env.step_dt, 0.02) or not math.isclose(env.cfg.sim.dt, 0.005):
        raise ValueError("runtime batch/control interval differs from the frozen condition")
    if not math.isclose(env.cfg.scene.height_scanner_base.update_period, 0.005):
        raise ValueError("safety patch does not refresh each physics step")
    if tuple(params["size"]) != (12.0, 4.0) or any(not math.isclose(params[name], value) for name, value in (
        ("flat_length_m", 3.0), ("collision_radius_m", 0.27), ("numerical_margin_m", 0.001),
    )):
        raise ValueError("runtime tile guard geometry changed")
    contact = manager.get_term_cfg("illegal_contact").params
    tilt = manager.get_term_cfg("bad_orientation_2").params
    if contact["threshold"] != 5.0 or contact["consecutive_steps"] != 3 or tilt["max_xy_norm"] != 0.65:
        raise ValueError("runtime safety termination thresholds changed")
    return {
        "body_names": body_names, "guard_body_ids": [int(i) for i in ids],
        "num_envs": env.num_envs, "step_dt": env.step_dt,
        "base_scan_update_period": env.cfg.scene.height_scanner_base.update_period,
        "termination_terms": list(manager.active_terms),
        "tile_guard": {name: params[name] for name in (
            "size", "flat_length_m", "collision_radius_m", "numerical_margin_m")},
        "contact_threshold_n": contact["threshold"], "contact_consecutive_steps": contact["consecutive_steps"],
        "max_projected_gravity_xy": tilt["max_xy_norm"],
    }


def run_worker(run_dir):
    import torch
    from rl_training.source_scan.frozen_actor import load_frozen_source_actor
    from scripts.reinforcement_learning.online_adaptation.run_source_qualification_scan import (
        _make_wrapped_env, _run_episode, _tensor_sha256,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    if _sha256(PREPARATION) != manifest["preparation_sha256"]:
        raise ValueError("preparation changed before worker launch")
    verify_hashes(manifest["input_sha256"])
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0", kit_args="--/log/flushStandardStreamOutput=true").app
    try:
        actor = load_frozen_source_actor(Path(manifest["checkpoint"]), device="cuda:0")
        before = _tensor_sha256(actor)
        cell = manifest["cell"]
        wrapped, _ = _make_wrapped_env(cell, 10000, 20000, device="cuda:0")
        try:
            env = wrapped.unwrapped
            write_json(run_dir / "runtime_readback.json", runtime_readback(env, manifest["expected_body_names"]))

            def observe(step, sample):
                # A terminal callback provides the sample before automatic reset.
                command = env.command_manager.get_command("base_velocity")[0, :3]
                if not torch.allclose(command, command.new_tensor([0.3, 0.0, 0.0]), atol=1e-6, rtol=0):
                    raise ValueError("runtime command changed outside the frozen range")
                with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": step, "sample": sample}, allow_nan=False) + "\n")

            # The runner seeds and resets again before the first policy decision.
            wrapped.seed(20000)
            wrapped.reset()
            (run_dir / "trace.log").write_text("initial-reset-complete\n")
            record = _run_episode(wrapped, actor, cell, 10000, 20000, 1,
                                  max_steps=2000, sample_observer=observe)
            record["episode_type"] = "g3_prerequisite_single_condition"
            after = _tensor_sha256(actor)
            unchanged = before == after and not any(p.requires_grad for p in actor.parameters())
            passed = unchanged and episode_passed(record)
            write_json(run_dir / "result.json", {
                "status": "single_condition_safety_passed" if passed else "failed",
                "scope": "one Flat L1 episode only; no shared-bank motion qualification",
                "g3_qualified": False, "policy_updated": not unchanged,
                "ppo_initialized": False, "actor_sha256_before": before,
                "actor_sha256_after": after, "record": record,
            })
            return 0 if passed else 1
        finally:
            wrapped.close()
    finally:
        app.close()


def supervise(run_dir):
    preparation = json.loads(PREPARATION.read_text())
    hashes = preparation["input_sha256"]
    verify_hashes(hashes)
    if _capture(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("refusing a dirty worktree")
    if _capture(["git", "-C", str(ROOT / "deep_robotics_model"), "status", "--porcelain"], cwd=ROOT):
        raise ValueError("refusing a dirty robot-model submodule")
    before_gpu, before_processes = _gpu_preflight(0, root=ROOT, minimum_available_mib=12288)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        **preparation, "revision": _capture(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "preparation_sha256": _sha256(PREPARATION),
        "gpu_before": before_gpu, "processes_before": before_processes,
        "startup_timeout_s": 120, "total_timeout_s": 240,
    }
    write_json(run_dir / "manifest.json", manifest)

    def sample_gpu(elapsed):
        value = _capture(["nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], cwd=ROOT, timeout_s=1)
        memory, utilization = [int(x.strip()) for x in value.split(",")]
        with (run_dir / "gpu_samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"elapsed_s": elapsed, "memory_used_mib": memory,
                                     "utilization_percent": utilization}) + "\n")

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    # Pin logical cuda:0 to the physical GPU that actually passed preflight.
    gpu_row = next(row for row in before_gpu.splitlines() if row.split(", ")[0] == "0")
    child_env["CUDA_VISIBLE_DEVICES"] = gpu_row.split(", ")[1]
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--run-dir", str(run_dir)]
    execution = run_monitored_process(
        command, console_path=run_dir / "console.log", trace_path=run_dir / "trace.log",
        startup_timeout_s=120, total_timeout_s=240, cwd=ROOT, env=child_env,
        sample_interval_s=1, sample_callback=sample_gpu,
    )
    errors = []
    if execution["exit_code"] != 0 or any(execution[key] is not None for key in (
        "stop_reason", "collision_cooking_failure", "fatal_exception",
    )):
        errors.append("worker_or_monitor_failed")
    try:
        verify_hashes(hashes)
        if _sha256(PREPARATION) != manifest["preparation_sha256"]:
            raise ValueError("preparation changed during execution")
        result = json.loads((run_dir / "result.json").read_text())
        if (result["status"] != "single_condition_safety_passed"
                or not episode_passed(result["record"]) or result["policy_updated"] is not False
                or result["ppo_initialized"] is not False or result["g3_qualified"] is not False
                or result["actor_sha256_before"] != result["actor_sha256_after"]):
            errors.append("episode_or_actor_failed")
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    try:
        after_gpu, after_processes = _gpu_preflight(0, root=ROOT, minimum_available_mib=12288)
        write_json(run_dir / "gpu_after.json", {"gpu": after_gpu, "processes": after_processes})
    except Exception as error:
        errors.append(str(error))
    outcome = {"status": "passed" if not errors else "failed", "execution": execution,
               "errors": errors, "g3_qualified": False}
    write_json(run_dir / "outcome.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False), flush=True)
    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        return run_worker(args.run_dir.resolve()) if args.worker else supervise(args.run_dir.resolve())
    except Exception:
        traceback.print_exc()
        print("[parallel-terrain-smoke] fatal-exception-recorded", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
