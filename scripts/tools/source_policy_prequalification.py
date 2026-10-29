"""One separately approved, pinned L1 safety episode; this does not run G3.

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
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/rl_training"))

from scripts.tools.source_parallel_terrain_startup_supervisor import (
    _capture, _gpu_preflight, _sha256, run_monitored_process,
)

PREPARATION = ROOT / "docs/lite3-source-downhill-preparation-20261029/preparation.json"


def select_case(preparation, case_id):
    """Only an explicit single case from the frozen plan may be executed."""
    matches = [case for case in preparation["cases"] if case["case_id"] == case_id]
    if not isinstance(case_id, str) or len(matches) != 1:
        raise ValueError("select exactly one registered --case-id; automatic sweeps are not supported")
    case = deepcopy(matches[0])
    cell = case["cell"]
    if cell["cell_id"] != case_id or type(cell["level"]) is not int or cell["level"] != 1:
        raise ValueError("only the pinned L1 cell is allowed")
    if case["isolated_mesh_matches_bank"] is not True or case["runtime_status"] != "not_run":
        raise ValueError("case must be a prepared, unrun bank-equivalent candidate")
    if (type(case["geometry_seed"]) is not int or case["geometry_seed"] != 10001
            or type(case["initial_state_seed"]) is not int or case["initial_state_seed"] < 0
            or case["bank_row"] != 0 or type(case["bank_column"]) is not int or not 1 <= case["bank_column"] <= 14):
        raise ValueError("case must use the pinned one-row bank and a remaining L1 column")
    if cell["episodes"] != 1 or cell["geometry_seed_range"] != [case["geometry_seed"]] * 2 or cell["initial_state_seed_range"] != [case["initial_state_seed"]] * 2:
        raise ValueError("run case must specify exactly one paired seed and one episode")
    from rl_training.source_scan.terrain_config import source_geometry_difficulty
    if source_geometry_difficulty(cell) != case["geometry_difficulty"]:
        raise ValueError("geometry realization differs from bank case")
    return case


def make_run_manifest(preparation, case, *, revision, gpu, processes, preparation_sha256):
    # Never expand the preparation dict into an execution manifest: its status,
    # simulation flags and multi-case registration fields describe another stage.
    return {
        "schema": "lite3_single_condition_run_spec_v4", "manifest_kind": "immutable_run_spec",
        "tile_bounds_diagnostics_required": True,
        "tile_bounds_schema": "lite3_tile_bounds_predicate_v2",
        "link_reference_verification_required": True,
        "revision": revision, "preparation_sha256": preparation_sha256,
        "case_id": case["case_id"], "cell": case["cell"],
        "geometry_seed": case["geometry_seed"], "initial_state_seed": case["initial_state_seed"],
        "episode_count": 1, "checkpoint": preparation["checkpoint"],
        "expected_body_names": preparation["expected_body_names"],
        "input_sha256": preparation["input_sha256"],
        "bank_geometry": {key: case[key] for key in (
            "bank_row", "bank_column", "subterrain", "geometry_difficulty", "mesh_fingerprint")},
        "gpu_before": gpu, "processes_before": processes,
        "gpu_admission": {"minimum_available_mib": 12288, "allow_existing_compute_processes": True,
                          "memory_basis": "memory.total - memory.used"},
        "startup_timeout_s": 120, "total_timeout_s": 240, "max_control_steps": 2000,
        "num_envs": 1, "gpu_index": 0, "command_b": [0.3, 0.0, 0.0],
        "qualification_scope": "one exact L1 tile and one state seed; no shared-bank motion/G3 qualification",
    }


def capture_kinematics(env):
    from rl_training.source_scan.rigid_body_reference import capture_root_reference_state
    from rl_training.source_scan.tile_bounds_diagnostics import json_state
    data = env.scene["robot"].data
    return {
        "schema": "lite3_root_kinematics_v2",
        "root_reference_state": json_state(capture_root_reference_state(data)),
        "legacy_field_semantics": {
            "root_quat_w_wxyz": "root_link_orientation_world",
            "root_lin_vel_w_mps": "root_com_velocity_world",
            "root_lin_vel_b_mps": "root_com_velocity_in_link_axes",
        },
        "root_quat_w_wxyz": data.root_quat_w[0].detach().cpu().tolist(),
        "root_lin_vel_w_mps": data.root_lin_vel_w[0].detach().cpu().tolist(),
        "root_lin_vel_b_mps": data.root_lin_vel_b[0].detach().cpu().tolist(),
        "projected_gravity_b": data.projected_gravity_b[0].detach().cpu().tolist(),
        "command_b_vx_vy_wz": env.command_manager.get_command("base_velocity")[0, :3].detach().cpu().tolist(),
    }


def sample_callbacks(env, run_dir, *, capture_tile_bounds=False):
    terminal = {}
    if capture_tile_bounds:
        from rl_training.source_scan.tile_bounds_diagnostics import TileBoundsRecorder
        if env.num_envs != 1 or getattr(env, "_source_tile_bounds_observer", None) is not None:
            raise ValueError("tile-bound diagnostics require one environment and no existing observer")
        env._source_tile_bounds_observer = TileBoundsRecorder(run_dir / "tile_bounds.jsonl")

    def before_reset(_ids, step):
        terminal[step] = capture_kinematics(env)

    def observe(step, sample):
        captured_before_reset = step in terminal
        kinematics = terminal.pop(step) if captured_before_reset else capture_kinematics(env)
        if any(not math.isfinite(x) or abs(x - target) > 1e-6
               for x, target in zip(kinematics["command_b_vx_vy_wz"], (0.3, 0.0, 0.0), strict=True)):
            raise ValueError("runtime command changed outside the frozen range")
        row = {"step": step, "sample": sample, "kinematics": kinematics,
               "kinematics_phase": "pre_reset" if captured_before_reset else "post_control_step"}
        with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    return before_reset, observe


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
    import torch
    from rl_training.source_scan.rigid_body_reference import (
        BODY_APIS, capture_body_reference_state, capture_root_reference_state, validate_body_reference_state,
    )
    from rl_training.source_scan.tile_bounds_diagnostics import json_state
    body_names = list(env.scene["robot"].body_names)
    manager = env.termination_manager
    expected_terms = {"source_terrain_tile_bounds", "illegal_contact", "bad_orientation_2"}
    if set(manager.active_terms) != expected_terms:
        raise ValueError("runtime termination terms differ from the frozen set")
    params = manager.get_term_cfg("source_terrain_tile_bounds").params
    guard = params["asset_cfg"]
    ids = list(range(len(body_names)))[guard.body_ids] if isinstance(guard.body_ids, slice) else list(guard.body_ids)
    if body_names != expected_body_names or ids != list(range(17)):
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
    data = env.scene["robot"].data
    reference = capture_body_reference_state(data)
    reference["state_dtype"] = str(reference["body_positions_w"].dtype)
    root_reference = capture_root_reference_state(data)
    conversion = validate_body_reference_state(reference)
    root_conversion = validate_body_reference_state(root_reference)
    if conversion["status"] != "verified" or root_conversion["status"] != "verified":
        raise ValueError("runtime link/COM reference state must be finite and verified")
    if any(not torch.allclose(root_reference[key], reference[key][:, :1], atol=2e-6, rtol=2e-6)
           for key in BODY_APIS):
        raise ValueError("runtime root reference differs from body index zero")
    return {
        "explicit_body_apis": dict(BODY_APIS),
        "body_reference_state": json_state(reference), "body_reference_review": conversion,
        "root_reference_state": json_state(root_reference), "root_reference_review": root_conversion,
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
    prepared_case = select_case(json.loads(PREPARATION.read_text()), manifest["case_id"])
    if (manifest["cell"] != prepared_case["cell"]
            or any(manifest[k] != prepared_case[k] for k in ("geometry_seed", "initial_state_seed"))
            or manifest["bank_geometry"] != {k: prepared_case[k] for k in manifest["bank_geometry"]}):
        raise ValueError("worker run specification differs from prepared case")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0", kit_args="--/log/flushStandardStreamOutput=true").app
    try:
        actor = load_frozen_source_actor(Path(manifest["checkpoint"]), device="cuda:0")
        before = _tensor_sha256(actor)
        cell = manifest["cell"]
        geometry_seed, initial_seed = manifest["geometry_seed"], manifest["initial_state_seed"]
        wrapped, _ = _make_wrapped_env(cell, geometry_seed, initial_seed, device="cuda:0")
        try:
            env = wrapped.unwrapped
            readback = runtime_readback(env, manifest["expected_body_names"])
            generator = env.cfg.scene.terrain.terrain_generator
            if generator.seed != geometry_seed or tuple(generator.difficulty_range) != (prepared_case["geometry_difficulty"],) * 2:
                raise ValueError("runtime generator differs from frozen bank realization")
            readback["geometry_seed"] = generator.seed
            readback["geometry_difficulty_range"] = list(generator.difficulty_range)
            write_json(run_dir / "runtime_readback.json", readback)
            before_reset, observe = sample_callbacks(env, run_dir, capture_tile_bounds=True)

            # The runner seeds and resets again before the first policy decision.
            wrapped.seed(initial_seed)
            wrapped.reset()
            (run_dir / "trace.log").write_text("initial-reset-complete\n")
            record = _run_episode(wrapped, actor, cell, geometry_seed, initial_seed, 1,
                                  max_steps=2000, sample_observer=observe, pre_reset_observer=before_reset)
            record["episode_type"] = "g3_prerequisite_single_condition"
            after = _tensor_sha256(actor)
            unchanged = before == after and not any(p.requires_grad for p in actor.parameters())
            passed = unchanged and episode_passed(record)
            write_json(run_dir / "result.json", {
                "status": "single_condition_safety_passed" if passed else "failed",
                "scope": "one pinned L1 episode only; no shared-bank motion qualification",
                "case_id": manifest["case_id"], "bank_geometry": manifest["bank_geometry"],
                "g3_qualified": False, "policy_updated": not unchanged,
                "ppo_initialized": False, "actor_sha256_before": before,
                "actor_sha256_after": after, "record": record,
            })
            from rl_training.source_scan.tile_bounds_diagnostics import validate_tile_bounds_trace
            diagnostic_review = validate_tile_bounds_trace(
                run_dir / "tile_bounds.jsonl", record, manifest["expected_body_names"], require_link_reference=True)
            write_json(run_dir / "tile_bounds_review.json", diagnostic_review)
            if diagnostic_review["reference_conversion"]["status"] != "verified":
                raise ValueError("nonfinite link-reference evidence is not verified")
            return 0 if passed else 1
        finally:
            wrapped.close()
    finally:
        app.close()


def supervise(run_dir, case_id=None):
    preparation = json.loads(PREPARATION.read_text())
    case = select_case(preparation, case_id)
    hashes = preparation["input_sha256"]
    verify_hashes(hashes)
    if _capture(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("refusing a dirty worktree")
    if _capture(["git", "-C", str(ROOT / "deep_robotics_model"), "status", "--porcelain"], cwd=ROOT):
        raise ValueError("refusing a dirty robot-model submodule")
    before_gpu, before_processes = _gpu_preflight(
        0, root=ROOT, minimum_available_mib=12288, allow_existing_compute_processes=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = make_run_manifest(preparation, case,
        revision=_capture(["git", "rev-parse", "HEAD"], cwd=ROOT),
        gpu=before_gpu, processes=before_processes, preparation_sha256=_sha256(PREPARATION))
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
        record = result["record"]
        from rl_training.source_scan.tile_bounds_diagnostics import validate_tile_bounds_trace
        diagnostic_review = validate_tile_bounds_trace(
            run_dir / "tile_bounds.jsonl", record, manifest["expected_body_names"], require_link_reference=True)
        if diagnostic_review["reference_conversion"]["status"] != "verified":
            errors.append("nonfinite_link_reference_unverifiable")
        if diagnostic_review != json.loads((run_dir / "tile_bounds_review.json").read_text()):
            errors.append("tile_bounds_review_mismatch")
        if result["case_id"] != case_id or result["bank_geometry"] != manifest["bank_geometry"] or any(record[key] != expected for key, expected in (
            ("cell_id", case_id), ("geometry_seed", case["geometry_seed"]),
            ("initial_state_seed", case["initial_state_seed"]), ("episode_index", 1),
        )):
            errors.append("case_provenance_mismatch")
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    try:
        after_gpu, after_processes = _gpu_preflight(
            0, root=ROOT, minimum_available_mib=12288, allow_existing_compute_processes=True)
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
    parser.add_argument("--case-id", help="exactly one frozen L1 case; no automatic sweep")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker and args.case_id is None:
        parser.error("--case-id is required for a separately approved single run")
    try:
        return run_worker(args.run_dir.resolve()) if args.worker else supervise(args.run_dir.resolve(), args.case_id)
    except Exception:
        traceback.print_exc()
        print("[parallel-terrain-smoke] fatal-exception-recorded", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
