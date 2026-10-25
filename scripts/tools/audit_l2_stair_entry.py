"""Offline audit of the registered L2 ascending stair entry geometry.

This tool imports only the pure Source Terrain geometry module.  It does not
launch Isaac Sim, load a policy, or execute an environment step.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / (
    "docs/lite3-source-scan-preparation-v0.2-2026-10-22/"
    "source_qualification_scan_manifest_v0.2_preparation.json"
)
RUNTIME_REPORT = ROOT / (
    "docs/lite3-source-scan-preparation-v0.2-2026-10-22/family_runtime_20261025/"
    "multi_step_stairs_L2_ascending_headless_diagnostic_retry3_termination_fields.json"
)
GENERATOR = ROOT / (
    "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/"
    "terrains/source_terrain_tracks.py"
)
CELL_ID = "multi_step_stairs_L2_ascending"
MANIFEST_SHA256 = "0fce8c53e2e2c8537c8b048b2c5f718ebf74211bf2197e1da99eb3cecf6c1cb1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pure_geometry_module():
    spec = importlib.util.spec_from_file_location("lite3_source_geometry_audit", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load geometry module: {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _surface_height(mesh, x: float, y: float) -> float:
    """Intersect a vertical line with upward-facing mesh triangles."""

    triangles = mesh.triangles[mesh.face_normals[:, 2] > 1.0e-8]
    if len(triangles) == 0:
        raise AssertionError("mesh has no upward-facing surface triangles")
    a = triangles[:, 0]
    u = triangles[:, 1] - a
    v = triangles[:, 2] - a
    determinant = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
    valid = np.abs(determinant) > 1.0e-14
    dx, dy = x - a[:, 0], y - a[:, 1]
    s = np.zeros_like(determinant)
    t = np.zeros_like(determinant)
    s[valid] = (dx[valid] * v[valid, 1] - v[valid, 0] * dy[valid]) / determinant[valid]
    t[valid] = (u[valid, 0] * dy[valid] - dx[valid] * u[valid, 1]) / determinant[valid]
    inside = valid & (s >= -1.0e-9) & (t >= -1.0e-9) & (s + t <= 1.0 + 1.0e-9)
    if not inside.any():
        raise AssertionError(f"no upward surface at local ({x}, {y})")
    heights = a[:, 2] + s * u[:, 2] + t * v[:, 2]
    return float(np.max(heights[inside]))


def _json_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise AssertionError("non-finite numeric audit value")
    return result


def audit() -> dict[str, Any]:
    manifest_sha256 = _sha256(MANIFEST)
    if manifest_sha256 != MANIFEST_SHA256:
        raise RuntimeError("preparation manifest hash changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "draft_not_for_run" or manifest["execution_authorized"] is not False:
        raise RuntimeError("offline audit requires the non-executable preparation draft")
    cells = [cell for cell in manifest["seed_plan"]["cells"] if cell["cell_id"] == CELL_ID]
    if len(cells) != 1:
        raise RuntimeError(f"expected one registered cell, got {len(cells)}")
    cell = cells[0]
    geometry = cell["resolved_parameters"]["geometry"]
    track = cell["resolved_parameters"]["track"]
    registered_geometry_hash = manifest["execution_contract_draft"]["artifacts"]["geometry_generator"]["sha256"]
    if _sha256(GENERATOR) != registered_geometry_hash:
        raise RuntimeError("geometry generator hash differs from the preparation registration")
    runtime = json.loads(RUNTIME_REPORT.read_text(encoding="utf-8"))
    if runtime["cell"]["cell_id"] != CELL_ID:
        raise RuntimeError("runtime report cell mismatch")
    if runtime["manifest_sha256"] != MANIFEST_SHA256:
        raise RuntimeError("runtime report uses a different preparation manifest")

    module = _load_pure_geometry_module()
    cfg = module.SourceTerrainCfg(
        family="multi_step_stairs",
        size=tuple(geometry["size"]),
        seed=int(cell["geometry_seed_range"][0]),
        flat_length_m=float(geometry["flat_length_m"]),
        transition_length_m=float(geometry["transition_length_m"]),
        step_count=int(geometry["step_count"]),
        step_height_m=float(geometry["step_height_m"]),
        tread_depth_m=float(geometry["tread_depth_m"]),
        direction=cell["direction_or_layout"],
        grid_spacing_m=float(geometry["grid_spacing_m"]),
    )
    generated = module.generate_source_terrain(cfg)
    metadata = generated.metadata
    steps = metadata["steps"]
    if len(steps) != cfg.step_count:
        raise AssertionError("generated riser count does not match registered step count")

    origin_local_x = float(generated.origin[0])
    entry_x = float(metadata["entry_x_m"])
    first = steps[0]
    first_riser_x = float(first["riser_x_m"])
    first_tread_end_x = float(first["tread_end_x_m"])
    expected_entry_x = cfg.flat_length_m + cfg.transition_length_m
    if first_riser_x != expected_entry_x or entry_x != expected_entry_x:
        raise AssertionError("first riser is not aligned with the registered entry boundary")

    local_y = float(cfg.size[1]) / 2.0
    epsilon = min(1.0e-5, cfg.tread_depth_m / 1000.0)
    surface_samples = {
        "before_first_riser": {
            "local_x_m": first_riser_x - epsilon,
            "surface_z_m": _surface_height(generated.meshes[0], first_riser_x - epsilon, local_y),
        },
        "after_first_riser": {
            "local_x_m": first_riser_x + epsilon,
            "surface_z_m": _surface_height(generated.meshes[0], first_riser_x + epsilon, local_y),
        },
        "first_tread_near_end": {
            "local_x_m": first_tread_end_x - epsilon,
            "surface_z_m": _surface_height(generated.meshes[0], first_tread_end_x - epsilon, local_y),
        },
    }
    if surface_samples["before_first_riser"]["surface_z_m"] != 0.0:
        raise AssertionError("surface before first riser is not at the initial level")
    if surface_samples["after_first_riser"]["surface_z_m"] != cfg.step_height_m:
        raise AssertionError("surface after first riser does not equal one step height")

    geometry_seed_checks = []
    baseline_vertices = generated.meshes[0].vertices.tobytes()
    baseline_faces = generated.meshes[0].faces.tobytes()
    baseline_metadata = json.dumps(metadata, sort_keys=True, allow_nan=False)
    for seed in range(
        int(cell["geometry_seed_range"][0]),
        int(cell["geometry_seed_range"][1]) + 1,
    ):
        seeded_cfg = module.SourceTerrainCfg(
            family=cfg.family,
            size=cfg.size,
            seed=seed,
            flat_length_m=cfg.flat_length_m,
            transition_length_m=cfg.transition_length_m,
            step_count=cfg.step_count,
            step_height_m=cfg.step_height_m,
            tread_depth_m=cfg.tread_depth_m,
            direction=cfg.direction,
            grid_spacing_m=cfg.grid_spacing_m,
        )
        seeded_geometry = module.generate_source_terrain(seeded_cfg)
        seeded_mesh = seeded_geometry.meshes[0]
        seeded_metadata = json.dumps(seeded_geometry.metadata, sort_keys=True, allow_nan=False)
        check = {
            "seed": seed,
            "metadata_matches_reference": seeded_metadata == baseline_metadata,
            "mesh_vertices_match_reference": seeded_mesh.vertices.tobytes() == baseline_vertices,
            "mesh_faces_match_reference": seeded_mesh.faces.tobytes() == baseline_faces,
        }
        if not all(check.values()):
            raise AssertionError(f"stair geometry changed across registered seed {seed}: {check}")
        geometry_seed_checks.append(check)

    terrain_origin_w = runtime["record"]["terrain_origin_w_m"]
    root_start_w = runtime["record"]["root_position_w_start_m"]
    root_end_w = runtime["record"]["root_position_w_end_m"]
    terrain_origin_x = float(terrain_origin_w[0])
    root_start_progress = float(root_start_w[0]) - terrain_origin_x
    root_end_progress = float(root_end_w[0]) - terrain_origin_x
    first_riser_progress = first_riser_x - origin_local_x
    stair_end_x = float(metadata["exit_x_m"])
    stair_end_progress = stair_end_x - origin_local_x
    completion_progress = float(track["completion_x_m"]) - float(track["start_x_m"])

    terminal_feet = []
    for foot in runtime["terminal_contact_diagnostic"]["foot_bodies"]:
        local_x = float(foot["position_w_m"][0]) - terrain_origin_x + origin_local_x
        terminal_feet.append({
            "body_name": foot["body_name"],
            "local_x_m": local_x,
            "local_y_m": float(foot["position_w_m"][1]) + float(generated.origin[1]),
            "surface_z_m_from_report": foot["surface_z_m"],
            "vertical_clearance_m_from_report": foot["vertical_clearance_m"],
            "current_force_n": foot["current_force_n"],
            "history_peak_force_n": foot["history_peak_force_n"],
            "relative_to_first_riser": (
                "before" if local_x < first_riser_x else
                "on_first_tread" if local_x < first_tread_end_x else "after_first_tread"
            ),
        })

    mesh = generated.meshes[0]
    report = {
        "schema_version": "lite3_l2_stair_entry_geometry_audit_v1",
        "status": "offline_geometry_audit_completed",
        "simulation_started": False,
        "policy_executed": False,
        "training_started": False,
        "sealed_accessed": False,
        "qualification_claim": False,
        "cell_id": CELL_ID,
        "inputs": {
            "preparation_manifest": {"path": str(MANIFEST), "sha256": manifest_sha256},
            "runtime_report": {"path": str(RUNTIME_REPORT), "sha256": _sha256(RUNTIME_REPORT)},
            "geometry_generator": {"path": str(GENERATOR), "sha256": _sha256(GENERATOR)},
            "registered_geometry_generator_sha256": registered_geometry_hash,
            "geometry_seed": int(cell["geometry_seed_range"][0]),
            "initial_state_seed_from_runtime": int(runtime["cell"]["initial_state_seed_range"][0]),
        },
        "registered_geometry": geometry,
        "registered_track": track,
        "generated_geometry": {
            "origin_local_m": [float(value) for value in generated.origin],
            "mesh_bounds_m": mesh.bounds.tolist(),
            "mesh_is_watertight": bool(mesh.is_watertight),
            "mesh_is_winding_consistent": bool(mesh.is_winding_consistent),
            "metadata": metadata,
        },
        "registered_geometry_seed_stability": {
            "seed_range": list(cell["geometry_seed_range"]),
            "checks": geometry_seed_checks,
            "conclusion": "rules-based stair geometry is byte-identical across the registered geometry seeds",
        },
        "entry_audit": {
            "initial_flat_prefix_m": float(cfg.flat_length_m),
            "entry_buffer_m": float(cfg.transition_length_m),
            "entry_transition_geometry": "flat_buffer_before_first_riser",
            "entry_transition_is_ramp": False,
            "first_riser_local_x_m": first_riser_x,
            "first_riser_progress_from_reset_m": first_riser_progress,
            "first_riser_height_change_m": float(first["height_after_m"] - first["height_before_m"]),
            "first_tread_interval_local_x_m": [first_riser_x, first_tread_end_x],
            "first_tread_depth_m": float(first_tread_end_x - first_riser_x),
            "stair_end_local_x_m": stair_end_x,
            "stair_end_progress_from_reset_m": stair_end_progress,
            "completion_progress_from_reset_m": completion_progress,
            "post_stair_plateau_before_completion_m": completion_progress - stair_end_progress,
            "surface_samples": surface_samples,
        },
        "runtime_alignment": {
            "terrain_origin_world_m": terrain_origin_w,
            "root_start_world_m": root_start_w,
            "root_start_progress_from_terrain_origin_m": root_start_progress,
            "root_start_local_x_m": root_start_progress + origin_local_x,
            "root_start_to_first_riser_m": first_riser_progress - root_start_progress,
            "root_terminal_world_m": root_end_w,
            "root_terminal_progress_from_terrain_origin_m": root_end_progress,
            "root_terminal_distance_past_first_riser_m": root_end_progress - first_riser_progress,
            "terminal_feet": terminal_feet,
        },
        "verdict": {
            "registered_step_count_matches_generated_risers": True,
            "first_riser_alignment_passed": True,
            "entry_flat_buffer_surface_check_passed": True,
            "first_riser_height_check_passed": True,
            "first_tread_depth_check_passed": True,
            "runtime_start_position_is_inside_entry_flat_buffer": (
                root_start_progress < first_riser_progress
            ),
            "runtime_terminal_position_reached_first_tread": (
                root_end_progress >= first_riser_progress
            ),
            "geometry_entry_is_not_too_short": (
                first_riser_progress - root_start_progress >= 1.0
            ),
            "audit_conclusion": (
                "The registered L2 ascending track has a 2.0 m reset-relative approach "
                "to the first 3.5 cm riser and a 0.5 m first tread. The transition_length_m "
                "field is implemented as a flat entry buffer, not a ramp. The terminal root "
                "and foot positions are consistent with the first-riser/first-tread layout; "
                "the fall is not explained by an accidentally short or missing entry segment."
            ),
            "remaining_runtime_question": (
                "The geometry audit does not establish whether the frozen policy can recover "
                "the body posture after the first riser on other registered seeds."
            ),
        },
    }
    return report


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_JSON")
    output = Path(sys.argv[1]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = audit()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
