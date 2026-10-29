"""Freeze isolated L1 policy cases matching the existing one-row bank on CPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools.source_terrain_mesh_audit import load_offline_pipeline


def mesh_fingerprint(meshes, origin):
    digest = hashlib.sha256()
    for mesh in meshes:
        for data in (np.asarray(mesh.vertices, dtype="<f8"), np.asarray(mesh.faces, dtype="<i8"),
                     np.asarray(mesh.visual.face_colors, dtype="u1")):
            digest.update(str(data.shape).encode())
            digest.update(data.tobytes())
    digest.update(np.asarray(origin, dtype="<f8").tobytes())
    return digest.hexdigest()


def build_cases(root=ROOT, isaaclab_root=Path("/home/tjx/IsaacLab")):
    factory, generator_type, _, sources = load_offline_pipeline(root, isaaclab_root)
    terrain_cfgs = factory("L1")
    registration_path = root / "docs/lite3-source-scan-preparation-v0.2-2026-10-22/source_qualification_scan_manifest_v0.2_preparation.json"
    registration = json.loads(registration_path.read_text())
    registered = [c for c in registration["seed_plan"]["cells"] if c["level"] == 1]
    captured = []
    for name, cfg in terrain_cfgs.items():
        cfg.proportion = 1 / 15
        original = cfg.function

        def capture(difficulty, cfg, *, name=name, original=original):
            meshes, origin = original(difficulty, cfg)
            captured.append({"subterrain": name, "difficulty": float(difficulty), "generator_seed": cfg.seed,
                             "fingerprint": mesh_fingerprint(meshes, origin)})
            return meshes, origin

        cfg.function = capture
    base = dict(seed=10001, num_rows=1, num_cols=15, size=(12.0, 4.0), border_width=0.0,
                use_cache=False, cache_dir="/tmp/unused_source_bank_case_cache", curriculum=True,
                difficulty_range=(0.0, 1.0), color_scheme="none", sub_terrains=terrain_cfgs)
    bank = generator_type(SimpleNamespace(**base), device="cpu")
    if len(captured) != 15 or bank.terrain_origins.shape != (1, 15, 3):
        raise ValueError("bank generation did not cover all 15 columns")
    bank_records = copy.deepcopy(captured)
    cases = []
    for column, (name, cfg) in enumerate(terrain_cfgs.items()):
        expected = bank_records[column]
        if name != expected["subterrain"]:
            raise ValueError("bank column order differs from registered source order")
        direction = ("forward_flat" if cfg.family == "flat" else "forward_lane" if cfg.family == "multi_scale_rough"
                     else cfg.friction_layout if cfg.family == "friction_variation" else cfg.direction)
        matches = [c for c in registered if c["terrain_family"] == cfg.family and c["direction_or_layout"] == direction]
        if len(matches) != 1:
            raise ValueError(f"ambiguous registration for {name}")
        original_cell = matches[0]
        cell = copy.deepcopy(original_cell)
        cell["episodes"] = 1
        cell["geometry_seed_range"] = [10001, 10001]
        initial_seed = original_cell["initial_state_seed_range"][0]
        cell["initial_state_seed_range"] = [initial_seed, initial_seed]
        cell["qualification_geometry_difficulty"] = expected["difficulty"]
        # Construct through the production cell adapter used by policy runs.
        # Its only Isaac-bound object is replaced by the offline config type.
        cell_namespace = {"SourceTerrainTrackCfg": type(cfg)}
        import ast
        path = root / "source/rl_training/rl_training/source_scan/terrain_config.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "source_subterrain_from_cell")
        function.body = [n for n in function.body if not isinstance(n, ast.ImportFrom)]
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec",
                     flags=__import__("__future__").annotations.compiler_flag), cell_namespace)
        isolated_cfg = cell_namespace["source_subterrain_from_cell"](cell, seed=10001)
        seen = []
        isolated_function = isolated_cfg.function

        def capture_isolated(difficulty, cfg):
            meshes, origin = isolated_function(difficulty, cfg)
            seen.append(mesh_fingerprint(meshes, origin))
            return meshes, origin

        isolated_cfg.function = capture_isolated
        isolated = generator_type(SimpleNamespace(**{**base, "num_cols": 1,
            "difficulty_range": (expected["difficulty"], expected["difficulty"]),
            "sub_terrains": {cell["cell_id"]: isolated_cfg}}), device="cpu")
        if seen != [expected["fingerprint"]]:
            raise ValueError(f"isolated cell does not reproduce bank column {column}: {name}")
        cases.append({"case_id": cell["cell_id"], "bank_row": 0, "bank_column": column,
                      "subterrain": name, "geometry_seed": 10001, "initial_state_seed": initial_seed,
                      "cell": cell, "geometry_difficulty": expected["difficulty"],
                      "mesh_fingerprint": expected["fingerprint"], "isolated_mesh_matches_bank": True,
                      "reference_registered_geometry_seed_range": original_cell["geometry_seed_range"],
                      "runtime_status": "not_run", "independent_runtime_approval_required": True})
        del isolated
    # Flat has no seeded roughness, but verify rather than assuming its old
    # isolated realization matches the bank's first column.
    generator_type(SimpleNamespace(**{**base, "seed": 10000, "num_cols": 1,
        "difficulty_range": (0.0, 0.0), "sub_terrains": {"flat": copy.deepcopy(terrain_cfgs["flat"])}}), device="cpu")
    historical_flat_matches = captured[-1]["fingerprint"] == bank_records[0]["fingerprint"]
    if not historical_flat_matches:
        raise ValueError("historical Flat geometry differs from current bank column zero")
    sources.append(root / "source/rl_training/rl_training/source_scan/terrain_config.py")
    return {"schema": "lite3_bank_equivalent_single_policy_cases_v1", "simulator_started": False,
            "bank_seed": 10001, "bank_grid": [1, 15], "tile_size_m": [12.0, 4.0],
            "registration_manifest_sha256": hashlib.sha256(registration_path.read_bytes()).hexdigest(),
            "historical_flat_geometry_matches_bank": historical_flat_matches,
            "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "fingerprint_semantics": "Pre-transform mesh vertices, faces, face colors/material tags and local origin, exact float64 bytes; geometry only, no PhysX cooking.",
            "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_cases()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"bank_columns": len(report["cases"]), "isolated_meshes_match": True}))
