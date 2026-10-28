"""Audit Source L1 terrain geometry and scaling without importing Isaac Sim.

Generate one representative instance of each of the fifteen configured tile
classes. Counts extrapolate the reserved-row design, not measured runtime
memory or PhysX cooking success. Representative seeds do not reconstruct the
failed sixty-environment map.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def audit(root: Path) -> dict:
    terrain_dir = root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains"
    geometry_path = terrain_dir / "source_terrain_tracks.py"
    config_path = terrain_dir.parent / "config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py"
    spec = importlib.util.spec_from_file_location("source_scaling_geometry", geometry_path)
    geometry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = geometry
    spec.loader.exec_module(geometry)

    # Execute the same pure factory used by the scene, excluding simulator
    # imports and classes. This matches the existing offline allocation audit.
    tree = ast.parse(config_path.read_text())
    nodes = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name == "_source_sub_terrains"
    ) or (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"SOURCE_TERRAIN_PROPORTIONS", "_LEVEL_CANDIDATES"}
            for target in node.targets
        )
    )]
    namespace = {"SourceTerrainTrackCfg": lambda **kwargs: SimpleNamespace(**kwargs)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(config_path), "exec"), namespace)
    configured = namespace["_source_sub_terrains"]("L1")
    tiles = []
    for column, (name, config) in enumerate(configured.items()):
        kwargs = vars(config).copy()
        kwargs.pop("proportion")
        cfg = geometry.SourceTerrainCfg(size=(12.0, 4.0), seed=10_001, **kwargs)
        generated = geometry.generate_source_terrain(cfg)
        faces = vertices = degenerate = bottom_faces = 0
        min_double_area = float("inf")
        all_finite = True
        all_watertight = True
        all_winding_consistent = True
        for mesh in generated.meshes:
            faces += len(mesh.faces)
            vertices += len(mesh.vertices)
            # Check float32 geometry at the far row of a centered 60-row map.
            # This is a sampled numeric check, not actual exported USD data.
            translated = (mesh.vertices + [59 * 12.0 - 360.0, column * 4.0 - 30.0, 0.0]).astype(np.float32)
            all_finite &= bool(np.isfinite(translated).all())
            triangles = translated[mesh.faces].astype(np.float64)
            double_area = np.linalg.norm(np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ), axis=1)
            degenerate += int(np.count_nonzero(double_area <= 1e-12))
            min_double_area = min(min_double_area, float(double_area.min()))
            bottom_faces += int(np.count_nonzero(np.all(
                mesh.vertices[mesh.faces, 2] == mesh.vertices[:, 2].min(), axis=1
            )))
            all_watertight &= bool(mesh.is_watertight)
            all_winding_consistent &= bool(mesh.is_winding_consistent)
        tiles.append({
            "name": name, "config": asdict(cfg), "triangles": faces,
            "vertices": vertices, "bottom_triangles": bottom_faces,
            "friction_material": generated.requires_region_material_binding,
            "finite_float32_at_far_row_60": all_finite,
            "degenerate_triangles_at_far_row_60": degenerate,
            "min_double_area_m2_at_far_row_60": min_double_area,
            "watertight": all_watertight, "winding_consistent": all_winding_consistent,
        })

    ordinary_per_row = sum(t["triangles"] for t in tiles if not t["friction_material"])
    friction_per_row = sum(t["triangles"] for t in tiles if t["friction_material"])
    report = {
        "scope": "representative L1 tile geometry and analytical count extrapolation",
        "simulator_started": False,
        "physx_cooking_tested": False,
        "failed_map_reconstructed": False,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (geometry_path, config_path)},
        "tiles": tiles,
        "ordinary_triangles_per_row_excluding_border": ordinary_per_row,
        "friction_triangles_per_row": friction_per_row,
        "bottom_triangles_per_row": sum(t["bottom_triangles"] for t in tiles),
        "extrapolation_excluding_generator_border": [{
            "num_envs": n, "tile_count": n * len(tiles),
            "map_size_m": [n * 12.0, len(tiles) * 4.0],
            "ordinary_triangles": n * ordinary_per_row,
            "all_triangles": n * (ordinary_per_row + friction_per_row),
            "active_tile_fraction": 1.0 / len(tiles),
        } for n in (32, 60, 120, 4096)],
        "recorded_ordinary_counts": [
            {"num_envs": n, "observed": count,
             "difference_from_tile_sum": count - n * ordinary_per_row}
            for n, count in ((32, 15_035_016), (60, 28_190_648))
        ],
        "limitations": [
            "One representative seed per class; not every failed-map tile was regenerated.",
            "Counts exclude generator borders; no exported USD/PhysX mesh validation.",
            "Finite, nondegenerate sample tiles do not prove full-map cooking validity.",
            "No measured training throughput or hardware environment limit can be inferred.",
        ],
    }
    if any(name == "pxr" or name.startswith(("isaaclab", "isaacsim", "omni.")) for name in sys.modules):
        raise RuntimeError("offline audit unexpectedly loaded a simulator module")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    report = audit(Path(__file__).resolve().parents[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: report[key] for key in (
        "ordinary_triangles_per_row_excluding_border", "bottom_triangles_per_row",
        "extrapolation_excluding_generator_border", "recorded_ordinary_counts",
    )}, indent=2))


if __name__ == "__main__":
    main()
