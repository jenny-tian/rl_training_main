"""Exercise Source generation and collision partitioning on CPU, without Kit.

The installed Isaac Lab generator class is read as Python AST, with its one
height-field type import replaced by a stand-in. Source uses trimesh tracks,
no flat-patch sampling and no cache. Cache hashing and USD creation are stubbed;
the real generation loops, seed derivation, transforms and importer methods run.
This checks mesh inputs, never PhysX cooking or runtime collision isolation.
"""

from __future__ import annotations

import __future__
import argparse
import ast
from contextlib import nullcontext
import copy
from dataclasses import fields
import hashlib
import importlib.util
import json
import logging
import numbers
import os
from pathlib import Path
import re
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import trimesh


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _definitions(path, namespace, *, names=None, replace_heightfield_import=False):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))
             and (names is None or getattr(node, "name", None) in names or (
                 isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets)
             ))]
    if replace_heightfield_import:
        class NoHeightfieldImport(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.module != "height_field" or node.level != 1:
                    raise RuntimeError("unexpected local import in geometry-only generator")
                return ast.copy_location(ast.Pass(), node)
        nodes = [NoHeightfieldImport().visit(node) for node in nodes]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


def load_offline_pipeline(root: Path, isaaclab_root: Path):
    terrain_dir = root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains"
    stock_path = isaaclab_root / "source/isaaclab/isaaclab/terrains/terrain_generator.py"
    geometry = _module(terrain_dir / "source_terrain_tracks.py", "mesh_audit_geometry")
    chunking = _module(terrain_dir / "_mesh_chunking.py", "mesh_audit_chunking")
    adapter = {"np": np, "numbers": numbers, "SourceTerrainCfg": geometry.SourceTerrainCfg,
               "SOURCE_TERRAIN_FAMILIES": geometry.SOURCE_TERRAIN_FAMILIES,
               "generate_source_terrain": geometry.generate_source_terrain}
    _definitions(terrain_dir / "source_terrain_isaac.py", adapter, names={
        "FRICTION_REGION_MARKER_ALPHA", "FRICTION_COEFFICIENT_SCALE", "FRICTION_COEFFICIENT_MAX_Q",
        "encode_friction_region_colour", "_mark_friction_regions",
        "generate_source_terrain_geometry", "source_terrain_track",
    })

    class TrackCfg(SimpleNamespace):
        def __init__(self, **kwargs):
            defaults = {f.name: f.default for f in fields(geometry.SourceTerrainCfg)}
            defaults.update(kwargs)
            defaults.update(flat_patch_sampling=None, function=adapter["source_terrain_track"])
            super().__init__(**defaults)

        def copy(self):
            return copy.deepcopy(self)

        def to_dict(self):
            return vars(self)

    config_path = terrain_dir.parent / "config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py"
    config = {"SourceTerrainTrackCfg": TrackCfg}
    _definitions(config_path, config, names={"SOURCE_TERRAIN_PROPORTIONS", "_LEVEL_CANDIDATES", "_source_sub_terrains"})
    stock = {
        "np": np, "torch": torch, "trimesh": trimesh, "os": os,
        "logger": logging.getLogger("offline_source_mesh_audit"),
        "Timer": lambda *_args: nullcontext(), "HfTerrainBaseCfg": type("UnusedHeightFieldCfg", (), {}),
        "dict_to_md5_hash": lambda _cfg: "unused_cache_disabled",
    }
    _definitions(stock_path, stock, names={"TerrainGenerator"}, replace_heightfield_import=True)
    source = {"TerrainGenerator": stock["TerrainGenerator"],
              "SOURCE_TILE_FACE_RANGES": chunking.SOURCE_TILE_FACE_RANGES,
              "validate_tile_face_ranges": chunking.validate_tile_face_ranges}
    _definitions(terrain_dir / "source_terrain_generator.py", source, names={"SourceTerrainGenerator"})
    importer = {
        "np": np, "torch": torch, "trimesh": trimesh,
        "TerrainImporter": object, "TerrainImporterCfg": object,
        "sim_utils": SimpleNamespace(create_prim=lambda *_args: None),
        "_LOG": logging.getLogger("offline_source_mesh_audit"),
        "SOURCE_TILE_FACE_RANGES": chunking.SOURCE_TILE_FACE_RANGES,
        "validate_tile_face_ranges": chunking.validate_tile_face_ranges,
        "balanced_face_chunk_slices": chunking.balanced_face_chunk_slices,
        "FRICTION_REGION_MARKER_ALPHA": adapter["FRICTION_REGION_MARKER_ALPHA"],
        "FRICTION_COEFFICIENT_SCALE": adapter["FRICTION_COEFFICIENT_SCALE"],
    }
    _definitions(terrain_dir / "source_terrain_importer.py", importer,
                 names={"SourceTerrainImporter", "_VISUAL_FACE_COLOUR", "MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES"})
    paths = [stock_path, config_path, *(terrain_dir / name for name in (
        "source_terrain_tracks.py", "source_terrain_isaac.py", "source_terrain_generator.py",
        "source_terrain_importer.py", "_mesh_chunking.py",
    ))]
    return config["_source_sub_terrains"], source["SourceTerrainGenerator"], importer, paths


def audit_map(root: Path, isaaclab_root: Path, num_envs: int) -> dict:
    factory, generator_type, importer_namespace, paths = load_offline_pipeline(root, isaaclab_root)
    sub_terrains = factory("L1")
    for cfg in sub_terrains.values():
        cfg.proportion = 1.0 / len(sub_terrains)
    cfg = SimpleNamespace(
        seed=10001, num_rows=num_envs, num_cols=len(sub_terrains), size=(12.0, 4.0),
        border_width=0.0, use_cache=False, cache_dir="/tmp/unused_source_mesh_audit_cache",
        curriculum=True, difficulty_range=(0.0, 1.0), color_scheme="none", sub_terrains=sub_terrains,
    )
    started = time.monotonic()
    generator = generator_type(cfg, device="cpu")
    generation_s = time.monotonic() - started
    mesh = generator.terrain_mesh
    importer_type = importer_namespace["SourceTerrainImporter"]
    importer = object.__new__(importer_type)
    importer.cfg = SimpleNamespace(
        prim_path="/World/ground", parallel_reset_sampler_enabled=True, terrain_generator=cfg,
        physics_material=SimpleNamespace(static_friction=1.0), visual_material=None,
    )
    importer.terrain_prim_paths = []
    importer._make_region_material = lambda coefficient: SimpleNamespace(static_friction=coefficient)
    records = []

    def capture(path, part, **kwargs):
        match = re.search(r"tile_r([0-9]+)_c([0-9]+)", path)
        assert match is not None, path
        row, col = map(int, match.groups())
        lower = np.array([row * 12.0 - num_envs * 6.0, col * 4.0 - 30.0])
        bounds = part.bounds
        assert np.all(bounds[0, :2] >= lower - 1e-8), path
        assert np.all(bounds[1, :2] <= lower + [12.0, 4.0] + 1e-8), path
        assert 0 < len(part.faces) <= importer_namespace["MAX_SOURCE_TERRAIN_COLLIDER_TRIANGLES"]
        assert part.is_watertight and part.is_winding_consistent and part.volume > 0, path
        assert np.isfinite(part.vertices).all() and part.area_faces.min() > 1e-12, path
        assert (np.asarray(part.visual.face_colors)[:, 3] == 255).all(), path
        records.append({"path": path, "triangles": len(part.faces),
                        "friction": kwargs["physics_material"].static_friction,
                        "bounds": bounds.tolist()})

    importer_namespace["create_prim_from_mesh"] = capture
    importer.import_mesh("terrain", mesh)
    assert sum(item["triangles"] for item in records) == len(mesh.faces)
    assert len(importer.terrain_prim_paths) == len(set(importer.terrain_prim_paths)) == len(records)
    if any(name == "pxr" or name.startswith(("isaaclab", "isaacsim", "omni.")) for name in sys.modules):
        raise RuntimeError("offline mesh audit loaded a simulator module")
    return {
        "scope": "CPU geometry generation and importer with USD/material bindings stubbed",
        "simulator_started": False, "physx_cooking_tested": False,
        "num_envs": num_envs, "geometry_seed": 10001,
        "tile_count": num_envs * 15, "origin_shape": list(generator.terrain_origins.shape),
        "map_triangles": len(mesh.faces), "collision_parts": len(records),
        "max_part_triangles": max(item["triangles"] for item in records),
        "all_parts_closed_finite_nondegenerate_within_tile": True,
        "all_faces_accounted_for": True, "zero_width_border_skipped": True,
        "generation_s": generation_s, "total_audit_s": time.monotonic() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "parts": records,
        "limitations": "No USD export, PhysX cooking, GPU usage, ray-caster runtime or training throughput measured.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/tjx/IsaacLab"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 120:
        parser.error("offline audit supports 1 to 120 rows")
    if args.output.exists():
        parser.error("refusing to overwrite an audit report")
    report = audit_map(Path(__file__).resolve().parents[2], args.isaaclab_root, args.num_envs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"parts", "source_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
