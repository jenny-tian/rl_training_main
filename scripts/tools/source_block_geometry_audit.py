"""Audit the real block geometry functions on CPU without launching Isaac Sim."""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import scipy.interpolate as interpolate
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
from source_terrain_mesh_audit import _definitions, load_offline_pipeline
from rl_training.source_curriculum.block import BLOCK_SIZE, BLOCK_SEED
from rl_training.source_curriculum.rows import NUM_ROWS, COLUMN_SPECS, column_fields, curriculum_row_parameters, row_from_difficulty, track_overrides


def load_block_function():
    isaac = Path("/home/tjx/IsaacLab/source/isaaclab/isaaclab/terrains")
    root = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains"
    factory, _, _, _ = load_offline_pipeline(ROOT, isaac.parents[3])
    native = {"np": np, "trimesh": trimesh, "copy": copy, "functools": functools, "interpolate": interpolate}
    _definitions(isaac / "trimesh/utils.py", native, names={"make_border"})
    _definitions(isaac / "trimesh/mesh_terrains.py", native,
                 names={"pyramid_stairs_terrain", "inverted_pyramid_stairs_terrain"})
    _definitions(isaac / "height_field/utils.py", native,
                 names={"height_field_to_mesh", "convert_height_field_to_mesh"})
    _definitions(isaac / "height_field/hf_terrains.py", native, names={"random_uniform_terrain"})
    base = factory("L2")
    namespace = dict(copy=copy, math=math, np=np, trimesh=trimesh, BLOCK_SIZE=BLOCK_SIZE, NUM_ROWS=NUM_ROWS,
                     curriculum_row_parameters=curriculum_row_parameters, row_from_difficulty=row_from_difficulty,
                     track_overrides=track_overrides, source_terrain_track=next(iter(base.values())).function,
                     HfRandomUniformTerrainCfg=lambda **kw: SimpleNamespace(**kw, function=native["random_uniform_terrain"]),
                     MeshPyramidStairsTerrainCfg=lambda **kw: SimpleNamespace(**kw, function=native["pyramid_stairs_terrain"]),
                     MeshInvertedPyramidStairsTerrainCfg=lambda **kw: SimpleNamespace(**kw, function=native["inverted_pyramid_stairs_terrain"]))
    _definitions(root / "source_block_terrain.py", namespace, names={"reference_block_rough", "source_block_terrain"})
    return namespace["source_block_terrain"], next(iter(base.values())), isaac / "trimesh/mesh_terrains.py"


def heights(meshes, points):
    """Independent vertical intersections with actual triangles (no ray backend)."""
    triangles = np.concatenate([m.triangles for m in meshes])
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v, w = b - a, c - a
    determinant = v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]
    good = np.abs(determinant) > 1e-12
    a, v, w, determinant = a[good], v[good], w[good], determinant[good]
    output = []
    for point in points:
        delta = np.asarray(point) - a[:, :2]
        u = (delta[:, 0] * w[:, 1] - delta[:, 1] * w[:, 0]) / determinant
        q = (v[:, 0] * delta[:, 1] - v[:, 1] * delta[:, 0]) / determinant
        inside = (u >= -1e-9) & (q >= -1e-9) & (u + q <= 1 + 1e-9)
        output.append(float(np.max((a[:, 2] + u * v[:, 2] + q * w[:, 2])[inside])))
    return np.asarray(output)


def audit():
    function, template, native_path = load_block_function()
    records, stairs = [], []
    # Match the installed curriculum generator's column-major draw order.
    # Its difficulty is a realization seed here; all physical parameters are L2.
    rng = np.random.default_rng(BLOCK_SEED)
    for column, (name, _) in enumerate(COLUMN_SPECS):
        for row in range(NUM_ROWS):
            cfg = copy.deepcopy(template)
            cfg.__dict__.update(column_fields(column), size=BLOCK_SIZE, seed=BLOCK_SEED)
            difficulty = (row + rng.uniform()) / NUM_ROWS
            meshes, origin = function(difficulty, cfg)
            vertices = np.concatenate([m.vertices for m in meshes])
            assert np.isfinite(vertices).all()
            assert np.all(vertices[:, :2] >= -1e-8) and np.all(vertices[:, :2] <= 8 + 1e-8)
            assert all(m.is_watertight and m.is_winding_consistent and m.area_faces.min() > 1e-12 for m in meshes)
            # The entire robot/scanner spawn footprint must be flat at origin z.
            spawn = heights(meshes, [(4+x, 4+y) for x in (-.55, 0., .55) for y in (-.55, 0., .55)])
            np.testing.assert_allclose(spawn, origin[2], atol=1e-8)
            assert np.allclose(origin[:2], (4., 4.))
            digest = hashlib.sha256(vertices.tobytes()).hexdigest()
            records.append(dict(column=column, name=name, replica=row, difficulty="L2", seed=BLOCK_SEED,
                                realization=difficulty, origin=origin.tolist(), triangles=sum(len(m.faces) for m in meshes),
                                vertices_sha256=digest, mesh_count=len(meshes), bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()]))
            if cfg.family == "slope":
                angle = (row % 4) * math.pi / 2
                rotation = np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
                if cfg.direction in ("uphill", "downhill"):
                    points = np.array(((5.75, 4.), (6.25, 4.)))
                    sign = 1 if cfg.direction == "uphill" else -1
                else:
                    points = np.array(((6., 3.75), (6., 4.25)))
                    sign = 1 if cfg.direction == "cross_slope_left" else -1
                actual = heights(meshes, (points - 4.) @ rotation.T + 4.)
                grade = float((actual[1] - actual[0]) / .5)
                np.testing.assert_allclose(grade, sign * math.tan(math.radians(8)), atol=1e-8)
                records[-1]["signed_challenge_grade"] = grade
            if column in (4, 5):
                offsets = np.array((0., .75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.4))
                expected = np.array((.175, .175, .14, .105, .07, .035, 0., 0.)) * (-1 if column == 4 else 1)
                for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    actual = heights(meshes, [(4 + dx*x, 4 + dy*x) for x in offsets])
                    np.testing.assert_allclose(actual, expected, atol=1e-8)
                stairs.append(dict(column=column, replica=row, height_changes=5, step_height_m=.035,
                                   tread_m=.5, centre_platform_m=2., four_directions_verified=True))
    for column in (6, 15, 16, 17, 18, 19):
        hashes = [r["vertices_sha256"] for r in records if r["column"] == column]
        assert len(set(hashes)) == NUM_ROWS
    return dict(all_passed=True, scope="CPU actual geometry; simulator bindings not exercised", tile_count=len(records),
                level="L2", native_stair_source=str(native_path),
                native_stair_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                stair_profiles=stairs, tiles=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("tiles", "stair_profiles")}))
