"""Compare B1.1 geometry with archived B1 and audit the actual native rough meshes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from source_block_geometry_audit import ROOT, load_block_function, heights
from rl_training.source_curriculum.block import BLOCK_SEED, BLOCK_SIZE
from rl_training.source_curriculum.rows import column_fields

TASK = "SourceBlock-ReferenceRough-Deeprobotics-Lite3-v0"


def audit():
    function, template, _ = load_block_function()
    baseline = json.loads((ROOT / "docs/lite3-block-l2-b1-20261111/geometry.json").read_text())
    records = []
    unchanged = 0
    legacy = 0
    for old in baseline["tiles"]:
        column, row = old["column"], old["replica"]
        cfg = copy.deepcopy(template)
        cfg.__dict__.update(column_fields(column), size=BLOCK_SIZE, seed=BLOCK_SEED)
        meshes, origin = function(old["realization"], cfg)
        digest = hashlib.sha256(np.concatenate([m.vertices for m in meshes]).tobytes()).hexdigest()
        assert digest == old["vertices_sha256"], (column, row, "B1 changed")
        np.testing.assert_allclose(origin, old["origin"], atol=1e-8)
        legacy += 1
        cfg.rough_profile = "reference_uniform"
        meshes, origin = function(old["realization"], cfg)
        vertices = np.concatenate([m.vertices for m in meshes])
        digest = hashlib.sha256(vertices.tobytes()).hexdigest()
        if column not in (6, 15):
            assert digest == old["vertices_sha256"], (column, row, "unrelated tile changed")
            unchanged += 1
            continue
        assert digest != old["vertices_sha256"]
        assert len(meshes) == 1 and len(meshes[0].faces) == 12800
        assert len(vertices) == 81 * 81
        assert np.isfinite(vertices).all() and meshes[0].is_winding_consistent
        assert (meshes[0].face_normals[:, 2] > 0).all()
        assert meshes[0].area_faces.min() > 0
        np.testing.assert_allclose(vertices.min(0), [0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(vertices.max(0), [8, 8, .06], atol=1e-7)
        # The native height field is an open surface, with one continuous
        # projected 8x8 m grid. No solid backing or artificial flat spawn pad.
        xy = np.round(vertices[:, :2], 5)
        assert len(np.unique(xy, axis=0)) == 81 * 81
        interior = (xy[:, 0] >= .3) & (xy[:, 0] <= 7.7) & (xy[:, 1] >= .3) & (xy[:, 1] <= 7.7)
        z = vertices[interior, 2]
        np.testing.assert_allclose([z.min(), z.max()], [.01, .06], atol=1e-7)
        np.testing.assert_allclose(z / .01, np.round(z / .01), atol=1e-6)
        assert np.all(vertices[~interior, 2] == 0.)
        assert len(z) == 75 * 75
        np.testing.assert_allclose(origin, [4, 4, .06], atol=1e-7)
        central = (abs(xy[:, 0] - 4) <= 1) & (abs(xy[:, 1] - 4) <= 1)
        assert vertices[central, 2].max() <= origin[2] + 1e-7
        # Independent triangle intersections exercise edges, the spawn area,
        # and all four exits, rather than only inspecting configuration fields.
        points = [(x, y) for x in (.01, .2, .3, 1., 3.45, 4., 4.55, 7., 7.7, 7.8, 7.99)
                  for y in (.01, .2, .3, 1., 3.45, 4., 4.55, 7., 7.7, 7.8, 7.99)]
        hits = heights(meshes, points)
        assert np.isfinite(hits).all() and hits.min() >= -1e-7 and hits.max() <= .06 + 1e-7
        records.append(dict(column=column, replica=row, realization=old["realization"],
            origin=origin.tolist(), vertices_sha256=digest, triangles=12800,
            interior_height_range_m=[float(z.min()), float(z.max())],
            interior_vertex_fraction=float(interior.mean()), ray_probes=len(points),
            square_footprint_m=[8., 8.], native_open_surface=True))
    assert len(records) == 20 and len({r["vertices_sha256"] for r in records}) == 20
    return dict(task=TASK, all_passed=True, scope="CPU mesh generation and independent vertical intersections",
                legacy_B1_tiles_identical=legacy, nonrough_B11_tiles_identical=unchanged,
                reference_rough_tiles=len(records), distinct_rough_realizations=len(records), tiles=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tiles"}))
