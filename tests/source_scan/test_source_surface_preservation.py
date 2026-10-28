"""Compare all non-bottom surfaces against the committed pre-optimization mesh."""

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "surface_preservation_geometry",
    ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_tracks.py",
)
terrain = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = terrain
SPEC.loader.exec_module(terrain)
BASELINE = json.loads((Path(__file__).parent / "fixtures/source_mesh_surfaces_368bb04.json").read_text())


@pytest.mark.parametrize("case", BASELINE["cases"], ids=lambda case: case["id"])
def test_top_sides_origin_and_material_regions_match_baseline(case):
    geometry = terrain.generate_source_terrain(terrain.SourceTerrainCfg(**case["config"]))
    surfaces = []
    for mesh in geometry.meshes:
        triangles = mesh.triangles
        bottom = np.all(triangles[:, :, 2] == mesh.bounds[0, 2], axis=1)
        surfaces.append(triangles[~bottom])
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0.0
        assert mesh.area_faces.min() > 1e-12
    # Exact bytes include face order and winding, not just point samples.
    digest = hashlib.sha256(np.concatenate(surfaces).astype("<f8", copy=False).tobytes()).hexdigest()
    assert digest == case["non_bottom_triangle_sha256"]
    assert sum(len(mesh.faces) for mesh in geometry.meshes) <= case["triangle_count"]
    assert sum(mesh.volume for mesh in geometry.meshes) == pytest.approx(case["volume"], rel=1e-12)
    assert sum(mesh.area for mesh in geometry.meshes) == pytest.approx(case["surface_area"], rel=1e-12)
    assert list(geometry.origin) == case["origin"]
    assert json.loads(json.dumps(geometry.metadata)) == case["metadata"]
    assert json.loads(json.dumps([asdict(r) for r in geometry.friction_regions])) == case["friction_regions"]
