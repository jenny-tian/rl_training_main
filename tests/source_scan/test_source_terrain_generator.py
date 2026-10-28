"""Geometry-only integration against the installed Isaac Lab generator loops."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh


ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = Path("/home/tjx/IsaacLab")
pytestmark = pytest.mark.skipif(
    not (ISAACLAB_ROOT / "source/isaaclab/isaaclab/terrains/terrain_generator.py").is_file(),
    reason="requires installed Isaac Lab source, but does not launch it",
)
SPEC = importlib.util.spec_from_file_location("source_generator_audit", ROOT / "scripts/tools/source_terrain_mesh_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_tile_metadata_preserves_stock_seed_stream_origins_and_triangle_order(monkeypatch):
    factory, source_type, importer, _paths = audit.load_offline_pipeline(ROOT, ISAACLAB_ROOT)
    terrains = factory("L1")
    terrains = {key: terrains[key] for key in ("flat", "multi_scale_rough", "friction_left_right_bands")}
    for cfg in terrains.values():
        cfg.proportion = 1 / 3
    cfg = SimpleNamespace(
        seed=10001, num_rows=3, num_cols=3, size=(12.0, 4.0), border_width=0.0, border_height=1.0,
        use_cache=False, cache_dir="/tmp/unused_source_generator_test_cache", curriculum=True,
        difficulty_range=(0.0, 1.0), color_scheme="none", sub_terrains=terrains,
    )
    stock_type = source_type.__bases__[0]
    border = {"np": np, "trimesh": trimesh}
    audit._definitions(
        ISAACLAB_ROOT / "source/isaaclab/isaaclab/terrains/trimesh/utils.py", border, names={"make_border"}
    )
    monkeypatch.setitem(stock_type._add_terrain_border.__globals__, "make_border", border["make_border"])
    stock = stock_type(copy.deepcopy(cfg))
    source = source_type(copy.deepcopy(cfg))
    has_area = stock.terrain_mesh.area_faces > 0.0
    assert (~has_area).sum() == 8
    np.testing.assert_array_equal(stock.terrain_mesh.triangles[has_area], source.terrain_mesh.triangles)
    np.testing.assert_array_equal(stock.terrain_mesh.visual.face_colors[has_area], source.terrain_mesh.visual.face_colors)
    np.testing.assert_array_equal(stock.terrain_origins, source.terrain_origins)
    ranges = source.terrain_mesh.metadata[importer["SOURCE_TILE_FACE_RANGES"]]
    assert [name for name, _, _ in ranges] == [f"tile_r{row:04d}_c{col:02d}" for col in range(3) for row in range(3)]
    assert ranges[0][1] == 0 and ranges[-1][2] == len(source.terrain_mesh.faces)
    for name, start, stop in ranges:
        row, col = int(name[6:10]), int(name[12:])
        triangles = source.terrain_mesh.vertices[source.terrain_mesh.faces[start:stop]]
        lower = np.array([row * 12 - 18, col * 4 - 6])
        assert np.all(triangles[:, :, :2].min(axis=(0, 1)) >= lower - 1e-8)
        assert np.all(triangles[:, :, :2].max(axis=(0, 1)) <= lower + [12, 4] + 1e-8)


def test_complete_two_row_map_reaches_material_and_tile_importer():
    report = audit.audit_map(ROOT, ISAACLAB_ROOT, 2)
    assert report["origin_shape"] == [2, 15, 3]
    assert report["tile_count"] == 30
    assert report["collision_parts"] == 48
    assert report["all_faces_accounted_for"]
    assert report["all_parts_closed_finite_nondegenerate_within_tile"]
    assert report["max_part_triangles"] < 100000
    assert {round(item["friction"], 6) for item in report["parts"]} == {0.7, 0.8, 0.9, 1.0}
