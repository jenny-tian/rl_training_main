import ast
import importlib.util
import math
import sys
from pathlib import Path

import pytest

from rl_training.source_curriculum.rows import (
    ANCHOR_ROWS, COLUMN_SPECS, FRICTION_RANGE, L1, L2, NUM_COLUMNS, OMNI_COLUMNS, column_fields,
    curriculum_row_parameters, row_from_difficulty, track_overrides)

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
T = V / "terrains"

# Plan key -> key in source_terrain_env_cfg._LEVEL_CANDIDATES (identical names today).
CANDIDATE_KEYS = {
    "rough_peak_valley_m": "rough_peak_valley_m",
    "slope_angle_deg": "slope_angle_deg",
    "step_count": "step_count",
    "step_height_m": "step_height_m",
    "slope_rough_peak_valley_m": "slope_rough_peak_valley_m",
    "slope_rough_angle_deg": "slope_rough_angle_deg",
    "exit_flat_length_m": "exit_flat_length_m",
    "tread_depth_m": "tread_depth_m",
    "friction_range": "friction_range",
}


def _level_candidates():
    tree = ast.parse((V / "config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py").read_text())
    for node in ast.walk(tree):
        targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
        if any(getattr(t, "id", None) == "_LEVEL_CANDIDATES" for t in targets if t is not None):
            return ast.literal_eval(node.value)
    raise AssertionError("_LEVEL_CANDIDATES not found")


def test_anchors_match_source_level_candidates():
    cand = _level_candidates()
    for anchor, level in ((L1, "L1"), (L2, "L2")):
        for key, value in anchor.items():
            assert cand[level][CANDIDATE_KEYS[key]] == pytest.approx(value), (level, key)
    assert tuple(cand["L2"][CANDIDATE_KEYS["friction_range"]]) == FRICTION_RANGE
    for num_rows, (a1, a2) in ANCHOR_ROWS.items():
        p1, p2 = curriculum_row_parameters(a1, num_rows), curriculum_row_parameters(a2, num_rows)
        for key in L1:
            assert p1[key] == pytest.approx(L1[key]) and p2[key] == pytest.approx(L2[key])
        assert p2["exit_flat_length_m"] == cand["L2"][CANDIDATE_KEYS["exit_flat_length_m"]]
        assert p2["tread_depth_m"] == cand["L2"][CANDIDATE_KEYS["tread_depth_m"]]


def test_spec_row_table():
    rows = [curriculum_row_parameters(r) for r in range(10)]
    assert [p["slope_angle_deg"] for p in rows] == pytest.approx([1, 2, 3, 4, 5, 5.6, 6.2, 6.8, 7.4, 8])
    assert [p["step_count"] for p in rows] == [3, 3, 3, 3, 3, 3, 4, 4, 5, 5]
    assert [100 * p["step_height_m"] for p in rows] == pytest.approx(
        [0.5, 0.875, 1.25, 1.625, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5])
    assert 100 * rows[0]["slope_rough_peak_valley_m"] == pytest.approx(0.25)
    assert 100 * rows[9]["slope_rough_peak_valley_m"] == pytest.approx(1.5)


@pytest.mark.parametrize("num_rows", [8, 10])
def test_monotonic_and_capped_at_l2(num_rows):
    rows = [curriculum_row_parameters(r, num_rows) for r in range(num_rows)]
    for key in L2:
        values = [p[key] for p in rows]
        assert values == sorted(values) and max(values) <= L2[key] + 1e-12
    assert all(type(p["step_count"]) is int for p in rows)
    assert {tuple(p["friction_range"]) for p in rows} == {FRICTION_RANGE}


def test_row_from_difficulty_edges_and_rejections():
    assert row_from_difficulty(0.0) == 0
    assert row_from_difficulty(0.4) == 4
    assert row_from_difficulty(0.39999999999) == 4  # epsilon absorbs float error
    assert row_from_difficulty(0.99) == 9
    assert row_from_difficulty(1.0) == 9
    assert row_from_difficulty(0.5, 8) == 4
    for bad in (True, -0.01, 1.01, math.nan, math.inf, "0.5", None):
        with pytest.raises(ValueError):
            row_from_difficulty(bad)
    with pytest.raises(ValueError):
        row_from_difficulty(0.5, 9)
    for bad in (-1, 10, 1.0, True):
        with pytest.raises(ValueError):
            curriculum_row_parameters(bad)


def test_track_overrides():
    p = curriculum_row_parameters(9)
    assert track_overrides("flat", p) == {} and track_overrides("friction_variation", p) == {}
    assert track_overrides("slope_plus_rough", p) == {
        "slope_angle_deg": 8.0, "rough_peak_valley_m": pytest.approx(0.015), "exit_flat_length_m": 2.0}
    assert track_overrides("multi_step_stairs", p)["step_count"] == 5
    with pytest.raises(ValueError, match="unknown Source family"):
        track_overrides("boxes", p)


def test_column_layout():
    assert len(COLUMN_SPECS) == 20 and OMNI_COLUMNS == (9, 10, 11, 12)
    families = [column_fields(c)["family"] for c in range(20)]
    assert all(f in ("flat", "friction_variation") for f in families[7:15])
    friction = [column_fields(c) for c in range(20) if families[c] == "friction_variation"]
    assert {(f["friction_range"], f["friction_region_count"]) for f in friction} == {(FRICTION_RANGE, 4)}
    assert sorted(set(families)) == sorted(["flat", "multi_scale_rough", "slope", "multi_step_stairs",
                                            "slope_plus_rough", "friction_variation"])


def _load_geometry_module():
    # source_terrain_tracks.py imports only numpy/trimesh, so no isaaclab skip is needed.
    name = "source_curriculum_geometry_under_test"
    spec = importlib.util.spec_from_file_location(name, T / "source_terrain_tracks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_every_cell_passes_source_geometry_validation():
    geometry = _load_geometry_module()
    tile_size = (12.0, 4.0)  # TerrainGeneratorCfg.size in source_terrain_env_cfg.py
    checked = 0
    for row in (0, 4, 9):
        params = curriculum_row_parameters(row)
        for column in range(NUM_COLUMNS):
            fields = column_fields(column)
            overrides = track_overrides(fields["family"], params)
            # SourceTerrainCfg.__post_init__ is the validation that SourceTerrainTrackCfg feeds.
            cfg = geometry.SourceTerrainCfg(size=tile_size, **fields, **overrides)
            checked += 1
            if 7 <= column <= 14:
                result = geometry.generate_source_terrain(cfg)
                top = max(float(mesh.vertices[:, 2].max()) for mesh in result.meshes)
                assert top == 0.0, (row, column, top)
    assert checked == 60
