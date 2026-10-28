"""Source geometry checks that require neither Isaac imports nor AppLauncher."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


# Load the pure module directly: task-package imports may register Isaac tasks.
MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_tracks.py"
)
SPEC = importlib.util.spec_from_file_location("source_geometry_under_test", MODULE_PATH)
terrain = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = terrain
SPEC.loader.exec_module(terrain)
Cfg = terrain.SourceTerrainCfg
generate = terrain.generate_source_terrain


def _cfg(family, **kwargs):
    values = dict(size=(8.0, 2.0), flat_length_m=1.5, seed=10001)
    if family == "multi_step_stairs":
        values["direction"] = "ascending"
    values.update(kwargs)
    return Cfg(family=family, **values)


def _surface_height(mesh, x, y):
    """Vertical intersection with upward-facing triangles, without ray extras."""
    triangles = mesh.triangles[mesh.face_normals[:, 2] > 1e-8]
    a = triangles[:, 0]
    u, v = triangles[:, 1] - a, triangles[:, 2] - a
    determinant = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
    dx, dy = x - a[:, 0], y - a[:, 1]
    s = (dx * v[:, 1] - v[:, 0] * dy) / determinant
    t = (u[:, 0] * dy - dx * u[:, 1]) / determinant
    inside = (s >= -1e-9) & (t >= -1e-9) & (s + t <= 1.0 + 1e-9)
    assert inside.any(), f"no surface at {(x, y)}"
    return float(np.max((a[:, 2] + s * u[:, 2] + t * v[:, 2])[inside]))


@pytest.mark.parametrize("family", terrain.SOURCE_TERRAIN_FAMILIES)
def test_all_families_have_finite_closed_outward_geometry_and_supported_start(family):
    cfg = _cfg(family)
    result = generate(cfg)
    assert result.family == family
    for mesh in result.meshes:
        assert np.isfinite(mesh.vertices).all()
        assert mesh.is_watertight
        assert mesh.is_winding_consistent
        assert mesh.volume > 0.0
        assert np.min(mesh.area_faces) > 1e-12
        assert mesh.bounds[0, 0] >= -1e-10
        assert mesh.bounds[1, 0] <= cfg.size[0] + 1e-10
    supported = any(
        mesh.bounds[0, 0] <= result.origin[0] <= mesh.bounds[1, 0]
        and mesh.bounds[0, 1] <= result.origin[1] <= mesh.bounds[1, 1]
        and _surface_height(mesh, *result.origin[:2]) == pytest.approx(result.origin[2])
        for mesh in result.meshes
    )
    assert supported
    json.dumps(result.metadata, allow_nan=False)


@pytest.mark.parametrize("family", terrain.SOURCE_TERRAIN_FAMILIES)
def test_seed_is_byte_deterministic(family):
    first, second = generate(_cfg(family)), generate(_cfg(family))
    assert first.metadata == second.metadata
    assert first.friction_regions == second.friction_regions
    for a, b in zip(first.meshes, second.meshes, strict=True):
        assert a.vertices.tobytes() == b.vertices.tobytes()
        assert a.faces.tobytes() == b.faces.tobytes()


@pytest.mark.parametrize("family", ["multi_scale_rough", "slope_plus_rough"])
def test_rough_seed_changes_geometry_and_flat_prefix_has_no_seam(family):
    cfg = _cfg(family)
    first = generate(cfg).meshes[0]
    other = generate(_cfg(family, seed=10002)).meshes[0]
    assert not np.array_equal(first.vertices, other.vertices)
    for x in (0.0, 0.75, 1.5):
        for y in (0.1, 1.0, 1.9):
            assert _surface_height(first, x, y) == pytest.approx(0.0, abs=1e-12)
    # A continuous heightfield has one top value at every shared XY vertex.
    upper = first.vertices[first.vertices[:, 2] > first.bounds[0, 2] + 1e-8]
    assert len(np.unique(upper[:, :2], axis=0)) == len(upper)


@pytest.mark.parametrize("height", [0.02, 0.03, 0.04])
def test_rough_contact_scale_statistics_match_mesh_vertices(height):
    cfg = _cfg("multi_scale_rough", rough_peak_valley_m=height)
    result = generate(cfg)
    mesh = result.meshes[0]
    top = mesh.vertices[mesh.vertices[:, 2] > mesh.bounds[0, 2] + 1e-8]
    # Independently reconstruct the registered half-window-stride audit.
    window = cfg.rough_audit_window_m
    start = cfg.flat_length_m + cfg.transition_length_m
    nx = int(np.ceil((cfg.size[0] - window - start) / (window / 2)))
    ny = int(np.ceil((cfg.size[1] - window) / (window / 2)))
    ranges = []
    for y0 in np.linspace(0.0, cfg.size[1] - window, ny + 1):
        for x0 in np.linspace(start, cfg.size[0] - window, nx + 1):
            mask = (
                (top[:, 0] >= x0 - 1e-10) & (top[:, 0] <= x0 + window + 1e-10)
                & (top[:, 1] >= y0 - 1e-10) & (top[:, 1] <= y0 + window + 1e-10)
            )
            ranges.append(np.ptp(top[mask, 2]))
    assert max(ranges) == pytest.approx(height)
    assert result.metadata["sampled_local_peak_valley_max_m"] == pytest.approx(max(ranges))
    assert result.metadata["sampled_local_peak_valley_p95_m"] == pytest.approx(np.percentile(ranges, 95))
    challenge = top[top[:, 0] >= start, 2]
    assert result.metadata["full_challenge_peak_valley_m"] == pytest.approx(np.ptp(challenge))
    assert len(result.metadata["rough_spatial_scales_m"]) >= 2


@pytest.mark.parametrize("angle", [5.0, 8.0, 12.0])
@pytest.mark.parametrize("direction", terrain.SLOPE_DIRECTIONS)
def test_slope_direction_angle_and_flat_transitions_from_mesh(angle, direction):
    cfg = _cfg("slope", slope_angle_deg=angle, direction=direction)
    result = generate(cfg)
    mesh = result.meshes[0]
    grade = np.tan(np.radians(angle))
    assert _surface_height(mesh, 0.75, 1.0) == pytest.approx(0.0)
    if direction in ("uphill", "downhill"):
        sign = 1 if direction == "uphill" else -1
        delta = _surface_height(mesh, 4.0, 1.0) - _surface_height(mesh, 3.0, 1.0)
        assert delta == pytest.approx(sign * grade, abs=1e-10)
        assert _surface_height(mesh, 4.0, 0.2) == pytest.approx(_surface_height(mesh, 4.0, 1.8))
    else:
        sign = 1 if direction == "cross_slope_left" else -1
        delta = _surface_height(mesh, 4.0, 1.5) - _surface_height(mesh, 4.0, 0.5)
        assert delta == pytest.approx(sign * grade)
        assert _surface_height(mesh, cfg.size[0], 0.2) == pytest.approx(0.0)
    # Smooth entrance joins the flat prefix without a vertical riser.
    assert _surface_height(mesh, cfg.flat_length_m, 0.3) == pytest.approx(0.0)


@pytest.mark.parametrize("family", ["slope", "slope_plus_rough"])
@pytest.mark.parametrize("direction", terrain.SLOPE_DIRECTIONS)
def test_slope_exit_transition_joins_a_level_terminal_buffer(family, direction):
    cfg = _cfg(
        family,
        size=(12.0, 4.0),
        exit_flat_length_m=2.0,
        direction=direction,
        slope_angle_deg=5.0,
    )
    result = generate(cfg)
    mesh = result.meshes[0]
    assert result.metadata["exit_transition_x_m"] == pytest.approx([9.5, 10.0])
    assert result.metadata["exit_flat_x_m"] == pytest.approx([10.0, 12.0])
    for y in (0.2, 1.0, 2.0, 3.8):
        flat_height = _surface_height(mesh, 10.0, y)
        assert _surface_height(mesh, 10.5, y) == pytest.approx(flat_height, abs=1e-10)
        assert _surface_height(mesh, 11.9, y) == pytest.approx(flat_height, abs=1e-10)
    assert mesh.is_watertight and mesh.is_winding_consistent


@pytest.mark.parametrize("direction", terrain.SLOPE_DIRECTIONS)
def test_slope_plus_rough_is_exactly_the_sum_on_shared_grid(direction):
    combined = generate(_cfg("slope_plus_rough", direction=direction))
    rough = generate(_cfg("multi_scale_rough"))
    slope = generate(_cfg("slope", direction=direction, grid_spacing_m=0.05))
    # Read actual surfaces so the check does not depend on common vertex order.
    for x, y in [(0.75, 0.5), (1.75, 1.0), (3.0, 0.5), (4.0, 1.5), (7.8, 1.0)]:
        expected = _surface_height(rough.meshes[0], x, y) + _surface_height(slope.meshes[0], x, y)
        assert _surface_height(combined.meshes[0], x, y) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("direction", terrain.STAIR_DIRECTIONS)
@pytest.mark.parametrize("count,height,depth", [(3, 0.02, 0.5), (5, 0.04, 0.4), (6, 0.05, 0.3)])
def test_stair_treads_risers_count_and_exit_from_mesh(direction, count, height, depth):
    cfg = _cfg("multi_step_stairs", direction=direction, step_count=count,
               step_height_m=height, tread_depth_m=depth)
    result = generate(cfg)
    mesh = result.meshes[0]
    entry = cfg.flat_length_m + cfg.transition_length_m
    previous = 0.0 if direction == "ascending" else count * height
    assert _surface_height(mesh, 0.75, 1.0) == pytest.approx(previous)
    for index in range(count):
        expected = (index + 1) * height if direction == "ascending" else (count - index - 1) * height
        edge = entry + index * depth
        assert _surface_height(mesh, edge - 1e-5, 1.0) == pytest.approx(previous)
        assert _surface_height(mesh, edge + 1e-5, 1.0) == pytest.approx(expected)
        assert _surface_height(mesh, edge + depth * 0.9, 1.0) == pytest.approx(expected)
        previous = expected
    assert _surface_height(mesh, 7.9, 1.0) == pytest.approx(previous)
    risers = mesh.triangles[np.abs(mesh.face_normals[:, 0]) > 0.99]
    assert len(np.unique(np.round(risers[:, :, 0], 10))) == count + 2  # track end walls
    assert mesh.is_watertight and mesh.is_winding_consistent


@pytest.mark.parametrize("layout", terrain.FRICTION_LAYOUTS)
@pytest.mark.parametrize("count", [2, 3, 4, 5, 7, 17])
def test_friction_regions_tile_track_without_gaps_overlap_or_material_claims(layout, count):
    cfg = _cfg("friction_variation", friction_layout=layout, friction_region_count=count,
               friction_range=(0.3, 1.0))
    result = generate(cfg)
    assert len(result.meshes) == len(result.friction_regions) == count
    total_area = 0.0
    for i, region in enumerate(result.friction_regions):
        x0, x1, y0, y1 = region.bounds_xy_m
        total_area += (x1 - x0) * (y1 - y0)
        np.testing.assert_allclose(result.meshes[region.mesh_index].bounds[:, :2], [[x0, y0], [x1, y1]])
        for other in result.friction_regions[i + 1:]:
            a0, a1, b0, b1 = other.bounds_xy_m
            overlap = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
            assert overlap == pytest.approx(0.0, abs=1e-12)
    assert total_area == pytest.approx(np.prod(cfg.size))
    values = [r.friction_coefficient_candidate for r in result.friction_regions]
    assert min(values) == pytest.approx(0.3)
    assert max(values) == pytest.approx(1.0)
    assert result.requires_region_material_binding
    assert result.metadata["effective_friction_confirmed"] is False
    with pytest.raises(RuntimeError, match="material-aware"):
        result.as_isaaclab_mesh_result()


@pytest.mark.parametrize("family", terrain.SOURCE_TERRAIN_FAMILIES[:-1])
def test_single_material_result_contract(family):
    result = generate(_cfg(family))
    meshes, origin = result.as_isaaclab_mesh_result()
    assert meshes == list(result.meshes)
    np.testing.assert_array_equal(origin, result.origin)


@pytest.mark.parametrize("kwargs", [
    {"family": "unknown"}, {"size": (0.0, 2.0)}, {"seed": -1}, {"seed": True},
    {"flat_length_m": 8.0}, {"transition_length_m": 0.0}, {"grid_spacing_m": float("nan")},
    {"family": "multi_scale_rough", "spatial_scales_m": (0.2, 0.2)},
    {"family": "multi_scale_rough", "rough_peak_valley_m": -0.01},
    {"family": "multi_scale_rough", "rough_audit_window_m": 4.0},
    {"family": "slope", "slope_angle_deg": 45.0},
    {"family": "slope", "direction": "ascending"},
    {"family": "slope", "exit_flat_length_m": -0.1},
    {"family": "slope", "exit_flat_length_m": True},
    {"family": "slope", "size": (4.5, 2.0), "flat_length_m": 1.5, "exit_flat_length_m": 2.0},
    {"family": "multi_step_stairs", "direction": "ascending", "step_count": 0},
    {"family": "multi_step_stairs", "direction": "ascending", "tread_depth_m": 4.0},
    {"family": "friction_variation", "friction_range": (1.0, 0.3)},
    {"family": "friction_variation", "friction_range": (float("nan"), 1.0)},
    {"family": "friction_variation", "friction_region_count": 1},
])
def test_invalid_parameters_are_rejected(kwargs):
    kwargs = dict(kwargs)
    family = kwargs.pop("family", "flat")
    with pytest.raises(ValueError):
        _cfg(family, **kwargs)


def test_default_slope_fits_budget_and_huge_grid_fails_before_allocation():
    assert generate(Cfg(family="slope")).meshes
    with pytest.raises(ValueError, match="vertex budget"):
        generate(_cfg("multi_scale_rough", grid_spacing_m=1e-8))


def test_geometry_module_does_not_import_simulation_runtime():
    import ast

    tree = ast.parse(MODULE_PATH.read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and name.split(".")[0] in {"isaaclab", "isaacsim", "omni", "pxr", "torch"}
                   for name in imports)
