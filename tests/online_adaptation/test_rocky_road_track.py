import importlib

import numpy as np
import pytest


def _api():
    module = importlib.import_module(
        "rl_training.tasks.manager_based.locomotion.velocity.terrains.rocky_road_track"
    )
    return module.RockyRoadTrackCfg, module.rocky_road_track


def _cfg(**overrides):
    cfg_type, _ = _api()
    values = dict(
        proportion=1.0,
        size=(10.0, 4.0),
        flat_length=3.0,
        transition_length=0.5,
        lane_width=2.0,
        rough_amplitude=0.10,
        grid_spacing=0.10,
        rock_count=24,
        rock_fraction=0.58,
        radius_x=(0.12, 0.32),
        radius_y=(0.10, 0.28),
        height_range=(0.045, 0.10),
        seed=101,
    )
    values.update(overrides)
    return cfg_type(**values)


def test_rocky_road_is_one_shared_mesh_and_bounded():
    _, generate = _api()
    meshes, origin = generate(1.0, _cfg())
    assert len(meshes) == 1
    assert origin.tolist() == [1.5, 2.0, 0.0]
    mesh = meshes[0]
    top = mesh.vertices[: mesh.vertices.shape[0] // 2]
    assert float(top[:, 2].max()) <= 0.10 + 1e-12
    assert float(top[:, 2].min()) >= -0.10 - 1e-12
    assert float(mesh.face_normals[:, 2].max()) > 0.5


def test_rocky_road_is_deterministic_and_has_both_rocks_and_potholes():
    _, generate = _api()
    first, _ = generate(0.5, _cfg(seed=17))
    second, _ = generate(0.5, _cfg(seed=17))
    assert first[0].vertices.tobytes() == second[0].vertices.tobytes()
    top = first[0].vertices[: first[0].vertices.shape[0] // 2, 2]
    assert np.any(top > 0.02)
    assert np.any(top < -0.02)


def test_rocky_road_validates_inputs():
    _, generate = _api()
    with pytest.raises(ValueError):
        generate(1.1, _cfg())
    with pytest.raises(ValueError):
        generate(1.0, _cfg(grid_spacing=0.0))
    with pytest.raises(ValueError):
        generate(1.0, _cfg(rough_amplitude=1.0))
    with pytest.raises(ValueError):
        generate(1.0, _cfg(height_range=(0.10, 0.05)))


def test_amplitude_sweep_scales_the_same_seeded_terrain_shape():
    _, generate = _api()
    large, _ = generate(1.0, _cfg(rough_amplitude=0.10))
    small, _ = generate(1.0, _cfg(rough_amplitude=0.06))
    large_top = large[0].vertices[: large[0].vertices.shape[0] // 2, 2]
    small_top = small[0].vertices[: small[0].vertices.shape[0] // 2, 2]
    np.testing.assert_allclose(small_top, large_top * 0.6, atol=1e-12)


def test_full_width_lane_has_no_flat_lateral_bypass_strip():
    _, generate = _api()
    meshes, _ = generate(1.0, _cfg(lane_width=4.0))
    mesh = meshes[0]
    top = mesh.vertices[: mesh.vertices.shape[0] // 2]
    rough = top[top[:, 0] > 3.5]
    # The exact outer edge is flat so the prism can close cleanly.  Every
    # interior lateral grid row must nevertheless contain rough geometry.
    interior_y = sorted(set(rough[(rough[:, 1] > 0.0) & (rough[:, 1] < 4.0), 1]))
    assert interior_y
    for y in interior_y:
        row = rough[np.isclose(rough[:, 1], y), 2]
        assert np.any(np.abs(row) > 1e-6)
