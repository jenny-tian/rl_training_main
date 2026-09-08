import importlib

import numpy as np
import pytest


def _api():
    module = importlib.import_module(
        "rl_training.tasks.manager_based.locomotion.velocity.terrains.continuous_rough_track"
    )
    return module.ContinuousRoughTrackCfg, module.continuous_rough_track


def _cfg(**overrides):
    cfg_type, _ = _api()
    values = dict(
        proportion=1.0,
        size=(20.0, 4.0),
        flat_length=3.0,
        transition_length=0.5,
        lane_width=2.0,
        rough_amplitude=0.02,
        correlation_length=0.4,
        seed=101,
    )
    values.update(overrides)
    return cfg_type(**values)


def test_continuous_track_has_flat_prefix_and_shared_heightfield():
    _, generate = _api()
    meshes, origin = generate(1.0, _cfg())
    assert origin.tolist() == [1.5, 2.0, 0.0]
    assert len(meshes) == 1
    heightfield = meshes[0]
    assert heightfield.vertices.shape[0] > 20
    tops = heightfield.vertices[: heightfield.vertices.shape[0] // 2, 2]
    assert np.max(tops) <= 0.02 + 1e-12
    assert np.min(tops) >= -0.02 - 1e-12


def test_same_seed_is_byte_deterministic_and_difficulty_does_not_hide_amplitude():
    _, generate = _api()
    first, _ = generate(0.5, _cfg(seed=77))
    second, _ = generate(0.5, _cfg(seed=77))
    third, _ = generate(1.0, _cfg(seed=77))
    for a, b in zip(first, second, strict=True):
        assert a.vertices.tobytes() == b.vertices.tobytes()
    first_top = np.max(np.abs(first[0].vertices[:, 2]))
    third_top = np.max(np.abs(third[0].vertices[:, 2]))
    assert third_top == pytest.approx(first_top)


def test_heightfield_tapers_to_flat_approach_and_side_strips():
    _, generate = _api()
    meshes, _ = generate(1.0, _cfg())
    top_count = meshes[0].vertices.shape[0] // 2
    top = meshes[0].vertices[:top_count]
    x_min, y_min = top[:, 0].min(), top[:, 1].min()
    y_max = top[:, 1].max()
    boundary = (top[:, 0] == x_min) | (top[:, 1] == y_min) | (top[:, 1] == y_max)
    np.testing.assert_allclose(top[boundary, 2], 0.0)


def test_adjacent_heightfield_columns_are_continuous():
    _, generate = _api()
    meshes, _ = generate(1.0, _cfg())
    mesh = meshes[0]
    top_count = mesh.vertices.shape[0] // 2
    top = mesh.vertices[:top_count]
    # Shared x/y coordinates have exactly one height value, unlike box cells.
    coordinates = {}
    for x, y, z in top:
        key = (round(float(x), 12), round(float(y), 12))
        coordinates.setdefault(key, []).append(float(z))
    assert coordinates
    assert all(len(values) == 1 for values in coordinates.values())


def test_heightfield_top_faces_face_upward():
    _, generate = _api()
    meshes, _ = generate(1.0, _cfg())
    mesh = meshes[0]
    normals = mesh.face_normals
    assert float(normals[:, 2].max()) > 0.5


def test_difficulty_is_validated_but_protocol_amplitude_is_preserved():
    _, generate = _api()
    meshes, _ = generate(0.0, _cfg())
    top_count = meshes[0].vertices.shape[0] // 2
    assert np.max(np.abs(meshes[0].vertices[:top_count, 2])) > 0.0
    with pytest.raises(ValueError):
        generate(1.1, _cfg())
    with pytest.raises(ValueError):
        generate(1.0, _cfg(correlation_length=0.0))
