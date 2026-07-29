import ast
import importlib
import inspect
import random
import textwrap

import numpy as np
import pytest


def _track_module():
    return importlib.import_module(
        "rl_training.tasks.manager_based.locomotion.velocity.terrains.flat_online_track"
    )


def _module_api():
    module = _track_module()

    return module.FlatOnlineTrackCfg, module.flat_online_track


def _cfg(**overrides):
    FlatOnlineTrackCfg, _ = _module_api()
    values = {
        "proportion": 1.0,
        "size": (20.0, 4.0),
        "flat_length": 3.0,
        "lane_width": 2.0,
        "step_height": 0.06,
        "step_length": 2.0,
        "rough_amplitude": 0.04,
        "rough_cell_width": 0.4,
        "seed": 1009,
    }
    values.update(overrides)
    return FlatOnlineTrackCfg(**values)


def _covering_meshes(meshes, x, y, tolerance=1e-10):
    return [
        mesh
        for mesh in meshes
        if mesh.bounds[0, 0] - tolerance <= x <= mesh.bounds[1, 0] + tolerance
        and mesh.bounds[0, 1] - tolerance <= y <= mesh.bounds[1, 1] + tolerance
    ]


def _rough_lane_meshes(meshes, cfg):
    rough_start = cfg.flat_length + cfg.step_length
    lane_min = (cfg.size[1] - cfg.lane_width) / 2
    lane_max = lane_min + cfg.lane_width
    return [
        mesh
        for mesh in meshes
        if mesh.bounds[0, 0] >= rough_start - 1e-10
        and mesh.bounds[0, 1] >= lane_min - 1e-10
        and mesh.bounds[1, 1] <= lane_max + 1e-10
    ]


def _assert_meshes_well_formed(meshes):
    assert meshes
    for mesh in meshes:
        assert np.isfinite(mesh.vertices).all()
        assert np.all(mesh.extents > 0.0)


def test_combined_track_has_flat_step_rough_lane_and_flat_side_strips():
    _, flat_online_track = _module_api()
    cfg = _cfg()

    meshes, origin = flat_online_track(0.73, cfg)

    _assert_meshes_well_formed(meshes)
    assert origin.dtype == np.float64
    assert 0.0 < origin[0] < cfg.flat_length
    np.testing.assert_array_equal(origin[1:], [cfg.size[1] / 2, 0.0])

    prefix_tops = [mesh.bounds[1, 2] for mesh in _covering_meshes(meshes, 2.0, 2.0)]
    assert max(prefix_tops) == pytest.approx(0.0)

    step_tops = [mesh.bounds[1, 2] for mesh in _covering_meshes(meshes, 4.0, 2.0)]
    assert max(step_tops) == pytest.approx(0.06)

    rough_tops = np.array([mesh.bounds[1, 2] for mesh in _rough_lane_meshes(meshes, cfg)])
    assert np.all(rough_tops >= -cfg.rough_amplitude)
    assert np.all(rough_tops <= cfg.rough_amplitude)
    assert np.any(rough_tops < 0.0)
    assert np.any(rough_tops > 0.0)

    for y in (0.5, 3.5):
        side_tops = [mesh.bounds[1, 2] for mesh in _covering_meshes(meshes, 10.0, y)]
        assert side_tops == pytest.approx([0.0])


def test_negative_rough_cells_have_no_zero_height_overlap():
    _, flat_online_track = _module_api()
    cfg = _cfg()
    meshes, _ = flat_online_track(0.5, cfg)

    negative_cells = [mesh for mesh in _rough_lane_meshes(meshes, cfg) if mesh.bounds[1, 2] < 0.0]
    assert negative_cells
    for cell in negative_cells:
        center = cell.bounds.mean(axis=0)
        covering = _covering_meshes(meshes, center[0], center[1])
        assert covering == [cell]
        assert all(mesh.bounds[1, 2] < 0.0 for mesh in covering)


def test_same_seed_produces_byte_equal_meshes_and_origin():
    _, flat_online_track = _module_api()

    first_meshes, first_origin = flat_online_track(0.0, _cfg())
    second_meshes, second_origin = flat_online_track(1.0, _cfg())

    np.testing.assert_array_equal(first_origin, second_origin)
    assert len(first_meshes) == len(second_meshes)
    for first, second in zip(first_meshes, second_meshes, strict=True):
        assert first.vertices.tobytes() == second.vertices.tobytes()
        assert first.faces.tobytes() == second.faces.tobytes()


def test_different_seed_changes_only_rough_cell_top_heights():
    _, flat_online_track = _module_api()
    first_cfg = _cfg(seed=1009)
    second_cfg = _cfg(seed=1010)

    first_meshes, _ = flat_online_track(0.5, first_cfg)
    second_meshes, _ = flat_online_track(0.5, second_cfg)
    rough_start = first_cfg.flat_length + first_cfg.step_length
    changed_rough_tops = []

    assert len(first_meshes) == len(second_meshes)
    for first, second in zip(first_meshes, second_meshes, strict=True):
        np.testing.assert_array_equal(first.faces, second.faces)
        is_rough_cell = first.bounds[0, 0] >= rough_start - 1e-10 and first.bounds[0, 1] >= 1.0 - 1e-10 and first.bounds[1, 1] <= 3.0 + 1e-10
        if not is_rough_cell:
            np.testing.assert_array_equal(first.vertices, second.vertices)
            continue

        np.testing.assert_array_equal(first.vertices[:, :2], second.vertices[:, :2])
        assert first.bounds[0, 2] == pytest.approx(-1.0)
        assert second.bounds[0, 2] == pytest.approx(-1.0)
        changed_rough_tops.append(first.bounds[1, 2] != second.bounds[1, 2])

    assert changed_rough_tops and all(changed_rough_tops)


def test_generation_does_not_change_global_numpy_or_python_rng_state():
    _, flat_online_track = _module_api()
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    flat_online_track(0.5, _cfg())

    numpy_after = np.random.get_state()
    python_after = random.getstate()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == python_after


def test_partial_cells_cover_lane_to_exact_boundaries_without_gaps_or_overshoot():
    _, flat_online_track = _module_api()
    cfg = _cfg(
        size=(6.25, 3.1),
        flat_length=1.1,
        step_length=0.9,
        lane_width=1.3,
        rough_cell_width=0.6,
    )

    meshes, _ = flat_online_track(0.5, cfg)
    cells = _rough_lane_meshes(meshes, cfg)
    x_intervals = sorted({(mesh.bounds[0, 0], mesh.bounds[1, 0]) for mesh in cells})
    y_intervals = sorted({(mesh.bounds[0, 1], mesh.bounds[1, 1]) for mesh in cells})
    lane_min = (cfg.size[1] - cfg.lane_width) / 2
    lane_max = lane_min + cfg.lane_width

    assert x_intervals[0][0] == pytest.approx(cfg.flat_length + cfg.step_length)
    assert x_intervals[-1][1] == pytest.approx(cfg.size[0])
    assert y_intervals[0][0] == pytest.approx(lane_min)
    assert y_intervals[-1][1] == pytest.approx(lane_max)
    for intervals in (x_intervals, y_intervals):
        for left, right in zip(intervals, intervals[1:], strict=False):
            assert left[1] == pytest.approx(right[0])
    assert max(right - left for left, right in x_intervals) <= cfg.rough_cell_width + 1e-10
    assert max(right - left for left, right in y_intervals) <= cfg.rough_cell_width + 1e-10
    assert all(mesh.bounds[1, 0] <= cfg.size[0] + 1e-10 for mesh in meshes)
    assert all(mesh.bounds[1, 1] <= cfg.size[1] + 1e-10 for mesh in meshes)


def test_zero_rough_amplitude_produces_zero_height_lane():
    _, flat_online_track = _module_api()
    cfg = _cfg(rough_amplitude=0.0)

    meshes, _ = flat_online_track(0.5, cfg)

    assert {mesh.bounds[1, 2] for mesh in _rough_lane_meshes(meshes, cfg)} == {0.0}


def test_zero_step_height_creates_no_raised_step_surface():
    _, flat_online_track = _module_api()
    cfg = _cfg(step_height=0.0)

    meshes, _ = flat_online_track(0.5, cfg)

    step_tops = [mesh.bounds[1, 2] for mesh in _covering_meshes(meshes, 4.0, 2.0)]
    assert step_tops == pytest.approx([0.0])


def test_tiny_finite_cell_width_is_rejected_before_mesh_materialization(monkeypatch):
    module = _track_module()
    monkeypatch.setattr(module, "_box", lambda *args: pytest.fail("mesh materialized"))
    cfg = _cfg(
        size=(0.0102, 1e-6),
        flat_length=0.0001,
        step_length=0.0001,
        lane_width=1e-6,
        rough_cell_width=1e-6,
    )

    with pytest.raises(ValueError, match="rough-cell grid"):
        module.flat_online_track(0.5, cfg)


def test_overflowing_y_grid_is_rejected_before_mesh_materialization(monkeypatch):
    module = _track_module()
    monkeypatch.setattr(module, "_box", lambda *args: pytest.fail("mesh materialized"))

    with pytest.raises(ValueError, match="rough-cell grid"):
        module.flat_online_track(
            0.5,
            _cfg(size=(20.0, 1e308), lane_width=1e308, rough_cell_width=0.4),
        )


def test_exact_rough_cell_budget_is_accepted(monkeypatch):
    module = _track_module()
    assert module._MAX_ROUGH_CELL_COUNT == 4096
    sentinel = object()
    monkeypatch.setattr(module, "_box", lambda *args: sentinel)
    cfg = _cfg(
        size=(67.0, 64.0),
        flat_length=1.0,
        step_length=2.0,
        step_height=0.0,
        lane_width=64.0,
        rough_cell_width=1.0,
        rough_amplitude=0.0,
    )

    meshes, _ = module.flat_online_track(0.5, cfg)

    assert len(meshes) == 2 + module._MAX_ROUGH_CELL_COUNT


def test_rough_cell_budget_is_rejected_above_boundary_before_materialization(monkeypatch):
    module = _track_module()
    monkeypatch.setattr(module, "_box", lambda *args: pytest.fail("mesh materialized"))
    cfg = _cfg(
        size=(68.0, 64.0),
        flat_length=1.0,
        step_length=2.0,
        lane_width=64.0,
        rough_cell_width=1.0,
    )

    with pytest.raises(ValueError, match="rough-cell grid"):
        module.flat_online_track(0.5, cfg)


def test_underflowing_y_ratio_still_covers_lane_with_one_partial_cell():
    _, flat_online_track = _module_api()
    cfg = _cfg(
        size=(20.0, 1e-200),
        lane_width=1e-200,
        rough_cell_width=1e200,
        rough_amplitude=0.0,
    )

    meshes, _ = flat_online_track(0.5, cfg)
    rough_cells = _rough_lane_meshes(meshes, cfg)

    assert len(rough_cells) == 1
    np.testing.assert_array_equal(
        rough_cells[0].bounds[:, :2],
        [[cfg.flat_length + cfg.step_length, 0.0], [cfg.size[0], cfg.lane_width]],
    )


def test_rejects_width_without_representable_interior_origin():
    _, flat_online_track = _module_api()
    minimum_positive = float.fromhex("0x0.0000000000001p-1022")
    cfg = _cfg(
        size=(20.0, minimum_positive),
        lane_width=minimum_positive,
    )

    with pytest.raises(ValueError, match="size width"):
        flat_online_track(0.5, cfg)


def test_geometry_uses_normalized_real_values():
    class ArithmeticTrapFloat(float):
        def _fail(self, _other=None):
            raise AssertionError("raw Real arithmetic used")

        __add__ = _fail
        __radd__ = _fail
        __sub__ = _fail
        __rsub__ = _fail
        __truediv__ = _fail
        __rtruediv__ = _fail
        __mul__ = _fail
        __rmul__ = _fail
        __neg__ = _fail

    _, flat_online_track = _module_api()
    trapped_cfg = _cfg(
        size=(ArithmeticTrapFloat(20.0), ArithmeticTrapFloat(4.0)),
        flat_length=ArithmeticTrapFloat(3.0),
        step_length=ArithmeticTrapFloat(2.0),
        step_height=ArithmeticTrapFloat(0.06),
        lane_width=ArithmeticTrapFloat(2.0),
        rough_cell_width=ArithmeticTrapFloat(0.4),
        rough_amplitude=ArithmeticTrapFloat(0.04),
    )

    trapped_meshes, trapped_origin = flat_online_track(0.5, trapped_cfg)
    normal_meshes, normal_origin = flat_online_track(0.5, _cfg())

    np.testing.assert_array_equal(trapped_origin, normal_origin)
    assert len(trapped_meshes) == len(normal_meshes)
    for trapped, normal in zip(trapped_meshes, normal_meshes, strict=True):
        np.testing.assert_array_equal(trapped.vertices, normal.vertices)
        np.testing.assert_array_equal(trapped.faces, normal.faces)


def test_source_precision_overflow_is_a_value_error():
    _, flat_online_track = _module_api()
    source_finite = np.longdouble("1e4000")
    assert np.isfinite(source_finite)

    with pytest.raises(ValueError, match="size"):
        flat_online_track(0.5, _cfg(size=(source_finite, 4.0)))


@pytest.mark.parametrize("difficulty", [True, False, -0.01, 1.01, float("nan"), float("inf"), "0.5", None])
def test_rejects_invalid_difficulty(difficulty):
    _, flat_online_track = _module_api()

    with pytest.raises(ValueError, match="difficulty"):
        flat_online_track(difficulty, _cfg())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"size": (20.0,)}, "size"),
        ({"size": (20.0, 4.0, 1.0)}, "size"),
        ({"size": {"length": 20.0, "width": 4.0}}, "size"),
        ({"size": (float("nan"), 4.0)}, "size"),
        ({"size": (20.0, False)}, "size"),
        ({"size": (20.0, 0.0)}, "size"),
        ({"flat_length": 0.0}, "flat_length"),
        ({"flat_length": float.fromhex("0x0.0000000000001p-1022")}, "flat_length"),
        ({"flat_length": True}, "flat_length"),
        ({"step_length": 0.0}, "step_length"),
        ({"step_length": float.fromhex("0x0.0000000000001p-1022")}, "step_length"),
        ({"flat_length": 18.0}, "rough segment"),
        ({"lane_width": 0.0}, "lane_width"),
        ({"lane_width": float.fromhex("0x0.0000000000001p-1022")}, "lane_width"),
        ({"lane_width": 4.1}, "lane_width"),
        ({"rough_cell_width": 0.0}, "rough_cell_width"),
        ({"rough_cell_width": float.fromhex("0x0.0000000000001p-1022")}, "rough_cell_width"),
        ({"rough_cell_width": float("inf")}, "rough_cell_width"),
        ({"step_height": -0.01}, "step_height"),
        ({"step_height": float("nan")}, "step_height"),
        ({"rough_amplitude": -0.01}, "rough_amplitude"),
        ({"rough_amplitude": 1.0}, "support depth"),
        ({"rough_amplitude": float("inf")}, "rough_amplitude"),
        ({"seed": True}, "seed"),
        ({"seed": 1.0}, "seed"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**63}, "seed"),
    ],
)
def test_rejects_invalid_configuration(overrides, match):
    _, flat_online_track = _module_api()

    with pytest.raises(ValueError, match=match):
        flat_online_track(0.5, _cfg(**overrides))


def test_package_exports_track_symbols_and_config_points_to_function():
    from rl_training.tasks.manager_based.locomotion.velocity import terrains

    FlatOnlineTrackCfg, flat_online_track = _module_api()
    cfg = _cfg()

    assert terrains.FlatOnlineTrackCfg is FlatOnlineTrackCfg
    assert terrains.flat_online_track is flat_online_track
    assert "FlatOnlineTrackCfg" in terrains.__all__
    assert "flat_online_track" in terrains.__all__
    assert cfg.function is flat_online_track

    source_tree = ast.parse(textwrap.dedent(inspect.getsource(FlatOnlineTrackCfg)))
    class_node = next(node for node in source_tree.body if isinstance(node, ast.ClassDef))
    function_declarations = [
        node
        for node in class_node.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "function" for target in node.targets)
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "function"
        )
    ]
    assert len(function_declarations) == 1
    assert isinstance(function_declarations[0], ast.Assign)
    assert ast.unparse(function_declarations[0].value) == "flat_online_track"
