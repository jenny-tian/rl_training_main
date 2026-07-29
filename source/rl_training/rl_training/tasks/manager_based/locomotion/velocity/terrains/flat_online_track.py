from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import trimesh

try:
    from isaaclab.terrains import SubTerrainBaseCfg
    from isaaclab.utils import configclass
except ModuleNotFoundError as error:
    if error.name != "pxr":
        raise

    @dataclass
    class SubTerrainBaseCfg:
        function: object = None
        proportion: float = 1.0
        size: tuple[float, float] = (10.0, 10.0)

    def configclass(cls):
        declared_function = cls.__dict__.get("function")
        dataclass_type = dataclass(cls)
        generated_init = dataclass_type.__init__

        def fallback_init(self, *args, **kwargs):
            if not args and "function" not in kwargs:
                kwargs["function"] = declared_function
            generated_init(self, *args, **kwargs)

        dataclass_type.__init__ = fallback_init
        return dataclass_type


_SUPPORT_DEPTH = 1.0
# Bounds mesh count and generation time independently of host memory and platform integer width.
_MAX_ROUGH_CELL_COUNT = 4096


@dataclass(frozen=True)
class _ValidatedTrackCfg:
    length: float
    width: float
    flat_length: float
    lane_width: float
    step_height: float
    step_length: float
    rough_amplitude: float
    rough_cell_width: float
    seed: int
    rough_x_cells: int
    rough_y_cells: int


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return normalized


def _cell_count(span: float, cell_width: float, axis: str) -> int:
    ratio = span / cell_width
    if not math.isfinite(ratio):
        raise ValueError(f"rough-cell grid {axis} count is not finite")
    nearest = round(ratio)
    if nearest >= 1 and math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12):
        return nearest
    return max(1, math.ceil(ratio))


def _validate_cfg(difficulty: float, cfg: "FlatOnlineTrackCfg") -> _ValidatedTrackCfg:
    difficulty_value = _finite_real(difficulty, "difficulty")
    if not 0.0 <= difficulty_value <= 1.0:
        raise ValueError(f"difficulty must be in [0, 1], got {difficulty}")

    size = cfg.size
    if isinstance(size, (str, bytes)):
        raise ValueError("size must contain exactly two finite positive dimensions")
    try:
        if len(size) != 2:
            raise ValueError("size must contain exactly two finite positive dimensions")
        length = _finite_real(size[0], "size")
        width = _finite_real(size[1], "size")
    except (TypeError, IndexError, KeyError) as error:
        raise ValueError("size must contain exactly two finite positive dimensions") from error
    if length <= 0.0 or width <= 0.0:
        raise ValueError("size dimensions must be positive")
    if not 0.0 < width / 2.0 < width:
        raise ValueError("size width must contain a representable interior origin")

    flat_length = _finite_real(cfg.flat_length, "flat_length")
    step_length = _finite_real(cfg.step_length, "step_length")
    lane_width = _finite_real(cfg.lane_width, "lane_width")
    rough_cell_width = _finite_real(cfg.rough_cell_width, "rough_cell_width")
    step_height = _finite_real(cfg.step_height, "step_height")
    rough_amplitude = _finite_real(cfg.rough_amplitude, "rough_amplitude")

    if flat_length <= 0.0:
        raise ValueError("flat_length must be positive")
    if not 0.0 < flat_length / 2.0 < flat_length:
        raise ValueError("flat_length must contain a representable interior origin")
    if step_length <= 0.0:
        raise ValueError("step_length must be positive")
    rough_start = flat_length + step_length
    if rough_start <= flat_length:
        raise ValueError("step_length must create a positive mesh dimension")
    if rough_start >= length:
        raise ValueError("flat_length and step_length must leave a non-empty rough segment")
    if not 0.0 < lane_width <= width:
        raise ValueError("lane_width must be positive and no wider than the terrain")
    lane_y_min = (width - lane_width) / 2.0
    if lane_y_min + lane_width <= lane_y_min:
        raise ValueError("lane_width must create a positive mesh dimension")
    if rough_cell_width <= 0.0:
        raise ValueError("rough_cell_width must be positive")
    if not math.isfinite((length - rough_start) / rough_cell_width):
        raise ValueError("rough_cell_width is too small to create positive mesh dimensions")
    if step_height < 0.0:
        raise ValueError("step_height must be nonnegative")
    if rough_amplitude < 0.0:
        raise ValueError("rough_amplitude must be nonnegative")
    if rough_amplitude >= _SUPPORT_DEPTH:
        raise ValueError(f"rough_amplitude must be less than the support depth {_SUPPORT_DEPTH}")
    if type(cfg.seed) is not int or not 0 <= cfg.seed <= 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")

    rough_x_cells = _cell_count(length - rough_start, rough_cell_width, "X")
    rough_y_cells = _cell_count(lane_width, rough_cell_width, "Y")
    rough_cell_count = rough_x_cells * rough_y_cells
    if rough_cell_count > _MAX_ROUGH_CELL_COUNT:
        raise ValueError(
            f"rough-cell grid has {rough_cell_count} cells; "
            f"maximum is {_MAX_ROUGH_CELL_COUNT}"
        )

    return _ValidatedTrackCfg(
        length=length,
        width=width,
        flat_length=flat_length,
        lane_width=lane_width,
        step_height=step_height,
        step_length=step_length,
        rough_amplitude=rough_amplitude,
        rough_cell_width=rough_cell_width,
        seed=cfg.seed,
        rough_x_cells=rough_x_cells,
        rough_y_cells=rough_y_cells,
    )


def _box(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> trimesh.Trimesh:
    if not x_min < x_max or not y_min < y_max or not z_min < z_max:
        raise ValueError("terrain mesh dimensions must be positive")
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    mesh.vertices[:, 0] = np.where(mesh.vertices[:, 0] < 0.0, x_min, x_max)
    mesh.vertices[:, 1] = np.where(mesh.vertices[:, 1] < 0.0, y_min, y_max)
    mesh.vertices[:, 2] = np.where(mesh.vertices[:, 2] < 0.0, z_min, z_max)
    return mesh


def _cell_intervals(
    start: float, stop: float, width: float, count: int
) -> Iterator[tuple[float, float]]:
    for index in range(count):
        left = start + index * width
        right = stop if index == count - 1 else min(start + (index + 1) * width, stop)
        if not left < right:
            raise ValueError("rough_cell_width is too small to create positive mesh dimensions")
        yield left, right


def flat_online_track(
    difficulty: float, cfg: "FlatOnlineTrackCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    values = _validate_cfg(difficulty, cfg)
    rough_start = values.flat_length + values.step_length
    lane_y_min = (values.width - values.lane_width) / 2.0
    lane_y_max = lane_y_min + values.lane_width

    meshes = [
        _box(0.0, values.flat_length, 0.0, values.width, -_SUPPORT_DEPTH, 0.0),
        _box(values.flat_length, rough_start, 0.0, values.width, -_SUPPORT_DEPTH, 0.0),
    ]
    if values.step_height > 0.0:
        meshes.append(
            _box(
                values.flat_length,
                rough_start,
                lane_y_min,
                lane_y_max,
                0.0,
                values.step_height,
            )
        )

    if lane_y_min > 0.0:
        meshes.append(
            _box(rough_start, values.length, 0.0, lane_y_min, -_SUPPORT_DEPTH, 0.0)
        )
    if lane_y_max < values.width:
        meshes.append(
            _box(
                rough_start,
                values.length,
                lane_y_max,
                values.width,
                -_SUPPORT_DEPTH,
                0.0,
            )
        )

    generator = np.random.default_rng(values.seed)
    x_intervals = _cell_intervals(
        rough_start,
        values.length,
        values.rough_cell_width,
        values.rough_x_cells,
    )
    for x_min, x_max in x_intervals:
        y_intervals = _cell_intervals(
            lane_y_min,
            lane_y_max,
            values.rough_cell_width,
            values.rough_y_cells,
        )
        for y_min, y_max in y_intervals:
            top = float(
                generator.uniform(-values.rough_amplitude, values.rough_amplitude)
            )
            meshes.append(_box(x_min, x_max, y_min, y_max, -_SUPPORT_DEPTH, top))

    origin = np.array(
        [values.flat_length / 2.0, values.width / 2.0, 0.0], dtype=np.float64
    )
    return meshes, origin


@configclass
class FlatOnlineTrackCfg(SubTerrainBaseCfg):
    function = flat_online_track
    flat_length: float = 3.0
    lane_width: float = 2.0
    step_height: float = 0.02
    step_length: float = 2.0
    rough_amplitude: float = 0.0
    rough_cell_width: float = 0.4
    seed: int = 0
