"""Continuous, spatially correlated rough terrain for reproducible rollouts.

The rough lane is one shared height-field mesh rather than independent boxes.
This keeps the collision surface C0-continuous while retaining a bounded,
seeded random profile for causal online-adaptation experiments.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
import trimesh

try:
    from isaaclab.terrains import SubTerrainBaseCfg
    from isaaclab.utils import configclass
except ModuleNotFoundError as error:
    if error.name != "pxr":
        raise
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class SubTerrainBaseCfg:
        function: object = None
        proportion: float = 1.0
        size: tuple[float, float] = (10.0, 10.0)

    def configclass(cls):
        declared_function = cls.__dict__.get("function")
        dataclass_type = _dataclass(cls)
        generated_init = dataclass_type.__init__

        def fallback_init(self, *args, **kwargs):
            if not args and "function" not in kwargs:
                kwargs["function"] = declared_function
            generated_init(self, *args, **kwargs)

        dataclass_type.__init__ = fallback_init
        return dataclass_type


_SUPPORT_DEPTH = 1.0
_MAX_GRID_VERTICES = 10_000


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _intervals(start: float, stop: float, width: float) -> np.ndarray:
    count = max(1, int(math.ceil((stop - start) / width)))
    return np.linspace(start, stop, count + 1, dtype=np.float64)


def _grid_with_boundaries(
    start: float, stop: float, width: float, boundaries: tuple[float, ...]
) -> np.ndarray:
    values = list(_intervals(start, stop, width))
    values.extend(float(value) for value in boundaries if start < value < stop)
    return np.asarray(sorted(set(values)), dtype=np.float64)


def _smooth_noise(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    noise = rng.uniform(-1.0, 1.0, size=(rows, cols))
    # A short separable blur gives neighbouring cells correlated heights while
    # avoiding a dependency on scipy in the terrain-generation process.
    for _ in range(3):
        padded = np.pad(noise, 1, mode="edge")
        noise = (
            padded[1:-1, 1:-1] * 0.5
            + (padded[:-2, 1:-1] + padded[2:, 1:-1]) * 0.125
            + (padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.125
        )
    peak = float(np.max(np.abs(noise)))
    return noise / peak if peak > 0.0 else noise


def _heightfield_prism(x: np.ndarray, y: np.ndarray, heights: np.ndarray) -> trimesh.Trimesh:
    rows, cols = heights.shape
    if rows != len(y) or cols != len(x):
        raise ValueError("heightfield shape does not match grid")
    top = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    top_vertices = np.column_stack((top, heights.reshape(-1)))
    bottom_vertices = np.column_stack((top, np.full(len(top), -_SUPPORT_DEPTH)))
    vertices = np.vstack((top_vertices, bottom_vertices))
    faces: list[tuple[int, int, int]] = []
    offset = len(top)
    for iy in range(rows - 1):
        for ix in range(cols - 1):
            a = iy * cols + ix
            b = a + 1
            c = a + cols
            d = c + 1
            # Top faces wind counter-clockwise as viewed from above (+Z).
            faces.extend(((a, b, c), (b, d, c), (offset + a, offset + c, offset + b), (offset + b, offset + c, offset + d)))
    for ix in range(cols - 1):
        a, b = ix, ix + 1
        faces.extend(((a, b, offset + a), (b, offset + b, offset + a)))
        a, b = (rows - 1) * cols + ix, (rows - 1) * cols + ix + 1
        faces.extend(((a, offset + a, b), (b, offset + a, offset + b)))
    for iy in range(rows - 1):
        a, b = iy * cols, (iy + 1) * cols
        faces.extend(((a, offset + a, b), (b, offset + a, offset + b)))
        a, b = iy * cols + cols - 1, (iy + 1) * cols + cols - 1
        faces.extend(((a, b, offset + a), (b, offset + b, offset + a)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


def continuous_rough_track(difficulty: float, cfg: "ContinuousRoughTrackCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    difficulty = _finite(difficulty, "difficulty")
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")
    length, width = (_finite(cfg.size[0], "size"), _finite(cfg.size[1], "size"))
    flat_length = _finite(cfg.flat_length, "flat_length")
    transition = _finite(cfg.transition_length, "transition_length")
    lane_width = _finite(cfg.lane_width, "lane_width")
    correlation = _finite(cfg.correlation_length, "correlation_length")
    # The protocol amplitude is the actual physical bound. Isaac Lab may pass
    # an implementation-dependent difficulty even when curriculum is off, so
    # do not silently rescale a declared +/-N cm evaluation terrain.
    amplitude = _finite(cfg.rough_amplitude, "rough_amplitude")
    if min(length, width, flat_length, transition, lane_width, correlation) <= 0.0:
        raise ValueError("terrain dimensions and correlation length must be positive")
    if flat_length + transition >= length or lane_width > width or amplitude >= _SUPPORT_DEPTH:
        raise ValueError("invalid continuous rough terrain dimensions")
    rough_start = flat_length + transition
    lane_min, lane_max = (width - lane_width) / 2.0, (width + lane_width) / 2.0
    x = _grid_with_boundaries(0.0, length, correlation, (rough_start,))
    y = _grid_with_boundaries(0.0, width, correlation, (lane_min, lane_max))
    if len(x) * len(y) > _MAX_GRID_VERTICES:
        raise ValueError("continuous rough heightfield exceeds vertex budget")
    rng = np.random.default_rng(cfg.seed)
    heights = np.zeros((len(y), len(x)), dtype=np.float64)
    noise = _smooth_noise(rng, len(y), len(x))
    rough_mask = (
        (np.asarray(y)[:, None] >= lane_min)
        & (np.asarray(y)[:, None] <= lane_max)
        & (np.asarray(x)[None, :] >= rough_start)
    )
    heights[rough_mask] = amplitude * noise[rough_mask]
    heights[:, np.isclose(x, rough_start)] = 0.0
    heights[np.isclose(y, lane_min) | np.isclose(y, lane_max), :] = 0.0

    meshes = [_heightfield_prism(x, y, heights)]
    origin = np.array([flat_length / 2.0, width / 2.0, 0.0], dtype=np.float64)
    return meshes, origin


@configclass
class ContinuousRoughTrackCfg(SubTerrainBaseCfg):
    function = continuous_rough_track
    flat_length: float = 3.0
    # Kept for compatibility with the online geometry protocol; continuous
    # terrain never creates a discrete step.
    step_height: float = 0.0
    step_length: float = 0.5
    transition_length: float = 0.5
    lane_width: float = 2.0
    rough_amplitude: float = 0.02
    correlation_length: float = 0.4
    seed: int = 0
