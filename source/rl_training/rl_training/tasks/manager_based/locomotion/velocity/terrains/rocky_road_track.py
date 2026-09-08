"""Continuous irregular rocky-road terrain for held-out robustness tests.

Unlike a smoothed noise field, this generator places deterministic, irregular
elliptic rocks and potholes on one shared height-field mesh.  The single mesh
keeps the collision surface continuous while the short spatial scale and
opposing height changes create realistic foot-placement disturbances.
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

    @dataclass
    class SubTerrainBaseCfg:
        function: object = None
        proportion: float = 1.0
        size: tuple[float, float] = (10.0, 10.0)

    def configclass(cls):
        declared_function = cls.__dict__.get("function")
        cls = dataclass(cls)
        generated_init = cls.__init__

        def fallback_init(self, *args, **kwargs):
            if not args and "function" not in kwargs:
                kwargs["function"] = declared_function
            generated_init(self, *args, **kwargs)

        cls.__init__ = fallback_init
        return cls


_SUPPORT_DEPTH = 1.0
_MAX_GRID_VERTICES = 40_000


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _grid(start: float, stop: float, spacing: float, boundaries: tuple[float, ...]) -> np.ndarray:
    count = max(1, int(math.ceil((stop - start) / spacing)))
    values = list(np.linspace(start, stop, count + 1, dtype=np.float64))
    values.extend(float(v) for v in boundaries if start < v < stop)
    return np.asarray(sorted(set(values)), dtype=np.float64)


def _heightfield_prism(x: np.ndarray, y: np.ndarray, heights: np.ndarray) -> trimesh.Trimesh:
    rows, cols = heights.shape
    top = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    top_vertices = np.column_stack((top, heights.reshape(-1)))
    bottom_vertices = np.column_stack((top, np.full(len(top), -_SUPPORT_DEPTH)))
    vertices = np.vstack((top_vertices, bottom_vertices))
    faces: list[tuple[int, int, int]] = []
    offset = len(top)
    for iy in range(rows - 1):
        for ix in range(cols - 1):
            a = iy * cols + ix
            b, c, d = a + 1, a + cols, a + cols + 1
            faces.extend(((a, b, c), (b, d, c),
                          (offset + a, offset + c, offset + b),
                          (offset + b, offset + c, offset + d)))
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


def _smoothstep(edge: np.ndarray) -> np.ndarray:
    edge = np.clip(edge, 0.0, 1.0)
    return edge * edge * (3.0 - 2.0 * edge)


def rocky_road_track(difficulty: float, cfg: "RockyRoadTrackCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    difficulty = _finite(difficulty, "difficulty")
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")
    length, width = _finite(cfg.size[0], "size"), _finite(cfg.size[1], "size")
    flat_length = _finite(cfg.flat_length, "flat_length")
    transition = _finite(cfg.transition_length, "transition_length")
    lane_width = _finite(cfg.lane_width, "lane_width")
    spacing = _finite(cfg.grid_spacing, "grid_spacing")
    amplitude = _finite(cfg.rough_amplitude, "rough_amplitude")
    if min(length, width, flat_length, transition, lane_width, spacing) <= 0.0:
        raise ValueError("terrain dimensions and grid spacing must be positive")
    if flat_length + transition >= length or lane_width > width:
        raise ValueError("invalid rocky-road dimensions")
    if amplitude <= 0.0 or amplitude >= _SUPPORT_DEPTH:
        raise ValueError("rough_amplitude must lie in (0, support depth)")
    height_min = _finite(cfg.height_range[0], "height_range")
    height_max = _finite(cfg.height_range[1], "height_range")
    if not 0.0 < height_min <= height_max:
        raise ValueError("height_range must contain positive ascending values")

    rough_start = flat_length + transition
    lane_min, lane_max = (width - lane_width) / 2.0, (width + lane_width) / 2.0
    x = _grid(0.0, length, spacing, (rough_start,))
    y = _grid(0.0, width, spacing, (lane_min, lane_max))
    if len(x) * len(y) > _MAX_GRID_VERTICES:
        raise ValueError("rocky-road heightfield exceeds vertex budget")
    xx, yy = np.meshgrid(x, y)
    heights = np.zeros_like(xx, dtype=np.float64)
    rng = np.random.default_rng(cfg.seed)

    # A small background undulation prevents the road from becoming a set of
    # isolated objects while keeping the dominant disturbance geometry explicit.
    background = rng.uniform(-1.0, 1.0, size=xx.shape)
    for _ in range(2):
        padded = np.pad(background, 1, mode="edge")
        background = (padded[1:-1, 1:-1] * 0.5
                      + (padded[:-2, 1:-1] + padded[2:, 1:-1]) * 0.125
                      + (padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.125)
    peak = float(np.max(np.abs(background)))
    if peak > 0.0:
        background /= peak
    rough = (xx >= rough_start) & (yy >= lane_min) & (yy <= lane_max)
    heights[rough] = 0.18 * amplitude * background[rough]

    # Draw irregular rocks and potholes.  The super-elliptic footprint gives
    # non-circular shapes; a smooth edge avoids non-physical mesh discontinuities.
    x_min = rough_start + 0.18
    x_max = length - 0.18
    y_min, y_max = lane_min + 0.12, lane_max - 0.12
    count = int(round(cfg.rock_count * (0.75 + 0.25 * difficulty)))
    for _ in range(max(1, count)):
        cx = float(rng.uniform(x_min, x_max))
        cy = float(rng.uniform(y_min, y_max))
        rx = float(rng.uniform(cfg.radius_x[0], cfg.radius_x[1]))
        ry = float(rng.uniform(cfg.radius_y[0], cfg.radius_y[1]))
        exponent = float(rng.uniform(2.2, 4.5))
        sign = 1.0 if rng.random() < cfg.rock_fraction else -1.0
        # Treat rough_amplitude as a scale, not merely a clipping threshold.
        # This preserves the same layout while making amplitude sweeps change
        # every rock and pothole proportionally (e.g. 0.10 -> 0.06 gives
        # 5-10 cm -> 3-6 cm rather than a field of clipped 6 cm plateaus).
        height = float(rng.uniform(height_min, height_max)) * amplitude / height_max
        radial = (np.abs((xx - cx) / rx) ** exponent + np.abs((yy - cy) / ry) ** exponent) ** (1.0 / exponent)
        footprint = _smoothstep((1.0 - radial) / 0.22)
        heights += sign * height * footprint * rough

    heights[:, np.isclose(x, rough_start)] = 0.0
    heights[np.isclose(y, lane_min) | np.isclose(y, lane_max), :] = 0.0
    heights = np.clip(heights, -amplitude, amplitude)
    mesh = _heightfield_prism(x, y, heights)
    origin = np.array([flat_length / 2.0, width / 2.0, 0.0], dtype=np.float64)
    return [mesh], origin


@configclass
class RockyRoadTrackCfg(SubTerrainBaseCfg):
    function = rocky_road_track
    flat_length: float = 3.0
    # Retained for the shared online geometry protocol.  This terrain has no
    # discrete step and only rough_amplitude is applied to its height bounds.
    step_height: float = 0.0
    transition_length: float = 0.5
    lane_width: float = 2.0
    rough_amplitude: float = 0.10
    grid_spacing: float = 0.10
    rock_count: int = 24
    rock_fraction: float = 0.58
    radius_x: tuple[float, float] = (0.12, 0.32)
    radius_y: tuple[float, float] = (0.10, 0.28)
    height_range: tuple[float, float] = (0.045, 0.10)
    seed: int = 101
