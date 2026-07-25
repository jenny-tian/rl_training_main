from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        function: Any = None
        proportion: float = 1.0
        size: tuple[float, float] = (10.0, 10.0)

    configclass = dataclass


def heldout_boxes_track(difficulty: float, cfg: "HeldoutBoxesTrackCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"difficulty must be in [0, 1], got {difficulty}")
    if not 0.0 < cfg.flat_length < cfg.size[0]:
        raise ValueError("flat_length must lie inside the terrain X extent")
    if cfg.grid_width <= 0.0:
        raise ValueError("grid_width must be positive")
    if not 0.0 < cfg.lane_width <= cfg.size[1]:
        raise ValueError("lane_width must be positive and no wider than the terrain")

    base_depth = 1.0
    base = trimesh.creation.box(
        (cfg.size[0], cfg.size[1], base_depth),
        trimesh.transformations.translation_matrix((cfg.size[0] / 2, cfg.size[1] / 2, -base_depth / 2)),
    )
    meshes: list[trimesh.Trimesh] = [base]
    max_height = cfg.grid_height_range[0] + difficulty * (
        cfg.grid_height_range[1] - cfg.grid_height_range[0]
    )
    generator = np.random.default_rng(getattr(cfg, "seed", 0))
    lane_y_min = (cfg.size[1] - cfg.lane_width) / 2
    lane_y_max = lane_y_min + cfg.lane_width

    x = cfg.flat_length
    while x < cfg.size[0] - 1e-9:
        cell_x = min(cfg.grid_width, cfg.size[0] - x)
        y = lane_y_min
        while y < lane_y_max - 1e-9:
            cell_y = min(cfg.grid_width, lane_y_max - y)
            top_height = float(generator.uniform(-max_height, max_height))
            column_height = base_depth + top_height
            column = trimesh.creation.box(
                (cell_x, cell_y, column_height),
                trimesh.transformations.translation_matrix(
                    (x + cell_x / 2, y + cell_y / 2, (top_height - base_depth) / 2)
                ),
            )
            meshes.append(column)
            y += cell_y
        x += cell_x

    origin = np.array([1.0, cfg.size[1] / 2, 0.0], dtype=np.float64)
    return meshes, origin


@configclass
class HeldoutBoxesTrackCfg(SubTerrainBaseCfg):
    function = heldout_boxes_track
    flat_length: float = 3.0
    lane_width: float = 2.0
    grid_width: float = 0.4
    grid_height_range: tuple[float, float] = (0.04, 0.12)
    seed: int = 0
