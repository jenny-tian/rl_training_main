"""Eight-metre L2 blocks, using unmodified Isaac Lab pyramid stair generators."""
from __future__ import annotations

import copy
import math
import numpy as np
import trimesh
from isaaclab.terrains import MeshPyramidStairsTerrainCfg, MeshInvertedPyramidStairsTerrainCfg
from isaaclab.utils import configclass

from rl_training.source_curriculum.block import BLOCK_SIZE, replica_rows
from rl_training.source_curriculum.rows import NUM_COLUMNS, NUM_ROWS, column_fields, curriculum_row_parameters, row_from_difficulty, track_overrides
from .source_terrain_isaac import SourceTerrainTrackCfg, source_terrain_track
from .source_terrain_importer import SourceTerrainImporter
from .source_terrain_importer_cfg import SourceTerrainImporterCfg


def source_block_terrain(difficulty, cfg):
    if tuple(cfg.size) != BLOCK_SIZE:
        raise ValueError("fixed-L2 blocks require size=(8, 8)")
    row = row_from_difficulty(difficulty, NUM_ROWS)
    if cfg.family == "multi_step_stairs":
        # Starting in the centre reverses the usual naming convention:
        # inverted = ascent, raised pyramid = descent.
        cls = MeshInvertedPyramidStairsTerrainCfg if cfg.direction == "ascending" else MeshPyramidStairsTerrainCfg
        stair = cls(size=BLOCK_SIZE, step_height_range=(.035, .035), step_width=.5,
                    platform_width=3., border_width=1., holes=False)
        return stair.function(difficulty, stair)
    track = copy.deepcopy(cfg)
    params = curriculum_row_parameters(NUM_ROWS - 1)
    for key, value in track_overrides(cfg.family, params).items():
        setattr(track, key, value)
    # Start on the central flat area. The challenge is x=5.5..6.5,
    # transition-out ends at x=7, leaving a safe exit before x=8.
    track.flat_length_m = 5.
    track.exit_flat_length_m = 1.
    meshes, _ = source_terrain_track(difficulty, track)
    angle = (row % 4) * math.pi / 2
    transform = trimesh.transformations.rotation_matrix(angle, (0., 0., 1.), point=(4., 4., 0.))
    for mesh in meshes:
        mesh.apply_transform(transform)
    return meshes, np.array((4., 4., 0.))


@configclass
class SourceBlockTrackCfg(SourceTerrainTrackCfg):
    function = source_block_terrain


def source_block_sub_terrains():
    result = {}
    params = curriculum_row_parameters(NUM_ROWS - 1)
    for column in range(NUM_COLUMNS):
        fields = column_fields(column)
        fields.update(track_overrides(fields["family"], params))
        fields.update(flat_length_m=5., exit_flat_length_m=1.)
        result[f"c{column:02d}"] = SourceBlockTrackCfg(proportion=1 / NUM_COLUMNS, **fields)
    return result


class SourceBlockTerrainImporter(SourceTerrainImporter):
    def _compute_env_origins_curriculum(self, num_envs, origins):
        super()._compute_env_origins_curriculum(num_envs, origins)
        self.terrain_levels = replica_rows(self.terrain_types)
        return origins[self.terrain_levels, self.terrain_types]

    def update_env_origins(self, env_ids, move_up, move_down):
        raise RuntimeError("L2 blocks have spatial replicas, no difficulty promotion or demotion")


@configclass
class SourceBlockTerrainImporterCfg(SourceTerrainImporterCfg):
    class_type: type = SourceBlockTerrainImporter
