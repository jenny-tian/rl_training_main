"""Source track whose parameters follow the curriculum row of each tile."""
from __future__ import annotations

import copy

from isaaclab.utils import configclass

from rl_training.source_curriculum.rows import (
    NUM_COLUMNS, NUM_ROWS, column_fields, curriculum_row_parameters, row_from_difficulty, track_overrides)

from .source_terrain_isaac import SourceTerrainTrackCfg, source_terrain_track


def source_curriculum_track(difficulty, cfg):
    row = row_from_difficulty(difficulty, cfg.curriculum_num_rows)
    params = curriculum_row_parameters(row, cfg.curriculum_num_rows)
    track = copy.deepcopy(cfg)
    for key, value in track_overrides(cfg.family, params).items():
        setattr(track, key, value)
    # The original difficulty still seeds the geometry, so every tile stays distinct.
    return source_terrain_track(difficulty, track)


@configclass
class SourceCurriculumTrackCfg(SourceTerrainTrackCfg):
    function = source_curriculum_track
    curriculum_num_rows: int = NUM_ROWS


def source_curriculum_sub_terrains(num_rows=NUM_ROWS):
    # Equal proportions make Isaac Lab's curriculum generator put sub-terrain k in column k.
    return {f"c{column:02d}": SourceCurriculumTrackCfg(proportion=1.0 / NUM_COLUMNS, curriculum_num_rows=num_rows,
                                                       **column_fields(column))
            for column in range(NUM_COLUMNS)}
