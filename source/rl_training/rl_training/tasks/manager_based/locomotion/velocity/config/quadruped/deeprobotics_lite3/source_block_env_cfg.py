"""Independent six-family block task. Every spatial replica uses fixed L2 geometry."""
import copy

from isaaclab.managers import CurriculumTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from rl_training.source_curriculum.block import BLOCK_SEED, BLOCK_SIZE, COMPLETION_M
from rl_training.tasks.manager_based.locomotion.velocity.mdp.source_block import (
    SourceBlockResets, SourceBlockVelocityCommandCfg, reset_source_block_root,
    source_block_bounds, source_block_completion)
from rl_training.tasks.manager_based.locomotion.velocity.terrains.source_block_terrain import (
    SourceBlockTerrainImporterCfg, source_block_sub_terrains)
from .source_curriculum_env_cfg import DeeproboticsLite3SourceCurriculumTargetedEnvCfg


@configclass
class DeeproboticsLite3SourceBlockL2EnvCfg(DeeproboticsLite3SourceCurriculumTargetedEnvCfg):
    source_fixed_difficulty: str = "L2"

    def __post_init__(self):
        super().__post_init__()
        old = self.scene.terrain
        terrain = SourceBlockTerrainImporterCfg()
        for name in old.__dataclass_fields__:
            if name != "class_type":
                setattr(terrain, name, copy.deepcopy(getattr(old, name)))
        terrain.terrain_generator.seed = BLOCK_SEED
        terrain.terrain_generator.size = BLOCK_SIZE
        terrain.terrain_generator.sub_terrains = source_block_sub_terrains()
        terrain.max_init_terrain_level = 9  # spatial spread only; geometry is L2 in every row
        self.scene.terrain = terrain
        old = self.commands.base_velocity
        command = SourceBlockVelocityCommandCfg()
        for name in old.__dataclass_fields__:
            if name != "class_type":
                setattr(command, name, copy.deepcopy(getattr(old, name)))
        self.commands.base_velocity = command
        self.source_completion_distance_m = COMPLETION_M
        self.curriculum.source_curriculum_levels = CurriculumTermCfg(func=SourceBlockResets)
        self.events.randomize_reset_base.func = reset_source_block_root
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-.1, .1), "y": (-.1, .1), "z": (0., 0.),
            "roll": (0., 0.), "pitch": (0., 0.), "yaw": (0., 0.)}
        self.terminations.source_curriculum_tile_bounds = TerminationTermCfg(
            func=source_block_bounds, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "omni": False})
        self.terminations.source_curriculum_omni_truncation = TerminationTermCfg(
            func=source_block_bounds, time_out=True,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "omni": True})
        self.terminations.source_curriculum_completion = TerminationTermCfg(
            func=source_block_completion, time_out=True)
