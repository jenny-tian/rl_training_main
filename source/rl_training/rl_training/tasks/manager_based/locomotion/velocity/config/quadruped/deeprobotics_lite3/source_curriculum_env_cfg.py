"""Lite3 Source curriculum task: 20 Source columns, rows from easier than L1 up to L2.

Tile columns walk straight along +x and move rows by outcome; the four omni
columns keep the Flat command mix on flat ground.
"""
from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG
from rl_training.lite3_stability_contract import LITE3_MAX_PROJECTED_GRAVITY_XY
from rl_training.source_curriculum.logic import omni_truncation_bounds
from rl_training.source_curriculum.rows import NUM_COLUMNS, NUM_ROWS, OMNI_COLUMNS
from rl_training.tasks.manager_based.locomotion.velocity.mdp import (
    SourceCurriculumVelocityCommandCfg,
    UniformThresholdVelocityCommandCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.terrains import (
    SourceTerrainImporterCfg,
    source_curriculum_sub_terrains,
)
from rl_training.tasks.manager_based.locomotion.velocity.terrains.source_terrain_generator import SourceTerrainGenerator

from .flat_env_cfg import DeeproboticsLite3FlatEnvCfg


def spawn_source_curriculum_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Keep the pre-reset GridCloner grid above terrain during PhysX warmup.

    Terrain origins are applied by the first environment reset. Until then,
    ordinary grid positions can put the robot inside an elevated slope. Only
    the USD spawn translation changes; default_root_state still comes from
    ArticulationCfg.init_state and resets use the registered 0.375 m height.
    """
    x, y, _z = translation if translation is not None else (0.0, 0.0, 0.0)
    return sim_utils.spawn_from_usd(prim_path, cfg, translation=(x, y, 3.0), orientation=orientation, **kwargs)


@configclass
class DeeproboticsLite3SourceCurriculumEnvCfg(DeeproboticsLite3FlatEnvCfg):
    # Record only: the Flat cfg already sets reward_mdp.gait_level = 1.0 at module level.
    reward_gait_level: float = 1.0
    source_omni_columns: tuple = OMNI_COLUMNS
    source_tile_lin_vel_x: tuple = (0.25, 0.5)
    source_completion_distance_m: float = 9.25

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0
        # PhysX requested 229478192 bytes with 2048 Source envs. Its default
        # 64 MiB stack drops contacts; reserve 256 MiB for this terrain task.
        self.sim.physx.gpu_collision_stack_size = 2**28
        # GPU env-ID replication exhausted 24 GiB at 2048 envs on this global
        # map. Authored USD collision groups used 5.1 GiB with the same scene.
        self.scene.replicate_physics = False
        self.scene.clone_in_fabric = False
        self.scene.filter_collisions = True
        self.scene.terrain = SourceTerrainImporterCfg(
            prim_path="/World/ground", terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                class_type=SourceTerrainGenerator, seed=20_261_109, curriculum=True, size=(12.0, 4.0),
                border_width=0.0, num_rows=NUM_ROWS, num_cols=NUM_COLUMNS, difficulty_range=(0.0, 1.0),
                use_cache=False, sub_terrains=source_curriculum_sub_terrains()),
            max_init_terrain_level=4, parallel_reset_sampler_enabled=False, parallel_reset_sampler_num_envs=None,
            shared_bank_rows=None, collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            debug_vis=False)
        self.scene.height_scanner = None
        self.scene.height_scanner_base = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name, update_period=self.sim.dt,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)), debug_vis=False,
            mesh_prim_paths=[MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground", merge_prim_meshes=True, track_mesh_transforms=False)])
        self.scene.robot.init_state.pos = (0.0, 0.0, DEEPROBOTICS_LITE3_CFG.init_state.pos[2])
        if self.scene.robot.spawn.func is not sim_utils.spawn_from_usd:
            raise ValueError("Source curriculum startup expects the standard USD robot spawner")
        self.scene.robot.spawn.func = spawn_source_curriculum_robot
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")

        old = self.commands.base_velocity
        if type(old) is not UniformThresholdVelocityCommandCfg:
            raise ValueError(f"expected UniformThresholdVelocityCommandCfg, got {type(old).__name__}")
        new = SourceCurriculumVelocityCommandCfg()
        for name in old.__dataclass_fields__:
            if name != "class_type":
                setattr(new, name, copy.deepcopy(getattr(old, name)))
        new.omni_columns = self.source_omni_columns
        new.tile_lin_vel_x = self.source_tile_lin_vel_x
        if not new.heading_command or tuple(new.ranges.ang_vel_z) != (-0.8, 0.8):
            raise ValueError("Flat command must keep the heading loop and ang_vel_z (-0.8, 0.8)")
        self.commands.base_velocity = new

        omni = self.source_omni_columns
        t = self.terminations
        t.terrain_out_of_bounds = None
        if hasattr(t, "source_terrain_tile_bounds"):
            t.source_terrain_tile_bounds = None
        nonfoot = [name for name in self.link_names if not name.endswith("_FOOT")]
        t.illegal_contact = DoneTerm(func=mdp.startup_aware_illegal_contact, params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=nonfoot), "threshold": 5.0,
            "consecutive_steps": 3})
        t.bad_orientation_2.func = mdp.startup_aware_bad_orientation_2
        t.bad_orientation_2.params["max_xy_norm"] = LITE3_MAX_PROJECTED_GRAVITY_XY
        t.source_curriculum_safety = DoneTerm(func=mdp.SourceCurriculumSafety, time_out=False)
        t.source_curriculum_tile_bounds = DoneTerm(func=mdp.source_curriculum_tile_bounds, time_out=False, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"), "size": (12.0, 4.0), "flat_length_m": 3.0,
            "collision_radius_m": 0.27, "numerical_margin_m": 0.001, "scanner_footprint_margin_m": 0.1,
            "omni_columns": omni})
        t.source_curriculum_completion = DoneTerm(func=mdp.source_curriculum_completion, time_out=True, params={
            "distance_m": self.source_completion_distance_m, "omni_columns": omni})
        t.source_curriculum_omni_truncation = DoneTerm(func=mdp.source_curriculum_omni_truncation, time_out=True,
                                                       params={"bounds": omni_truncation_bounds(), "omni_columns": omni})

        for name in list(vars(self.curriculum)):
            if not name.startswith("_"):
                setattr(self.curriculum, name, None)
        self.curriculum.source_curriculum_levels = CurrTerm(func=mdp.SourceCurriculumLevels,
                                                            params={"omni_columns": omni})

        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.2, 0.2), "y": (-0.1, 0.1), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0),
            "yaw": (-0.2, 0.2)}
        material = self.events.randomize_rigid_body_material
        material.params["static_friction_range"] = (0.5, 1.1)
        material.params["dynamic_friction_range"] = (0.4, 1.0)
        material.params["restitution_range"] = (0.0, 0.0)
        material.params["make_consistent"] = True
        self.disable_zero_weight_rewards()


__all__ = ["DeeproboticsLite3SourceCurriculumEnvCfg"]
