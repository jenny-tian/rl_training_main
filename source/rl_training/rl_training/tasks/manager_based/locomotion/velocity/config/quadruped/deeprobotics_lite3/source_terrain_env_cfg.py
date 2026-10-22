"""Lite3 Source Terrain scene used for bridge smoke tests and future source runs.

This is a geometry/integration configuration, not an approved training
distribution.  It creates the six registered families in columns with the
draft Pilot proportions while keeping every physical difficulty at its
explicit level-one candidate.  Baseline scanning and Source qualification
must decide the eventual training ranges and sampler.
"""

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns

from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG
from rl_training.online_adaptation.terrain_guards import (
    deterministic_rigid_body_material_params,
    deterministic_root_reset_params,
)
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.terrains import (
    SourceTerrainImporterCfg,
    SourceTerrainTrackCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import TerminationsCfg

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


SOURCE_TERRAIN_PROPORTIONS = {
    "flat": 0.10,
    "multi_scale_rough": 0.20,
    "slope": 0.20,
    "multi_step_stairs": 0.25,
    "slope_plus_rough": 0.15,
    "friction_variation": 0.10,
}

_SOURCE_COLUMN_COUNT = 120


_LEVEL_CANDIDATES = {
    "L1": {
        "rough_peak_valley_m": 0.02,
        "slope_angle_deg": 5.0,
        "step_count": 3,
        "step_height_m": 0.02,
        "tread_depth_m": 0.5,
        "slope_rough_peak_valley_m": 0.01,
        "slope_rough_angle_deg": 5.0,
        "friction_range": (0.7, 1.0),
    },
    "L2": {
        "rough_peak_valley_m": 0.03,
        "slope_angle_deg": 8.0,
        "step_count": 5,
        "step_height_m": 0.035,
        "tread_depth_m": 0.5,
        "slope_rough_peak_valley_m": 0.015,
        "slope_rough_angle_deg": 8.0,
        "friction_range": (0.5, 1.0),
    },
    "L3": {
        "rough_peak_valley_m": 0.04,
        "slope_angle_deg": 12.0,
        "step_count": 5,
        "step_height_m": 0.05,
        "tread_depth_m": 0.4,
        "slope_rough_peak_valley_m": 0.02,
        "slope_rough_angle_deg": 10.0,
        "friction_range": (0.3, 1.0),
    },
}


def _source_sub_terrains(level: str = "L1") -> dict[str, SourceTerrainTrackCfg]:
    """Build the draft 6-family mix at one candidate scan level."""

    try:
        candidate = _LEVEL_CANDIDATES[level]
    except KeyError as error:
        raise ValueError(f"source terrain level must be one of {tuple(_LEVEL_CANDIDATES)}") from error

    terrains: dict[str, SourceTerrainTrackCfg] = {
        "flat": SourceTerrainTrackCfg(
            proportion=0.10,
            family="flat",
        ),
        "multi_scale_rough": SourceTerrainTrackCfg(
            proportion=0.20,
            family="multi_scale_rough",
            rough_peak_valley_m=candidate["rough_peak_valley_m"],
        ),
    }

    slope_directions = ("uphill", "downhill", "cross_slope_left", "cross_slope_right")
    for direction in slope_directions:
        terrains[f"slope_{direction}"] = SourceTerrainTrackCfg(
            proportion=SOURCE_TERRAIN_PROPORTIONS["slope"] / len(slope_directions),
            family="slope",
            slope_angle_deg=candidate["slope_angle_deg"],
            direction=direction,
        )

    for direction in ("ascending", "descending"):
        terrains[f"stairs_{direction}"] = SourceTerrainTrackCfg(
            proportion=SOURCE_TERRAIN_PROPORTIONS["multi_step_stairs"] / 2,
            family="multi_step_stairs",
            step_count=candidate["step_count"],
            step_height_m=candidate["step_height_m"],
            tread_depth_m=candidate["tread_depth_m"],
            direction=direction,
        )

    slope_rough_weights = {
        "uphill": SOURCE_TERRAIN_PROPORTIONS["slope_plus_rough"] / 3,
        "downhill": SOURCE_TERRAIN_PROPORTIONS["slope_plus_rough"] / 3,
        "cross_slope_left": SOURCE_TERRAIN_PROPORTIONS["slope_plus_rough"] / 6,
        "cross_slope_right": SOURCE_TERRAIN_PROPORTIONS["slope_plus_rough"] / 6,
    }
    for direction, proportion in slope_rough_weights.items():
        terrains[f"slope_rough_{direction}"] = SourceTerrainTrackCfg(
            proportion=proportion,
            family="slope_plus_rough",
            slope_angle_deg=candidate["slope_rough_angle_deg"],
            direction=direction,
            rough_peak_valley_m=candidate["slope_rough_peak_valley_m"],
        )

    friction_layouts = ("left_right_bands", "front_back_bands", "alternating_regions")
    for layout in friction_layouts:
        terrains[f"friction_{layout}"] = SourceTerrainTrackCfg(
            proportion=SOURCE_TERRAIN_PROPORTIONS["friction_variation"] / len(friction_layouts),
            family="friction_variation",
            friction_range=candidate["friction_range"],
            friction_layout=layout,
            friction_region_count=4,
        )
    return terrains


@configclass
class DeeproboticsLite3SourceTerrainEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """Six-family Source terrain bridge configuration for Lite3."""

    startup_base_height: float = DEEPROBOTICS_LITE3_CFG.init_state.pos[2]
    source_terrain_level: str = "L1"

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.terrain = SourceTerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=10_001,
                curriculum=True,
                size=(10.0, 4.0),
                border_width=0.0,
                num_rows=1,
                num_cols=_SOURCE_COLUMN_COUNT,
                # For this one-row layout Isaac Lab uses the per-column
                # difficulty draw only as a deterministic child-seed source.
                # It does not scale the physical candidate parameters.
                difficulty_range=(0.0, 1.0),
                use_cache=False,
                sub_terrains=_source_sub_terrains(self.source_terrain_level),
            ),
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        # The source importer places ordinary terrain and spatial friction
        # regions under separate mesh prims. The stock RayCaster reads only
        # one mesh, so merge all meshes below the terrain root for height scans.
        source_height_meshes = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground",
                merge_prim_meshes=True,
                track_mesh_transforms=False,
            )
        ]
        self.scene.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            update_period=self.decimation * self.sim.dt,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.07, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=source_height_meshes,
        )
        self.scene.height_scanner_base = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            update_period=self.decimation * self.sim.dt,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
            debug_vis=False,
            mesh_prim_paths=source_height_meshes,
        )
        # This is a simulator-facing geometry config.  Training/evaluation
        # launchers must choose and record the parallel environment count.
        self.scene.num_envs = 1
        self.scene.robot.init_state.pos = (0.0, 0.0, self.startup_base_height)

        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.curriculum.gait_level = None
        self.curriculum.command_levels = None

        self.events.randomize_rigid_body_material.params.update(
            deterministic_rigid_body_material_params()
        )
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_reset_base.params = deterministic_root_reset_params()

        self.terminations.time_out = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.illegal_contact = TerminationsCfg().illegal_contact
        self.terminations.illegal_contact.func = mdp.startup_aware_illegal_contact
        self.terminations.illegal_contact.params["threshold"] = 5.0
        self.terminations.illegal_contact.params["consecutive_steps"] = 3
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            link for link in self.link_names if not link.endswith("_FOOT")
        ]
        self.terminations.bad_orientation_2.func = mdp.startup_aware_bad_orientation_2
        self.disable_zero_weight_rewards()


__all__ = [
    "DeeproboticsLite3SourceTerrainEnvCfg",
    "SOURCE_TERRAIN_PROPORTIONS",
    "_source_sub_terrains",
]
