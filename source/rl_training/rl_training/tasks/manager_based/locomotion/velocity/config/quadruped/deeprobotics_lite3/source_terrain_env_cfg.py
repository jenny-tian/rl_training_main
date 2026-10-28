"""Lite3 Source Terrain scene used for bridge smoke tests and future source runs.

This is a geometry/integration configuration, not an approved training
distribution.  It creates the six registered families in columns with the
draft Pilot proportions while keeping every physical difficulty at its
explicit level-one candidate.  Baseline scanning and Source qualification
must decide the eventual training ranges and sampler.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG
from rl_training.online_adaptation.terrain_guards import (
    deterministic_rigid_body_material_params,
    deterministic_root_reset_params,
)
from rl_training.source_scan.parallel_terrain_allocation import (
    set_uniform_subterrain_proportions,
)
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.terrains import (
    SourceTerrainImporterCfg,
    SourceTerrainTrackCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.terrains.source_terrain_generator import SourceTerrainGenerator
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    TerminationsCfg,
)

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
_SOURCE_TILE_COLLISION_RADIUS_CANDIDATE_M = 0.27
_SOURCE_TILE_NUMERICAL_MARGIN_M = 0.001


_LEVEL_CANDIDATES = {
    "L1": {
        "rough_peak_valley_m": 0.02,
        "slope_angle_deg": 5.0,
        "exit_flat_length_m": 2.0,
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
        "exit_flat_length_m": 2.0,
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
        "exit_flat_length_m": 2.0,
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
        raise ValueError(
            f"source terrain level must be one of {tuple(_LEVEL_CANDIDATES)}"
        ) from error

    terrains: dict[str, SourceTerrainTrackCfg] = {
        "flat": SourceTerrainTrackCfg(
            proportion=0.10,
            family="flat",
        ),
        "multi_scale_rough": SourceTerrainTrackCfg(
            proportion=0.20,
            family="multi_scale_rough",
            rough_peak_valley_m=candidate["rough_peak_valley_m"],
            exit_flat_length_m=candidate["exit_flat_length_m"],
        ),
    }

    slope_directions = ("uphill", "downhill", "cross_slope_left", "cross_slope_right")
    for direction in slope_directions:
        terrains[f"slope_{direction}"] = SourceTerrainTrackCfg(
            proportion=SOURCE_TERRAIN_PROPORTIONS["slope"] / len(slope_directions),
            family="slope",
            slope_angle_deg=candidate["slope_angle_deg"],
            exit_flat_length_m=candidate["exit_flat_length_m"],
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
            exit_flat_length_m=candidate["exit_flat_length_m"],
        )

    friction_layouts = ("left_right_bands", "front_back_bands", "alternating_regions")
    for layout in friction_layouts:
        terrains[f"friction_{layout}"] = SourceTerrainTrackCfg(
            proportion=SOURCE_TERRAIN_PROPORTIONS["friction_variation"]
            / len(friction_layouts),
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
    source_reset_sampler_enabled: bool = False
    source_reset_sampler_seed: int | None = None
    source_parallel_sampler_enabled: bool = False
    source_parallel_sampler_num_envs: int | None = None
    source_parallel_sampler_seed: int | None = None
    source_shared_bank_rows: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.source_shared_bank_rows is not None and (
            not self.source_parallel_sampler_enabled
            or type(self.source_shared_bank_rows) is not int
            or type(self.source_parallel_sampler_num_envs) is not int
            or not 1 <= self.source_shared_bank_rows <= self.source_parallel_sampler_num_envs
        ):
            raise ValueError("source_shared_bank_rows requires the parallel sampler and 1..num_envs rows")
        if self.source_reset_sampler_enabled and self.source_parallel_sampler_enabled:
            raise ValueError(
                "single-env and parallel Source samplers are mutually exclusive"
            )
        if self.source_parallel_sampler_enabled:
            if (
                isinstance(self.source_parallel_sampler_num_envs, bool)
                or not isinstance(self.source_parallel_sampler_num_envs, int)
                or self.source_parallel_sampler_num_envs < 1
            ):
                raise ValueError(
                    "source_parallel_sampler_num_envs must be explicitly set to a positive integer "
                    "when source_parallel_sampler_enabled is True"
                )
            if (
                isinstance(self.source_parallel_sampler_seed, bool)
                or not isinstance(self.source_parallel_sampler_seed, int)
                or self.source_parallel_sampler_seed < 0
            ):
                raise ValueError(
                    "source_parallel_sampler_seed must be explicitly set to a nonnegative integer "
                    "when source_parallel_sampler_enabled is True"
                )

        sub_terrains = _source_sub_terrains(self.source_terrain_level)
        if self.source_parallel_sampler_enabled:
            set_uniform_subterrain_proportions(sub_terrains)
            terrain_num_rows = self.source_shared_bank_rows or self.source_parallel_sampler_num_envs
            terrain_num_cols = len(sub_terrains)
            scene_num_envs = self.source_parallel_sampler_num_envs
        else:
            terrain_num_rows = 1
            terrain_num_cols = _SOURCE_COLUMN_COUNT
            scene_num_envs = 1

        self.scene.terrain = SourceTerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                class_type=SourceTerrainGenerator,
                seed=10_001,
                curriculum=True,
                size=(12.0, 4.0),
                border_width=0.0,
                num_rows=terrain_num_rows,
                num_cols=terrain_num_cols,
                # Source Terrain uses difficulty only as a deterministic
                # child-seed source; it does not scale physical candidate
                # parameters. Parallel map rows are ownership slots, not L2+.
                difficulty_range=(0.0, 1.0),
                use_cache=False,
                sub_terrains=sub_terrains,
            ),
            max_init_terrain_level=0,
            parallel_reset_sampler_enabled=self.source_parallel_sampler_enabled,
            parallel_reset_sampler_num_envs=(
                scene_num_envs if self.source_parallel_sampler_enabled else None
            ),
            shared_bank_rows=self.source_shared_bank_rows,
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
        self.scene.num_envs = scene_num_envs
        self.scene.robot.init_state.pos = (0.0, 0.0, self.startup_base_height)
        if self.source_shared_bank_rows is not None:
            # Use the installed Cloner's inspectable USD collision groups.
            # The GPU env-ID replication path is a separate future candidate.
            self.scene.replicate_physics = False
            self.scene.clone_in_fabric = False
            self.scene.filter_collisions = True
            self.scene.robot.collision_group = 0
            self.scene.height_scanner.update_mesh_ids = True
            self.scene.height_scanner_base.update_mesh_ids = True
            from rl_training.source_scan.shared_terrain_bank import validate_shared_scene_cfg

            validate_shared_scene_cfg(self.scene)

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
        if self.source_reset_sampler_enabled:
            if (
                isinstance(self.source_reset_sampler_seed, bool)
                or not isinstance(self.source_reset_sampler_seed, int)
                or self.source_reset_sampler_seed < 0
            ):
                raise ValueError(
                    "source_reset_sampler_seed must be explicitly set to a nonnegative integer "
                    "when source_reset_sampler_enabled is True"
                )
            self.curriculum.source_terrain_reset_sampler = CurrTerm(
                func=mdp.source_terrain_reset_sampler,
                params={"sampler_seed": self.source_reset_sampler_seed},
            )
        elif self.source_parallel_sampler_enabled:
            self.curriculum.source_terrain_reset_sampler = None
            self.curriculum.source_terrain_parallel_reset_sampler = CurrTerm(
                func=mdp.source_terrain_parallel_reset_sampler,
                params={"sampler_seed": self.source_parallel_sampler_seed},
            )
        else:
            self.curriculum.source_terrain_reset_sampler = None
            self.curriculum.source_terrain_parallel_reset_sampler = None

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
        self.terminations.source_terrain_tile_bounds = DoneTerm(
            func=mdp.source_terrain_tile_bounds,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "size": tuple(float(value) for value in self.scene.terrain.terrain_generator.size),
                "flat_length_m": float(sub_terrains["flat"].flat_length_m),
                # Static audit of the pinned Lite3 PhysX USD found a maximum
                # per-body local collision envelope of 0.2644 m in the torso
                # box. This candidate is still subject to joint-sweep and
                # runtime qualification; it is not the whole-robot footprint.
                "collision_radius_m": _SOURCE_TILE_COLLISION_RADIUS_CANDIDATE_M,
                "numerical_margin_m": _SOURCE_TILE_NUMERICAL_MARGIN_M,
            },
            time_out=False,
        )
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
