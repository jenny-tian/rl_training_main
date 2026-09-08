from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG
from rl_training.online_adaptation.terrain_guards import (
    deterministic_rigid_body_material_params,
    deterministic_root_reset_params,
)
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.terrains import (
    ContinuousRoughTrackCfg,
    RockyRoadTrackCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import TerminationsCfg

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


@configclass
class DeeproboticsLite3ContinuousRoughEnvCfg(DeeproboticsLite3RoughEnvCfg):
    startup_base_height: float = DEEPROBOTICS_LITE3_CFG.init_state.pos[2]

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=101, curriculum=False, size=(10.0, 4.0), border_width=0.0,
            num_rows=1, num_cols=1, use_cache=False,
            sub_terrains={"continuous_rough": ContinuousRoughTrackCfg(
                proportion=1.0, flat_length=3.0, transition_length=0.5,
                lane_width=2.0, rough_amplitude=0.02,
                correlation_length=0.4, seed=101,
            )},
        )
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.robot.init_state.pos = (0.0, 0.0, self.startup_base_height)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
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
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            link for link in self.link_names if not link.endswith("_FOOT")
        ]
        self.terminations.bad_orientation_2.func = mdp.startup_aware_bad_orientation_2
        self.disable_zero_weight_rewards()


@configclass
class DeeproboticsLite3GrassEnvCfg(DeeproboticsLite3ContinuousRoughEnvCfg):
    """Grass-like variant: lower friction and very small height variation."""

    def __post_init__(self):
        super().__post_init__()
        grass = self.scene.terrain.terrain_generator.sub_terrains["continuous_rough"]
        grass.rough_amplitude = 0.01
        grass.correlation_length = 0.5
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=0.55, dynamic_friction=0.45, restitution=0.05,
        )


@configclass
class DeeproboticsLite3RockyRoadEnvCfg(DeeproboticsLite3ContinuousRoughEnvCfg):
    """Held-out irregular rock and pothole road for baseline stress tests."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=101, curriculum=False, size=(10.0, 4.0), border_width=0.0,
            num_rows=1, num_cols=1, use_cache=False,
            sub_terrains={"rocky_road": RockyRoadTrackCfg(
                proportion=1.0, flat_length=3.0, transition_length=0.5,
                # Cover the complete track width.  Only the longitudinal
                # approach remains flat; there are no lateral bypass strips.
                lane_width=4.0, rough_amplitude=0.10, grid_spacing=0.10,
                # Dense enough that the policy cannot simply route through a
                # long flat gap between isolated rock clusters.
                rock_count=84, rock_fraction=0.52,
                radius_x=(0.16, 0.38), radius_y=(0.13, 0.32),
                height_range=(0.05, 0.10), seed=101,
            )},
        )
        self.scene.terrain.max_init_terrain_level = 0
