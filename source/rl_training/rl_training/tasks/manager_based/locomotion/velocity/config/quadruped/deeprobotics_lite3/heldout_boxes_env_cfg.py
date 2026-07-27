from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from rl_training.online_adaptation.terrain_guards import deterministic_root_reset_params
from rl_training.tasks.manager_based.locomotion.velocity.terrains import HeldoutBoxesTrackCfg

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


@configclass
class DeeproboticsLite3HeldoutBoxesEnvCfg(DeeproboticsLite3RoughEnvCfg):
    heldout_start_x: float = -7.0
    track_boundary_x: float = -7.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=7,
            curriculum=False,
            size=(20.0, 4.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            use_cache=False,
            sub_terrains={
                "heldout_boxes": HeldoutBoxesTrackCfg(
                    proportion=1.0,
                    flat_length=3.0,
                    lane_width=2.0,
                    grid_width=0.4,
                    grid_height_range=(0.04, 0.12),
                    seed=7,
                )
            },
        )
        self.scene.terrain.max_init_terrain_level = 0

        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.illegal_contact = None

        self.curriculum.terrain_levels = None
        self.curriculum.gait_level = None
        self.curriculum.command_levels = None

        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_reset_joints = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_reset_base.params = deterministic_root_reset_params()
        self.events.randomize_push_robot = None

        self.disable_zero_weight_rewards()
