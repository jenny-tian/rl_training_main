"""EA-LPL scene: frozen base policy, optional heading hold, explicit columns and reset table."""
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rl_training.ea_lpl.reset_table import DEFAULT_RESET_RANGES, reset_root_state_from_table
from .source_adaptation_env_cfg import DeeproboticsLite3SourceAdaptationEnvCfg


@configclass
class DeeproboticsLite3EaLplEnvCfg(DeeproboticsLite3SourceAdaptationEnvCfg):
    source_allowed_levels: tuple = ("L1", "L2")
    ea_heading_hold: bool = True
    ea_heading_stiffness: float = 0.5
    ea_max_yaw_rate: float = 0.8
    ea_columns: tuple | None = None
    ea_reset_table_seed: int | None = None
    ea_reset_namespace: int = 0
    ea_reset_slots: tuple | None = None

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.rel_standing_envs = 0.0
        command.rel_zero_vel_envs = 0.0
        command.rel_only_lin_y_envs = 0.0
        command.rel_only_ang_z_envs = 0.0
        command.ranges.lin_vel_x = (0.3, 0.3)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.heading = (0.0, 0.0)
        if self.ea_heading_hold:
            command.heading_command = True
            command.heading_control_stiffness = self.ea_heading_stiffness
            command.rel_heading_envs = 1.0
            command.rel_only_lin_x_envs = 0.0
            command.ranges.ang_vel_z = (-self.ea_max_yaw_rate, self.ea_max_yaw_rate)
        else:
            command.heading_command = False
            command.rel_heading_envs = 0.0
            command.rel_only_lin_x_envs = 1.0
            command.ranges.ang_vel_z = (0.0, 0.0)
        if self.ea_columns is not None:
            columns = tuple(int(c) for c in self.ea_columns)
            if len(columns) != self.source_parallel_sampler_num_envs or any(not 0 <= c < 15 for c in columns):
                raise ValueError("ea_columns must list one registered column per environment")
            self.curriculum.source_terrain_parallel_reset_sampler.params["fixed_columns"] = columns
        if self.ea_reset_table_seed is not None:
            slots = None if self.ea_reset_slots is None else tuple(int(s) for s in self.ea_reset_slots)
            if slots is not None and len(slots) != self.source_parallel_sampler_num_envs:
                raise ValueError("ea_reset_slots must list one slot per environment")
            if not 0 <= int(self.ea_reset_namespace) < 9:
                raise ValueError("namespace 9 is reserved for sealed runs")
            self.events.randomize_reset_base.func = reset_root_state_from_table
            self.events.randomize_reset_base.params = {
                "asset_cfg": SceneEntityCfg("robot"),
                "table_seed": int(self.ea_reset_table_seed),
                "namespace": int(self.ea_reset_namespace),
                "slots": slots,
                "ranges": dict(DEFAULT_RESET_RANGES),
            }
