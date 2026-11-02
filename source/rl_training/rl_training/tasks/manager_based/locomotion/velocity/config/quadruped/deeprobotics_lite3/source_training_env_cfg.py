"""Opt-in L1 training configuration; historical scan configuration is separate."""
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from rl_training.source_scan.training_lifecycle import SourceRunSeeds
from rl_training.source_scan.training_env import (
    source_training_completion, source_training_safety, source_training_timeout,
)
from .source_terrain_env_cfg import DeeproboticsLite3SourceTerrainEnvCfg


@configclass
class DeeproboticsLite3SourceTrainingEnvCfg(DeeproboticsLite3SourceTerrainEnvCfg):
    source_training_run: int = 1

    def __post_init__(self):
        seeds = SourceRunSeeds(self.source_training_run)
        if not self.source_parallel_sampler_enabled or self.source_terrain_level != "L1":
            raise ValueError("Source training requires explicit parallel sampling and L1 only")
        self.source_parallel_sampler_seed = seeds.sampler
        super().__post_init__()
        self.seed = seeds.ppo
        self.scene.terrain.terrain_generator.seed = seeds.geometry
        self.episode_length_s = 40.0
        self.is_finite_horizon = False
        self.observations.critic.enable_corruption = False
        command = self.commands.base_velocity
        command.rel_zero_vel_envs = 0.0
        command.rel_only_lin_y_envs = 0.0
        command.rel_only_lin_x_envs = 1.0
        command.rel_only_ang_z_envs = 0.0
        self.terminations.source_training_safety = DoneTerm(func=source_training_safety)
        self.terminations.source_training_completion = DoneTerm(func=source_training_completion)
        self.terminations.source_training_timeout = DoneTerm(func=source_training_timeout, time_out=True)
