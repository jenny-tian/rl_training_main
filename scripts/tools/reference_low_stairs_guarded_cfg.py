"""只在已批准的能力保持试验中启用持续转向覆盖。"""
import torch

from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import UniformThresholdVelocityCommand
from reference_low_stairs_preservation import split_heading_turn_samples


class ContinuousTurnCommand(UniformThresholdVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.turn_generator = torch.Generator(device=self.device).manual_seed(env.cfg.seed + 137)
        self.changed_left = self.changed_right = 0

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        choice = torch.rand(len(ids), generator=self.turn_generator, device=self.device)
        magnitude = torch.rand(len(ids), generator=self.turn_generator, device=self.device)
        changed = split_heading_turn_samples(self, ids, choice, magnitude)
        self.changed_left += int((self.vel_command_b[changed, 2] > 0).sum())
        self.changed_right += int((self.vel_command_b[changed, 2] < 0).sum())


def configure(reference):
    from reference_six_terrain_cfg import replace_terrain
    from reference_low_stairs_cfg import apply_terrain, apply_rewards
    from reference_low_stairs_clearance_cfg import apply_clearance
    from reference_low_stairs_repair import apply_repair
    cfg, terrain = apply_terrain(replace_terrain(reference))
    cfg, rewards = apply_rewards(cfg)
    cfg, clearance = apply_clearance(cfg)
    cfg, yaw = apply_repair(cfg, 'yaw_airtime')
    assert cfg.commands.base_velocity.class_type is UniformThresholdVelocityCommand
    cfg.commands.base_velocity.class_type = ContinuousTurnCommand
    return cfg, dict(terrain=terrain, rewards=rewards, clearance=clearance, yaw=yaw,
                    commands=dict(original_pure_heading_retained=.5, fixed_left=.25, fixed_right=.25,
                                  magnitude_rad_s=[.2, .8], standing_unchanged=True))
