"""Commands, fixed-L2 resets, bounds, and exact exposure counts for block training."""
from __future__ import annotations

import math
import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from rl_training.source_curriculum.block import (
    BlockStats, STAIR_EDGES_M, block_axis, block_completed, block_progress)
from rl_training.source_curriculum.logic import wrap_to_pi
from rl_training.source_curriculum.rows import NUM_ROWS, OMNI_COLUMNS
from rl_training.source_scan.tile_bounds import source_tile_bounds_terminated
from .commands import UniformThresholdVelocityCommand
from .source_curriculum import SourceCurriculumVelocityCommand, SourceCurriculumVelocityCommandCfg


class BlockState:
    def __init__(self, env):
        self.env = env
        self.episode_numbers = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.axes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.stats = BlockStats(env.device, rough_profile=env.cfg.source_block_rough_profile)
        self.omni = torch.isin(env.scene.terrain.terrain_types, torch.tensor(OMNI_COLUMNS, device=env.device))
        self.foot_ids = env.scene["robot"].find_bodies(".*_FOOT")[0]
        sensor = env.scene["contact_forces"]
        self.foot_sensor_ids = sensor.find_bodies(".*_FOOT")[0]
        self.nonfoot_sensor_ids = [i for i, n in enumerate(sensor.body_names) if not n.endswith("_FOOT")]
        self.support_progress = torch.full((env.num_envs, 4), -float("inf"), device=env.device)
        self.crossed_edges = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.last_step = 0

    def progress(self):
        robot = self.env.scene["robot"]
        return block_progress(robot.data.root_link_pos_w[:, :2],
                              robot.data.body_link_pos_w[:, self.foot_ids, :2],
                              self.env.scene.env_origins[:, :2], self.axes)

    def observe(self):
        env = self.env
        if self.last_step == env.common_step_counter:
            return
        self.last_step = env.common_step_counter
        forces = env.scene["contact_forces"].data.net_forces_w.norm(dim=-1)
        contact = forces[:, self.foot_sensor_ids] > 1.
        self.stats.observe(env.scene.terrain.terrain_types, self.axes, contact.any(-1),
                           (forces[:, self.nonfoot_sensor_ids] > 1.).any(-1), self.omni,
                           env.command_manager.get_command("base_velocity"))
        _, feet = self.progress()
        self.support_progress = torch.maximum(self.support_progress, torch.where(contact, feet, -float("inf")))
        edges = torch.tensor(STAIR_EDGES_M, device=env.device)
        crossed = (self.support_progress.amin(-1)[:, None] >= edges + .025).sum(-1)
        self.crossed_edges = torch.maximum(self.crossed_edges, crossed)


class SourceBlockVelocityCommand(SourceCurriculumVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env.source_block_state = BlockState(env)

    def _resample_command(self, env_ids):
        UniformThresholdVelocityCommand._resample_command(self, env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[self.tile_mask[ids]]
        if ids.numel():
            low, high = self.cfg.tile_lin_vel_x
            self.vel_command_b[ids, 0] = low + (high - low) * torch.rand(ids.numel(), device=self.device)
            self.vel_command_b[ids, 1] = 0.
            target = self._env.source_block_state.axes[ids] * (math.pi / 2)
            self.heading_target[ids] = target
            self.is_heading_env[ids] = True
            self.is_standing_env[ids] = False
            error = wrap_to_pi(target - self.robot.data.heading_w[ids])
            self.vel_command_b[ids, 2] = (self.cfg.heading_control_stiffness * error).clamp(*self.cfg.ranges.ang_vel_z)


@configclass
class SourceBlockVelocityCommandCfg(SourceCurriculumVelocityCommandCfg):
    class_type: type = SourceBlockVelocityCommand


def reset_source_block_root(env, env_ids, pose_range, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    roots = asset.data.default_root_state[env_ids].clone()
    ranges = torch.tensor([pose_range.get(k, (0., 0.)) for k in ("x", "y", "z", "roll", "pitch", "yaw")], device=env.device)
    sample = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    yaw = sample[:, 5] + env.source_block_state.axes[env_ids] * (math.pi / 2)
    orientations = math_utils.quat_mul(roots[:, 3:7], math_utils.quat_from_euler_xyz(sample[:, 3], sample[:, 4], yaw))
    # Flat/stair origins are the true centre height. Reference rough origins
    # are the central 2x2 m maximum, safely above every jittered starting foot.
    # Add the default 0.375 m to that support height; never assume world z=0.
    positions = roots[:, :3] + env.scene.env_origins[env_ids] + sample[:, :3]
    ranges = torch.tensor([velocity_range.get(k, (0., 0.)) for k in ("x", "y", "z", "roll", "pitch", "yaw")], device=env.device)
    velocities = roots[:, 7:13] + math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    asset.write_root_pose_to_sim(torch.cat((positions, orientations), -1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def source_block_bounds(env, asset_cfg, omni=False):
    state = env.source_block_state
    state.observe()
    robot, terrain = env.scene[asset_cfg.name], env.scene.terrain
    outside = source_tile_bounds_terminated(
        robot.data.body_link_pos_w[:, asset_cfg.body_ids], robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids],
        terrain.terrain_origins, terrain.terrain_levels, terrain.terrain_types,
        size=(8., 8.), flat_length_m=8., collision_radius_m=.27, numerical_margin_m=.001,
        control_period_s=env.step_dt)
    root = robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]
    predicted = root + robot.data.root_link_lin_vel_w[:, :2] * env.step_dt
    outside |= (root.abs() + .1 > 4.).any(-1) | (predicted.abs() + .1 > 4.).any(-1)
    return outside & (state.omni if omni else ~state.omni)


def source_block_completion(env):
    state = env.source_block_state
    return block_completed(*state.progress(), state.omni)


class SourceBlockResets(ManagerTermBase):
    """Record endings before reset; cycle replicas, keeping every block at L2."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.column_stats = env.source_block_state.stats

    def __call__(self, env, env_ids):
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        state, terrain, tm = env.source_block_state, env.scene.terrain, env.termination_manager
        ended = ids[(env.episode_length_buf[ids] > 0) & tm.dones[ids]]
        if ended.numel():
            self.column_stats.record_endings(
                terrain.terrain_types[ended], terrain.terrain_levels[ended], state.axes[ended],
                tm.get_term("source_curriculum_completion")[ended], tm.terminated[ended],
                tm.time_outs[ended], state.crossed_edges[ended],
                {name: tm.get_term(name)[ended] for name in tm.active_terms})
        initialized = state.episode_numbers[ids] >= 0
        terrain.terrain_levels[ids] = (terrain.terrain_levels[ids] + initialized.long()) % NUM_ROWS
        terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
        state.episode_numbers[ids] += 1
        state.axes[ids] = block_axis(terrain.terrain_levels[ids], terrain.terrain_types[ids], state.episode_numbers[ids], ids)
        state.support_progress[ids] = -float("inf")
        state.crossed_edges[ids] = 0
        self.column_stats.record_starts(terrain.terrain_types[ids], terrain.terrain_levels[ids], state.axes[ids])
        return self.column_stats.report()
