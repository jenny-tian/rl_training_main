"""Source curriculum terms: tile commands, masked terminations, safety and level updates."""
from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

from rl_training.source_curriculum.logic import (
    FamilyStats, OMNI_FAMILY_ID, apply_tile_commands, column_family_ids, curriculum_moves,
    omni_levels, outside_bounds, safety_unsafe)
from rl_training.source_curriculum.rows import OMNI_COLUMNS

from .commands import UniformThresholdVelocityCommand, UniformThresholdVelocityCommandCfg
from .terminations import source_terrain_tile_bounds

__all__ = ["SourceCurriculumVelocityCommand", "SourceCurriculumVelocityCommandCfg",
           "source_curriculum_tile_bounds", "source_curriculum_completion",
           "source_curriculum_omni_truncation", "SourceCurriculumSafety", "SourceCurriculumLevels"]


def _omni_mask(env, omni_columns):
    types = env.scene.terrain.terrain_types
    columns = torch.as_tensor(tuple(omni_columns), device=types.device, dtype=types.dtype)
    return torch.isin(types, columns)


def _env_ids(env_ids, num_envs, device):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


class SourceCurriculumVelocityCommand(UniformThresholdVelocityCommand):
    """Omni envs keep the Flat command mix; tile envs walk straight along +x."""

    def __init__(self, cfg, env):
        if cfg.command_acceleration_limits is not None:
            # The slew path keeps the sample in target_vel_command_b and restores
            # vel_command_b, which would silently undo the tile override.
            raise ValueError("SourceCurriculumVelocityCommand requires command_acceleration_limits=None")
        super().__init__(cfg, env)
        self.tile_mask = ~_omni_mask(env, cfg.omni_columns)
        self.commanded_distance = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        self.commanded_distance[slice(None) if env_ids is None else env_ids] = 0.0
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = _env_ids(env_ids, self.num_envs, self.device)
        tile = ids[self.tile_mask[ids]]
        apply_tile_commands(self.vel_command_b, self.heading_target, self.is_heading_env, self.is_standing_env,
                            tile, heading=self.robot.data.heading_w[tile], lin_vel_x=self.cfg.tile_lin_vel_x,
                            stiffness=self.cfg.heading_control_stiffness, ang_range=self.cfg.ranges.ang_vel_z)

    def _update_metrics(self):
        super()._update_metrics()
        self.commanded_distance += self.vel_command_b[:, 0] * self._env.step_dt


@configclass
class SourceCurriculumVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    class_type: type = SourceCurriculumVelocityCommand
    omni_columns: tuple = OMNI_COLUMNS
    tile_lin_vel_x: tuple = (0.25, 0.5)


def source_curriculum_tile_bounds(env, asset_cfg, size, flat_length_m, collision_radius_m, numerical_margin_m,
                                  scanner_footprint_margin_m, omni_columns):
    base = source_terrain_tile_bounds(env, asset_cfg=asset_cfg, size=size, flat_length_m=flat_length_m,
                                      collision_radius_m=collision_radius_m, numerical_margin_m=numerical_margin_m,
                                      scanner_footprint_margin_m=scanner_footprint_margin_m)
    return base & ~_omni_mask(env, omni_columns)


def source_curriculum_completion(env, distance_m, omni_columns):
    robot = env.scene["robot"]
    progress = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return (progress >= distance_m) & ~_omni_mask(env, omni_columns)


def source_curriculum_omni_truncation(env, bounds, omni_columns):
    xy = env.scene["robot"].data.root_pos_w[:, :2]
    return outside_bounds(xy, bounds) & _omni_mask(env, omni_columns)


class SourceCurriculumSafety(ManagerTermBase):
    """Evaluator unsafe formula as a termination (R2, R9, R16)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._adapter = None
        self._limits = None

    def reset(self, env_ids=None):
        self.contact_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env):
        if self._adapter is None:
            from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
            from rl_training.online_adaptation.safety import SafetyLimits
            self._adapter = Lite3IsaacLabAdapter(env, safety_height_source="base_scan")
            self._limits = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4, device=env.device))
        timed_out = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
        state = self._adapter.safety_state(timed_out)
        unsafe, self.contact_steps = safety_unsafe(state.projected_gravity_xy, state.base_height, state.joint_vel,
                                                   state.nonfoot_contact, state.control_timed_out,
                                                   self.contact_steps, self._limits)
        forces = self._adapter.contact_sensor.data.net_forces_w
        return unsafe | ~torch.isfinite(forces).flatten(1).all(-1)


class SourceCurriculumLevels(ManagerTermBase):
    """Moves tile envs by outcome; re-draws omni rows 1..num_rows-2 at every reset (R3, R5, R6)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        omni_columns = cfg.params["omni_columns"]
        self.family_by_column = column_family_ids(omni_columns, device=env.device)
        types = env.scene.terrain.terrain_types
        if int(types.max()) >= self.family_by_column.numel():
            raise ValueError(f"terrain has column {int(types.max())} but the Source curriculum defines only "
                             f"{self.family_by_column.numel()} columns")
        self.env_family_ids = self.family_by_column[types.long()]
        self.omni = self.env_family_ids == OMNI_FAMILY_ID
        self.stats = FamilyStats(env.device)

    def __call__(self, env, env_ids, omni_columns):
        terrain = env.scene.terrain
        ids = _env_ids(env_ids, env.num_envs, env.device)
        tile = ids[~self.omni[ids]]
        omni = ids[self.omni[ids]]
        if tile.numel():
            tm = env.termination_manager
            completed = tm.get_term("source_curriculum_completion")[tile]
            failed = tm.terminated[tile]
            timed_out = tm.get_term("time_out")[tile]
            progress = env.scene["robot"].data.root_pos_w[tile, 0] - terrain.env_origins[tile, 0]
            distance = env.command_manager.get_term("base_velocity").commanded_distance[tile]
            up, down = curriculum_moves(completed, failed, timed_out, progress, distance)
            self.stats.update(self.env_family_ids[tile], completed, failed, timed_out)
            terrain.update_env_origins(tile, up, down)
        if omni.numel():
            levels = omni_levels(omni.numel(), num_rows=terrain.max_terrain_level, device=env.device)
            terrain.terrain_levels[omni] = levels
            terrain.env_origins[omni] = terrain.terrain_origins[levels, terrain.terrain_types[omni]]
        return self.stats.report(terrain.terrain_levels, self.env_family_ids, top_row=terrain.max_terrain_level - 1)
