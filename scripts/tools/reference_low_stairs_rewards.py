"""Training-only stair shaping, with pure tensor rules for behavioral checks.

The actor receives no terrain information. Local surface heights are privileged
reward inputs; this module is not an online-learning or deployment reward.
"""
from __future__ import annotations

import torch

FOOT_NAMES = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
FOOT_RADIUS = .022  # Sphere collision radius in the unchanged Lite3 asset.
MARGIN = .025


def clearance_cost(bottom, obstacle_top, force, foot_velocity, direction, enabled):
    """Penalize insufficient clearance only for feet swinging forward.

    No positive score for holding a foot high; height beyond the margin is free.
    Use total force, not just vertical force, to exclude a foot pressing a wall.
    """
    airborne = force.norm(dim=-1) < 1.
    forward = (foot_velocity[..., :2] * direction[:, None]).sum(-1).clamp(0., .5) / .5
    deficit = ((obstacle_top + MARGIN - bottom).clamp_min(0.) / .05).clamp_max(1.)
    return (deficit.square() * forward * airborne * enabled).mean(-1)


def progress_reward(velocity, direction, command_speed, support, upright, enabled):
    progress = (velocity[:, :2] * direction).sum(-1) / command_speed.clamp_min(.1)
    return progress.clamp(-1., 1.) * enabled * upright * (support.sum(-1) >= 2)


class LandingState:
    """Credit higher, supported landings once per foot and elevation per episode."""
    def __init__(self, count, device):
        shape = (count, 4)
        self.initialized = torch.zeros(shape, dtype=torch.bool, device=device)
        self.was_supported = torch.zeros_like(self.initialized)
        self.air_time = torch.zeros(shape, device=device)
        self.support_z = torch.zeros(shape, device=device)
        self.highest_credited_z = torch.zeros(shape, device=device)
        self.peak_bottom = torch.zeros(shape, device=device)
        self.support_xy = torch.zeros((*shape, 2), device=device)
        self.support_root_xy = torch.zeros_like(self.support_xy)
        self.direction = torch.zeros_like(self.support_xy)
        self.pending = torch.zeros_like(self.initialized)
        self.pending_z = torch.zeros(shape, device=device)
        self.dwell = torch.zeros(shape, device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.initialized[ids] = False
        self.was_supported[ids] = False
        self.air_time[ids] = 0.
        self.pending[ids] = False
        self.dwell[ids] = 0.

    def update(self, *, feet, bottom, ground, support, root_xy, direction, enabled, upright, dt):
        # Ground and positions share world coordinates. Reset seeds baselines from
        # actual support, so a new episode on a high platform earns no landing.
        first = support & ~self.initialized
        self.highest_credited_z = torch.where(first, ground, self.highest_credited_z)
        self.initialized |= first
        self.peak_bottom = torch.where(self.was_supported, bottom, torch.maximum(self.peak_bottom, bottom))
        airborne = ~support
        self.air_time = torch.where(airborne, self.air_time + dt, self.air_time)
        displacement = ((feet[..., :2] - self.support_xy) * self.direction).sum(-1)
        body_progress = ((root_xy[:, None] - self.support_root_xy) * self.direction).sum(-1)
        touchdown = support & ~self.was_supported & (self.air_time >= .04 - 1e-6) & ~first
        candidate = (touchdown & self.initialized & enabled[:, None] & upright[:, None]
                  & (support.sum(-1) >= 2)[:, None]
                  & (ground > self.support_z + .008)
                  & (ground > self.highest_credited_z + .008)
                  & (displacement >= .04) & (body_progress >= .01)
                  & (self.peak_bottom >= ground + .015)
                  & ((bottom - ground).abs() <= .025))
        self.pending_z = torch.where(candidate, ground, self.pending_z)
        self.pending = ((self.pending | candidate) & support & enabled[:, None] & upright[:, None]
                        & (support.sum(-1) >= 2)[:, None] & ((ground-self.pending_z).abs()<.005))
        self.dwell = torch.where(self.pending, self.dwell + dt, 0.)
        landed = self.pending & (self.dwell >= .06 - 1e-6)
        self.pending &= ~landed
        self.dwell = torch.where(landed, 0., self.dwell)
        self.highest_credited_z = torch.where(landed, ground, self.highest_credited_z)
        self.support_z = torch.where(support, ground, self.support_z)
        self.support_xy = torch.where(support[..., None], feet[..., :2], self.support_xy)
        self.support_root_xy = torch.where(support[..., None], root_xy[:, None], self.support_root_xy)
        self.direction = torch.where(support[..., None], direction[:, None], self.direction)
        self.air_time = torch.where(support, 0., self.air_time)
        self.was_supported = support.clone()
        # Manager multiplies rates by dt. One successful foot earns 1/4 unit,
        # independently of control frequency; four feet earn one unit per level.
        return landed.float().mean(-1) / dt


def sample(env):
    """One shared set of terrain-only rays per control step, all four feet."""
    tick = env.common_step_counter
    cached = getattr(env, '_low_stairs_sample', None)
    if cached is not None and cached['tick'] == tick:
        return cached
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh
    from reference_six_terrain_cfg import column_names, FAMILIES

    robot = env.scene['robot']; sensor = env.scene['contact_forces']
    if not hasattr(env, '_low_stairs_bindings'):
        foot_ids, foot_names = robot.find_bodies(FOOT_NAMES, preserve_order=True)
        sensor_ids, sensor_names = sensor.find_bodies(FOOT_NAMES, preserve_order=True)
        assert foot_names == sensor_names == FOOT_NAMES
        columns = torch.tensor([FAMILIES[name] == 'stairs' for name in column_names(env.cfg)], device=env.device)
        env._low_stairs_bindings = (foot_ids, sensor_ids, columns)
    foot_ids, sensor_ids, columns = env._low_stairs_bindings
    data = robot.data
    feet = data.body_pos_w[:, foot_ids]
    force = sensor.data.net_forces_w[:, sensor_ids]
    command = env.command_manager.get_command('base_velocity')[:, :2]
    speed = command.norm(dim=-1)
    heading = data.heading_w
    direction = torch.stack((command[:, 0]*heading.cos() - command[:, 1]*heading.sin(),
                             command[:, 0]*heading.sin() + command[:, 1]*heading.cos()), -1)
    direction = direction / speed.clamp_min(.1)[:, None]
    stairs = columns[env.scene.terrain.terrain_types]
    enabled = stairs & (speed > .1)
    offsets = feet.new_tensor([0., .06, .12, .18, .24])
    starts = feet[:, :, None].expand(-1, -1, len(offsets), -1).clone()
    starts[..., :2] += direction[:, None, None] * offsets[None, None, :, None]
    starts[..., 2] += 1.
    ray_direction = torch.zeros_like(starts); ray_direction[..., 2] = -1.
    mesh = RayCaster.meshes[env.cfg.scene.terrain.prim_path]
    hits = raycast_mesh(starts.reshape(-1, 3), ray_direction.reshape(-1, 3), mesh)[0].reshape_as(starts)
    if not torch.isfinite(hits).all():
        raise FloatingPointError('nonfinite low-stair reward rays')
    ground = hits[:, :, 0, 2]
    obstacle = hits[..., 2].amax(-1)
    bottom = feet[..., 2] - FOOT_RADIUS
    upright = -data.projected_gravity_b[:, 2] > .7
    support = (force[..., 2] > 1.) & ((bottom - ground).abs() <= .025)
    result = dict(tick=tick, feet=feet.clone(), bottom=bottom, ground=ground, obstacle=obstacle,
        force=force, foot_velocity=data.body_lin_vel_w[:, foot_ids], direction=direction,
        speed=speed, enabled=enabled, upright=upright, support=support,
        root_xy=data.root_pos_w[:, :2], velocity=data.root_lin_vel_w,
        upcoming=(obstacle-ground > .008) & enabled[:, None])
    env._low_stairs_sample = result
    return result


def stair_clearance(env):
    x = sample(env)
    return clearance_cost(x['bottom'], x['obstacle'], x['force'], x['foot_velocity'],
                          x['direction'], x['upcoming'] & x['upright'][:, None])


def stair_progress(env):
    x = sample(env)
    return progress_reward(x['velocity'], x['direction'], x['speed'], x['support'], x['upright'], x['enabled'])


def phase_outside_stair_translation(env, command_name, asset_cfg, std=.1, command_threshold=.1,
                                   cycle_time=.4, phase_offsets=(0.,1.,1.,0.), gait_span=-.008,
                                   gait_psi=.15, gait_delta=.03, x_offset=0., stance_span=.2,
                                   stand_ref_z_offset=-.2, velocity_weight=.5):
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards import phase_foot_trajectory_exp
    kwargs = dict(command_name=command_name,asset_cfg=asset_cfg,std=std,command_threshold=command_threshold,
        cycle_time=cycle_time,phase_offsets=phase_offsets,gait_span=gait_span,gait_psi=gait_psi,gait_delta=gait_delta,
        x_offset=x_offset,stance_span=stance_span,stand_ref_z_offset=stand_ref_z_offset,velocity_weight=velocity_weight)
    return phase_foot_trajectory_exp(env, **kwargs) * ~sample(env)['enabled']


def airtime_without_stair_short_step_penalty(env, command_name, sensor_cfg, threshold, cmd_threshold=.1):
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards import feet_air_time_lin_xy_cmd
    from reference_reward_variants import feet_air_time_lin_xy_nonnegative
    original = feet_air_time_lin_xy_cmd(env, command_name, sensor_cfg, threshold, cmd_threshold)
    adjusted = feet_air_time_lin_xy_nonnegative(env, command_name, sensor_cfg, threshold, cmd_threshold)
    return torch.where(sample(env)['enabled'], adjusted, original)
