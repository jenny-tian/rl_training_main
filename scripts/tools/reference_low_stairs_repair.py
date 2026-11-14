"""Isolated, opt-in repair candidates for the frozen low-stair checkpoint.

No training entry point selects these automatically. The original training
sources stay unchanged so their saved manifests remain verifiable.
"""
from __future__ import annotations

import copy

import torch

from reference_low_stairs_rewards import FOOT_RADIUS, sample
from reference_reward_variants import plain

VARIANTS = ('riser_contact', 'yaw_airtime')


def pure_yaw(command):
    return (command[:, :2].norm(dim=-1) < .1) & (command[:, 2].abs() > .1)


def riser_contact_cost(force, direction, wall_distance, wall_normal, enabled):
    """Bounded cost for force opposing motion against a nearby vertical face.

    Missed horizontal rays contain infinity. They are explicitly masked before
    arithmetic. Floor friction, distant faces and non-stair/yaw-only motion do
    not qualify. This is not a positive reward for keeping the feet raised.
    """
    valid = torch.isfinite(wall_distance) & torch.isfinite(wall_normal).all(-1)
    normal = torch.where(valid[..., None], wall_normal, 0.)
    face = ((wall_distance >= 0.) & (wall_distance <= FOOT_RADIUS + .008)
            & (normal[..., 2].abs() < .2)
            & ((normal[..., :2] * direction[:, None]).sum(-1).abs() > .5))
    opposing = -(force[..., :2] * direction[:, None]).sum(-1)
    force_cost = ((opposing - 5.) / 25.).clamp(0., 1.)
    return (force_cost * valid * face * enabled[:, None]).mean(-1)


def wall_sample(env):
    cached = getattr(env, '_low_stairs_wall_sample', None)
    if cached is not None and cached['tick'] == env.common_step_counter:
        return cached
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh
    x = sample(env)
    rays = torch.zeros_like(x['feet'])
    # Pure yaw has a zero planar direction. Give inactive rays a valid unit
    # direction; the enabled mask prevents these from scoring.
    direction = torch.where(x['enabled'][:, None], x['direction'],
                            x['direction'].new_tensor([1., 0.]))
    rays[..., :2] = direction[:, None]
    hits, distance, normals, _ = raycast_mesh(
        x['feet'], rays, RayCaster.meshes[env.cfg.scene.terrain.prim_path],
        max_dist=.04, return_distance=True, return_normal=True)
    cached = dict(tick=env.common_step_counter, distance=distance, normal=normals, hits=hits)
    env._low_stairs_wall_sample = cached
    return cached


def stair_riser_contact(env):
    x, wall = sample(env), wall_sample(env)
    return riser_contact_cost(x['force'], x['direction'], wall['distance'], wall['normal'],
                              x['enabled'] & x['upright'])


def yaw_airtime_without_short_step_penalty(env, command_name, sensor_cfg, threshold, cmd_threshold=.1):
    from reference_reward_variants import feet_air_time_yaw_nonnegative
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards import feet_air_time_ang_z_cmd_lite3
    original = feet_air_time_ang_z_cmd_lite3(env, command_name, sensor_cfg, threshold, cmd_threshold)
    adjusted = feet_air_time_yaw_nonnegative(env, command_name, sensor_cfg, threshold, cmd_threshold)
    return torch.where(pure_yaw(env.command_manager.get_command(command_name)), adjusted, original)


def apply_repair(original, variant):
    """Change one reward term relative to the already configured low-stair run."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown low-stair repair: {variant}')
    from isaaclab.managers import RewardTermCfg
    cfg = copy.deepcopy(original)
    before = plain(original)
    assert {'stair_clearance', 'stair_progress', 'stair_landing'} <= set(before['rewards'])
    if variant == 'riser_contact':
        assert 'stair_riser_contact' not in before['rewards']
        cfg.rewards.stair_riser_contact = RewardTermCfg(func=stair_riser_contact, weight=-1.)
        expected = {'stair_riser_contact'}
    else:
        term = cfg.rewards.feet_air_time_ang_z_lite3
        assert term.weight == 5. and term.params['threshold'] == .5 and term.params['cmd_threshold'] == .1
        assert term.func.__name__ == 'feet_air_time_ang_z_cmd_lite3'
        term.func = yaw_airtime_without_short_step_penalty
        expected = {'feet_air_time_ang_z_lite3'}
    after = plain(cfg)
    a, b = copy.deepcopy(before), copy.deepcopy(after)
    ra, rb = a.pop('rewards'), b.pop('rewards')
    changed = {k for k in ra.keys() | rb.keys() if ra.get(k) != rb.get(k)}
    assert a == b and changed == expected and plain(original) == before
    return cfg, dict(status='passed', variant=variant, non_reward_environment_exact=True,
                     input_config_unchanged=True,
                     changes={k:dict(before=ra.get(k), after=rb.get(k)) for k in sorted(changed)})
