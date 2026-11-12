"""Explicit reward overrides for the read-only Lite3 reference environment.

The reference preset is an exact copy. Every other preset changes only the
listed reward terms; commands, terrain, observations and PPO are untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json

import torch

BASE_HEIGHT_M = 0.32
VARIANTS = {
    'reference': (),
    'stand_gate': ('stand_still',),
    'height_target': ('base_height_l2',),
    'core': ('stand_still', 'base_height_l2'),
    'core_no_phase': ('stand_still', 'base_height_l2', 'phase_foot_trajectory_exp'),
    'core_airtime_nonnegative': ('stand_still', 'base_height_l2', 'feet_air_time_lin_xy', 'feet_air_time_ang_z_lite3'),
    'core_no_gait': ('stand_still', 'base_height_l2', 'feet_gait'),
}


def stand_still_joint_deviation_se2(env, command_name, asset_cfg,
                                   linear_threshold=0.1, angular_threshold=0.1):
    """Apply the standing penalty only when translation AND yaw are small.

    Keep separate thresholds in m/s and rad/s instead of mixing their units.
    Preserve the original joint selection and L1 deviation calculation.
    """
    data = env.scene[asset_cfg.name].data
    deviation = (data.joint_pos[:, asset_cfg.joint_ids]
                 - data.default_joint_pos[:, asset_cfg.joint_ids]).abs().sum(dim=1)
    command = env.command_manager.get_command(command_name)
    standing = ((command[:, :2].norm(dim=1) < linear_threshold)
                & (command[:, 2].abs() < angular_threshold))
    return deviation * standing


def nonnegative_landing_reward(sensor, body_ids, dt, threshold):
    """Clamp EACH foot's short-swing deficit before summing first landings.

    Short steps and static contacts receive zero. A swing longer than the
    unchanged threshold earns the original positive reward when it lands.
    """
    first_contact = sensor.compute_first_contact(dt)[:, body_ids]
    air_time = sensor.data.last_air_time[:, body_ids]
    return ((air_time - threshold).clamp_min(0.) * first_contact).sum(dim=1)


def _gait_scale(env):
    # Resolve only after reference_lite3_loader has installed its namespace.
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards import get_gait_level_tensor
    return get_gait_level_tensor(env)


def feet_air_time_lin_xy_nonnegative(env, command_name, sensor_cfg, threshold, cmd_threshold=0.1):
    reward = nonnegative_landing_reward(env.scene.sensors[sensor_cfg.name], sensor_cfg.body_ids,
                                        env.step_dt, threshold)
    moving = env.command_manager.get_command(command_name)[:, :2].norm(dim=1) > cmd_threshold
    return reward * moving * _gait_scale(env)


def feet_air_time_yaw_nonnegative(env, command_name, sensor_cfg, threshold, cmd_threshold=0.1):
    reward = nonnegative_landing_reward(env.scene.sensors[sensor_cfg.name], sensor_cfg.body_ids,
                                        env.step_dt, threshold)
    turning = env.command_manager.get_command(command_name)[:, 2].abs() > cmd_threshold
    return reward * turning * _gait_scale(env)


def plain(cfg):
    return json.loads(json.dumps(cfg.to_dict(), default=str))


def apply_reward_variant(reference_cfg, variant):
    """Return an isolated config and a complete, serializable change audit."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown reward variant: {variant}')
    cfg = copy.deepcopy(reference_cfg)
    before = plain(reference_cfg)
    changed = VARIANTS[variant]
    if 'stand_still' in changed:
        term = cfg.rewards.stand_still
        if term.params['command_threshold'] != .1 or term.weight != -.5:
            raise ValueError('standing baseline changed; repeat reference audit')
        term.func = stand_still_joint_deviation_se2
        term.params.pop('command_threshold')
        term.params.update(linear_threshold=.1, angular_threshold=.1)
    if 'base_height_l2' in changed:
        term = cfg.rewards.base_height_l2
        if term.params['target_height'] != .55 or term.weight != -50.:
            raise ValueError('height baseline changed; repeat reference audit')
        term.params['target_height'] = BASE_HEIGHT_M
    if variant == 'core_no_phase':
        cfg.rewards.phase_foot_trajectory_exp.weight = 0.
    if variant == 'core_no_gait':
        cfg.rewards.feet_gait.weight = 0.
    if variant == 'core_airtime_nonnegative':
        cfg.rewards.feet_air_time_lin_xy.func = feet_air_time_lin_xy_nonnegative
        cfg.rewards.feet_air_time_ang_z_lite3.func = feet_air_time_yaw_nonnegative
    after = plain(cfg)
    # Scope is checked against complete configs, not a hand-picked subset.
    stripped_before, stripped_after = copy.deepcopy(before), copy.deepcopy(after)
    rewards_before = stripped_before.pop('rewards')
    rewards_after = stripped_after.pop('rewards')
    if stripped_before != stripped_after:
        raise ValueError('reward override changed another environment setting')
    actual = tuple(k for k in rewards_before if rewards_before[k] != rewards_after[k])
    if set(actual) != set(changed) or rewards_before.keys() != rewards_after.keys():
        raise ValueError(f'reward override scope differs: {actual}')
    if plain(reference_cfg) != before:
        raise ValueError('reference config was mutated')
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    audit = dict(status='passed', variant=variant, non_reward_environment_exact=True,
                 reference_config_unchanged=True, before_sha256=digest(before), after_sha256=digest(after),
                 changes={name: dict(before=rewards_before[name], after=rewards_after[name]) for name in changed})
    return cfg, audit
