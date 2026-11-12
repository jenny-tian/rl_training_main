"""Explicit low-stair experiment: isolated terrain and reward changes."""
from __future__ import annotations

import copy

from isaaclab.managers import ManagerTermBase, RewardTermCfg
from reference_reward_variants import apply_reward_variant, plain
import reference_low_stairs_rewards as rewards


class StairLandingReward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.state = rewards.LandingState(env.num_envs, env.device)

    def reset(self, env_ids=None):
        self.state.reset(env_ids)

    def __call__(self, env):
        x = rewards.sample(env)
        return self.state.update(**{k:x[k] for k in ('feet','bottom','ground','support','root_xy','direction','enabled','upright')},
                                 dt=env.step_dt)


def apply_terrain(original):
    cfg = copy.deepcopy(original)
    before = plain(original)
    for name in ('stairs_up', 'stairs_down'):
        sub = cfg.scene.terrain.terrain_generator.sub_terrains[name]
        assert tuple(sub.step_height_range) == (.05, .23) and sub.step_width == .3
        sub.step_height_range = (.02, .05)
    # Start at the lowest native row. The native first reset can advance it by
    # one row; both rows remain in the 2-3 cm introductory band.
    cfg.scene.terrain.max_init_terrain_level = 0
    after = plain(cfg)
    expected = copy.deepcopy(before)
    expected['scene']['terrain']['max_init_terrain_level'] = 0
    for name in ('stairs_up', 'stairs_down'):
        expected['scene']['terrain']['terrain_generator']['sub_terrains'][name]['step_height_range'] = [.02, .05]
    assert after == expected and plain(original) == before
    return cfg, dict(status='passed',profile='low_stairs',step_height_range_m=[.02,.05],
        tread_depth_m=.3,max_init_terrain_level=0,native_layout_and_progression_preserved=True,
        observation_and_ppo_unchanged=True,initial_level_change_applies_to_all_six_families=True,
        online_candidate_heights_m=[.06,.07,.08],later_validation_height_m=.10,
        checkpoint_initialization='scratch_only',reference_unchanged=True)


def apply_rewards(original):
    cfg, core_audit = apply_reward_variant(original, 'core')
    before = plain(original)
    cfg.rewards.phase_foot_trajectory_exp.func = rewards.phase_outside_stair_translation
    cfg.rewards.feet_air_time_lin_xy.func = rewards.airtime_without_stair_short_step_penalty
    cfg.rewards.stair_clearance = RewardTermCfg(func=rewards.stair_clearance, weight=-1.)
    cfg.rewards.stair_progress = RewardTermCfg(func=rewards.stair_progress, weight=1.)
    cfg.rewards.stair_landing = RewardTermCfg(func=StairLandingReward, weight=1.)
    after = plain(cfg)
    expected_changes = {'stand_still','base_height_l2','phase_foot_trajectory_exp','feet_air_time_lin_xy',
                        'stair_clearance','stair_progress','stair_landing'}
    a, b = copy.deepcopy(before), copy.deepcopy(after)
    ra, rb = a.pop('rewards'), b.pop('rewards')
    actual = {k for k in set(ra)|set(rb) if ra.get(k) != rb.get(k)}
    assert a == b and actual == expected_changes and plain(original) == before
    return cfg, dict(status='passed',variant='low_stairs',core_audit=core_audit,
        non_reward_environment_exact=True,reference_config_unchanged=True,
        changes={k:dict(before=ra.get(k),after=rb.get(k)) for k in sorted(actual)},
        stair_scope='stairs columns with planar command speed > 0.1 m/s; pure yaw and other terrain families keep their gait terms',
        foot_radius_m=rewards.FOOT_RADIUS,clearance_margin_m=rewards.MARGIN,
        reward_information='privileged simulator terrain for offline base training only; actor remains proprioceptive')
