"""只替换净空项的可选配置；原训练及 A/B 变体保持原样。"""
from __future__ import annotations

import copy

from isaaclab.managers import ManagerTermBase
from reference_low_stairs_clearance import ClearanceState, clearance_sample
from reference_low_stairs_rewards import sample
from reference_reward_variants import plain


class StairClearanceReward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.state = ClearanceState(env.num_envs, env.device)

    def reset(self, env_ids=None):
        self.state.reset(env_ids)
        for name in ('_low_stairs_sample', '_persistent_clearance_sample'):
            if hasattr(self._env, name):
                delattr(self._env, name)

    def __call__(self, env):
        x = sample(env)
        walls = clearance_sample(env)
        return self.state.update(**{k: x[k] for k in (
            'bottom', 'obstacle', 'upcoming', 'support', 'direction', 'enabled', 'upright')},
            wall=walls['wall'], dt=env.step_dt, tick=env.common_step_counter).mean(-1)


def apply_clearance(original):
    cfg = copy.deepcopy(original)
    before = plain(original)
    term = cfg.rewards.stair_clearance
    assert term.func.__name__ == 'stair_clearance' and term.weight == -1. and term.params == {}
    # 这次准备只接受未加入 A/B 奖励的父模型配置。
    assert 'stair_riser_contact' not in before['rewards']
    assert cfg.rewards.feet_air_time_ang_z_lite3.func.__name__ == 'feet_air_time_ang_z_cmd_lite3'
    term.func = StairClearanceReward
    after = plain(cfg)
    expected = copy.deepcopy(before)
    expected['rewards']['stair_clearance'] = after['rewards']['stair_clearance']
    assert after == expected and before != after and plain(original) == before
    return cfg, dict(status='passed', changed_terms=['stair_clearance'], weight=-1.,
        non_reward_environment_exact=True, turning_rewards_unchanged=True,
        checkpoint_initialization='reviewed parent; no A500 continuation',
        before=before['rewards']['stair_clearance'], after=after['rewards']['stair_clearance'])
