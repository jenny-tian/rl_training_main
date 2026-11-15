"""已批准的单项候选：保留原函数副作用，只屏蔽纯转向相位得分。"""
from __future__ import annotations

import copy
import torch

from reference_low_stairs_repair import pure_yaw

FUNCTION = 'reference_low_stairs_yaw_phase_off_cfg:phase_outside_stair_translation_and_pure_yaw'


def gate_phase(original, command):
    mask = pure_yaw(command)
    return torch.where(mask, torch.zeros_like(original), original), mask


def phase_outside_stair_translation_and_pure_yaw(
    env, command_name, asset_cfg, std=.1, command_threshold=.1,
    cycle_time=.4, phase_offsets=(0.,1.,1.,0.), gait_span=-.008,
    gait_psi=.15, gait_delta=.03, x_offset=0., stance_span=.2,
    stand_ref_z_offset=-.2, velocity_weight=.5,
):
    from reference_low_stairs_rewards import phase_outside_stair_translation
    # 每次仍只调用一次既有函数，先初始化/复用相位参考及地形缓存，再门控。
    original = phase_outside_stair_translation(env, command_name=command_name, asset_cfg=asset_cfg,
        std=std, command_threshold=command_threshold, cycle_time=cycle_time, phase_offsets=phase_offsets,
        gait_span=gait_span, gait_psi=gait_psi, gait_delta=gait_delta, x_offset=x_offset,
        stance_span=stance_span, stand_ref_z_offset=stand_ref_z_offset, velocity_weight=velocity_weight)
    value, mask = gate_phase(original, env.command_manager.get_command(command_name))
    env._yaw_phase_off_sample = dict(tick=env.common_step_counter, original=original.detach(),
                                     value=value.detach(), mask=mask)
    return value


def apply_phase_gate(original):
    from reference_reward_variants import plain
    before = plain(original)
    term = before['rewards']['phase_foot_trajectory_exp']
    assert term['func'] == 'reference_low_stairs_rewards:phase_outside_stair_translation'
    assert term['weight'] == 2.
    cfg = copy.deepcopy(original)
    cfg.rewards.phase_foot_trajectory_exp.func = phase_outside_stair_translation_and_pure_yaw
    after = plain(cfg); expected = copy.deepcopy(before)
    expected['rewards']['phase_foot_trajectory_exp']['func'] = FUNCTION
    assert after == expected and plain(original) == before
    return cfg, dict(status='passed', only_changed_reward='phase_foot_trajectory_exp',
                     only_changed_field='func', original_called_once=True,
                     pure_yaw_xy_below=.1, pure_yaw_abs_wz_above=.1,
                     old_function=term['func'], new_function=FUNCTION, weight_unchanged=2.)


class RuntimeAudit:
    """同一次原生奖励计算内重建关闭前分数，不二次调用有状态奖励。"""
    def __init__(self, base):
        self.base=base; self.steps=0; self.frames=0; self.pure_frames=0; self.removed_frames=0
        self.max_term_error=0.; self.max_total_error=0.; self.max_non_yaw_error=0.
        self.reference_initialized=True
        self.index=list(base.reward_manager.active_terms).index('phase_foot_trajectory_exp')
        self.weight=base.reward_manager.get_term_cfg('phase_foot_trajectory_exp').weight
        assert self.weight==2.

    def capture(self, reward, dt):
        base=self.base; x=base._yaw_phase_off_sample
        assert x['tick']==base.common_step_counter
        rates=base.reward_manager._step_reward
        original=x['original']*self.weight; expected=x['value']*self.weight
        assert torch.equal(expected[~x['mask']], original[~x['mask']])
        assert torch.count_nonzero(expected[x['mask']])==0
        term_error=float((rates[:,self.index]-expected).abs().max())
        assert term_error < 2e-5
        # 同一批状态的关闭前反事实：其他各项使用刚计算一次的原值。
        before=rates.clone(); before[:,self.index]=original
        expected_removed=original*x['mask']*dt
        total_error=float((before.sum(-1)*dt-reward-expected_removed).abs().max())
        assert total_error < 2e-5
        assert torch.allclose(reward,rates.sum(-1)*dt,atol=2e-5,rtol=1e-5)
        self.reference_initialized &= (base.phase_foot_ref_body.shape==(1,4,3))
        self.steps+=1; self.frames+=base.num_envs; self.pure_frames+=int(x['mask'].sum())
        self.removed_frames+=int((x['mask']&(original!=0)).sum())
        self.max_term_error=max(self.max_term_error,term_error)
        self.max_total_error=max(self.max_total_error,total_error)

    def summary(self):
        return dict(steps=self.steps,frames=self.frames,pure_yaw_frames=self.pure_frames,
                    removed_nonzero_frames=self.removed_frames,non_yaw_exact=True,
                    max_term_error=self.max_term_error,max_total_error=self.max_total_error,
                    reference_initialized=self.reference_initialized,
                    comparison='same-state counterfactual replaces only phase column; no duplicate reward calls')
