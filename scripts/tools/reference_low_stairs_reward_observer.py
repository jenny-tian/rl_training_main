"""旁路解释当前训练状态机；只调用历史类，不修改其行为。"""
from __future__ import annotations

import torch

from reference_low_stairs_clearance import ClearanceState, LANDING_DWELL
from reference_low_stairs_rewards import LandingState, progress_reward

LANDING_RULES = ('touchdown', 'initialized', 'enabled', 'upright', 'two_supports',
    'higher_than_last', 'higher_than_credited', 'foot_forward', 'body_forward', 'peak_clear', 'near_ground')


def state_copy(state, prefix):
    return {prefix+k:v.clone() for k,v in vars(state).items() if isinstance(v,torch.Tensor)}


class RewardObserver:
    def __init__(self, count, device):
        self.clearance = ClearanceState(count, device)
        self.landing = LandingState(count, device)

    def reset(self, ids=None):
        self.clearance.reset(ids)
        self.landing.reset(ids)

    def observe(self, x, walls, dt, tick):
        c,l=self.clearance,self.landing
        assert c.last_tick.ne(tick).all(), '旁路状态每个控制步只能更新一次'
        before=state_copy(c,'clear_before_') | state_copy(l,'land_before_')
        valid=(x['enabled'] & x['upright'])[:,None]
        aligned=(c.direction*x['direction'][:,None]).sum(-1)>.5
        kept=c.active & valid & aligned
        supported=x['support'] & ~walls['wall']
        start=valid & x['upcoming'] & ~supported & ~kept
        candidate_active=kept | start
        candidate_target=torch.where(start,x['obstacle'],c.target)
        supported_dwell=torch.where(candidate_active & supported,c.dwell+dt,0.)
        release=candidate_active & (supported_dwell>=LANDING_DWELL-1e-6)

        first=x['support'] & ~l.initialized
        peak=torch.where(l.was_supported,x['bottom'],torch.maximum(l.peak_bottom,x['bottom']))
        airtime=torch.where(~x['support'],l.air_time+dt,l.air_time)
        touchdown=x['support'] & ~l.was_supported & (airtime>=.04-1e-6) & ~first
        displacement=((x['feet'][...,:2]-l.support_xy)*l.direction).sum(-1)
        body_progress=((x['root_xy'][:,None]-l.support_root_xy)*l.direction).sum(-1)
        rules=dict(touchdown=touchdown, initialized=l.initialized | first,
            enabled=x['enabled'][:,None].expand_as(first), upright=x['upright'][:,None].expand_as(first),
            two_supports=(x['support'].sum(-1)>=2)[:,None].expand_as(first),
            higher_than_last=x['ground']>l.support_z+.008,
            higher_than_credited=x['ground']>torch.where(first,x['ground'],l.highest_credited_z)+.008,
            foot_forward=displacement>=.04, body_forward=body_progress>=.01,
            peak_clear=peak>=x['ground']+.015, near_ground=(x['bottom']-x['ground']).abs()<=.025)
        qualifies=torch.stack([rules[k] for k in LANDING_RULES]).all(0)
        pending_z=torch.where(qualifies,x['ground'],l.pending_z)
        pending=(l.pending | qualifies) & x['support'] & valid
        pending &= (x['support'].sum(-1)>=2)[:,None] & ((x['ground']-pending_z).abs()<.005)
        pending_dwell=torch.where(pending,l.dwell+dt,0.)
        awarded=pending & (pending_dwell>=.06-1e-6)

        cost=c.update(**{k:x[k] for k in ('bottom','obstacle','upcoming','support','direction','enabled','upright')},
            wall=walls['wall'],dt=dt,tick=tick)
        landing=l.update(**{k:x[k] for k in ('feet','bottom','ground','support','root_xy','direction','enabled','upright')},dt=dt)
        assert torch.equal(c.active,candidate_active & ~release)
        assert torch.equal(l.pending,pending & ~awarded)
        assert torch.equal(landing,awarded.float().mean(-1)/dt)
        assert torch.equal(c.target,candidate_target)
        values={k:x[k] for k in ('feet','bottom','ground','obstacle','force','foot_velocity','direction',
            'speed','enabled','upright','support','root_xy','velocity','upcoming')}
        values.update(wall=walls['wall'],wall_distance=walls['distance'],wall_normal=walls['normal'],
            clear_start=start, clear_release_support=release,
            clear_release_lower=release & (x['ground']<candidate_target-.008),
            clear_release_scope=before['clear_before_active'] & ~x['enabled'][:,None],
            clear_release_tilt=before['clear_before_active'] & x['enabled'][:,None] & ~x['upright'][:,None],
            clear_release_direction=before['clear_before_active'] & valid & ~aligned,
            land_touch_first=first,land_raw_touch=x['support'] & ~before['land_before_was_supported'] & ~first,
            land_airtime_at_check=airtime,land_peak_at_check=peak,land_displacement=displacement,
            land_body_progress=body_progress,land_qualifies=qualifies,land_awarded=awarded,
            land_dwell_at_check=pending_dwell,**{f'land_rule_{k}':v for k,v in rules.items()},
            clearance_rate=-cost.mean(-1),landing_rate=landing,
            progress_rate=progress_reward(x['velocity'],x['direction'],x['speed'],x['support'],x['upright'],x['enabled']))
        values.update(before)
        values.update(state_copy(c,'clear_after_'))
        values.update(state_copy(l,'land_after_'))
        return values
