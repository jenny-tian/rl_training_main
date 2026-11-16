"""单项峰值门槛试验仍须拒绝原地踏步、瞬间接触、单足支撑及重复高度。"""
import ast
import inspect
import textwrap

import pytest
import torch

from reference_low_stairs_landing_candidate import LandingPeak5mmState
from reference_low_stairs_rewards import LandingState


def test_only_peak_constant_differs_from_historic_update():
    old=ast.parse(textwrap.dedent(inspect.getsource(LandingState.update)))
    new=ast.parse(textwrap.dedent(inspect.getsource(LandingPeak5mmState.update)))
    constants=[n for n in ast.walk(old) if isinstance(n,ast.Constant) and n.value==.015]
    assert len(constants)==1
    constants[0].value=.005
    assert ast.dump(old)==ast.dump(new)


def frame():
    return dict(feet=torch.zeros(1,4,3),bottom=torch.zeros(1,4),ground=torch.zeros(1,4),
        support=torch.ones(1,4,dtype=torch.bool),root_xy=torch.zeros(1,2),
        direction=torch.tensor([[1.,0.]]),enabled=torch.tensor([True]),upright=torch.tensor([True]))


def swing(state,x,*,dt=.02,peak=.06,foot=.1,body=.04,air=.08):
    x['support'][0,0]=False;x['bottom'][0,0]=peak;x['feet'][0,0,0]+=foot;x['root_xy'][0,0]+=body
    for _ in range(round(air/dt)):assert not state.update(**x,dt=dt).any()
    x['support'][0,0]=True;x['ground'][0,0]=.05;x['bottom'][0,0]=.05


@pytest.mark.parametrize('dt',[.01,.02])
def test_10mm_peak_gains_once_only_after_60ms_and_reset_cannot_pay(dt):
    for cls,expected in ((LandingState,0.),(LandingPeak5mmState,.25)):
        s=cls(1,'cpu');x=frame();s.update(**x,dt=dt);swing(s,x,dt=dt)
        returns=[s.update(**x,dt=dt).item()*dt for _ in range(round(.06/dt))]
        assert not any(returns[:-1]) and sum(returns)==pytest.approx(expected)
        swing(s,x,dt=dt)
        assert sum(s.update(**x,dt=dt).item()*dt for _ in range(round(.06/dt)))==0.
        s.reset([0]);x['ground'][:]=.5;x['bottom'][:]=.5
        assert not s.update(**x,dt=dt).any() and not s.air_time.any()


@pytest.mark.parametrize('override',[
    {'peak':.054},{'foot':.039},{'body':.009},{'air':.02},
])
def test_other_landing_requirements_still_reject(override):
    s=LandingPeak5mmState(1,'cpu');x=frame();s.update(**x,dt=.02);swing(s,x,**override)
    assert sum(s.update(**x,dt=.02).item() for _ in range(4))==0.


@pytest.mark.parametrize('mode',['single_support','disabled','tilted','lost_support'])
def test_pending_cancels_on_support_or_scope_loss(mode):
    s=LandingPeak5mmState(1,'cpu');x=frame();s.update(**x,dt=.02);swing(s,x)
    assert not s.update(**x,dt=.02).any() and s.pending[0,0]
    if mode=='single_support':x['support'][0,1:]=False
    elif mode=='disabled':x['enabled'][:]=False
    elif mode=='tilted':x['upright'][:]=False
    else:x['support'][0,0]=False
    assert not s.update(**x,dt=.02).any() and not s.pending.any()


def test_flat_hop_and_descent_receive_no_credit():
    for level in (0.,-.05):
        s=LandingPeak5mmState(1,'cpu');x=frame();s.update(**x,dt=.02);swing(s,x)
        x['ground'][0,0]=level;x['bottom'][0,0]=level
        assert sum(s.update(**x,dt=.02).item() for _ in range(4))==0.
