"""Behavioral reward checks: stepping, stalls, wall contacts and reset exploits."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_low_stairs_rewards import LandingState, clearance_cost, progress_reward


def test_clearance_is_local_and_does_not_pay_for_high_feet_or_stance():
    bottom = torch.tensor([[.02,.075,.15,.02]])
    obstacle = torch.full((1,4), .05)
    force = torch.zeros(1,4,3); force[0,3,2] = 20.
    velocity = torch.zeros_like(force); velocity[...,0] = .5
    direction = torch.tensor([[1.,0.]])
    enabled = torch.ones(1,4,dtype=torch.bool)
    cost = clearance_cost(bottom,obstacle,force,velocity,direction,enabled)
    assert cost.item() == pytest.approx(.25)
    # Adding elevation to an entire stair flight does not change clearance.
    assert torch.allclose(cost,clearance_cost(bottom+2.,obstacle+2.,force,velocity,direction,enabled))
    assert clearance_cost(bottom,obstacle,force,velocity*0,direction,enabled).item() == 0.
    assert clearance_cost(bottom,obstacle,force,velocity,direction,enabled*False).item() == 0.
    force[0,0,0] = 25.  # Pushing the vertical riser is not an airborne swing.
    assert clearance_cost(bottom,obstacle,force,velocity,direction,enabled).item() == 0.


def test_progress_distinguishes_forward_stall_reverse_and_unsupported_jump():
    v=torch.tensor([[.4,0.,0.],[0.,0.,0.],[-.4,0.,0.],[.4,0.,0.]])
    support=torch.ones(4,4,dtype=torch.bool); support[-1]=False
    result=progress_reward(v,torch.tensor([[1.,0.]]).repeat(4,1),torch.full((4,),.4),
                           support,torch.ones(4,dtype=torch.bool),torch.ones(4,dtype=torch.bool))
    assert torch.equal(result,torch.tensor([1.,0.,-1.,0.]))


def frame():
    return dict(feet=torch.zeros(1,4,3),bottom=torch.zeros(1,4),ground=torch.zeros(1,4),
                support=torch.ones(1,4,dtype=torch.bool),root_xy=torch.zeros(1,2),
                direction=torch.tensor([[1.,0.]]),enabled=torch.tensor([True]),upright=torch.tensor([True]))


def step(state, x, dt=.02):
    return state.update(**x,dt=dt).item()*dt


def perform_step(state,x,dt=.02,*,height=.03,advance=.1,body_advance=.04,peak=.07):
    x['support'][0,0]=False
    x['bottom'][0,0]=peak
    x['feet'][0,0,0]+=advance
    x['root_xy'][0,0]+=body_advance
    for _ in range(round(.08/dt)): assert step(state,x,dt)==0.
    x['support'][0,0]=True;x['ground'][0,0]=height;x['bottom'][0,0]=height
    return sum(step(state,x,dt) for _ in range(round(.06/dt)))


@pytest.mark.parametrize('dt',[.01,.02])
def test_supported_higher_landing_is_credited_once_and_rate_is_dt_independent(dt):
    state=LandingState(1,'cpu');x=frame();assert step(state,x,dt)==0.
    assert perform_step(state,x,dt)==pytest.approx(.25)
    for _ in range(10): assert step(state,x,dt)==0.
    assert perform_step(state,x,dt,height=.03)==0.  # Jump again on same tread.
    assert perform_step(state,x,dt,height=.06,peak=.1)==pytest.approx(.25)


@pytest.mark.parametrize('override',[
    dict(height=0.),dict(advance=0.),dict(body_advance=0.),dict(peak=.03),
])
def test_no_credit_for_flat_hopping_marching_or_insufficient_clearance(override):
    state=LandingState(1,'cpu');x=frame();step(state,x)
    assert perform_step(state,x,**override)==0.


def test_no_repeat_credit_after_descending_and_returning_to_same_tread():
    state=LandingState(1,'cpu');x=frame();step(state,x)
    assert perform_step(state,x)==pytest.approx(.25)
    assert perform_step(state,x,height=0.)==0.
    assert perform_step(state,x,height=.03)==0.


def test_brief_contact_cannot_earn_landing_credit():
    state=LandingState(1,'cpu');x=frame();step(state,x)
    x['support'][0,0]=False;x['bottom'][0,0]=.07;x['feet'][0,0,0]=.1;x['root_xy'][0,0]=.04
    for _ in range(4):assert step(state,x)==0.
    x['support'][0,0]=True;x['ground'][0,0]=.03;x['bottom'][0,0]=.03
    assert step(state,x)==0.
    x['support'][0,0]=False
    assert step(state,x)==0. and not state.pending.any()


def test_reset_on_high_platform_cannot_pay_or_inherit_airtime():
    state=LandingState(1,'cpu');x=frame();step(state,x)
    perform_step(state,x)
    x['support'][0,0]=False;step(state,x)
    state.reset([0]);x['support'][:]=True;x['ground'][:]=1.;x['bottom'][:]=1.
    assert step(state,x)==0.
    assert not state.air_time.any()
    assert perform_step(state,x,height=1.03,peak=1.07)==pytest.approx(.25)


@pytest.mark.parametrize('key',['enabled','upright'])
def test_no_landing_credit_outside_stair_motion_or_when_tilted(key):
    state=LandingState(1,'cpu');x=frame();step(state,x);x[key][:]=False
    assert perform_step(state,x)==0.


@pytest.mark.parametrize('extra',[
    ['--terrain-profile','low_stairs'],
    ['--reward-variant','low_stairs'],
    ['--terrain-profile','low_stairs','--reward-variant','low_stairs','--resume-final','--updates','500'],
    ['--terrain-profile','low_stairs','--reward-variant','low_stairs'],
])
def test_new_training_requires_matching_profiles_fresh_weights_and_explicit_budget(extra,monkeypatch,tmp_path):
    from reference_six_train import main
    monkeypatch.setattr(sys,'argv',['reference_six_train.py','--mode','train','--output',str(tmp_path/'unstarted'),*extra])
    with pytest.raises(SystemExit) as failure:main()
    assert failure.value.code==2 and not (tmp_path/'unstarted').exists()


def test_low_stair_evaluation_excludes_online_heights_and_preserves_original_protocol():
    from reference_isaac_stairs_diagnose import protocol, HEIGHT_SETS
    old=protocol(20);low=protocol(20,HEIGHT_SETS['low_stairs'])
    assert len(old)==len(low)==240
    assert {r['height'] for r in old}=={0.,.05,.1,.15}
    assert {r['height'] for r in low}=={0.,.02,.03,.05}
    for a,b in zip(old,low):
        assert {k:v for k,v in a.items() if k!='height'}=={k:v for k,v in b.items() if k!='height'}
