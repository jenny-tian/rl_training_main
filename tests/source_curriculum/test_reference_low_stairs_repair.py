"""Repair candidates must distinguish riser contact and retain unrelated gait behavior."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_low_stairs_repair import pure_yaw, riser_contact_cost, yaw_airtime_without_short_step_penalty
from reference_low_stairs_failure_audit import stair_features, longest


def contact_case():
    force=torch.zeros(1,4,3);force[0,0,0]=-30.
    normal=torch.zeros_like(force);normal[...,0]=-1.
    return dict(force=force,direction=torch.tensor([[1.,0.]]),wall_distance=torch.full((1,4),.022),
                wall_normal=normal,enabled=torch.tensor([True]))


def test_riser_cost_is_bounded_and_rotation_invariant():
    x=contact_case();assert riser_contact_cost(**x).item()==pytest.approx(.25)
    x['force'][...,0]=-300.
    assert riser_contact_cost(**x).item()==1.
    # Rotate world coordinates by 90 degrees; the command-relative cost is unchanged.
    for key in ('force','wall_normal'):
        x[key][...,1]=x[key][...,0];x[key][...,0]=0.
    x['direction']=torch.tensor([[0.,1.]])
    assert riser_contact_cost(**x).item()==1.


@pytest.mark.parametrize('case',['miss','floor','far','away','force_noise','disabled'])
def test_riser_cost_does_not_penalize_non_collision_cases(case):
    x=contact_case()
    if case=='miss':x['wall_distance'][:]=float('inf');x['wall_normal'][:]=float('inf')
    elif case=='floor':x['wall_normal'][:]=torch.tensor([0.,0.,1.])
    elif case=='far':x['wall_distance'][:]=.035
    elif case=='away':x['force']*=-1
    elif case=='force_noise':x['force']*=.1
    elif case=='disabled':x['enabled'][:]=False
    result=riser_contact_cost(**x)
    assert torch.isfinite(result).all() and result.item()==0.


def test_pure_yaw_has_separate_linear_and_angular_thresholds():
    cmd=torch.tensor([[0.,0.,0.],[0.,0.,.2],[0.,0.,-.5],[.3,0.,.5],[.11,0.,-.5],[0.,0.,.1]])
    assert pure_yaw(cmd).tolist()==[False,True,True,False,False,False]


def test_yaw_repair_removes_only_short_pure_turn_landing_penalties(monkeypatch):
    from reference_reward_code_audit import definitions, REWARD
    ns=dict(torch=torch,gait_level=1.)
    definitions(REWARD,['feet_air_time_ang_z_cmd_lite3','get_gait_level_tensor'],ns)
    module=NS(**ns)
    monkeypatch.setitem(sys.modules,'rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards',module)
    cmd=torch.tensor([[0.,0.,.5],[0.,0.,-.5],[.3,0.,.5],[.4,0.,0.],[0.,0.,0.]])
    contact=torch.tensor([[True,True,False,False]]).repeat(5,1)
    # Mixed short/long landings ensure per-foot clamping before summation.
    sensor=NS(data=NS(last_air_time=torch.tensor([[.2,.7,0.,0.]]).repeat(5,1)),
              compute_first_contact=lambda dt:contact)
    scene=NS(sensors={'feet':sensor})
    env=NS(scene=scene,step_dt=.02,num_envs=5,device='cpu',command_manager=NS(get_command=lambda name:cmd))
    args=(env,'base_velocity',NS(name='feet',body_ids=list(range(4))),.5)
    result=yaw_airtime_without_short_step_penalty(*args)
    assert torch.allclose(result,torch.tensor([.2,.2,-.1,0.,0.]),atol=1e-6)
    contact[:]=False
    assert yaw_airtime_without_short_step_penalty(*args).eq(0.).all()


def test_stair_audit_uses_foot_radius_and_separates_floor_and_wall():
    feet=np.array([[[1.178,.1,.03],[1.178,-.1,.08],[1.10,.1,.022],[1.25,-.1,.072]]])
    force=np.array([[[-30.,0.,0.],[-30.,0.,0.],[0.,0.,30.],[0.,0.,30.]]])
    x=stair_features(feet,force,dict(fronts=[1.2],tops=[.05]))
    assert x['collision'].tolist()==[[True,False,False,False]]
    assert x['wall'].tolist()==[[True,False,False,False]]
    assert x['support'].tolist()==[[False,False,True,True]]
    assert longest([False,True,True,False,True])==.04
