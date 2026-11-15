import copy
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from reference_low_stairs_motion_analyze import (
    airtime_rates,DT,inverse_rotation,phase_function,definitions,REWARD,Entity,Scene,
    selected_mask,paired_groups)


def test_airtime_matches_source_and_distinguishes_pure_yaw_from_arc():
    ns=dict(torch=torch,SceneEntityCfg=Entity,gait_level=1.)
    definitions(REWARD,['get_gait_level_tensor','feet_air_time_ang_z_cmd_lite3'],ns)
    command=np.array([[0,0,.8],[0,0,-.8],[.3,0,.8],[0,0,0]],dtype=np.float32)
    last=np.array([[.2,.7,.1,.1]]*4,dtype=np.float32)
    current=np.array([[.02,.01,.2,0]]*4,dtype=np.float32)
    original,repaired,first=airtime_rates(command,last,current)
    sensor=NS(data=NS(last_air_time=torch.tensor(last)),compute_first_contact=lambda _:torch.tensor(first))
    env=NS(scene=Scene(contact_forces=sensor),step_dt=DT,num_envs=4,device='cpu',command_manager=NS(get_command=lambda _:torch.tensor(command)))
    expected=ns['feet_air_time_ang_z_cmd_lite3'](env,'base_velocity',Entity('contact_forces',body_ids=list(range(4))),.5)*5
    assert np.allclose(original,expected.numpy())
    assert np.allclose(repaired[:2],1.) and repaired[2]==original[2]<0 and repaired[3]==0


def test_short_swing_has_no_positive_repaired_reward_and_static_has_none():
    commands=np.array([[0,0,.8],[0,0,-.8]],dtype=np.float32)
    original,repaired,_=airtime_rates(commands,np.full((2,4),.2),np.full((2,4),.01))
    assert (original<0).all() and not repaired.any()
    original,repaired,_=airtime_rates(commands,np.full((2,4),.2),np.full((2,4),1.))
    assert not original.any() and not repaired.any()


def test_inverse_rotation_for_known_yaw():
    q=torch.tensor([[math.sqrt(.5),0,0,math.sqrt(.5)]])
    assert torch.allclose(inverse_rotation(q,torch.tensor([[1.,0,0]])),torch.tensor([[0.,-1,0]]),atol=2e-7)


def test_phase_intermediate_capture_preserves_exact_source_reward():
    ns=dict(torch=torch,math=math,SceneEntityCfg=Entity,gait_level=1.,math_utils=NS(quat_apply_inverse=inverse_rotation))
    definitions(REWARD,['phase_foot_trajectory_exp','get_gait_level_tensor','_bernstein_torch','_bezier_curve_torch','_bezier_curve_derivative_torch'],ns)
    data=NS(root_pos_w=torch.zeros(5,3),root_quat_w=torch.tensor([[1.,0,0,0]]*5),root_lin_vel_w=torch.zeros(5,3),
            body_pos_w=torch.tensor([[[.2,.1,-.3],[.2,-.1,-.3],[-.2,.1,-.3],[-.2,-.1,-.3]]]*5),body_lin_vel_w=torch.zeros(5,4,3))
    env=NS(num_envs=5,device='cpu',scene={'robot':NS(data=data)},step_dt=.02,episode_length_buf=torch.arange(1,6),
           command_manager=NS(get_command=lambda _:torch.tensor([[0,0,.8]]*5)))
    params=dict(command_name='base_velocity',asset_cfg=Entity('robot',body_ids=list(range(4))))
    before=ns['phase_foot_trajectory_exp'](env,**params)
    after,phase,position,velocity,_,_=phase_function()(env,**params)
    assert torch.equal(before,after) and position.shape==velocity.shape==(5,4,3)
    assert ((phase>=0)&(phase<2)).all()


def test_failed_window_and_missing_repeat_are_not_included_in_paired_means():
    trace={'active':np.ones((1000,1),bool),'terminated':np.zeros((1000,1),bool)}
    assert selected_mask(trace,0,4,8).sum()==200
    trace['terminated'][399,0]=True
    assert selected_mask(trace,0,4,8) is None
    rows=[dict(policy=p,profile='nominal',case='fast_left',window='steady',repeat=i,value=float(i))
          for p in ('parent','zero') for i in range(3)]
    rows=[r for r in rows if not (r['policy']=='zero' and r['repeat']==2)]
    row,=paired_groups(rows)
    assert row['n']==2 and row['parent_value']==row['zero_value']==.5
