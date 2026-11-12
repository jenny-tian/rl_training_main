"""Execute exact source reward functions on controlled tensors without Isaac imports.

Only module imports/type dependencies are replaced. Selected AST definitions are
compiled unchanged, and source hashes/line numbers are recorded for review.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from reference_lite3_loader import ROOT, REFERENCE
from reference_stairs_eval import save,sha

REWARD = REFERENCE/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py'
STAND = Path('/home/tjx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py')
COMMAND_BASE = Path('/home/tjx/IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py')
COMMAND = REWARD.with_name('commands.py')


class Entity:
    def __init__(self,name='robot',**kwargs):
        self.name=name
        self.__dict__.update(kwargs)


class Scene(dict):
    @property
    def sensors(self): return self


def definitions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return {n.name:dict(path=str(path),line=n.lineno,sha256=sha(path)) for n in nodes}


def audit():
    def identity_rotation(quat,value):
        assert torch.allclose(quat,quat.new_tensor([1.,0.,0.,0.]).expand_as(quat))
        return value
    ns=dict(torch=torch,math=math,SceneEntityCfg=Entity,ManagerTermBase=object,gait_level=1.,
        math_utils=NS(quat_apply_inverse=identity_rotation))
    funcs=['get_gait_level_tensor','update_gait_level_from_terrain_mean','track_lin_vel_xy_exp','track_ang_vel_z_exp',
           'GaitReward','feet_air_time_lin_xy_cmd','feet_air_time_ang_z_cmd_lite3','phase_foot_trajectory_exp',
           '_bernstein_torch','_bezier_curve_torch','_bezier_curve_derivative_torch']
    sources=definitions(REWARD,funcs,ns)
    ns['mdp']=NS(joint_deviation_l1=lambda env,cfg:torch.abs(env.scene[cfg.name].data.joint_pos-env.scene[cfg.name].data.default_joint_pos).sum(-1))
    sources.update(definitions(STAND,['stand_still_joint_deviation_l1'],ns))
    commands=torch.tensor([[0.,0.,0.],[0.,0.,.5],[0.,0.,-.5],[.4,0.,0.]])
    data=NS(joint_pos=torch.full((4,12),.1),default_joint_pos=torch.zeros(4,12),
            root_lin_vel_b=torch.zeros(4,3),root_ang_vel_b=torch.zeros(4,3))
    env=NS(num_envs=4,device='cpu',scene=Scene(robot=NS(data=data)),command_manager=NS(get_command=lambda name:commands))
    stand=ns['stand_still_joint_deviation_l1'](env,'base_velocity',.1)*-.5
    assert torch.allclose(stand,torch.tensor([-.6,-.6,-.6,0.]))
    stationary=ns['track_lin_vel_xy_exp'](env,math.sqrt(.5),'base_velocity')*4
    yaw_stationary=ns['track_ang_vel_z_exp'](env,math.sqrt(.5),'base_velocity')*1.5
    # Exact pair synchronization helper on a no-liftoff contact history.
    contact=NS(data=NS(current_air_time=torch.zeros(1,4),current_contact_time=torch.full((1,4),10.)))
    gait=object.__new__(ns['GaitReward']); gait.contact_sensor=contact; gait.std=math.sqrt(.5); gait.max_err=.2
    sync=gait._sync_reward_func(0,3)*gait._sync_reward_func(1,2)
    async_product=gait._async_reward_func(0,1)*gait._async_reward_func(3,2)*gait._async_reward_func(0,2)*gait._async_reward_func(1,3)
    static_gait=float((sync*async_product*.5).item())
    assert static_gait>.3
    # A 0.2 s swing is below the configured 0.5 s landing threshold.
    landing=torch.tensor([[True,False,False,False]]).repeat(4,1)
    sensor=NS(data=NS(last_air_time=torch.full((4,4),.2)),compute_first_contact=lambda dt:landing)
    env.scene['feet']=sensor; env.step_dt=.02
    air=ns['feet_air_time_lin_xy_cmd'](env,'base_velocity',Entity('feet',body_ids=list(range(4))),.5)*5
    assert abs(float(air[-1])+1.5)<1e-6
    air_yaw=ns['feet_air_time_ang_z_cmd_lite3'](env,'base_velocity',Entity('feet',body_ids=list(range(4))),.5)*5
    assert torch.allclose(air_yaw,torch.tensor([0.,-1.5,-1.5,0.]))
    landing[:]=False
    no_liftoff=ns['feet_air_time_lin_xy_cmd'](env,'base_velocity',Entity('feet',body_ids=list(range(4))),.5)*5
    assert torch.all(no_liftoff==0)
    # Fixed foot positions across a whole cycle; command direction/height absent.
    n=400
    foot=torch.tensor([[.2,.16,-.32],[.2,-.16,-.32],[-.15,.16,-.32],[-.15,-.16,-.32]])
    static=NS(body_pos_w=foot.repeat(n,1,1),root_pos_w=torch.zeros(n,3),root_quat_w=torch.tensor([1.,0.,0.,0.]).repeat(n,1),
              body_lin_vel_w=torch.zeros(n,4,3),root_lin_vel_w=torch.zeros(n,3))
    cmd=torch.tensor([.4,0.,0.]).repeat(n,1)
    phase_env=NS(num_envs=n,device='cpu',step_dt=.425/n,episode_length_buf=torch.arange(n),
                 scene={'robot':NS(data=static)},command_manager=NS(get_command=lambda _:cmd))
    saved=json.loads((ROOT/'logs/reference_six/training/scratch_seed42_10000/configuration_audit.json').read_text())['candidate_environment']
    params=copy.deepcopy(saved['rewards']['phase_foot_trajectory_exp']['params'])
    params['asset_cfg']=Entity('robot',body_ids=list(range(4)))
    phase_forward=ns['phase_foot_trajectory_exp'](phase_env,**params)*2
    cmd[:]=torch.tensor([0.,0.,.5])
    phase_turn=ns['phase_foot_trajectory_exp'](phase_env,**params)*2
    assert torch.equal(phase_forward,phase_turn)
    cached=phase_env.phase_foot_ref_body.clone()
    static.body_pos_w[:,:,2]+=.1
    ns['phase_foot_trajectory_exp'](phase_env,**params)
    assert torch.equal(cached,phase_env.phase_foot_ref_body)
    levels=[0.,.01,1.,2.,2.97583,3.]
    scales={str(x):ns['update_gait_level_from_terrain_mean'](x) for x in levels}
    urdf=ROOT/'deep_robotics_model/Lite3/Lite3_urdf/urdf/Lite3.urdf'
    tree=ET.parse(urdf).getroot()
    joints={joint.get('name'):joint for joint in tree.findall('joint')}
    def offset(name): return [float(v) for v in joints[name].find('origin').get('xyz').split()]
    hip_offset=offset('FL_HipY_joint')[1]
    upper=abs(offset('FL_Knee_joint')[2])
    foot_joint=next(j for j in joints.values() if j.find('child').get('link')=='FL_FOOT')
    lower=abs(float(foot_joint.find('origin').get('xyz').split()[2]))
    foot=tree.find("link[@name='FL_FOOT']/collision/geometry/sphere")
    radius=float(foot.get('radius'))
    # Level torso: hip abduction rotates a lateral offset perpendicular to
    # the sagittal leg chain. This ignores joint limits, so is an upper bound.
    max_height=math.sqrt(hip_offset**2+(upper+lower)**2)+radius
    assert max_height<.55
    command_audit,command_sources=probe_commands(saved['commands']['base_velocity'])
    sources.update(command_sources)
    return dict(status='passed',sources=sources,command_sampling=command_audit,
        standstill=dict(commands=commands.tolist(),weighted_penalties=stand.tolist(),
                        finding='Yaw commands are ignored by the active standing penalty gate.'),
        stationary_tracking=dict(linear_rates=stationary.tolist(),yaw_rates=yaw_stationary.tolist()),
        static_gait=dict(weighted_rate=static_gait,full_weight=.5,ratio=static_gait/.5,
                         finding='Error clipping leaves a positive reward for long static contact.'),
        air_time=dict(short_swing_s=.2,threshold_s=.5,landing_rates=air.tolist(),
                      yaw_landing_rates=air_yaw.tolist(),no_liftoff_rates=no_liftoff.tolist()),
        static_phase=dict(mean_weighted_rate=float(phase_forward.mean()),minimum=float(phase_forward.min()),
                          maximum=float(phase_forward.max()),full_weight=2.,forward_turn_exactly_equal=True,
                          reference_unchanged_after_pose_change=True,params={k:v for k,v in params.items() if k!='asset_cfg'}),
        global_curriculum_scales=scales,
        level_standing_height=dict(source=str(urdf),source_sha256=sha(urdf),hip_lateral_offset=hip_offset,
            upper_leg=upper,lower_leg=lower,foot_radius=radius,
            loose_upper_bound_m=max_height,target_height_m=.55,
            assumptions='Level torso, level ground, at least one supporting foot; ignore joint limits. Not a bound on airborne or pitched states.'),
        caveat='These reproduce local function behavior. They do not prove a unique cause of trained behavior or predict outcomes after reward changes.')


def probe_commands(cfg):
    ns=dict(torch=torch,CommandTerm=object,
            math_utils=NS(wrap_to_pi=lambda x:(x+math.pi)%(2*math.pi)-math.pi))
    sources=definitions(COMMAND_BASE,['UniformVelocityCommand'],ns)
    ns['mdp']=NS(UniformVelocityCommand=ns['UniformVelocityCommand'])
    sources.update(definitions(COMMAND,['UniformThresholdVelocityCommand'],ns))
    term=object.__new__(ns['UniformThresholdVelocityCommand'])
    n=100000; term.device='cpu'; term.num_envs=n
    term.cfg=NS(**{k:v for k,v in cfg.items() if k not in ('ranges',)})
    term.cfg.ranges=NS(**cfg['ranges'])
    term.vel_command_b=torch.zeros(n,3); term.heading_target=torch.zeros(n)
    term.is_heading_env=torch.zeros(n,dtype=torch.bool); term.is_standing_env=torch.zeros(n,dtype=torch.bool)
    term.robot=NS(data=NS(heading_w=torch.zeros(n)))
    term._full_ranges_lin_vel_x=cfg['ranges']['lin_vel_x']; term._full_ranges_lin_vel_y=cfg['ranges']['lin_vel_y']
    torch.manual_seed(20261114); term._resample_command(torch.arange(n)); term._update_command()
    cmd=term.command
    pure=(cmd[:,:2].norm(dim=1)<1e-8)&(cmd[:,2].abs()>1e-8)
    low=(cmd[:,:2].norm(dim=1)>0)&(cmd[:,:2].norm(dim=1)<=.2)
    nonheading_yaw=(~term.is_heading_env)&(cmd[:,2].abs()>1e-8)
    result=dict(samples=n,pure_yaw=int(pure.sum()),pure_yaw_heading_controlled=int((pure&term.is_heading_env).sum()),
                constant_yaw_nonheading=int(nonheading_yaw.sum()),low_nonzero_xy_after_special_overrides=int(low.sum()),
                standing=int((cmd.norm(dim=1)==0).sum()))
    assert result['constant_yaw_nonheading']==0 and result['pure_yaw_heading_controlled']==result['pure_yaw']
    assert result['low_nonzero_xy_after_special_overrides']>0
    return result,sources


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=audit(); save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
