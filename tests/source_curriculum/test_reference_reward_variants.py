"""Reward semantics and scope checks independent of simulator startup."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
import reference_reward_variants as variants
from reference_reward_resume import equal_state


class Config:
    def __init__(self):
        self.rewards=NS(
            stand_still=NS(func='original_stand',weight=-.5,params=dict(command_name='base_velocity',
                asset_cfg=NS(name='robot',joint_ids=slice(None)),command_threshold=.1)),
            base_height_l2=NS(func='original_height',weight=-50.,params=dict(target_height=.55,sensor_cfg='base_scan')),
            phase_foot_trajectory_exp=NS(func='original_phase',weight=2.,params=dict(velocity_weight=0.)),
            feet_gait=NS(func='original_gait',weight=.5,params=dict(max_err=.2)),
            feet_air_time_lin_xy=NS(func='original_linear_air',weight=5.,params=dict(threshold=.5)),
            feet_air_time_ang_z_lite3=NS(func='original_yaw_air',weight=5.,params=dict(threshold=.5)),
            track_lin_vel_xy_exp=NS(func='original_tracking',weight=4.,params=dict(std=.707)))
        self.commands=dict(heading=True,range=[-1.5,1.5])
        self.scene=dict(terrain='six',robot='Lite3')
        self.observations=dict(actor=45,critic=393)
    def to_dict(self):
        def serialize(value):
            if isinstance(value,NS): return serialize(vars(value))
            if isinstance(value,dict): return {k:serialize(v) for k,v in value.items()}
            if callable(value): return f'{value.__module__}:{value.__name__}'
            return copy.deepcopy(value)
        return serialize(vars(self))


def test_standing_gate_keeps_selected_joints_and_requires_both_commands_small():
    commands=torch.tensor([[0.,0.,0.],[0.,0.,.2],[0.,0.,-.2],[.2,0.,0.],[0.,-.2,0.],
                           [.06,.06,.09],[0.,0.,.1],[0.,0.,-.1],[.1,0.,0.]])
    n=len(commands)
    data=NS(joint_pos=torch.tensor([[99.,.3,99.,-.2]]).repeat(n,1),default_joint_pos=torch.zeros(n,4))
    env=NS(scene={'robot':NS(data=data)},command_manager=NS(get_command=lambda _:commands))
    value=variants.stand_still_joint_deviation_se2(env,'base_velocity',NS(name='robot',joint_ids=[1,3]))
    assert torch.allclose(value,torch.tensor([.5,0.,0.,0.,0.,.5,0.,0.,0.]))


def test_short_landings_cannot_cancel_long_landings_or_pay_while_airborne():
    sensor=NS(data=NS(last_air_time=torch.tensor([[.2,.8,.9,8.]])),
              compute_first_contact=lambda _:torch.tensor([[True,True,False,False]]))
    reward=variants.nonnegative_landing_reward(sensor,[0,1,2],.02,.5)
    assert reward.item()==pytest.approx(.3)
    sensor.compute_first_contact=lambda _:torch.zeros(1,4,dtype=torch.bool)
    assert variants.nonnegative_landing_reward(sensor,slice(None),.02,.5).item()==0.


def test_air_time_preserves_linear_yaw_gates_weights_and_curriculum(monkeypatch):
    commands=torch.tensor([[0.,0.,0.],[.4,0.,0.],[0.,0.,.5],[0.,0.,-.5],[.4,0.,.5]])
    sensor=NS(data=NS(last_air_time=torch.full((5,4),.7)),
              compute_first_contact=lambda _:torch.ones(5,4,dtype=torch.bool))
    env=NS(scene=NS(sensors={'feet':sensor}),step_dt=.02,command_manager=NS(get_command=lambda _:commands))
    monkeypatch.setattr(variants,'_gait_scale',lambda _:torch.tensor(.25))
    cfg=NS(name='feet',body_ids=[0,2])
    linear=variants.feet_air_time_lin_xy_nonnegative(env,'base_velocity',cfg,.5)
    yaw=variants.feet_air_time_yaw_nonnegative(env,'base_velocity',cfg,.5)
    assert torch.allclose(linear,torch.tensor([0.,.1,0.,0.,.1]))
    assert torch.allclose(yaw,torch.tensor([0.,0.,.1,.1,.1]))


@pytest.mark.parametrize('name',tuple(variants.VARIANTS))
def test_variants_are_isolated_and_change_only_the_declared_terms(name):
    original=Config();before=variants.plain(original)
    cfg,audit=variants.apply_reward_variant(original,name)
    assert variants.plain(original)==before
    assert set(audit['changes'])==set(variants.VARIANTS[name])
    for key in ('commands','scene','observations'):
        assert cfg.to_dict()[key]==original.to_dict()[key]
    if name=='reference': assert variants.plain(cfg)==before
    else:
        for term in ('stand_still','base_height_l2','feet_air_time_lin_xy','feet_air_time_ang_z_lite3'):
            assert getattr(cfg.rewards,term).weight==getattr(original.rewards,term).weight
    # Selecting a variant must not affect a later copy of the reference preset.
    fresh,_=variants.apply_reward_variant(original,'reference')
    assert variants.plain(fresh)==before


def test_core_keeps_gait_terms_and_height_function_unchanged():
    cfg,audit=variants.apply_reward_variant(Config(),'core')
    assert cfg.rewards.base_height_l2.params['target_height']==.32
    assert cfg.rewards.base_height_l2.func=='original_height'
    assert cfg.rewards.phase_foot_trajectory_exp.weight==2.
    assert cfg.rewards.feet_gait.weight==.5
    assert cfg.rewards.feet_air_time_lin_xy.func=='original_linear_air'
    assert cfg.rewards.stand_still.params['linear_threshold']==.1
    assert cfg.rewards.stand_still.params['angular_threshold']==.1


def test_unknown_or_changed_baselines_fail():
    with pytest.raises(ValueError,match='unknown'):
        variants.apply_reward_variant(Config(),'typo')
    cfg=Config();cfg.rewards.base_height_l2.params['target_height']=.4
    with pytest.raises(ValueError,match='height baseline'):
        variants.apply_reward_variant(cfg,'core')


def test_resume_state_comparison_covers_optimizer_moments_and_groups():
    state={'state':{0:{'step':torch.tensor(200000.),'exp_avg':torch.tensor([.1,.2])}},
           'param_groups':[{'lr':.0002562890625,'params':[0]}]}
    assert equal_state(state,copy.deepcopy(state))
    changed=copy.deepcopy(state);changed['state'][0]['exp_avg'][0]=0.
    assert not equal_state(state,changed)
    changed=copy.deepcopy(state);changed['param_groups'][0]['lr']=.001
    assert not equal_state(state,changed)


@pytest.mark.parametrize('extra',[
    ['--reward-variant','reference','--updates','1'],
    ['--reward-variant','core','--updates','501'],
    ['--reward-variant','core'],
])
def test_resume_cli_rejects_original_rewards_and_unbounded_training(extra,monkeypatch,tmp_path):
    from reference_six_train import main
    monkeypatch.setattr(sys,'argv',['reference_six_train.py','--mode','train','--resume-final',
                                  '--output',str(tmp_path/'unstarted'),*extra])
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code==2
    assert not (tmp_path/'unstarted').exists()
