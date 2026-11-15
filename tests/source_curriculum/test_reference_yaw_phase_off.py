import copy
from types import SimpleNamespace as NS

import pytest
import torch

from reference_low_stairs_yaw_phase_off_cfg import gate_phase, phase_outside_stair_translation_and_pure_yaw
from reference_low_stairs_yaw_phase_off_train import expected_configuration, checked_configuration, directory


def test_gate_exact_boundaries_both_signs_and_translation():
    cmd=torch.tensor([[0,0,.2],[0,0,-.2],[0,0,.1],[0,0,-.1],[.1,0,.8],
                      [0,.1,-.8],[.06,.06,.8],[.08,.08,.8],[0,0,0]],dtype=torch.float64)
    original=torch.arange(1,10,dtype=torch.float64)
    value,mask=gate_phase(original,cmd)
    assert mask.tolist()==[True,True,False,False,False,False,True,False,False]
    assert torch.equal(value[~mask],original[~mask]) and torch.count_nonzero(value[mask])==0


def test_calls_original_once_even_if_every_reward_is_gated(monkeypatch):
    import reference_low_stairs_rewards as rewards
    calls=[];reference=torch.zeros(1,4,3)
    def original(env,**kw):
        calls.append(kw)
        if not hasattr(env,'phase_foot_ref_body'):env.phase_foot_ref_body=reference
        return torch.tensor([.5,.75])
    monkeypatch.setattr(rewards,'phase_outside_stair_translation',original)
    env=NS(common_step_counter=1,command_manager=NS(get_command=lambda _:torch.tensor([[0,0,.8],[0,0,-.8]])))
    for _ in range(2):assert not phase_outside_stair_translation_and_pure_yaw(env,'base_velocity',None).any()
    assert len(calls)==2 and env.phase_foot_ref_body is reference
    assert calls[0]['stance_span']==.2 and calls[0]['cycle_time']==.4


@pytest.mark.parametrize('field',['reward','optimizer','terrain'])
def test_configuration_rejects_unapproved_changes(monkeypatch,field):
    import reference_low_stairs_yaw_phase_off_train as module
    monkeypatch.setattr(module,'save',lambda *a:None)
    good=expected_configuration();assert checked_configuration(good)==good
    bad=copy.deepcopy(good)
    if field=='reward':bad['environment']['rewards']['stair_clearance']['weight']*=2
    elif field=='optimizer':bad['agent']['algorithm']['learning_rate']*=2
    else:bad['environment']['scene']['terrain']['max_init_terrain_level']=1
    with pytest.raises(AssertionError):checked_configuration(bad)


@pytest.mark.parametrize('phase',[-1,2,100])
def test_budget_disallows_additional_training_phase(phase):
    with pytest.raises(ValueError):directory(phase)
