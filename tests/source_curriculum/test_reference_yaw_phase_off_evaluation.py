import copy
from types import SimpleNamespace as NS

import pytest
import torch

from reference_low_stairs_yaw_phase_off_evaluate import protocol_equal
from reference_low_stairs_yaw_phase_off_cfg import RuntimeAudit


@pytest.mark.parametrize('field',['physics','sources','action_mode'])
def test_pairing_rejects_unapproved_protocol_differences(field):
    a=dict(model='old',checkpoint='a',action_mode='mean',action_std=[.1],configuration={'physics':1},sources=[])
    b=dict(a,model='new',checkpoint='b',action_std=[.2])
    assert protocol_equal(a,b)['effective_config_exact']
    b=copy.deepcopy(b)
    if field=='physics':b['configuration']['physics']=2
    elif field=='sources':b['sources']=[dict(path='/unapproved.py',sha256='a'*64)]
    else:b['action_mode']='sampled'
    with pytest.raises(AssertionError):protocol_equal(a,b)


def test_runtime_checks_same_state_gate_and_reward_sum():
    rates=torch.tensor([[1.,0.,-.5],[2.,1.5,-.25]])
    manager=NS(active_terms=['track','phase_foot_trajectory_exp','cost'],_step_reward=rates,
               get_term_cfg=lambda _:NS(weight=2.))
    base=NS(reward_manager=manager,num_envs=2,common_step_counter=3,phase_foot_ref_body=torch.zeros(1,4,3),
            _yaw_phase_off_sample=dict(tick=3,original=torch.tensor([.5,.75]),value=torch.tensor([0.,.75]),mask=torch.tensor([True,False])))
    audit=RuntimeAudit(base);audit.capture(rates.sum(-1)*.02,.02)
    assert audit.summary()['removed_nonzero_frames']==1
    with pytest.raises(AssertionError):audit.capture(rates.sum(-1)*.02+.1,.02)
    base._yaw_phase_off_sample['tick']=2
    with pytest.raises(AssertionError):audit.capture(rates.sum(-1)*.02,.02)
