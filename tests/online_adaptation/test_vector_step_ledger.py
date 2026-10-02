import json
import pytest
from rl_training.vector_step_ledger import VectorStepLedger


def test_vector_steps_charge_every_environment_and_cannot_exceed_budget(tmp_path):
    p=tmp_path/'ledger.json';ledger=VectorStepLedger(p,num_envs=8,max_transitions=16,metadata={})
    for _ in range(2):ledger.reserve();ledger.confirm()
    with pytest.raises(ValueError,match='exceeds'):ledger.reserve()
    ledger.finish()
    s=json.loads(p.read_text());assert s['charged']==s['confirmed']==16 and s['uncertain_transitions']==0
    with pytest.raises(ValueError,match='exists'):VectorStepLedger(p,num_envs=8,max_transitions=16,metadata={})


def test_failed_step_keeps_whole_batch_charged_as_uncertain(tmp_path):
    ledger=VectorStepLedger(tmp_path/'ledger.json',num_envs=8,max_transitions=16,metadata={})
    ledger.reserve()
    with pytest.raises(ValueError):ledger.reserve()
    with pytest.raises(ValueError):ledger.finish()
    ledger.finish(aborted=True)
    assert ledger.state['charged']==8 and ledger.state['confirmed']==0 and ledger.state['uncertain_transitions']==8


def test_proxy_charges_before_environment_and_preserves_uncertain_failure(tmp_path):
    from rl_training.vector_step_ledger import ChargedVecEnv
    ledger=VectorStepLedger(tmp_path/'ledger.json',num_envs=8,max_transitions=16,metadata={})
    class Env:
        num_envs=8
        def get_observations(self):return 'obs'
        def step(self,action):
            assert ledger.state['charged']==8 and ledger.state['confirmed']==0
            raise RuntimeError('physics failed')
    proxy=ChargedVecEnv(Env(),ledger)
    assert proxy.get_observations()=='obs'
    with pytest.raises(RuntimeError):proxy.step('action')
    ledger.finish(aborted=True)
    assert ledger.state['uncertain_transitions']==8


def test_proxy_callbacks_observe_confirmed_steps_without_extra_calls(tmp_path):
    from rl_training.vector_step_ledger import ChargedVecEnv
    ledger=VectorStepLedger(tmp_path/'ledger.json',num_envs=1,max_transitions=1,metadata={})
    class Env:
        num_envs=1
        def get_observations(self):return 'initial'
        def step(self,action):return ('next',2,False,{})
    records=[]
    proxy=ChargedVecEnv(Env(),ledger,before_step=lambda obs,act:(obs,act),
                        after_step=lambda pre,result:records.append((pre,result,ledger.state['confirmed'])))
    proxy.get_observations();assert proxy.step('action')[0]=='next'
    assert records==[(('initial','action'),('next',2,False,{}),1)]


def test_emitted_clipping_preserves_requested_action_for_ppo_likelihood(tmp_path):
    import torch
    from rl_training.vector_step_ledger import ChargedVecEnv
    from rl_training.lite3_joint_conventions import clip_lite3_policy_action
    ledger=VectorStepLedger(tmp_path/'ledger.json',num_envs=1,max_transitions=1,metadata={})
    received=[]
    class Env:
        num_envs=1
        def get_observations(self):return None
        def step(self,action):
            received.append(action.clone());return (None,0,False,{})
    requested=torch.full((1,12),20.);original=requested.clone()
    proxy=ChargedVecEnv(Env(),ledger,action_transform=clip_lite3_policy_action)
    proxy.step(requested)
    assert torch.equal(requested,original)
    assert torch.equal(received[0],clip_lite3_policy_action(original))
    assert ledger.state['confirmed']==1
