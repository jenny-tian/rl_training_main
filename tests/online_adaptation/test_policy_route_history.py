import copy

import pytest
import torch
from tensordict import TensorDict

from rl_training.policy_route_history import RouteHistoryTracker, RouteHistoryVecEnv, expand_actor_history_inputs, insert_route_history, strip_route_history
from rl_training.online_adaptation.flowdagger import FrozenPPOExpert, _mlp
from rl_training.policy_flat_anchor import FrozenFlatMean


def options(enabled=True):
    return dict(enabled=enabled,dt=.02,lateral_scale_m=1.65,speed_scale_m_s=.3,clip_m_s=1.)


def test_partial_reset_causal_difference_reflection_clamp_and_zero_control():
    tracker=RouteHistoryTracker(options())
    obs=torch.zeros(3,48);obs[:,9]=torch.tensor([.2,-.2,.5])
    first=tracker.advance(obs,torch.zeros(3,dtype=torch.bool))
    assert not first[:,12].any() and tracker.snapshot['history_reset'].all()
    obs[:,9]+=torch.tensor([.002,-.002,100.])
    second=tracker.advance(obs,torch.tensor([False,False,True]))
    torch.testing.assert_close(second[:,12],torch.tensor([.55,-.55,0.]),atol=1e-5,rtol=0)
    obs[:,9]+=torch.tensor([10.,-10.,.002])
    third=tracker.advance(obs,torch.zeros(3,dtype=torch.bool))
    torch.testing.assert_close(third[:2,12],torch.tensor([1/.3,-1/.3]))
    assert abs(third[2,12])<1.  # Previous reset starts a fresh difference, not a jump across episodes.
    control=RouteHistoryTracker(options(False));control.advance(torch.zeros_like(obs),torch.ones(3,dtype=torch.bool))
    assert not control.advance(obs,torch.zeros(3,dtype=torch.bool))[:,12].any()
    assert control.snapshot['history_velocity'].abs().max()==1
    torch.testing.assert_close(strip_route_history(third),obs,atol=0,rtol=0)
    torch.testing.assert_close(third[:,-12:],obs[:,-12:],atol=0,rtol=0)


def test_wrapper_repeated_reads_do_not_advance_and_critic_is_unchanged():
    class Env:
        num_envs=2
        def __init__(self):self.steps=0
        def get_observations(self):
            obs=torch.zeros(2,48);obs[:,9]=self.steps*.002
            return TensorDict({'policy':obs,'critic':torch.ones(2,51)*self.steps},batch_size=[2])
        def step(self,action):
            self.steps+=1
            return self.get_observations(),torch.zeros(2),torch.tensor([False,self.steps==1]),{}
    env=Env();wrapper=RouteHistoryVecEnv(env,options());rng=torch.get_rng_state().clone()
    initial=wrapper.get_observations();again=wrapper.get_observations()
    assert initial is again and env.steps==0
    obs,_,done,_=wrapper.step(torch.zeros(2,12))
    torch.testing.assert_close(obs['policy'][:,12],torch.tensor([.55,0.]),atol=1e-6,rtol=0)
    snapshot=wrapper.history_snapshot();wrapper.get_observations();wrapper.get_observations()
    for key,value in snapshot.items():torch.testing.assert_close(value,wrapper.history_snapshot()[key],atol=0,rtol=0)
    assert env.steps==1 and obs['critic'].shape==(2,51) and torch.equal(obs['critic'],torch.ones(2,51))
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('bad',[{'dt':0.},{'speed_scale_m_s':float('nan')},{'clip_m_s':-1.},{'enabled':1}])
def test_invalid_history_options_fail_before_physics(bad):
    with pytest.raises(ValueError):RouteHistoryTracker(dict(options(),**bad))


def test_zero_input_expansion_preserves_mapping_and_original_teacher_targets():
    torch.manual_seed(43)
    original={'actor_state_dict':{'mlp.'+k:v for k,v in _mlp((48,512,256,128,12)).state_dict().items()}}
    saved=copy.deepcopy(original)
    rng=torch.get_rng_state().clone();expanded=expand_actor_history_inputs(original)
    assert torch.equal(rng,torch.get_rng_state())
    states=torch.randn(64,48);history=torch.linspace(-3.,3.,64)
    extended=insert_route_history(states,history)
    old=FrozenPPOExpert(original['actor_state_dict']);new=FrozenPPOExpert(expanded['actor_state_dict'])
    torch.testing.assert_close(old(states),new(extended),atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(old(states),old(extended),atol=0,rtol=0)
    teacher=FrozenFlatMean(original)
    torch.testing.assert_close(teacher(states),teacher(extended),atol=0,rtol=0)
    for key,value in saved['actor_state_dict'].items():torch.testing.assert_close(value,original['actor_state_dict'][key],atol=0,rtol=0)
    assert not expanded['actor_state_dict']['mlp.0.weight'][:,12].any()


def test_history_column_can_receive_gradient_without_teacher_changes():
    torch.manual_seed(19)
    base={'actor_state_dict':{'mlp.'+k:v for k,v in _mlp((48,512,256,128,12)).state_dict().items()}}
    actor=FrozenPPOExpert(expand_actor_history_inputs(base)['actor_state_dict']).requires_grad_(True)
    states=insert_route_history(torch.randn(32,48),torch.ones(32))
    actor(states).square().mean().backward()
    assert actor.mlp[0].weight.grad[:,12].abs().sum()>0


def test_rsl49_rollout_ppo_and48_reference_replay_integrate():
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from rsl_rl.algorithms import PPO
    from rl_training.policy_mean_anchor import MeanActionReplay, update_with_mean_anchor
    torch.manual_seed(83)
    obs=TensorDict({'policy':torch.randn(8,49),'critic':torch.randn(8,51)},batch_size=[8])
    groups={'actor':['policy'],'critic':['critic']}
    actor=MLPModel(obs,groups,'actor',12,hidden_dims=[16,8],distribution_cfg={'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.05,'std_type':'log'})
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=[16,8])
    storage=RolloutStorage('rl',8,4,obs,[12],device='cpu')
    algorithm=PPO(actor,critic,storage,num_learning_epochs=2,num_mini_batches=2,schedule='fixed',learning_rate=3e-6,device='cpu')
    with torch.no_grad():
        for _ in range(4):
            algorithm.act(obs);algorithm.process_env_step(obs,torch.randn(8),torch.zeros(8,dtype=torch.long),{})
        algorithm.compute_returns(obs)
    reference=FrozenPPOExpert({'mlp.'+k:v for k,v in _mlp((48,512,256,128,12)).state_dict().items()})
    replay=MeanActionReplay(reference,{'correction':obs['policy']},{'correction':8},coefficient=100.,seed=7,device='cpu')
    records=[];loss=update_with_mean_anchor(algorithm,replay,gradient_observer=records.append)
    assert len(records)==1 and replay.calls==4 and replay.rows==32
    assert all(torch.isfinite(torch.tensor(v)) for v in loss.values())
    assert all(p.grad is None for p in reference.parameters())
