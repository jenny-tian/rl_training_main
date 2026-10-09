import copy
from types import SimpleNamespace

import pytest
import torch

from rl_training.policy_learning import warmstart_actor_only


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=torch.nn.Linear(3,2)
        self.distribution=torch.nn.Module()
        self.distribution.log_std_param=torch.nn.Parameter(torch.zeros(2))


def fixture():
    actor,critic=Actor(),torch.nn.Linear(3,1)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()))
    runner=SimpleNamespace(alg=SimpleNamespace(actor=actor,critic=critic,optimizer=optimizer),current_learning_iteration=0)
    payload={'actor_state_dict':copy.deepcopy(Actor().state_dict()),'critic_state_dict':{'unrelated':torch.ones(7)},
             'optimizer_state_dict':{'unrelated':1},'iter':999}
    return runner,payload


def test_actor_mean_copied_without_critic_optimizer_rng_or_checkpoint_mutation():
    runner,payload=fixture();saved=copy.deepcopy(payload);critic=copy.deepcopy(runner.alg.critic.state_dict())
    rng=torch.get_rng_state().clone()
    report=warmstart_actor_only(runner,payload,initial_std=.2)
    assert report['mean_state_preserved'] and not report['critic_loaded'] and not report['optimizer_loaded']
    assert torch.equal(rng,torch.get_rng_state())
    assert not runner.alg.optimizer.state and runner.current_learning_iteration==0
    for k,v in critic.items():assert torch.equal(v,runner.alg.critic.state_dict()[k])
    for k,v in saved['actor_state_dict'].items():
        assert torch.equal(v,payload['actor_state_dict'][k])
        if k.startswith('mlp.'):assert torch.equal(v,runner.alg.actor.state_dict()[k])
    torch.testing.assert_close(runner.alg.actor.distribution.log_std_param.exp(),torch.full((2,),.2))


@pytest.mark.parametrize('failure',['shape','nan','std','std_overflow','std_underflow','iteration','optimizer'])
def test_invalid_initialization_rejected_before_actor_changes(failure):
    runner,payload=fixture();before=copy.deepcopy(runner.alg.actor.state_dict());std=None
    if failure=='shape':payload['actor_state_dict']['mlp.weight']=torch.ones(2,4)
    elif failure=='nan':payload['actor_state_dict']['mlp.weight'][0,0]=float('nan')
    elif failure=='std':std=-1
    elif failure=='std_overflow':std=1e300
    elif failure=='std_underflow':std=1e-300
    elif failure=='iteration':runner.current_learning_iteration=1
    else:runner.alg.optimizer.state[next(runner.alg.actor.parameters())]={'step':1}
    with pytest.raises(ValueError):warmstart_actor_only(runner,payload,initial_std=std)
    for k,v in before.items():assert torch.equal(v,runner.alg.actor.state_dict()[k])


def test_mean_only_checkpoint_explicitly_initializes_only_exploration():
    runner,payload=fixture()
    del payload['actor_state_dict']['distribution.log_std_param']
    saved=copy.deepcopy(payload['actor_state_dict'])
    critic=copy.deepcopy(runner.alg.critic.state_dict())
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        warmstart_actor_only(runner,payload,initial_std=.05)
    with pytest.raises(ValueError,match='explicit exploration'):
        warmstart_actor_only(runner,payload,allow_mean_only=True)
    report=warmstart_actor_only(runner,payload,initial_std=.05,allow_mean_only=True)
    assert report['distribution_initialized_from_mean_only']
    assert torch.equal(rng,torch.get_rng_state()) and not runner.alg.optimizer.state
    assert payload['actor_state_dict'].keys()==saved.keys()
    for k,v in saved.items():
        assert torch.equal(v,payload['actor_state_dict'][k]) and torch.equal(v,runner.alg.actor.state_dict()[k])
    for k,v in critic.items():assert torch.equal(v,runner.alg.critic.state_dict()[k])
    torch.testing.assert_close(runner.alg.actor.distribution.log_std_param.exp(),torch.full((2,),.05))
