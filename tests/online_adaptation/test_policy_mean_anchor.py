import copy

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from rl_training.policy_mean_anchor import MeanActionReplay, update_with_mean_anchor


def fixture(normalize=False):
    torch.manual_seed(47)
    obs=TensorDict({'policy':torch.randn(8,48),'critic':torch.randn(8,51)},batch_size=[8])
    groups={'actor':['policy'],'critic':['critic']}
    actor=MLPModel(obs,groups,'actor',12,hidden_dims=[16,8],distribution_cfg={
        'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.05,'std_type':'log'})
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=[16,8])
    storage=RolloutStorage('rl',8,4,obs,[12],device='cpu')
    algorithm=PPO(actor,critic,storage,num_learning_epochs=2,num_mini_batches=2,schedule='fixed',
                  learning_rate=3e-6,normalize_advantage_per_mini_batch=normalize,device='cpu')
    with torch.no_grad():
        for _ in range(4):
            algorithm.act(obs)
            algorithm.process_env_step(obs,torch.randn(8),torch.zeros(8,dtype=torch.long),{})
        algorithm.compute_returns(obs)
    replay=MeanActionReplay(copy.deepcopy(actor.mlp),{'flat':obs['policy'].clone()},{'flat':8},
                            coefficient=0.,seed=97,device='cpu')
    return algorithm,replay


@pytest.mark.parametrize('normalize',[False,True])
def test_zero_coefficient_matches_installed_ppo_parameters_losses_and_rng(normalize):
    original,_=fixture(normalize)
    extended,replay=fixture(normalize)
    torch.manual_seed(53)
    expected=original.update()
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(53)
    actual=update_with_mean_anchor(extended,replay)
    assert torch.equal(expected_rng,torch.get_rng_state())
    assert replay.calls==0
    for key in expected:assert actual[key]==expected[key]
    for before,after in ((original.actor,extended.actor),(original.critic,extended.critic)):
        for key,value in before.state_dict().items():
            torch.testing.assert_close(value,after.state_dict()[key],atol=0,rtol=0)


def test_mean_loss_has_actor_gradient_but_no_teacher_or_exploration_gradient():
    algorithm,replay=fixture()
    actor=algorithm.actor
    teacher_before=copy.deepcopy(replay.teacher.state_dict())
    replay.observations['flat'].zero_()
    with torch.no_grad():
        [layer for layer in actor.mlp.modules() if isinstance(layer,torch.nn.Linear)][-1].bias.add_(.1)
    start=float(replay.loss(actor).detach())
    optimizer=torch.optim.Adam(actor.parameters(),lr=.001)
    for _ in range(5):
        optimizer.zero_grad()
        loss=replay.loss(actor)
        loss.backward()
        assert actor.distribution.log_std_param.grad is None
        assert all(p.grad is None for p in replay.teacher.parameters())
        optimizer.step()
    assert float(replay.loss(actor).detach())<start
    for key,value in teacher_before.items():assert torch.equal(value,replay.teacher.state_dict()[key])


def test_unsupported_ppo_features_fail_instead_of_silently_changing_algorithm():
    algorithm,replay=fixture()
    algorithm.schedule='adaptive'
    with pytest.raises(ValueError,match='fixed-LR'):
        update_with_mean_anchor(algorithm,replay)
