import copy
import hashlib

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


def test_demo_targets_follow_sampled_rows_and_override_only_correction():
    class Teacher(torch.nn.Module):
        def forward(self,x):return x[:,:12]
    class Actor(torch.nn.Module):
        obs_normalization=False
        obs_groups=['policy']
        obs_dim=48
        def __init__(self):
            super().__init__()
            self.bias=torch.nn.Parameter(torch.zeros(()))
        def forward(self,obs):return obs['policy'][:,:12]+self.bias
    states={'correction':torch.arange(3).float()[:,None].expand(3,48)/10,
            'flat':torch.arange(5).float()[:,None].expand(5,48)/10}
    target=(states['correction'][:,:12]+.25).clone().requires_grad_()
    replay=MeanActionReplay(Teacher(),states,{'correction':17,'flat':19},coefficient=1.,seed=313,device='cpu',fixed_targets={'correction':target})
    actor=Actor()
    loss=replay.loss(actor)
    torch.testing.assert_close(loss,torch.tensor(.25**2*17/36))
    loss.backward()
    assert actor.bias.grad<0 and target.grad is None
    generator=torch.Generator().manual_seed(313)
    indices=torch.randint(3,(17,),generator=generator)
    expected=hashlib.sha256(target.detach()[indices].numpy().tobytes()).hexdigest()
    assert replay.fixed_target_rows==17 and replay.fixed_target_digest.hexdigest()==expected
    torch.testing.assert_close(target.detach(),states['correction'][:,:12]+.25)


@pytest.mark.parametrize('failure',['unknown','rows','nan'])
def test_invalid_fixed_target_alignment_is_rejected(failure):
    observations={'correction':torch.zeros(3,48)}
    targets={'correction':torch.zeros(3,12)}
    if failure=='unknown':targets={'validation':torch.zeros(3,12)}
    elif failure=='rows':targets['correction']=torch.zeros(2,12)
    else:targets['correction'][0,0]=float('nan')
    with pytest.raises(ValueError):
        MeanActionReplay(torch.nn.Identity(),observations,{'correction':1},coefficient=1.,seed=1,device='cpu',fixed_targets=targets)


def test_zero_tail_preserves_existing_mean_update_rng_draws_and_optimizer():
    original,old_replay=fixture()
    extended,replay=fixture()
    old_replay.coefficient=replay.coefficient=100.
    tail=MeanActionReplay(copy.deepcopy(replay.teacher),{'post_recovery':torch.ones(7,48)},
                         {'post_recovery':3},coefficient=0.,seed=109,device='cpu')
    torch.manual_seed(59)
    expected=update_with_mean_anchor(original,old_replay)
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(59)
    actual=update_with_mean_anchor(extended,replay,tail_replay=tail)
    assert tail.calls==0 and actual.pop('tail_anchor')==0.
    assert actual==expected and torch.equal(torch.get_rng_state(),expected_rng)
    assert old_replay.draw_digest.hexdigest()==replay.draw_digest.hexdigest()
    for before,after in ((original.actor,extended.actor),(original.critic,extended.critic)):
        for key,value in before.state_dict().items():torch.testing.assert_close(value,after.state_dict()[key],atol=0,rtol=0)
    assert [v['step'].item() for v in original.optimizer.state.values()]==[v['step'].item() for v in extended.optimizer.state.values()]


def test_tail_gradient_is_additive_without_diluting_existing_loss_or_changing_draws():
    original,old_replay=fixture()
    extended,replay=fixture()
    old_replay.coefficient=replay.coefficient=100.
    # Zero LR isolates the exact per-mini-batch gradients from later weight drift.
    for algorithm in (original,extended):
        for group in algorithm.optimizer.param_groups:group['lr']=0.
    tail_states=torch.ones(7,48)
    teacher=copy.deepcopy(replay.teacher)
    with torch.no_grad():
        [layer for layer in teacher.modules() if isinstance(layer,torch.nn.Linear)][-1].bias.add_(.2)
    tail=MeanActionReplay(teacher,{'post_recovery':tail_states},{'post_recovery':3},coefficient=25.,seed=109,device='cpu')
    parameter=next(extended.actor.mlp.parameters())
    predicted=extended.actor(TensorDict({'policy':tail_states},batch_size=[7]))
    expected_tail=torch.autograd.grad(25.*(predicted-teacher(tail_states)).square().mean(),parameter)[0]
    before,after=[],[]
    h0=next(original.actor.mlp.parameters()).register_hook(lambda g:before.append(g.detach().clone()))
    h1=parameter.register_hook(lambda g:after.append(g.detach().clone()))
    torch.manual_seed(61)
    old_losses=update_with_mean_anchor(original,old_replay)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(61)
    new_losses=update_with_mean_anchor(extended,replay,tail_replay=tail)
    h0.remove();h1.remove()
    assert len(before)==len(after)==4 and tail.calls==4 and tail.rows==12
    assert torch.equal(rng,torch.get_rng_state())
    assert old_replay.draw_digest.hexdigest()==replay.draw_digest.hexdigest()
    for key,value in old_losses.items():assert new_losses[key]==value
    for old,new in zip(before,after):torch.testing.assert_close(new-old,expected_tail,atol=2e-6,rtol=2e-5)
    assert all(p.grad is None for p in tail.teacher.parameters())
