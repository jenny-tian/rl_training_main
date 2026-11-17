import copy

import numpy as np
import pytest
import torch

from rl_training.ea_lpl.c1a import Session,episode_return,exploration_table,loo_advantages


def batch(head,noises,returns):
    result = []
    for noise,value in zip(noises,returns):
        head.begin(noise)
        result.append(head.finish(value))
    return result


def test_original_reward_uses_final_distance_and_nine_meter_scale():
    assert episode_return(9.3,True,False)==2.
    assert episode_return(4.5,False,False)==.5
    assert episode_return(-1.,False,True)==-1.
    assert episode_return(3.,True,False)==pytest.approx(4/3)
    # Bounds exits were not safety endings in the original schema.
    assert episode_return(3.,False,False)==pytest.approx(1/3)
    with pytest.raises(ValueError): episode_return(float('nan'),False,False)
    with pytest.raises(ValueError): episode_return(1.,True,True)


def test_episode_holds_requested_code_and_incomplete_batch_never_updates():
    head = Session(True)
    expected = head.begin([.2,-.3])
    expected[:] = 99.  # Return is a copy, not the live request.
    for _ in range(1000): assert np.array_equal(head.requested_code(),[.12,-.18])
    with pytest.raises(ValueError): head.begin([0.,0.])
    head.finish(.7)
    batch(head,[[1.,1.],[1.,-1.]],[.4,.8])
    assert head.updates==0 and len(head.batch)==3 and torch.count_nonzero(head.b)==0
    head.begin([-.5,.5])
    assert head.snapshot()['active'] is not None and head.updates==0


def test_loo_excludes_own_return():
    first = loo_advantages([1.,2.,3.,4.])
    second = loo_advantages([11.,2.,3.,4.])
    assert second[0]-first[0]==10.
    assert torch.allclose(first,torch.tensor([-2.,-2/3,2/3,2.],dtype=torch.float64))
    with pytest.raises(ValueError): loo_advantages([1.])


def test_gradient_and_multiple_adam_updates_match_autograd():
    head = Session(True)
    parameter = torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter],lr=.3)
    rng = np.random.default_rng(10)
    for _ in range(5):
        noises = rng.normal(size=(4,2)); returns = rng.normal(size=4)
        actions = parameter.detach().clone()+.6*torch.tensor(noises)
        advantages = loo_advantages(returns)
        loss = -(torch.distributions.Normal(parameter,.6).log_prob(actions).sum(-1)*advantages).mean()
        optimizer.zero_grad(); loss.backward()
        expected_gradient = parameter.grad.clone()
        optimizer.step()
        with torch.no_grad(): parameter.clamp_(-2.5,2.5)
        stats = batch(head,noises,returns)[-1]
        assert np.allclose(stats['gradient'],expected_gradient.numpy(),rtol=0,atol=1.e-14)
        assert torch.allclose(head.b,parameter,rtol=0,atol=1.e-14)
        assert torch.allclose(head.m,optimizer.state[parameter]['exp_avg'],rtol=0,atol=1.e-14)
        assert torch.allclose(head.v,optimizer.state[parameter]['exp_avg_sq'],rtol=0,atol=1.e-14)


def test_frozen_and_smoke_are_unchanged_and_sessions_independent():
    frozen,updated,other = Session(False),Session(True),Session(True)
    noise = [[-1.,0.],[0.,1.],[1.,0.],[0.,-1.]]
    for head in (frozen,updated): batch(head,noise,[-1.,0.,1.,0.])
    assert frozen.updates==other.updates==0 and frozen.digest()==other.digest()
    assert updated.updates==1 and updated.digest()!=other.digest()
    other.begin([1.,0.]); other.finish(1.,learn=False)
    assert other.updates==0 and not other.batch


def test_parameter_version_and_nonfinite_input_fail_without_consuming_episode():
    head = Session(True)
    head.begin([1.,0.])
    with pytest.raises(ValueError): head.finish(float('inf'))
    assert head.active is not None and not head.batch
    head.finish(1.)
    head.begin([0.,1.]); head.batch[0]['version']=99
    with pytest.raises(ValueError,match='one current'): head.finish(1.)
    assert head.updates==0 and len(head.batch)==1 and head.active is not None
    head.b[0]=1.
    with pytest.raises(ValueError,match='inside episode'): head.requested_code()


def test_snapshot_is_independent_and_loads_for_deterministic_evaluation(tmp_path):
    head = Session(True)
    batch(head,[[1.,0.],[-1.,0.],[1.,0.],[-1.,0.]],[1.,-1.,1.,-1.])
    path = tmp_path/'head.pt'
    torch.save(head.snapshot(),path)
    snapshot = torch.load(path,weights_only=True)
    evaluation = Session(False); evaluation.load_evaluation_mean(snapshot)
    assert np.array_equal(evaluation.begin(),head.b.numpy())
    evaluation.finish(1.,learn=False)
    assert evaluation.updates==0 and not evaluation.batch
    head.b.add_(.1)
    assert not torch.equal(head.b,snapshot['b'])
    invalid = copy.deepcopy(snapshot); invalid['b'][0]=3.
    with pytest.raises(ValueError): Session(False).load_evaluation_mean(invalid)


def test_mean_bound_and_noise_table_reproducibility():
    head = Session(True)
    for _ in range(30): batch(head,[[1.,1.],[-1.,-1.],[1.,1.],[-1.,-1.]],[1.,-1.,1.,-1.])
    assert torch.equal(head.b,torch.tensor([2.5,2.5],dtype=torch.float64))
    assert np.array_equal(exploration_table(),exploration_table())
    assert exploration_table().shape==(60,400,2)
