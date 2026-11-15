"""系数为零仍走相同采样路径，且不改变原损失的梯度和 Adam 更新。"""
import copy
import json

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from reference_low_stairs_guarded_ppo import Preserver as WeightedPreserver
from reference_low_stairs_zero_preservation_train import Preserver, directory, checked_configuration, GUARDED_OUTPUT
from reference_stairs_eval import sha


@pytest.fixture
def bank(tmp_path):
    path = tmp_path/'bank.npz'
    rng = np.random.default_rng(91)
    np.savez(path, observation=rng.normal(size=(75,45)).astype('float32'),
             parent_action=rng.normal(size=(75,12)).astype('float32'), group=np.repeat(np.arange(5),15))
    path.with_suffix('.json').write_text(json.dumps(dict(bank_sha256=sha(path), seed=20261116, diagnostic_bank_used=False)))
    return path


def test_zero_weight_keeps_sample_cursor_and_rejects_weighted_state(bank):
    zero, weighted = Preserver(bank,'cpu'), WeightedPreserver(bank,'cpu')
    zero.calls = weighted.calls = 1000
    a,b = zero.sample(),weighted.sample()
    assert torch.equal(a[0]['policy'],b[0]['policy'])
    assert all(torch.equal(x,y) for x,y in zip(a[1:],b[1:]))
    assert zero.state() == dict(weighted.state(),coefficient=0.)
    other = Preserver(bank,'cpu');other.restore(zero.state())
    assert other.state() == zero.state()
    with pytest.raises(ValueError): zero.restore(weighted.state())


def test_zero_auxiliary_preserves_primary_gradients_and_adam_updates(bank):
    torch.manual_seed(71)
    obs = TensorDict({'policy':torch.randn(8,45)},batch_size=[8])
    old = MLPModel(obs,{'actor':['policy']},'actor',12,hidden_dims=[16],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution.GaussianDistribution','init_std':1.})
    new = copy.deepcopy(old)
    a,b = (torch.optim.Adam(m.parameters(),lr=1e-3) for m in (old,new))
    p = Preserver(bank,'cpu')
    for _ in range(3):
        old(obs,stochastic_output=True);new(obs,stochastic_output=True)
        cached = [v.clone() for v in new.output_distribution_params]
        a.zero_grad();b.zero_grad()
        loss = p.loss(new)
        assert loss.item()==0. and all(torch.equal(x,y) for x,y in zip(cached,new.output_distribution_params))
        old(obs).square().mean().backward()
        (new(obs).square().mean()+loss).backward()
        for x,y in zip(old.parameters(),new.parameters()):
            assert (x.grad is None and y.grad is None) or torch.equal(x.grad,y.grad)
        a.step();b.step()
        assert all(torch.equal(x,y) for x,y in zip(old.parameters(),new.parameters()))
    assert p.calls==3 and p.loss_sum==0.


@pytest.mark.parametrize('phase',[-1,2,3])
def test_no_additional_training_phase(phase):
    with pytest.raises(ValueError): directory(phase)


def test_configuration_rejects_reward_or_optimizer_changes(monkeypatch):
    import reference_low_stairs_zero_preservation_train as module
    monkeypatch.setattr(module,'save',lambda *args:None)
    old = json.loads((GUARDED_OUTPUT/'train_001_100/configuration.json').read_text())
    correct=copy.deepcopy(old);correct['agent']['max_iterations']=100
    assert checked_configuration(correct)==correct
    changed=copy.deepcopy(correct);changed['agent']['algorithm']['learning_rate'] *= 2
    with pytest.raises(AssertionError): checked_configuration(changed)
    changed=copy.deepcopy(correct);changed['environment']['rewards']['stair_clearance']['weight'] *= 2
    with pytest.raises(AssertionError): checked_configuration(changed)
