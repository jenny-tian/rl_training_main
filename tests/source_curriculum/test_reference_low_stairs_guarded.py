import ast
import copy
import json

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from reference_low_stairs_guarded_ppo import Preserver, adapted_update, PER_GROUP
from reference_stairs_eval import sha


@pytest.fixture
def bank(tmp_path):
    path = tmp_path / 'training_bank.npz'
    rng = np.random.default_rng(9)
    np.savez(path, observation=rng.normal(size=(75,45)).astype('float32'),
             parent_action=rng.normal(size=(75,12)).astype('float32'), group=np.repeat(np.arange(5),15))
    path.with_suffix('.json').write_text(json.dumps(dict(bank_sha256=sha(path), seed=20261116,
                                                        diagnostic_bank_used=False)))
    return path


def test_native_update_changes_only_one_loss_assignment():
    _, original, adapted = adapted_update()
    a, b = ast.parse(original), ast.parse(adapted)
    loop = next(n for n in ast.walk(b) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'batch')
    added = [n for n in loop.body if isinstance(n, ast.Assign) and 'capability_preserver' in ast.unparse(n)]
    assert len(added) == 1
    index = loop.body.index(added[0])
    assert ast.unparse(loop.body[index+1]) == 'self.optimizer.zero_grad()'
    assert ast.unparse(loop.body[index+2]) == 'loss.backward()'
    loop.body.remove(added[0])
    assert ast.dump(a) == ast.dump(b)


def test_bank_identity_and_cursor_round_trip(bank):
    p = Preserver(bank, 'cpu')
    p.calls = 123
    other = Preserver(bank, 'cpu'); other.restore(p.state())
    a, b = p.sample(), other.sample()
    assert torch.equal(a[0]['policy'], b[0]['policy'])
    assert all(torch.equal(x,y) for x,y in zip(a[1:],b[1:]))
    assert torch.bincount(a[2]).tolist() == [PER_GROUP]*5
    wrong = dict(p.state(), coefficient=9.)
    with pytest.raises(ValueError): other.restore(wrong)
    with pytest.raises(ValueError): other.restore(dict(p.state(), calls=-1))
    manifest = json.loads(bank.with_suffix('.json').read_text())
    manifest['diagnostic_bank_used'] = True
    bank.with_suffix('.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): Preserver(bank, 'cpu')


def test_auxiliary_gradient_cache_and_zero_weight(bank):
    torch.manual_seed(77)
    obs = TensorDict({'policy':torch.randn(4,45)}, batch_size=[4])
    actor = MLPModel(obs, {'actor':['policy']}, 'actor', 12, hidden_dims=[16],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution.GaussianDistribution', 'init_std':1.})
    actor(obs, stochastic_output=True)
    cached = tuple(x.clone() for x in actor.output_distribution_params)
    p = Preserver(bank, 'cpu')
    loss = p.loss(actor); loss.backward()
    assert loss > 0 and not p.target.requires_grad
    assert any(x.grad is not None and x.grad.abs().sum() > 0 for x in actor.mlp.parameters())
    assert all(torch.isfinite(x.grad).all() for x in actor.parameters() if x.grad is not None)
    assert all(torch.equal(x,y) for x,y in zip(cached,actor.output_distribution_params))
    actor.zero_grad(set_to_none=True)
    zero = Preserver(bank, 'cpu', coefficient=0.)
    zero.loss(actor).backward()
    assert all(x.grad is None or torch.count_nonzero(x.grad) == 0 for x in actor.parameters())


def test_teacher_self_loss_zero(bank):
    p = Preserver(bank, 'cpu')
    obs, _, _ = p.sample()
    model = torch.nn.Linear(45,12)
    p.target = model(p.obs).detach()
    class Actor(torch.nn.Module):
        def forward(self, data): return model(data['policy'])
    assert p.loss(Actor()).item() == 0.


def test_midpoint_gate_keeps_failure_counts_and_rejects_missing_windows():
    from reference_low_stairs_guarded_evaluate import turn_stop_checks
    cases = ('low_left','low_right','turn_left','turn_right','fast_left','fast_right')
    old = {'records':[dict(case=case,outcome='survived',duration_s=20.) for case in (*cases,'standing')]}
    pairs = [dict(case=case,window='steady',n=20,baseline_mae=.2,candidate_mae=.2,mae_delta=0.) for case in cases]
    assert all(c['passed'] for c in turn_stop_checks(old,old,pairs,'nominal'))
    new = copy.deepcopy(old); new['records'][0].update(outcome='orientation_failure',duration_s=.5)
    assert sum(not c['passed'] for c in turn_stop_checks(old,new,pairs,'nominal')) == 2
    pairs[1].update(candidate_mae=.23,mae_delta=.03)
    assert not all(c['passed'] for c in turn_stop_checks(old,old,pairs,'nominal'))
    pairs[1].update(candidate_mae=.2,mae_delta=0.,n=9)
    assert not all(c['passed'] for c in turn_stop_checks(old,old,pairs,'nominal'))
