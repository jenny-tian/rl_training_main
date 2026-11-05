"""Check observer neutrality and bounded accounting without Isaac or production writes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.tools.source_training_numerics_check import make_runner
from rl_training.source_scan.fixed_lr_trial import MinibatchMonitor, WalkingProbeMonitor, fixed_low_lr_plan
from rl_training.source_scan.training_budget_ledger import BudgetLedger

ROOT = Path(__file__).resolve().parents[2]


def collected():
    runner = make_runner(compact=True, rollout_length=4)
    alg = runner.alg
    alg.schedule = 'fixed'
    alg.learning_rate = 1e-5
    alg.num_learning_epochs = alg.num_mini_batches = 2
    for group in alg.optimizer.param_groups:
        group['lr'] = 1e-5
    obs = runner.env.get_observations()
    with torch.inference_mode():
        for _ in range(4):
            actions = alg.act(obs)
            obs,reward,done,extras = runner.env.step(actions)
            alg.process_env_step(obs,reward,done,extras)
        alg.compute_returns(obs)
    return runner


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key],right[key])
    elif isinstance(left, (list,tuple)):
        assert len(left) == len(right)
        for a,b in zip(left,right):
            equal(a,b)
    else:
        assert left == right


def test_real_ppo_monitor_is_bit_exact_and_preserves_rng():
    reference = collected()
    monitored = collected()
    torch.manual_seed(810)
    expected = reference.alg.update()
    expected_rng = torch.random.get_rng_state().clone()
    rows = []
    monitor = MinibatchMonitor(monitored.alg,rows.append)
    torch.manual_seed(810)
    with monitor:
        monitor.begin_update(1)
        actual = monitored.alg.update()
        monitor.finish_update()
    assert actual == expected
    assert torch.equal(expected_rng,torch.random.get_rng_state())
    for key in ('actor','critic'):
        equal(getattr(reference.alg,key).state_dict(),getattr(monitored.alg,key).state_dict())
    equal(reference.alg.optimizer.state_dict(),monitored.alg.optimizer.state_dict())
    assert len(rows) == 4
    assert [r['optimizer_step'] for r in rows] == [1,2,3,4]
    assert all(r['learning_rates'] == [1e-5] for r in rows)
    assert rows[0]['rollout_kl_before_mean'] == 0
    assert rows[0]['rollout_kl_after_mean'] > 0


def test_monitor_refuses_lr_drift_and_restores_generator_after_error():
    runner = collected()
    alg = runner.alg
    original = alg.storage.mini_batch_generator
    before = deepcopy(alg.actor.state_dict())
    with pytest.raises(ValueError,match='learning rate changed'):
        with MinibatchMonitor(alg,lambda row:None) as monitor:
            monitor.begin_update(1)
            alg.optimizer.param_groups[0]['lr'] = 0.001
            alg.update()
    assert alg.storage.mini_batch_generator == original
    assert alg.source_optimizer_steps_completed == 0
    equal(before,alg.actor.state_dict())


def test_walking_probe_has_known_shift_and_does_not_change_rng_or_distribution():
    runner = collected()
    actor = runner.alg.actor
    obs = torch.zeros(8,45)
    cached = tuple(p.clone() for p in actor.output_distribution_params)
    rng = torch.random.get_rng_state().clone()
    probe = WalkingProbeMonitor(actor,{100:{'observation':obs}},'cpu')
    assert probe.measure(actor,0)[0]['action_shift_rms'] == 0
    with torch.no_grad():
        actor.mlp[-1].bias.add_(0.1)
    measured = probe.measure(actor,1)[0]
    assert measured['action_shift_rms'] == pytest.approx(0.1,abs=1e-7)
    assert measured['original_policy_kl_mean'] == pytest.approx(0.06,abs=1e-6)
    assert torch.equal(rng,torch.random.get_rng_state())
    equal(cached,actor.output_distribution_params)


@pytest.fixture
def ledger(tmp_path):
    value = BudgetLedger(tmp_path/'ledger.json')
    value.path.write_bytes((ROOT/'docs/lite3-source-two-update-trial-20261105/raw/runtime/budget_after.json').read_bytes())
    return value


def reserve(ledger,**overrides):
    args = dict(run=1,attempt_id='cpu-fixed-rate',output_dir=ledger.path.parent/'trial',revision='c'*40,
        expected_reference=ledger.reference(),expected_plan=fixed_low_lr_plan())
    args.update(overrides)
    return ledger.reserve_fixed_lr_trial(**args)


def test_new_increment_preserves_history_and_is_never_refunded(ledger):
    before = ledger.read()
    row = reserve(ledger)
    after = ledger.read()
    assert after['entries'][:-1] == before['entries']
    assert after['ceilings'] == before['ceilings']
    assert ledger.charged(1,after) == 271104
    for n,updates in [(5761,2),(5760,3)]:
        with pytest.raises(ValueError):
            ledger.finish(row['attempt_id'],accepted=True,attempted=n,returned=n,stored=n,updates=updates)
    ledger.finish(row['attempt_id'],accepted=False,attempted=120,returned=0,stored=0,updates=0)
    with pytest.raises(ValueError):
        reserve(ledger,attempt_id='another-name')
    assert ledger.charged(1,ledger.read()) == 271104


@pytest.mark.parametrize('fault',['stale','ongoing','baseline','duplicate','run','learning_rate','schedule','updates'])
def test_reservation_refuses_plan_or_history_drift_without_writing(ledger,fault):
    args = {}
    if fault in ('ongoing','baseline','duplicate'):
        data = ledger.read()
        row = data['entries'][-1]
        if fault == 'ongoing': row['state'] = 'reserved'
        if fault == 'baseline': row['reserved_transitions'] += 1
        if fault == 'duplicate': row['attempt_id'] = 'cpu-fixed-rate'
        ledger.path.write_text(json.dumps(data))
    elif fault == 'stale':
        args['expected_reference'] = dict(ledger.reference(),sha256='f'*64)
    elif fault == 'run':
        args['run'] = 2
    else:
        plan = fixed_low_lr_plan()
        key,value = dict(learning_rate=('learning_rate',0.001),schedule=('schedule','adaptive'),updates=('max_updates',3))[fault]
        plan[key] = value
        args['expected_plan'] = plan
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        reserve(ledger,**args)
    assert ledger.path.read_bytes() == before
