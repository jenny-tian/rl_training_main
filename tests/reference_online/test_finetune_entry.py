"""微调对照在运行入口里的登记：变体、快照、任务表、安全过滤的上限。方案见 docs/lite3-online-finetune-20261124/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')


def load(monkeypatch, variant, target):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    for name in ('C1A2_SOURCE', 'C1A2_CANDIDATE', 'C1A2_LEAD', 'C1A2_RATE'): monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    return importlib.import_module('reference_online_c1a2')


@pytest.mark.parametrize('variant', ['ft', 'lora'])
@pytest.mark.parametrize('target', ['stairs_up_07', 'boxes_07', 'stairs_up_075', 'boxes_08'])
def test_filtered_fine_tuning_runs_one_robot_hour_under_the_simulation_limits(monkeypatch, variant, target):
    entry = load(monkeypatch, variant, target)
    p = entry.protocol()
    assert entry.FINETUNE and not entry.UNFILTERED and entry.REPLICAS == 4
    assert entry.TRAIN_STEPS == 45000 and entry.MARKS == (0, 1250, 2500, 5000, 10000, 20000, 45000) and entry.PRIMARY == 10000
    assert p['residual_rate_limit'] == .04 and p['residual_limits'] == [.24, .20, .20] * 4
    assert p['namespaces'] == dict(smoke=60, train=61, target=62, retention=63)
    assert p['finetune']['mode'] == dict(ft='full', lora='lora')[variant] and p['finetune']['learning_rate'] == dict(ft=1e-3, lora=1e-2)[variant]
    assert p['finetune']['horizon'] == 250 and p['train_control_steps'] == 45000
    assert entry.OUTPUT.parent.name == dict(ft='online_finetune_full_20261124', lora='online_finetune_lora_20261124')[variant]


@pytest.mark.parametrize('variant', ['ft', 'lora'])
def test_filtered_fine_tuning_evaluates_every_snapshot_and_retention_at_two_marks(monkeypatch, variant):
    names = [j['name'] for j in load(monkeypatch, variant, 'boxes_07').jobs()]
    assert names[:2] == ['smoke', 'train']
    for mark in (1250, 2500, 5000, 10000, 20000, 45000):
        assert sum(n.startswith(f'target_{mark}_') for n in names) == 4
    assert sum(n.startswith('retention_10000_') for n in names) == 20 and sum(n.startswith('retention_45000_') for n in names) == 20
    assert len(names) == 2 + 24 + 40


@pytest.mark.parametrize('variant,mode', [('ftu', 'full'), ('lorau', 'lora')])
def test_unfiltered_fine_tuning_lifts_only_the_residual_limits(monkeypatch, variant, mode):
    entry = load(monkeypatch, variant, 'stairs_up_07')
    p = entry.protocol()
    assert entry.FINETUNE and entry.UNFILTERED and p['finetune']['mode'] == mode
    assert p['residual_rate_limit'] == 100. and p['residual_limits'] == [100.] * 12
    assert p['namespaces'] == dict(smoke=60, train=61, target=62, retention=63)
    assert entry.OUTPUT.parent.name == dict(ftu='online_finetune_full_unfiltered_20261124', lorau='online_finetune_lora_unfiltered_20261124')[variant]


@pytest.mark.parametrize('target', ['stairs_up_075', 'boxes_08', 'stairs_up_06'])
def test_unfiltered_fine_tuning_is_limited_to_the_two_7_cm_targets(monkeypatch, target):
    with pytest.raises(AssertionError): load(monkeypatch, 'ftu', target)


def test_existing_variants_keep_their_settings(monkeypatch):
    entry = load(monkeypatch, 'h1', 'boxes_07')
    assert not entry.FINETUNE and entry.TRAIN_STEPS == 10000 and entry.MARKS == (0, 1250, 2500, 5000, 10000)
    assert 'finetune' not in entry.protocol() and len(entry.jobs()) == 2 + 16 + 20


def test_the_registered_ppo_settings_are_the_ones_the_module_uses(monkeypatch):
    from rl_training.ea_lpl import finetune
    f = load(monkeypatch, 'ft', 'boxes_07').protocol()['finetune']
    assert (f['gamma'], f['lam'], f['clip'], f['entropy'], f['value'], f['epochs'], f['minibatches'], f['max_grad_norm'], f['desired_kl']) == (
        finetune.GAMMA, finetune.LAM, finetune.CLIP, finetune.ENTROPY, finetune.VALUE, finetune.EPOCHS, finetune.MINIBATCHES, finetune.MAX_GRAD_NORM, finetune.DESIRED_KL)
    assert f['learning_rate_range'] == [finetune.LR_LOW, finetune.LR_HIGH]


def test_the_evaluation_pool_can_be_limited_to_listed_gpus(monkeypatch):
    load(monkeypatch, 'ft', 'boxes_07')
    sys.modules.pop('reference_online_c1a2_pool', None)
    pool = importlib.import_module('reference_online_c1a2_pool')
    free = {0: 20000, 1: 9000, 2: 15000, 3: 100}
    assert pool.allowed(free, None) == free
    assert pool.allowed(free, '1') == {1: 9000} and pool.allowed(free, '1,3') == {1: 9000, 3: 100}
    assert pool.choose(pool.allowed(free, '1,3'), [], 0., 7000)[0] == 1
