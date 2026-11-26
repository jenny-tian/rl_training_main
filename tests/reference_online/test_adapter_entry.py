"""S1 小模型在运行入口里的登记：变体、目标、快照、任务表。方案见 docs/lite3-online-adapter-20261126/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
VARIANTS = ('mfa', 'mfb', 'mfc', 'mra', 'mrb', 'mrc')


def load(monkeypatch, variant, target):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    for name in ('C1A2_SOURCE', 'C1A2_CANDIDATE', 'C1A2_LEAD', 'C1A2_RATE', 'C1A2_REFLEX'): monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    return importlib.import_module('reference_online_c1a2')


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('target', ['stairs_up_07', 'slope_rough_up_x11'])
def test_adapter_variants_follow_the_registered_run(monkeypatch, variant, target):
    entry = load(monkeypatch, variant, target)
    p = entry.protocol()
    assert entry.ADAPTER == dict(kind=dict(f='film', r='residual')[variant[1]], update=variant[2], level0=True, limit_scale=1.)
    assert entry.REPLICAS == 4 and entry.TRAIN_STEPS == 20000 and entry.MARKS == (0, 1250, 2500, 5000, 10000, 20000)
    assert entry.PRIMARY == 10000 and entry.RETENTION_MARKS == (10000,)
    assert p['residual_rate_limit'] == .04 and p['residual_limits'] == [.24, .20, .20] * 4
    assert p['namespaces'] == dict(smoke=60, train=61, target=62, retention=63) and p['train_control_steps'] == 20000
    assert p['adapter']['kind'] == entry.ADAPTER['kind'] and p['adapter']['update'] == variant[2] and p['adapter']['level0'] == 'stall correction (C1-b-R)'
    assert p['adapter']['std'] == .1 and p['adapter']['calibration'] == 1000 and p['adapter']['track_gain'] == .2
    assert p['adapter']['lr_low'] == 1e-6 and p['adapter']['residual_limits'] == p['residual_limits']
    assert entry.OUTPUT.parts[-2:] == (f'online_adapter_{variant}_20261126', target)
    assert target in entry.TRAINABLE
    names = [j['name'] for j in entry.jobs()]
    assert names[:2] == ['smoke', 'train'] and len(names) == 2 + 5 * 4 + 20
    assert sum(n.startswith('retention_10000_') for n in names) == 20


def test_the_group_without_level_zero_only_drops_the_stall_correction(monkeypatch):
    entry = load(monkeypatch, 'mfcz', 'stairs_up_07')
    assert entry.ADAPTER == dict(kind='film', update='c', level0=False, limit_scale=1.) and entry.protocol()['adapter']['level0'] is None
    assert entry.OUTPUT.parts[-2:] == ('online_adapter_mfcz_20261126', 'stairs_up_07')
    same = load(monkeypatch, 'mfc', 'stairs_up_07')
    assert entry.rows('train') == same.rows('train') and entry.rows('target', part=2) == same.rows('target', part=2)


def test_adapter_runs_share_starts_with_the_stall_correction_alone(monkeypatch):
    r2 = load(monkeypatch, 'r2', 'stairs_up_07')
    rows = {kind: r2.rows(kind) for kind in ('train', 'target', 'retention')}
    noise = r2.NOISE_SEED
    entry = load(monkeypatch, 'mfa', 'stairs_up_07')
    assert {kind: entry.rows(kind) for kind in rows} == rows and entry.NOISE_SEED == noise
    assert entry.protocol()['reset_seed'] == r2.protocol()['reset_seed']


@pytest.mark.parametrize('target', ['boxes_07', 'stairs_up_06', 'slope_rough_up_x12'])
def test_adapter_variants_are_limited_to_the_two_targets(monkeypatch, target):
    with pytest.raises(AssertionError): load(monkeypatch, 'mfa', target)


def test_stall_correction_alone_can_run_on_the_rough_slope_as_the_reference(monkeypatch):
    entry = load(monkeypatch, 'r2', 'slope_rough_up_x11')
    assert entry.STALL and entry.TRAIN_STEPS == 10000 and entry.MARKS == (0, 1250, 2500, 5000, 10000) and 'slope_rough_up_x11' in entry.TRAINABLE
    assert entry.protocol()['residual_rate_limit'] == .04
    assert entry.OUTPUT.parts[-2:] == ('online_c1b_stall_rate04_20261124', 'slope_rough_up_x11')
    with pytest.raises(AssertionError): load(monkeypatch, 'r1', 'slope_rough_up_x11')
    with pytest.raises(AssertionError): load(monkeypatch, 'h1', 'slope_rough_up_x11')


def test_existing_variants_keep_their_settings(monkeypatch):
    entry = load(monkeypatch, 'r2', 'stairs_up_07')
    assert entry.ADAPTER is None and entry.TRAIN_STEPS == 10000 and entry.PRIMARY == 2500
    entry = load(monkeypatch, 'm4', 'stairs_up_06')
    assert entry.ADAPTER is None and entry.RUNS['m4'] == 'online_c1a_m4_20261118'
    entry = load(monkeypatch, 'sa1', 'boxes_07')
    assert entry.ADAPTER is None and entry.SARA and entry.MARKS == (0, 1250, 2500, 5000, 10000)


def test_trial_runs_are_short_and_kept_out_of_the_registered_output(monkeypatch):
    monkeypatch.setenv('C1A2_TRIAL', '1500')
    entry = load(monkeypatch, 'mfa', 'stairs_up_07')
    assert entry.TRAIN_STEPS == 1500 and entry.OUTPUT.parts[-2:] == ('online_adapter_mfa_20261126', 'stairs_up_07_trial')
    assert next(j for j in entry.jobs() if j['name'] == 'train')['steps'] == 1500
    monkeypatch.delenv('C1A2_TRIAL')
    assert load(monkeypatch, 'mfa', 'stairs_up_07').TRAIN_STEPS == 20000
