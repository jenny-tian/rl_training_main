"""S2′ 在运行入口里的登记：变体、目标、训练步数、快照、任务表、幅度上限。方案见 docs/lite3-online-adapter2-20261126/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
DOWN = ('slope_rough_down_x10', 'stairs_down_085')
NARROW, WIDE = [.24, .20, .20] * 4, [.48, .40, .40] * 4


def load(monkeypatch, variant, target):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    for name in ('C1A2_SOURCE', 'C1A2_CANDIDATE', 'C1A2_LEAD', 'C1A2_RATE', 'C1A2_REFLEX', 'C1A2_TRIAL'): monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    return importlib.import_module('reference_online_c1a2')


def names(entry):
    return [j['name'] for j in entry.jobs()]


@pytest.mark.parametrize('variant,level0', [('mrs', True), ('mrsz', False)])
@pytest.mark.parametrize('target', DOWN)
def test_averaged_adapter_on_the_downhill_targets_trains_to_the_primary_mark(monkeypatch, variant, level0, target):
    entry = load(monkeypatch, variant, target)
    p = entry.protocol()
    assert entry.ADAPTER == dict(kind='residual', update='s', level0=level0, limit_scale=1.)
    assert entry.TRAIN_STEPS == 10000 and entry.MARKS == (0, 1250, 2500, 5000, 10000) and entry.PRIMARY == 10000 and entry.RETENTION_MARKS == (10000,)
    assert not entry.SKIP_RETENTION and entry.REPLICAS == 4 and target in entry.TRAINABLE
    assert p['residual_limits'] == NARROW and p['residual_rate_limit'] == .04 and p['train_control_steps'] == 10000
    assert p['adapter']['update'] == 's' and p['adapter']['ema'] == .996 and p['adapter']['decay_steps_s'] == 1000 and p['adapter']['limits'] == NARROW
    assert p['adapter']['level0'] == ('stall correction (C1-b-R)' if level0 else None) and p['namespaces'] == dict(smoke=60, train=61, target=62, retention=63)
    assert entry.OUTPUT.parts[-2:] == (f'online_adapter_{variant}_20261126', target)
    jobs = names(entry)
    assert jobs[:2] == ['smoke', 'train'] and len(jobs) == 2 + 4 * 4 + 20 and sum(n.startswith('retention_10000_') for n in jobs) == 20


def test_averaged_adapter_without_stall_correction_runs_longer_on_the_stairs(monkeypatch):
    entry = load(monkeypatch, 'mrsz', 'stairs_up_07')
    assert entry.TRAIN_STEPS == 20000 and entry.MARKS == (0, 1250, 2500, 5000, 10000, 20000) and entry.PRIMARY == 10000
    assert len(names(entry)) == 2 + 5 * 4 + 20
    with_level0 = load(monkeypatch, 'mrs', 'stairs_up_07')
    assert with_level0.TRAIN_STEPS == 10000 and with_level0.MARKS == (0, 1250, 2500, 5000, 10000)


def test_the_plain_directed_control_without_stall_correction_keeps_the_first_stage_settings(monkeypatch):
    entry = load(monkeypatch, 'mrbz', 'stairs_up_07')
    assert entry.ADAPTER == dict(kind='residual', update='b', level0=False, limit_scale=1.)
    assert entry.TRAIN_STEPS == 20000 and entry.MARKS == (0, 1250, 2500, 5000, 10000, 20000)
    assert entry.OUTPUT.parts[-2:] == ('online_adapter_mrbz_20261126', 'stairs_up_07')


def test_the_wide_control_doubles_only_the_amplitude_limit(monkeypatch):
    entry = load(monkeypatch, 'mrsw', 'stairs_up_07')
    p = entry.protocol()
    assert entry.ADAPTER == dict(kind='residual', update='s', level0=True, limit_scale=2.)
    assert p['residual_limits'] == pytest.approx(WIDE) and p['adapter']['limits'] == pytest.approx(WIDE)
    assert p['residual_rate_limit'] == .04  # 每步变化上限不变
    assert entry.TRAIN_STEPS == 10000 and entry.MARKS == (0, 1250, 2500, 5000, 10000)
    assert entry.OUTPUT.parts[-2:] == ('online_adapter_mrsw_20261126', 'stairs_up_07')
    jobs = names(entry)
    assert len(jobs) == 2 + 16 + 20  # 保持评估也在 2 倍上限下做
    for target in DOWN + ('slope_rough_up_x11', 'boxes_07'):
        with pytest.raises(AssertionError): load(monkeypatch, 'mrsw', target)


def test_stall_correction_alone_is_the_reference_on_the_downhill_targets_without_retention(monkeypatch):
    for target in DOWN:
        entry = load(monkeypatch, 'r2', target)
        assert entry.STALL and entry.ADAPTER is None and entry.TRAIN_STEPS == 10000 and entry.MARKS == (0, 1250, 2500, 5000, 10000)
        assert entry.SKIP_RETENTION and entry.protocol()['residual_limits'] == NARROW and entry.protocol()['residual_rate_limit'] == .04
        assert entry.OUTPUT.parts[-2:] == ('online_c1b_stall_rate04_20261124', target)
        jobs = names(entry)
        assert not any(n.startswith('retention') for n in jobs) and len(jobs) == 2 + 16
    for target in ('stairs_up_07', 'slope_rough_up_x11'):  # 已有的 r2 运行不变
        entry = load(monkeypatch, 'r2', target)
        assert not entry.SKIP_RETENTION and sum(n.startswith('retention_2500_') for n in names(entry)) == 20


@pytest.mark.parametrize('target', ('stairs_up_07',) + DOWN)
def test_every_variant_on_a_target_shares_the_starts_of_the_stall_correction(monkeypatch, target):
    reference = load(monkeypatch, 'r2', target)
    rows = {kind: reference.rows(kind) for kind in ('train', 'target', 'retention')}
    noise, reset = reference.NOISE_SEED, reference.protocol()['reset_seed']
    for variant in ('mrs', 'mrsz') + (('mrsw', 'mrbz') if target == 'stairs_up_07' else ()):
        entry = load(monkeypatch, variant, target)
        assert {kind: entry.rows(kind) for kind in rows} == rows and entry.NOISE_SEED == noise and entry.protocol()['reset_seed'] == reset


@pytest.mark.parametrize('variant,target', [('mrs', 'stairs_up_075'), ('mrsz', 'slope_rough_up_x11'), ('mrs', 'stairs_up_06'), ('mrbz', 'stairs_down_085'),
                                            ('mfb', 'stairs_down_085'), ('mrc', 'slope_rough_down_x10'), ('r1', 'stairs_down_085'),
                                            ('h1', 'slope_rough_down_x10'), ('s2n', 'stairs_down_085'), ('sa1', 'slope_rough_down_x10'),
                                            # 坡度 0.34 的粗糙下坡在单地块的地图上太容易，不再是目标
                                            ('mrs', 'slope_rough_down_x085'), ('mrsz', 'slope_rough_down_x085'), ('r2', 'slope_rough_down_x085')])
def test_combinations_outside_the_registered_runs_are_rejected(monkeypatch, variant, target):
    with pytest.raises(AssertionError): load(monkeypatch, variant, target)


@pytest.mark.parametrize('variant', ['mfa', 'mfb', 'mfc', 'mra', 'mrb', 'mrc', 'mrcz'])
@pytest.mark.parametrize('target', ['stairs_up_07', 'slope_rough_up_x11'])
def test_first_stage_variants_are_unchanged(monkeypatch, variant, target):
    entry = load(monkeypatch, variant, target)
    p = entry.protocol()
    assert entry.TRAIN_STEPS == 20000 and entry.MARKS == (0, 1250, 2500, 5000, 10000, 20000) and entry.PRIMARY == 10000
    assert entry.ADAPTER['limit_scale'] == 1. and p['residual_limits'] == NARROW and p['adapter']['limits'] == NARROW
    assert entry.OUTPUT.parts[-2:] == (f'online_adapter_{variant}_20261126', target) and not entry.SKIP_RETENTION
    assert len(names(entry)) == 2 + 20 + 20


def test_other_existing_variants_are_unchanged(monkeypatch):
    entry = load(monkeypatch, 'r2', 'stairs_up_07')
    assert entry.TRAIN_STEPS == 10000 and entry.PRIMARY == 2500 and entry.protocol()['residual_limits'] == NARROW
    entry = load(monkeypatch, 'sa1', 'boxes_07')
    assert entry.SARA and entry.MARKS == (0, 1250, 2500, 5000, 10000) and not entry.SKIP_RETENTION
    entry = load(monkeypatch, 'ftu', 'stairs_up_07')
    assert entry.protocol()['residual_limits'] == [100.] * 12


def test_trial_runs_stay_out_of_the_registered_output_for_new_variants(monkeypatch):
    load(monkeypatch, 'mrs', 'stairs_down_085')
    monkeypatch.setenv('C1A2_TRIAL', '1500')
    sys.modules.pop('reference_online_c1a2', None)
    entry = importlib.import_module('reference_online_c1a2')
    assert entry.TRAIN_STEPS == 1500 and entry.OUTPUT.parts[-2:] == ('online_adapter_mrs_20261126', 'stairs_down_085_trial')
