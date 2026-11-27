"""S2 的候选目标：地形参数和在运行入口里的登记。方案见 docs/lite3-online-s2-20261127/PLAN.md。"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
NEW = ('rough_09', 'rough_10', 'rough_11', 'slope_up_55', 'slope_up_57', 'slope_up_60', 'slope_rough_up_x12',
       'slope_rough_down_x11', 'slope_rough_down_x12', 'stairs_down_09', 'stairs_down_10')
CANDIDATES = ('boxes_07', 'boxes_08', 'slope_rough_up_x11') + NEW


@pytest.fixture
def targets(monkeypatch):
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2_targets', None)
    return importlib.import_module('reference_online_c1a2_targets')


def load(monkeypatch, variant, target, rule=None):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    for name in ('C1A2_SOURCE', 'C1A2_CANDIDATE', 'C1A2_LEAD', 'C1A2_RATE', 'C1A2_REFLEX', 'C1A2_TRIAL', 'C1A2_RULE', 'C1A2_TOP', 'C1A2_TOP_CONFIRM'): monkeypatch.delenv(name, raising=False)
    if rule: monkeypatch.setenv('C1A2_RULE', rule)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    return importlib.import_module('reference_online_c1a2')


def training_map():
    slope = lambda: SimpleNamespace(slope_range=(0., .4), proportion=.1)
    rough_slope = lambda: SimpleNamespace(slope_range=(0., .4), noise_range=(.01, .06), proportion=.05)
    return dict(rough=SimpleNamespace(noise_range=(.01, .06), proportion=.2), slope_up=slope(), slope_down=slope(),
                slope_rough_up=rough_slope(), slope_rough_down=rough_slope(), stairs_down=SimpleNamespace(step_height_range=(.05, .23), proportion=.1))


def test_every_new_candidate_uses_the_parameters_of_the_screening_tool(targets):
    sys.modules.pop('reference_frozen_rebaseline', None)
    screening = importlib.import_module('reference_frozen_rebaseline')
    for name in NEW + ('slope_rough_up_x11', 'slope_rough_down_x10', 'stairs_down_085'):
        result = targets.target_sub_terrains(name, training_map())
        spec = dict(screening.terrain_spec(name)); source = spec.pop('source')
        assert list(result) == [name] and result[name].proportion == 1.
        for key, value in spec.items(): assert tuple(getattr(result[name], key)) == tuple(value), (name, key)
        fresh = training_map()
        assert result[name].__dict__.keys() == fresh[source].__dict__.keys()


@pytest.mark.parametrize('name,source,field,value', [('rough_10', 'rough', 'noise_range', (.01, .10)), ('slope_up_57', 'slope_up', 'slope_range', (.57, .57)),
                                                    ('slope_rough_down_x12', 'slope_rough_down', 'slope_range', (.48, .48)), ('stairs_down_10', 'stairs_down', 'step_height_range', (.10, .10))])
def test_examples_of_the_new_targets(targets, name, source, field, value):
    captured = training_map()
    result = targets.target_sub_terrains(name, captured)
    assert result[name] is captured[source] and getattr(result[name], field) == pytest.approx(value)


def test_descending_stair_heights_cover_the_candidates(targets):
    assert targets.DOWN_STAIR_HEIGHT == dict(stairs_down_085=.085, stairs_down_09=.09, stairs_down_10=.10)


@pytest.mark.parametrize('source,field', [('rough', 'noise_range'), ('slope_up', 'slope_range')])
def test_a_map_whose_training_range_changed_is_refused(targets, source, field):
    captured = training_map(); setattr(captured[source], field, (0., .9))
    with pytest.raises(ValueError):
        targets.target_sub_terrains(dict(rough='rough_10', slope_up='slope_up_57')[source], captured)


@pytest.mark.parametrize('target', CANDIDATES)
def test_the_rule_control_can_probe_every_candidate(monkeypatch, target):
    entry = load(monkeypatch, 'r2', target, rule='directed')
    assert entry.S2_CANDIDATES == CANDIDATES
    names = [j['name'] for j in entry.jobs()]
    assert names == [f'target_0_{p:02d}' for p in range(4)]  # 保持评估已经在 7 cm 上阶做过，这里不重复
    assert entry.OUTPUT.parts[-3:] == ('online_directed_rule_20261126', target, 'directed')
    assert entry.protocol()['residual_limits'] == [.24, .20, .20] * 4 and entry.protocol()['residual_rate_limit'] == .04


@pytest.mark.parametrize('target', CANDIDATES)
def test_the_second_stage_configuration_can_train_on_every_candidate(monkeypatch, target):
    entry = load(monkeypatch, 'mrs', target)
    assert entry.ADAPTER == dict(kind='residual', update='s', level0=True, limit_scale=1.) and target in entry.TRAINABLE
    assert entry.TRAIN_STEPS == 10000 and entry.MARKS == (0, 1250, 2500, 5000, 10000) and entry.PRIMARY == 10000
    names = [j['name'] for j in entry.jobs()]
    assert len(names) == 2 + 16 + 20 and entry.OUTPUT.parts[-2:] == ('online_adapter_mrs_20261126', target)
    reference = load(monkeypatch, 'r2', target)
    assert reference.rows('target', part=3) == entry.rows('target', part=3) and reference.rows('train') == entry.rows('train')
    assert reference.TRAIN_STEPS == 10000 and reference.MARKS == (0, 1250, 2500, 5000, 10000)


@pytest.mark.parametrize('target', NEW)
def test_the_stall_correction_reference_skips_retention_on_new_targets(monkeypatch, target):
    entry = load(monkeypatch, 'r2', target)
    assert entry.SKIP_RETENTION and not any(n.startswith('retention') for n in (j['name'] for j in entry.jobs()))


def test_existing_reference_runs_keep_their_retention(monkeypatch):
    for target in ('boxes_07', 'slope_rough_up_x11', 'stairs_up_07'):
        assert not load(monkeypatch, 'r2', target).SKIP_RETENTION


@pytest.mark.parametrize('variant,target', [('mrsz', 'rough_10'), ('mrsw', 'boxes_07'), ('mfb', 'slope_up_57'), ('mrbz', 'stairs_down_10'), ('h1', 'rough_10'), ('s2n', 'slope_up_57'), ('mrs', 'rough_12'), ('mrs', 'stairs_up_075')])
def test_other_combinations_stay_rejected(monkeypatch, variant, target):
    with pytest.raises(AssertionError): load(monkeypatch, variant, target)
