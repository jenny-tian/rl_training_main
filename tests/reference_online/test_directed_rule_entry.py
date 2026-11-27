"""不学习的规则对照在运行入口里的登记。方案见 docs/lite3-online-adapter2-20261126/PLAN.md 第 12 节。"""
import importlib
import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
TARGETS = ('stairs_up_07', 'slope_rough_down_x10', 'stairs_down_085')


def load(monkeypatch, variant, target, rule='directed'):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    for name in ('C1A2_SOURCE', 'C1A2_CANDIDATE', 'C1A2_LEAD', 'C1A2_RATE', 'C1A2_REFLEX', 'C1A2_TRIAL', 'C1A2_RULE'): monkeypatch.delenv(name, raising=False)
    if rule: monkeypatch.setenv('C1A2_RULE', rule)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    return importlib.import_module('reference_online_c1a2')


@pytest.mark.parametrize('target', TARGETS)
def test_the_rule_control_only_evaluates_with_the_filter_of_the_stall_correction(monkeypatch, target):
    entry = load(monkeypatch, 'r2', target)
    p = entry.protocol()
    assert entry.RULE == 'directed' and entry.SOURCE_MARK == 0
    assert entry.OUTPUT.parts[-3:] == ('online_directed_rule_20261126', target, 'directed')
    assert p['residual_limits'] == [.24, .20, .20] * 4 and p['residual_rate_limit'] == .04
    assert p['directed_rule'] == dict(track_gain=.2, slow=.02, limits=[.24, .20, .20] * 4, delay_steps=1, level0=None)
    names = [j['name'] for j in entry.jobs()]
    assert names[:4] == [f'target_0_{p:02d}' for p in range(4)] and 'train' not in names and 'smoke' not in names
    assert all(j['mark'] == 0 for j in entry.jobs())
    # 保持地形与目标无关，只在 7 cm 上阶的目录下做一次
    assert sum(n.startswith('retention_0_') for n in names) == (20 if target == 'stairs_up_07' else 0)
    assert len(names) == (24 if target == 'stairs_up_07' else 4)


@pytest.mark.parametrize('target', TARGETS)
def test_the_rule_control_shares_starts_with_the_runs_it_is_compared_to(monkeypatch, target):
    rule = load(monkeypatch, 'r2', target)
    rows = {kind: rule.rows(kind, part=1) for kind in ('target', 'retention')}
    reset = rule.protocol()['reset_seed']
    for variant in ('r2', 'mrsz'):
        other = load(monkeypatch, variant, target, rule=None)
        assert {kind: other.rows(kind, part=1) for kind in rows} == rows and other.protocol()['reset_seed'] == reset
        assert other.RULE is None and 'directed_rule' not in other.protocol()


@pytest.mark.parametrize('variant,target,rule', [('s2n', 'stairs_up_07', 'directed'), ('mrsz', 'stairs_down_085', 'directed'),
                                                ('r2', 'stairs_up_075', 'directed'), ('r2', 'boxes_06', 'directed'), ('r2', 'stairs_up_07', 'other')])
def test_the_rule_control_is_limited_to_the_registered_targets(monkeypatch, variant, target, rule):
    with pytest.raises(AssertionError): load(monkeypatch, variant, target, rule=rule)


def test_runs_without_the_switch_are_unchanged(monkeypatch):
    entry = load(monkeypatch, 'r2', 'stairs_down_085', rule=None)
    assert entry.RULE is None and entry.OUTPUT.parts[-2:] == ('online_c1b_stall_rate04_20261124', 'stairs_down_085')
    assert [j['name'] for j in entry.jobs()][:2] == ['smoke', 'train']
