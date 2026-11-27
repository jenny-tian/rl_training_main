"""上阶最后一级的脚本化检查在运行入口里的登记。方案见 docs/lite3-online-s2-20261127/PLAN.md 第 5 节。"""
import importlib
import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
GROUPS = ('none', 'hind_half', 'hind_full', 'all_half', 'h15', 'h15_hind_half')
SWITCHES = ('C1A2_SOURCE', 'C1A2_CANDIDATE', 'C1A2_LEAD', 'C1A2_RATE', 'C1A2_REFLEX', 'C1A2_TRIAL', 'C1A2_RULE', 'C1A2_TOP', 'C1A2_TOP_CONFIRM')


def load(monkeypatch, variant='r2', target='stairs_up_07', top=None, confirm=False, **extra):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    for name in SWITCHES: monkeypatch.delenv(name, raising=False)
    if top: monkeypatch.setenv('C1A2_TOP', top)
    if confirm: monkeypatch.setenv('C1A2_TOP_CONFIRM', '1')
    for name, value in extra.items(): monkeypatch.setenv(name, value)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    return importlib.import_module('reference_online_c1a2')


@pytest.mark.parametrize('group', GROUPS)
def test_a_probe_group_only_evaluates_the_target_with_the_stall_parameters_at_ten_thousand_steps(monkeypatch, group):
    entry = load(monkeypatch, top=group)
    p = entry.protocol()
    assert entry.TOP == group and not entry.TOP_CONFIRM
    assert entry.TOP_SOURCE.parts[-4:] == ('online_c1b_stall_rate04_20261124', 'stairs_up_07', 'train', 'at_10000.pt') and entry.TOP_SOURCE_MARK == 10000
    names = [j['name'] for j in entry.jobs()]
    assert names == [f'target_0_{p:02d}' for p in range(4)]  # 没有训练，也不做保持评估
    assert entry.OUTPUT.parts[-3:] == ('online_top_probe_20261127', 'stairs_up_07', group)
    assert p['top_probe']['group'] == group and p['top_probe']['min_base_height'] == (.15 if group.startswith('h15') else .20)
    assert p['top_probe']['parameters'] == 'r2 stairs_up_07 at 10000' and p['top_probe']['feature'] == 'top'
    assert p['residual_limits'] == [.24, .20, .20] * 4 and p['residual_rate_limit'] == .04


def test_screening_uses_the_starts_of_the_stall_correction_evaluation(monkeypatch):
    reference = load(monkeypatch)
    rows = [reference.rows('target', part=p) for p in range(4)]
    entry = load(monkeypatch, top='hind_half')
    assert [entry.rows('target', part=p) for p in range(4)] == rows


def test_confirmation_uses_new_start_slots_in_the_same_namespace(monkeypatch):
    screening = load(monkeypatch, top='hind_half')
    first = [r for p in range(4) for r in screening.rows('target', part=p)]
    namespace = screening.protocol()['namespaces']['target']
    entry = load(monkeypatch, top='hind_half', confirm=True)
    rows = [r for p in range(4) for r in entry.rows('target', part=p)]
    assert entry.TOP_CONFIRM and entry.protocol()['namespaces']['target'] == namespace and entry.protocol()['top_probe']['confirmation']
    assert entry.OUTPUT.parts[-3:] == ('online_top_probe_20261127', 'stairs_up_07', 'confirm_hind_half')
    assert len(rows) == len(first) == 480
    assert not {r['slot'] for r in rows} & {r['slot'] for r in first}                       # 起点槽位全部是新的
    assert [dict(r, slot=0) for r in rows] == [dict(r, slot=0) for r in first]              # 其余（会话、组、重复编号）不变
    assert len({r['slot'] for r in rows}) == 240                                            # 每对环境共用一个槽位
    assert all(a['slot'] == b['slot'] and (a['arm'], b['arm']) == ('Frozen', 'Updated') for a, b in zip(rows[::2], rows[1::2]))
    # 不与训练、保持评估和其他脚本化检查用过的槽位重合
    used = {r['slot'] for kind in ('train', 'retention') for part in range(20 if kind == 'retention' else 1) for r in load(monkeypatch).rows(kind, part=part)}
    assert not {r['slot'] for r in rows} & used and min(r['slot'] for r in rows) >= 500000 and max(r['slot'] for r in rows) < 900000


@pytest.mark.parametrize('kwargs', [dict(top='hind_half', target='boxes_07'), dict(top='hind_half', target='stairs_down_085'), dict(top='hind_half', variant='mrs'),
                                    dict(top='unknown'), dict(confirm=True), dict(top='hind_half', C1A2_RULE='directed'), dict(top='hind_half', C1A2_TRIAL='1500')])
def test_combinations_outside_the_registered_check_are_rejected(monkeypatch, kwargs):
    with pytest.raises(AssertionError): load(monkeypatch, **kwargs)


def test_runs_without_the_switch_are_unchanged(monkeypatch):
    entry = load(monkeypatch)
    assert entry.TOP is None and not entry.TOP_CONFIRM and 'top_probe' not in entry.protocol()
    assert entry.OUTPUT.parts[-2:] == ('online_c1b_stall_rate04_20261124', 'stairs_up_07') and [j['name'] for j in entry.jobs()][:2] == ['smoke', 'train']
