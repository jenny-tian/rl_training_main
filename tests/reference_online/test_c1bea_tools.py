"""C1-b-EA 运行工具里不依赖仿真的部分：任务表、按会话拆分、快照合并、早期快照重建、GPU 选择。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')


def load(monkeypatch, variant, target='stairs_up_06'):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    monkeypatch.syspath_prepend(TOOLS)
    for name in ('reference_online_c1a2', 'reference_online_c1a2_merge', 'reference_online_c1bea_early', 'reference_online_c1a2_pool'):
        sys.modules.pop(name, None)
    return importlib.import_module('reference_online_c1a2')


def test_ea_variant_keeps_the_s2n_protocol_and_splits_training_by_session(monkeypatch):
    s2n = load(monkeypatch, 's2n'); reference = s2n.protocol(); old_jobs = [j['name'] for j in s2n.jobs()]
    old_rows = s2n.rows('train'); old_target = s2n.rows('target', part=2)
    assert old_jobs[:2] == ['smoke', 'train'] and old_jobs[-8:] == [f'target_{m}_{p:02d}' for m in (5000, 10000) for p in range(4)]
    assert all(j.get('extra') for j in s2n.jobs()[-8:]) and reference['marks'] == [0, 20000, 40000, 60000]
    ea = load(monkeypatch, 's2ea'); protocol = ea.protocol()
    same = ('sigma', 'learning_rate', 'residual_rate_limit', 'residual_limits', 'active_parameters', 'namespaces', 'noise_seed',
            'reset_seed', 'reward', 'pairs', 'train_control_steps', 'primary_mark', 'parameters', 'replicas', 'command', 'heading_feedback')
    assert all(protocol[k] == reference[k] for k in same) and protocol['algorithm'] == 'C1-b-S-EA'
    assert protocol['marks'] == [0, 5000, 10000, 20000, 40000, 60000] and ea.OUTPUT.parent.name == 'online_c1b_ea_20261120'
    names = [j['name'] for j in ea.jobs()]
    assert names[:6] == ['smoke', 'train_00', 'train_01', 'train_02', 'train_03', 'train'] and ea.jobs()[5]['kind'] == 'merge'
    assert sum(n.startswith('target_') for n in names) == 20 and sum(n.startswith('retention_60000_') for n in names) == 20
    parts = [ea.rows('train', part=p) for p in range(4)]
    assert all(len(p) == 120 and [r['env_id'] for r in p] == list(range(120)) for p in parts)
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'env_id'} for r in rows]
    assert sum((strip(p) for p in parts), []) == strip(old_rows)  # 拆分只改变环境编号，会话、起点槽位、组别都不变
    assert ea.rows('target', part=2) == old_target and len(ea.rows('smoke')) == 480


def test_merge_orders_sessions_and_rejects_inconsistent_parts(monkeypatch):
    load(monkeypatch, 's2ea')
    merge = importlib.import_module('reference_online_c1a2_merge')
    part = lambda p, **change: dict(dict(sessions=list(range(15 * p, 15 * p + 15)), control_steps=5000, base_sha256='b', noise_sha256='n', kind='train',
        heads=[dict(w=[s, arm], updated=bool(arm)) for s in range(15 * p, 15 * p + 15) for arm in (0, 1)]), **change)
    heads = merge.merge_heads([part(p) for p in range(4)])
    assert [h['w'] for h in heads] == [[s, arm] for s in range(60) for arm in (0, 1)]
    with pytest.raises(ValueError): merge.merge_heads([part(p) for p in (0, 2, 1, 3)])
    with pytest.raises(ValueError): merge.merge_heads([part(0), part(1), part(2), part(3, control_steps=10000)])
    with pytest.raises(ValueError): merge.merge_heads([part(0), part(1), part(2), part(3, heads=part(3)['heads'][::-1])])


def test_early_snapshot_takes_the_last_update_not_later_than_the_mark(monkeypatch):
    load(monkeypatch, 's2n')
    early = importlib.import_module('reference_online_c1bea_early')
    record = lambda session, step, updates, value, arm='Updated': dict(session=session, arm=arm, control_step=step, updates=updates, b_after=[0, 0, value, value])
    records = [record(0, 900, 0, 0.), record(0, 1053, 1, .3), record(0, 2100, 2, .5), record(0, 5001, 3, .9),
               record(1, 1053, 1, -.2), record(1, 1053, 0, 0.), record(1, 6000, 2, .7), record(2, 1053, 1, 9., arm='Frozen')]
    w = early.reconstruct(records, 5000, sessions=3)
    assert w.shape == (3, 2, 4) and not w[:, 0].any() and np.allclose(w[:, 1, 2], [.5, -.2, 0.])
    assert np.allclose(early.reconstruct(records, 6000, sessions=3)[:, 1, 3], [.9, .7, 0.])


def test_gpu_choice_counts_jobs_whose_memory_is_not_visible_yet(monkeypatch):
    load(monkeypatch, 's2ea')
    pool = importlib.import_module('reference_online_c1a2_pool')
    free = {0: 4000, 1: 20000, 2: 14000, 3: 12000}
    assert pool.choose(free, [], 1000.)[0] == 1
    gpu, pending = pool.choose(free, [[1, 990.], [1, 995.], [1, 100.]], 1000.)  # 两个刚派到 GPU 1，一个早已占上显存
    assert gpu == 2 and pending == [[1, 990.], [1, 995.]]
    assert pool.choose({0: 4000, 1: 12000}, [[1, 999.]], 1000.)[0] is None  # 12000 - 3500 < 9000
