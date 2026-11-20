"""C1-b-EA 运行工具里不依赖仿真的部分：任务表、早期快照重建、GPU 选择。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')


def load(monkeypatch, variant, target='stairs_up_06'):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    monkeypatch.syspath_prepend(TOOLS)
    for name in ('reference_online_c1a2', 'reference_online_c1bea_early', 'reference_online_c1a2_pool'):
        sys.modules.pop(name, None)
    return importlib.import_module('reference_online_c1a2')


def test_ea_variant_keeps_the_s2n_protocol_and_evaluates_registered_marks_first(monkeypatch):
    s2n = load(monkeypatch, 's2n'); reference = s2n.protocol(); old_jobs = [j['name'] for j in s2n.jobs()]
    old_rows = {kind: s2n.rows(kind, part=part) for kind, part in (('train', 0), ('smoke', 0), ('target', 2), ('retention', 7))}
    assert old_jobs[:2] == ['smoke', 'train'] and old_jobs[-8:] == [f'target_{m}_{p:02d}' for m in (5000, 10000) for p in range(4)]
    assert len(old_jobs) == 2 + 12 + 20 + 8 and reference['marks'] == [0, 20000, 40000, 60000]
    assert [bool(j.get('extra')) for j in s2n.jobs()] == [False] * 34 + [True] * 8
    ea = load(monkeypatch, 's2ea'); protocol = ea.protocol()
    same = ('sigma', 'learning_rate', 'residual_rate_limit', 'residual_limits', 'active_parameters', 'namespaces', 'noise_seed',
            'reset_seed', 'reward', 'pairs', 'train_control_steps', 'primary_mark', 'parameters', 'replicas', 'command', 'heading_feedback')
    assert all(protocol[k] == reference[k] for k in same) and protocol['algorithm'] == 'C1-b-S-EA'
    assert protocol['marks'] == [0, 5000, 10000, 20000, 40000, 60000] and ea.OUTPUT.parent.name == 'online_c1b_ea_20261120'
    assert [j['name'] for j in ea.jobs()] == old_jobs  # 任务表与做法二相同：登记的快照和保持评估在前，早期快照在后
    assert all(ea.rows(kind, part=part) == rows for (kind, part), rows in zip((('train', 0), ('smoke', 0), ('target', 2), ('retention', 7)), old_rows.values()))


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
    assert pool.choose({0: 4000, 1: 12000}, [[1, 999.]], 1000., minimum=7000)[0] == 1
