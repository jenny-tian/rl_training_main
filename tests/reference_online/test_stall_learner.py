"""C1-b-R 受阻修正里不依赖仿真的部分。方案见 docs/lite3-online-stall-20261123/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_training.ea_lpl import c1b_r, c1b_s, c1b_w

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
SWING = np.array([False, False, False, False])  # 四条腿都在摆动


def window(group, front_speed, rear_speed, rough=1., steps=100, finish=False):
    if not group.running:
        codes = [group.begin(k) for k in range(4)]
    for t in range(steps):
        for k in range(4):
            group.observe(k, SWING, np.array([front_speed, front_speed, rear_speed, rear_speed]), rough, .008)
            if finish and t == steps - 1: group.finish(k, 1.)
        entry = group.tick()
    return entry


def session(updated=True):
    return c1b_r.StallSession(updated, c1b_w.exploration_table()[0])


def test_blocked_swing_raises_only_that_leg_group_and_no_noise_is_added():
    group = session()
    assert np.allclose([group.begin(k) for k in range(4)], 0.)
    entry = window(group, 0., .5)  # 前腿全受阻，后腿不受阻
    assert entry['updated'] and np.allclose(entry['stall'], [1., 0.]) and group.updates == 1
    assert np.allclose(group.w, [0., 0., 1.5 * .7, 0.])  # 前腿加 GAIN * (1 - 0.3)，后腿减到下限 0
    assert all(np.allclose(group.requested_code(k), group.w) for k in range(4))
    entry = window(group, 0., 0.)
    assert np.allclose(group.w, [0., 0., 2.1, 1.05])
    window(group, 0., 0.)
    assert np.allclose(group.w, [0., 0., c1b_s.BOUND, 2.1])  # 上限
    for _ in range(4): window(group, .5, .5)
    assert np.allclose(group.w, [0., 0., c1b_s.BOUND - 4 * .45, 2.1 - 4 * .45])  # 不受阻时每个窗口减 GAIN * 0.3


def test_flat_ground_windows_do_not_update_and_a_frozen_group_keeps_zero():
    group = session()
    entry = window(group, 0., 0., rough=0.)
    assert entry['used'] == [] and not entry['updated'] and torch.count_nonzero(group.w) == 0 and group.updates == 0
    frozen = session(False)
    entry = window(frozen, 0., 0.)
    assert not entry['updated'] and torch.count_nonzero(frozen.w) == 0 and frozen.updates == 0 and np.allclose(entry['stall'], [1., 1.])
    group = session()
    window(group, 0., 0., finish=True)
    head = c1b_s.ParallelSession(False, replicas=1)
    head.load_evaluation_mean(group.snapshot())
    assert torch.equal(head.w, group.w) and group.episode == 1


@pytest.mark.parametrize('target', ['stairs_up_07', 'boxes_07'])
def test_entry_registers_the_stall_variant(monkeypatch, target):
    monkeypatch.setenv('C1A2_VARIANT', 'r1'); monkeypatch.setenv('C1A2_TARGET', target)
    for name in ('C1A2_SOURCE', 'C1A2_CANDIDATE', 'C1A2_LEAD', 'C1A2_RATE'): monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    module = importlib.import_module('reference_online_c1a2')
    p = module.protocol()
    assert module.PRIMARY == 2500 == p['primary_mark'] and p['marks'] == [0, 1250, 2500, 5000, 10000] and p['algorithm'] == 'C1-b-R'
    assert p['active_parameters'] == ['front_swing', 'rear_swing'] and p['residual_rate_limit'] == .02 and p['namespaces']['target'] == 62
    assert 'online_c1b_stall_20261123' in str(module.OUTPUT) and module.WINDOW_INITIAL is None
    sys.modules.pop('reference_online_c1a2', None)
