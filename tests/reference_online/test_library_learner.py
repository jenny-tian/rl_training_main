"""C1-b-L 参数库选择里不依赖仿真的部分。方案见 docs/lite3-online-library-20261122/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_training.ea_lpl import c1b_l, c1b_s, c1b_w

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
FRONT_SWING = np.array([False, False, True, True])
LIBRARY = torch.tensor([[0, 0, 1., 1.], [0, 0, 2., 1.5], [.5, -.5, 1., 0], [0, 0, 0, 0]], dtype=torch.float64)


def episode(group, speed_of_entry, steps=150):
    """每个环境前腿摆动时的足端速度由它当前用的库参数决定。"""
    codes = [group.begin(k) for k in range(4)]
    entries = [group.entry(k) for k in range(4)] if group.selecting else None
    for t in range(steps):
        for k in range(4):
            group.observe(k, FRONT_SWING, np.full(4, speed_of_entry[entries[k]] if entries else .5), 1., .008)
            if t == steps - 1: group.finish(k, 1.)
        group.tick()
    return codes, entries


def test_selection_rotates_entries_without_noise_and_picks_the_best_score():
    group = c1b_l.LibrarySession(True, c1b_w.exploration_table()[0], LIBRARY)
    speed = [0., .5, 0., 0.]  # 只有编号 1 的摆动不受阻
    codes, entries = episode(group, speed)
    assert entries == [0, 1, 2, 3] and np.allclose(codes, LIBRARY) and group.selecting and group.updates == 0
    codes, entries = episode(group, speed)
    assert entries == [1, 2, 3, 0] and np.allclose(codes, LIBRARY[[1, 2, 3, 0]])
    assert not group.selecting and group.selection['entry'] == 1 and group.selection['applied']
    assert group.selection['windows'] == [4, 4, 4, 4] and np.allclose(group.selection['scores'], [1., 1.5, 1., 1.])  # (前腿 u + 后腿 u) / 2 + 前进量 1
    assert torch.equal(group.w, LIBRARY[1]) and group.updates == 0
    assert np.allclose(group.requested_code(0) if group.running else group.begin(0), LIBRARY[1] + .7 * group.noise[group.windows, 0])


def test_after_selection_it_learns_like_the_window_learner_and_snapshot_loads():
    group = c1b_l.LibrarySession(True, c1b_w.exploration_table()[1], LIBRARY)
    for _ in range(2): episode(group, [.5, 0., 0., 0.])
    assert group.selection['entry'] == 0
    for k in range(4): group.begin(k)
    for t in range(100):
        for k in range(4): group.observe(k, FRONT_SWING, np.full(4, .5 * (k < 2)), 1., .008)
        entry = group.tick()
    assert entry['updated'] and group.updates == 1 and not torch.equal(group.w, LIBRARY[0])
    head = c1b_s.ParallelSession(False, replicas=1)
    head.load_evaluation_mean(group.snapshot())
    assert torch.equal(head.w, group.w) and group.snapshot()['selection']['entry'] == 0


def test_a_group_that_does_not_update_records_the_choice_but_keeps_its_parameters():
    group = c1b_l.LibrarySession(False, c1b_w.exploration_table()[2], LIBRARY)
    for _ in range(2): episode(group, [0., 0., .5, 0.])
    assert group.selection['entry'] == 2 and not group.selection['applied'] and torch.count_nonzero(group.w) == 0
    with pytest.raises(ValueError): c1b_l.LibrarySession(True, c1b_w.exploration_table()[0], LIBRARY[:3])
    with pytest.raises(ValueError): c1b_l.LibrarySession(True, c1b_w.exploration_table()[0], LIBRARY * 9)


@pytest.mark.parametrize('variant,target,primary,active', [('c3', 'stairs_up_07', 2500, None), ('c3', 'boxes_07', 2500, None),
                                                           ('w2', 'boxes_07', 5000, ['front_swing', 'rear_swing']), ('w2', 'stairs_up_07', 5000, ['front_swing', 'rear_swing'])])
def test_entry_registers_the_new_targets(monkeypatch, variant, target, primary, active):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target); monkeypatch.delenv('C1A2_SOURCE', raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    module = importlib.import_module('reference_online_c1a2')
    p = module.protocol()
    assert module.PRIMARY == primary == p['primary_mark'] and p['namespaces'] == dict(smoke=60, train=61, target=62, retention=63)
    assert p.get('active_parameters') == active and p['residual_rate_limit'] == .02 and module.WINDOW_INITIAL is None
    assert p['noise_seed'] == (2026112201 if variant == 'c3' else 2026112101) and ('library' in p) == (variant == 'c3')
    assert target in str(module.OUTPUT) and ('library' in str(module.OUTPUT)) == (variant == 'c3')
    sys.modules.pop('reference_online_c1a2', None)


def test_transfer_onto_a_new_target_goes_to_its_own_folder(monkeypatch):
    monkeypatch.setenv('C1A2_VARIANT', 's2n'); monkeypatch.setenv('C1A2_TARGET', 'boxes_07'); monkeypatch.setenv('C1A2_SOURCE', 'bs:boxes_06:40000:pooled')
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    module = importlib.import_module('reference_online_c1a2')
    assert 'online_library_transfer_20261122/boxes_07/from_bs_boxes_06_40000_pooled' in str(module.OUTPUT)
    assert [j['kind'] for j in module.jobs()] == ['target'] * 4 and module.protocol()['namespaces']['target'] == 62
    sys.modules.pop('reference_online_c1a2', None)
