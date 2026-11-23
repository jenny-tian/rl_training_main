"""C1-b-D 持续选择里不依赖仿真的部分。方案见 docs/lite3-online-select-20261123/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_training.ea_lpl import c1b_d, c1b_s, c1b_w

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
FRONT_SWING = np.array([False, False, True, True])
LIBRARY = torch.tensor([[0, 0, 1., 1.], [0, 0, 2., 1.5], [.5, -.5, 1., 0], [0, 0, 0, 0]], dtype=torch.float64)


def session(updated=True, **kwargs):
    return c1b_d.SelectSession(updated, c1b_w.exploration_table()[0], LIBRARY, seed=[7, 0], **kwargs)


def episode(group, succeeds, steps=150, speed=.5):
    """succeeds：哪些库编号的尝试算成功。返回这个回合各环境用的参数和候选编号。"""
    codes, arms = [group.begin(k) for k in range(4)], list(group.assignment)
    for t in range(steps):
        for k in range(4):
            group.observe(k, FRONT_SWING, np.full(4, speed), 1., .008)
            if t == steps - 1: group.finish(k, 1., success=arms[k] in succeeds)
        group.tick()
    return codes, arms


def test_first_two_rounds_rotate_the_library_without_noise_and_the_leader_follows_success():
    group = session()
    codes, arms = episode(group, {1})
    assert arms == [0, 1, 2, 3] and np.allclose(codes, LIBRARY) and group.round == 1
    assert group.leader == 1 and torch.equal(group.w, LIBRARY[1]) and group.updates == 1
    codes, arms = episode(group, {1})
    assert arms == [1, 2, 3, 0] and np.allclose(codes, LIBRARY[[1, 2, 3, 0]])
    assert group.tries == [2, 2, 2, 2] and group.wins == [0, 2, 0, 0] and group.selection['entry'] == 1
    assert np.allclose(group.selection['scores'], [.015, 1.015, .015, .015])  # 成功 + 0.01 *（(前腿 0 + 后腿 1) / 2 + 前进量 1）
    assert len(group.history) == 2 and group.history[1]['success'] == [True, False, False, False]


def test_later_rounds_sample_by_success_and_use_exactly_the_candidate_parameters():
    group, other = session(), session()
    used = []
    for _ in range(12):
        codes, arms = episode(group, {2}); episode(other, {2})
        assert np.allclose(codes, torch.stack([group.arms[a] for a in arms]))  # 不加扰动
        used += arms
    assert group.leader == 2 and torch.equal(group.w, LIBRARY[2]) and len(group.arms) == 4
    assert used[8:].count(2) > 30 and sum(group.tries) == 48  # 两轮之后几乎都在试成功的那一组
    assert group.history == other.history  # 同一个种子给出同样的分配
    for k in range(4): group.begin(k)
    for k in range(4): group.observe(k, FRONT_SWING, np.full(4, .5), 1., .008)
    assert set(group.finish(0, 1., success=True)) >= {'candidate', 'leader', 'code'} and 'arm' not in group.finish(1, 1.)
    assert c1b_w.Slot(group, 2).finish(1., learn=True, posture=False, success=True)['candidate'] == group.assignment[2]  # 采集循环经过 Slot 调用
    assert c1b_d.SelectSession(True, c1b_w.exploration_table()[0], LIBRARY, seed=[7, 1]).rng.random() != session().rng.random()


def test_window_score_only_breaks_ties():
    group = session()
    codes, arms = [group.begin(k) for k in range(4)], list(group.assignment)
    for t in range(150):
        for k in range(4):
            group.observe(k, FRONT_SWING, np.full(4, .5 * (k == 2)), 1., .008)  # 只有环境 2 的摆动不受阻
            if t == 149: group.finish(k, 1., success=k == 0)
        group.tick()
    assert group.leader == 0  # 成功的一组领先，窗口反馈更高的一组不会超过它
    group = session()
    for k in range(4): group.begin(k)
    for t in range(150):
        for k in range(4):
            group.observe(k, FRONT_SWING, np.full(4, .5 * (k == 2)), 1., .008)
            if t == 149: group.finish(k, 1.)
        group.tick()
    assert group.leader == 2  # 都不成功时按窗口反馈
    group = session()
    episode(group, set(), steps=20)  # 回合太短，没有窗口反馈，也没有成功
    assert group.leader == 3 and torch.count_nonzero(group.w) == 0 and group.selection['scores'] == [0.] * 4


def test_expansion_adds_candidates_from_the_two_leading_entries_once():
    plain, group = session(), session(expand=True)
    for _ in range(4): episode(plain, {1, 2}); episode(group, {1, 2})
    assert len(plain.arms) == 4 and len(group.arms) == 7 and group.added == [4, 5, 6] and group.tries[4:] == [0, 0, 0]
    order = sorted(range(4), key=lambda k: -group.selection['scores'][k])[:2]
    a, b = torch.tanh(LIBRARY[order[0]]), torch.tanh(LIBRARY[order[1]])
    assert torch.allclose(torch.tanh(group.arms[4]), (a + b) / 2)
    assert torch.allclose(torch.tanh(group.arms[5]), torch.stack([a[0], a[1], a[2] * 1.2, a[3] * 1.2]).clamp(max=c1b_d.LIMIT))
    assert torch.allclose(torch.tanh(group.arms[6]), torch.stack([a[0], a[1], a[2] * .8, a[3] * .8]))
    assert all((arm.abs() <= c1b_s.BOUND + 1e-9).all() for arm in group.arms)
    for _ in range(6): episode(group, {4})
    assert len(group.arms) == 7 and group.leader == 4 and torch.equal(group.w, group.arms[4]) and group.tries[4] >= 4 and group.wins[4] == group.tries[4]
    zero = [torch.zeros(4, dtype=torch.float64)] * 2
    assert len(c1b_d.candidates(*zero)) == 3 and all(torch.count_nonzero(row) == 0 for row in c1b_d.candidates(*zero))


def test_a_group_that_does_not_update_keeps_zero_and_snapshots_load_for_evaluation():
    group = session(False)
    for _ in range(3): episode(group, {0})
    assert group.selection['entry'] == 0 and not group.selection['applied'] and torch.count_nonzero(group.w) == 0 and group.updates == 0
    group = session()
    assert group.selection is None and group.snapshot()['selection'] is None
    for _ in range(3): episode(group, {0})
    head = c1b_s.ParallelSession(False, replicas=1)
    head.load_evaluation_mean(group.snapshot())
    assert torch.equal(head.w, LIBRARY[0]) and group.snapshot()['selection']['entry'] == 0 and 'history' not in group.snapshot()['selection']
    with pytest.raises(ValueError): c1b_d.SelectSession(True, c1b_w.exploration_table()[0], LIBRARY[:3], seed=0)
    with pytest.raises(ValueError): c1b_d.SelectSession(True, c1b_w.exploration_table()[0], LIBRARY * 9, seed=0)


@pytest.mark.parametrize('variant', ['d1', 'd2'])
@pytest.mark.parametrize('target', ['stairs_up_07', 'boxes_07'])
def test_entry_registers_the_selection_variants(monkeypatch, variant, target):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    monkeypatch.delenv('C1A2_SOURCE', raising=False); monkeypatch.delenv('C1A2_CANDIDATE', raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    module = importlib.import_module('reference_online_c1a2')
    p = module.protocol()
    assert module.PRIMARY == 10000 == p['primary_mark'] and p['namespaces'] == dict(smoke=60, train=61, target=62, retention=63)
    assert p['algorithm'] == 'C1-b-D' and p['noise_seed'] == 2026112201 and p['select_seed'] == 2026112301 and p['residual_rate_limit'] == .02
    assert p['expand_after_round'] == (4 if variant == 'd2' else None) and p['marks'] == [0, 1250, 2500, 5000, 10000]
    assert ('select_expand' in str(module.OUTPUT)) == (variant == 'd2') and target in str(module.OUTPUT)
    assert [j['name'] for j in module.jobs() if j['kind'] == 'retention'][0] == 'retention_10000_00'
    sys.modules.pop('reference_online_c1a2', None)


def test_candidate_evaluation_uses_the_target_starts_and_its_own_folder(monkeypatch):
    monkeypatch.setenv('C1A2_VARIANT', 's2n'); monkeypatch.setenv('C1A2_TARGET', 'boxes_07'); monkeypatch.setenv('C1A2_CANDIDATE', '21m')
    monkeypatch.delenv('C1A2_SOURCE', raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    module = importlib.import_module('reference_online_c1a2')
    assert 'online_select_candidates_20261123/boxes_07/21m' in str(module.OUTPUT)
    assert [j['name'] for j in module.jobs()] == [f'target_0_{p:02d}' for p in range(4)] and module.protocol()['namespaces']['target'] == 62
    assert module.protocol()['candidate'] == '21m'
    sys.modules.pop('reference_online_c1a2', None)
