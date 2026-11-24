"""C1-b-RD（受阻修正加持续选择）里不依赖仿真的部分。方案见 docs/lite3-online-hybrid-20261124/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_training.ea_lpl import c1b_rd, c1b_s, c1b_w

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
SWING = np.zeros(4, bool)  # 四条腿都在摆动


def episode(group, succeeds, steps=300, speed=0.):
    codes, arms = [group.begin(k) for k in range(4)], list(group.assignment)
    for t in range(steps):
        for k in range(4):
            group.observe(k, SWING, np.full(4, speed), 1., .008)
            if t == steps - 1: group.finish(k, 1., success=arms[k] in succeeds)
        group.tick()
    return codes, arms


def session(updated=True):
    return c1b_rd.StallSelectSession(updated, c1b_w.exploration_table()[0], seed=[7, 0])


def test_first_episode_learns_swing_from_stall_then_candidates_are_built_from_it():
    group = session()
    codes, _ = episode(group, set())  # 3 个窗口都全受阻：每个窗口加 1.5 * 0.7
    assert np.allclose(codes, 0.) and not group.warm and group.warm_updates == 3
    s = min(3 * 1.05, c1b_s.BOUND)
    assert torch.allclose(torch.stack(group.arms), torch.tensor([[0, 0, s, s], [.8, -.8, s, s], [.8, -.8, s / 2, s / 2], [0, 0, 0, 0]], dtype=torch.float64))
    assert torch.allclose(group.w, group.arms[0]) and group.tries == [0] * 4 and group.round == 0 and group.selection is None
    codes, arms = episode(group, {1})
    assert arms == [0, 1, 2, 3] and np.allclose(codes, torch.stack(group.arms)) and group.leader == 1 and torch.equal(group.w, group.arms[1])
    codes, arms = episode(group, {1})
    assert arms == [1, 2, 3, 0] and group.tries == [2] * 4 and group.wins == [0, 2, 0, 0]
    for _ in range(6): codes, arms = episode(group, {1})
    assert group.tries[1] > 12 and group.selection['entry'] == 1 and group.selection['warm_updates'] == 3
    head = c1b_s.ParallelSession(False, replicas=1)
    head.load_evaluation_mean(group.snapshot())
    assert torch.equal(head.w, group.arms[1])


def test_unimpeded_swing_keeps_zero_and_a_frozen_group_never_changes():
    group = session()
    episode(group, set(), speed=.5)
    assert torch.count_nonzero(group.arms[0]) == 0 and torch.allclose(group.arms[1], torch.tensor([.8, -.8, 0, 0], dtype=torch.float64))
    frozen = session(False)
    for _ in range(4): episode(frozen, {1})
    assert torch.count_nonzero(frozen.w) == 0 and frozen.updates == 0 and frozen.selection['entry'] == 1 and not frozen.selection['applied']


@pytest.mark.parametrize('variant,algorithm,primary', [('h1', 'C1-b-RD', 10000), ('r2', 'C1-b-R', 2500)])
def test_entry_registers_the_variants_at_rate_004(monkeypatch, variant, algorithm, primary):
    for name in ('C1A2_SOURCE', 'C1A2_CANDIDATE', 'C1A2_LEAD', 'C1A2_RATE'): monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', 'boxes_07')
    sys.modules.pop('reference_online_c1a2', None)
    module = importlib.import_module('reference_online_c1a2')
    p = module.protocol()
    assert p['algorithm'] == algorithm and p['residual_rate_limit'] == .04 and module.PRIMARY == primary and 'library' not in p
    assert p['namespaces']['target'] == 62 and '20261124' in str(module.OUTPUT)
    sys.modules.pop('reference_online_c1a2', None)
