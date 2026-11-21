"""C1-b-W 窗口学习器里不依赖仿真的部分。方案见 docs/lite3-online-window-20261121/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_training.ea_lpl import c1b_s, c1b_w

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
ALL_STANCE, FRONT_SWING = np.ones(4, bool), np.array([False, False, True, True])


def run_window(group, velocity, distance=.008, steps=c1b_w.WINDOW, posture=None):
    """velocity[k]：环境 k 的前腿摆动时足端速度。每步前腿摆动、后腿着地。"""
    entry = None
    for t in range(steps):
        for k in sorted(group.running):
            group.observe(k, FRONT_SWING, np.full(4, velocity[k]), 1., distance)
            if posture is not None and k == posture and t == steps - 1: group.finish(k, -3., posture=True)
        entry = group.tick()
    return entry


def test_feedback_is_unimpeded_ratio_plus_progress():
    assert np.allclose(c1b_w.window_feedback([40, 20], [10, 20], 90, .36, False), [.75 + .5, 0. + .5])
    assert np.allclose(c1b_w.window_feedback([0, 10], [0, 0], 50, 5., True), [0 + 1 - 3, 1 + 1 - 3])
    with pytest.raises(ValueError): c1b_w.window_feedback([1, 1], [2, 0], 10, 0., False)


def test_noise_changes_every_window_and_update_follows_the_formula():
    noise = c1b_w.exploration_table()[0]
    group = c1b_w.WindowSession(True, noise, active=['front_swing', 'rear_swing'])
    codes = [group.begin(k) for k in range(4)]
    assert np.allclose(codes, .7 * noise[0] * [0, 0, 1, 1])
    velocity = [.5, .5, 0., 0.]  # 前两个环境前腿摆动顺畅，后两个受阻
    entry = run_window(group, velocity)
    assert entry['used'] == [0, 1, 2, 3] and entry['updated'] and group.updates == 1
    feedback = np.array([[2., 1.], [2., 1.], [1., 1.], [1., 1.]])  # 后腿没有摆动，u 记 0，前进量 1
    assert np.allclose(entry['feedback'], feedback)
    advantage = feedback - (feedback.sum(0) - feedback) / 3
    step = (advantage[:, [0, 1, 0, 1]] * noise[0] * [0, 0, 1, 1]).mean(0)
    assert np.allclose(group.w, .6 * .7 * step) and group.w[:2].abs().sum() == 0
    assert np.allclose(group.requested_code(0), group.w + .7 * noise[1, 0] * np.array([0, 0, 1, 1]))


def test_settle_steps_are_not_counted_and_short_windows_are_skipped():
    group = c1b_w.WindowSession(True, c1b_w.exploration_table()[0])
    for k in range(4): group.begin(k)
    for t in range(c1b_w.SETTLE + c1b_w.MINIMUM_STEPS - 1):
        for k in range(4): group.observe(k, FRONT_SWING, np.full(4, .5), 1., .008)
        assert group.tick() is None
    assert group.acc[0]['steps'] == c1b_w.MINIMUM_STEPS - 1 and group.acc[0]['alive'] == c1b_w.SETTLE + c1b_w.MINIMUM_STEPS - 1
    for k in range(4): group.finish(k, 1.)
    entry = group.tick()  # 4 个环境都结束，窗口提前关闭；统计步数不够，不更新
    assert entry['used'] == [] and not entry['updated'] and group.updates == 0 and group.episode == 1 and group.clock == 0
    assert group.begin(0) is not None


def test_posture_failure_is_penalised_in_its_window_and_finished_replicas_drop_out():
    group = c1b_w.WindowSession(True, c1b_w.exploration_table()[1])
    for k in range(4): group.begin(k)
    entry = run_window(group, [.5] * 4, posture=3)
    assert np.allclose(np.asarray(entry['feedback'])[:, 0], [2, 2, 2, -1]) and entry['posture'] == [False, False, False, True]
    entry = run_window(group, [.5] * 4)
    assert entry['used'] == [0, 1, 2] and entry['updated']
    with pytest.raises(ValueError): group.requested_code(3)
    with pytest.raises(ValueError): group.begin(3)


def test_frozen_group_gets_noise_but_never_moves():
    group = c1b_w.WindowSession(False, c1b_w.exploration_table()[2])
    for k in range(4): group.begin(k)
    assert np.abs(group.requested_code(0)).sum() > 0
    entry = run_window(group, [.5, 0., .5, 0.])
    assert 'step' in entry and not entry['updated'] and group.updates == 0 and torch.count_nonzero(group.w) == 0
    assert run_window(group, [.5] * 4, steps=3) is None  # 不学习时也不更新
    for k in range(4): group.finish(k, 0.)
    assert group.tick(learn=False)['window'] == 1


def test_snapshot_loads_into_the_evaluation_head_and_initial_parameters_are_used():
    initial = torch.tensor([.3, -.2, 1., .5], dtype=torch.float64)
    group = c1b_w.WindowSession(True, c1b_w.exploration_table()[0], initial=initial)
    for k in range(4): group.begin(k)
    run_window(group, [.5, .5, 0., 0.])
    assert group.updates == 1 and not torch.equal(group.w, initial)
    head = c1b_s.ParallelSession(False, replicas=1)
    head.load_evaluation_mean(group.snapshot())
    assert torch.equal(head.w, group.w)
    with pytest.raises(ValueError): c1b_w.WindowSession(True, c1b_w.exploration_table()[0], initial=[9., 0, 0, 0])


@pytest.mark.parametrize('target,primary,namespace,active', [('stairs_up_06', 5000, 62, ['front_swing', 'rear_swing']),
                                                              ('stairs_up_065', 5000, 62, ['front_swing', 'rear_swing']), ('boxes_06', 10000, 52, None)])
def test_entry_registers_the_window_variant(monkeypatch, target, primary, namespace, active):
    monkeypatch.setenv('C1A2_VARIANT', 'w2'); monkeypatch.setenv('C1A2_TARGET', target); monkeypatch.delenv('C1A2_SOURCE', raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    module = importlib.import_module('reference_online_c1a2')
    p = module.protocol()
    assert module.MARKS == (0, 1250, 2500, 5000, 10000) and module.PRIMARY == primary == p['primary_mark']
    assert p['namespaces']['target'] == namespace and p.get('active_parameters') == active and p['residual_rate_limit'] == .02
    assert p['train_control_steps'] == 10000 and p['noise_seed'] == 2026112101
    jobs = module.jobs()
    assert jobs[1]['steps'] == 10000 and sum(j['kind'] == 'target' for j in jobs) == 16
    assert {j['mark'] for j in jobs if j['kind'] == 'retention'} == {primary}
    assert bool(module.WINDOW_INITIAL) == (target == 'boxes_06')
    sys.modules.pop('reference_online_c1a2', None)
