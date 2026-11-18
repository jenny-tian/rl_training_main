import numpy as np
import pytest
import torch

from rl_training.ea_lpl.c1a import Session, loo_advantages
from rl_training.ea_lpl.c1a_m4 import ParallelSession, Slot


def play(group, noises, returns, order=(0, 1, 2, 3)):
    for k in range(4): group.begin(k, noises[k])
    return [group.finish(k, returns[k]) for k in order]


def test_round_is_one_plain_gradient_step():
    noises = np.array([[1., 0.], [-1., .5], [0., 2.], [.3, -.7]])
    returns = [1.5, .2, 1., 0.]
    group = ParallelSession(True)
    stats = play(group, noises, returns)
    advantages = loo_advantages(returns).numpy()
    gradient = -(advantages[:, None] * noises).mean(0)  # sigma = 1
    assert [s['updated'] for s in stats] == [False, False, False, True]
    assert np.allclose(stats[-1]['gradient'], gradient) and np.allclose(group.b.numpy(), -.6 * gradient)
    assert group.updates == 1 and not group.batch and not group.active


def test_finish_order_does_not_change_update():
    noises = np.random.default_rng(0).standard_normal((4, 2)); returns = [1., 2., 0., .5]
    a, b = ParallelSession(True), ParallelSession(True)
    play(a, noises, returns); play(b, noises, returns, order=(3, 1, 0, 2))
    assert torch.allclose(a.b, b.b)


def test_codes_use_unit_sigma_and_one_mean_per_round():
    group = ParallelSession(True)
    group.b = torch.tensor([.5, -.5], dtype=torch.float64)
    assert np.allclose(group.begin(0, [1., -2.]), [1.5, -2.5])
    with pytest.raises(ValueError): group.begin(0, [0., 0.])
    group.finish(0, 1.)
    with pytest.raises(ValueError): group.begin(0, [0., 0.])  # 同一轮不能再跑一次


def test_mean_is_clamped():
    group = ParallelSession(True)
    for _ in range(40): play(group, np.array([[3., 0.]] * 2 + [[-3., 0.]] * 2), [2., 2., -1., -1.])
    assert group.b[0] == 2.5 and group.updates == 40


def test_frozen_and_no_learn_never_move():
    frozen, idle = ParallelSession(False), ParallelSession(True)
    play(frozen, np.ones((4, 2)), [1., 0., 2., 3.])
    for k in range(4): idle.begin(k, [1., 1.])
    for k in range(4): idle.finish(k, float(k), learn=False)
    assert not frozen.b.any() and frozen.updates == 0 and not idle.b.any() and idle.updates == 0 and not idle.batch


def test_slots_share_one_mean_and_snapshot_loads_for_evaluation():
    group = ParallelSession(True); slots = [Slot(group, k) for k in range(4)]
    for s in slots: s.begin([1., -1.])
    for s, value in zip(slots, [1., 0., .5, 2.]): s.finish(value)
    assert slots[2].updates == 1 and slots[0].digest() == group.digest()
    head = Session(False); head.load_evaluation_mean(slots[0].snapshot())
    assert np.allclose(head.begin(None), group.b.numpy())
