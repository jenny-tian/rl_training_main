import numpy as np
import pytest
import torch

from rl_training.ea_lpl.basis import basis
from rl_training.ea_lpl.c1a import loo_advantages
from rl_training.ea_lpl.c1b_s import BOUND, ParallelSession, Slot, exploration_table, residual_request
from rl_training.ea_lpl.swing_lift import swing_request

BIG = 20.  # tanh(20) = 1


def test_zero_weights_and_flat_ground_give_zero():
    stance = torch.tensor([[True, False, True, False]])
    assert not residual_request(torch.zeros(1, 4), stance, torch.ones(1)).any()
    assert not residual_request(torch.full((1, 4), 2.), stance, torch.zeros(1)).any()


def test_static_part_matches_sag_basis():
    w = torch.tensor([[.7, -1.2, 0., 0.]])
    stance = torch.tensor([[True, False, False, True]])
    expected = basis('sag') @ torch.tanh(w[0, :2])
    assert torch.allclose(residual_request(w, stance, torch.ones(1))[0], expected, atol=1e-6)


def test_swing_part_matches_the_scripted_probe():
    stance = torch.tensor([[True, False, False, True], [False, False, True, False]])
    rough = torch.tensor([1., .5])
    rear = residual_request(torch.tensor([[0., 0., 0., BIG]] * 2), stance, rough)
    both = residual_request(torch.tensor([[0., 0., BIG, BIG]] * 2), stance, rough)
    assert torch.allclose(rear, swing_request(stance, rough, ['R', 'R'], 1.), atol=1e-6)
    assert torch.allclose(both, swing_request(stance, rough, ['Q', 'Q'], 1.), atol=1e-6)


def test_sum_is_clipped_to_the_residual_limit_and_sign_is_free():
    stance = torch.zeros(1, 4, dtype=torch.bool)
    r = residual_request(torch.full((1, 4), BIG), stance, torch.ones(1))
    assert r.abs().max() == pytest.approx(.20) and torch.allclose(r[0, :3], torch.tensor([0., -.10, .20]))
    n = residual_request(torch.tensor([[0., 0., -BIG, 0.]]), stance, torch.ones(1))[0]
    assert torch.allclose(n[:3], torch.tensor([0., .10, -.20])) and not n[6:].any()
    with pytest.raises(ValueError): residual_request(torch.zeros(1, 4), stance.float(), torch.ones(1))
    with pytest.raises(ValueError): residual_request(torch.zeros(1, 4), stance, torch.tensor([1.2]))


def play(group, noises, returns, order=(0, 1, 2, 3)):
    for k in range(4): group.begin(k, noises[k])
    return [group.finish(k, returns[k]) for k in order]


def test_round_is_one_plain_step():
    noises = np.random.default_rng(2).standard_normal((4, 4)); returns = [1.5, -2.4, 1., 0.]
    group = ParallelSession(True)
    assert np.allclose(group.begin(0, noises[0]), .7 * noises[0]); group.finish(0, 0., learn=False)
    stats = play(group, noises, returns)
    step = .6 * .7 * (loo_advantages(returns).numpy()[:, None] * noises).mean(0)
    assert [s['updated'] for s in stats] == [False, False, False, True] and np.allclose(group.w.numpy(), step)
    other = ParallelSession(True); play(other, noises, returns, order=(2, 0, 3, 1))
    assert torch.allclose(group.w, other.w)


def test_clamp_frozen_and_round_rules():
    group, frozen = ParallelSession(True), ParallelSession(False)
    noises = np.array([[3.] * 4] * 2 + [[-3.] * 4] * 2)
    for _ in range(40): play(group, noises, [2., 2., -1., -1.])
    play(frozen, noises, [2., 2., -1., -1.])
    assert (group.w == BOUND).all() and not frozen.w.any() and frozen.updates == 0
    fresh = ParallelSession(True); fresh.begin(0, np.zeros(4))
    with pytest.raises(ValueError): fresh.begin(0, np.zeros(4))
    fresh.finish(0, 1.)
    with pytest.raises(ValueError): fresh.begin(0, np.zeros(4))
    with pytest.raises(ValueError): fresh.requested_code(0)


def test_snapshot_loads_for_evaluation():
    group = ParallelSession(True); slots = [Slot(group, k) for k in range(4)]
    for s in slots: s.begin(np.ones(4))
    for s, value in zip(slots, [1., 0., .5, 2.]): s.finish(value)
    evaluator = Slot(ParallelSession(False, replicas=1), 0)
    evaluator.load_evaluation_mean(group.snapshot())
    assert np.allclose(evaluator.begin(), group.w.numpy()) and evaluator.digest() == group.digest()
    bad = group.snapshot(); bad['w'] = bad['w'] + 1e-3
    with pytest.raises(ValueError): ParallelSession(False).load_evaluation_mean(bad)


def test_exploration_table():
    assert exploration_table().shape == (60, 400, 4) and np.array_equal(exploration_table(), exploration_table())
