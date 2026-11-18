import numpy as np
import pytest
import torch

from rl_training.ea_lpl.c1a import loo_advantages
from rl_training.ea_lpl.c1b_h import (BOUND, FEATURES, ParallelSession, Slot, TerrainFeatures, exploration_table,
                                      scripted)

X, Y = np.meshgrid(np.round(np.arange(-.8, .81, .1), 3), np.round(np.arange(-.5, .51, .1), 3), indexing='xy')
XY = np.column_stack((X.ravel(), Y.ravel()))


def stairs(x, first=1.2, tread=.3, rise=.06, steps=7):
    return rise * np.clip(np.floor((x - first) / tread + 1e-9) + 1, 0, steps)


def at(body_x, ground=stairs):
    features = TerrainFeatures(XY)
    return features(ground(XY[:, 0] + body_x)[None])[0].numpy()


def test_grid_has_187_rays_and_flat_ground_gives_zero():
    assert XY.shape == (187, 2)
    features = TerrainFeatures(XY)
    assert not features(np.zeros((3, 187))).any() and not features(np.full((1, 187), .37)).any()
    assert not features((-2e-6 * np.random.default_rng(0).random((2, 187)))).any()


def test_smooth_slope_gives_zero():
    slope = (.2 * XY[:, 0] + .1 * XY[:, 1])[None]
    assert np.allclose(TerrainFeatures(XY)(slope).numpy(), 0, atol=1e-9)


def test_stair_sections_activate_their_own_feature():
    rough, entry, middle, top = (FEATURES.index(k) for k in FEATURES)
    start, approach, mid, summit, beyond = at(.2), at(.9), at(2.1), at(3.15), at(4.2)
    assert not start.any()
    assert approach[entry] > .9 and approach[middle] < .1 and approach[top] < .1
    assert mid[middle] > .9 and mid[entry] < .1 and mid[top] < .1 and mid[rough] == 1
    assert summit[top] > .9 and summit[entry] < .1 and summit[middle] < .1
    assert not beyond.any()


def test_downward_steps_only_show_roughness():
    value = at(2.1, lambda x: -stairs(x))
    assert value[0] == 1 and not value[1:].any()


def test_missing_rays_are_filled_and_counted():
    features = TerrainFeatures(XY)
    z = np.zeros((2, 187)); z[1, 5] = np.inf
    assert not features(z).any() and features.missing == 1
    with pytest.raises(ValueError): features(np.full((1, 187), np.inf))


def test_code_is_weights_times_features_and_zero_on_flat():
    group = ParallelSession(True)
    noise = np.arange(8.) / 10
    assert not group.begin(0, noise).any()
    f = [1., 0., .5, .25]
    assert np.allclose(group.choose(0, f), .7 * noise.reshape(2, 4) @ np.array(f))
    assert np.allclose(group.requested_code(0), group.choose(0, f))
    assert not group.choose(0, [0., 0., 0., 0.]).any() and not group.requested_code(0).any()
    with pytest.raises(ValueError): group.choose(0, [0., 0., 0., 1.5])
    with pytest.raises(ValueError): group.choose(1, f)


def play(group, noises, returns, order=(0, 1, 2, 3)):
    for k in range(4): group.begin(k, noises[k])
    return [group.finish(k, returns[k]) for k in order]


def test_round_is_one_plain_step_on_the_parameters():
    noises = np.random.default_rng(1).standard_normal((4, 8)); returns = [1.5, -2.4, 1., 0.]
    group = ParallelSession(True)
    stats = play(group, noises, returns)
    advantages = loo_advantages(returns).numpy()
    step = .6 * .7 * (advantages[:, None] * noises).mean(0)
    assert [s['updated'] for s in stats] == [False, False, False, True]
    assert np.allclose(group.W.numpy().ravel(), step) and group.updates == 1 and not group.batch and not group.active
    other = ParallelSession(True); play(other, noises, returns, order=(3, 1, 0, 2))
    assert torch.allclose(group.W, other.W)


def test_parameters_are_clamped_and_frozen_never_moves():
    group, frozen = ParallelSession(True), ParallelSession(False)
    noises = np.array([[3.] * 8] * 2 + [[-3.] * 8] * 2)
    for _ in range(40): play(group, noises, [2., 2., -1., -1.])
    play(frozen, noises, [2., 2., -1., -1.])
    assert (group.W == BOUND).all() and not frozen.W.any() and frozen.updates == 0
    idle = ParallelSession(True)
    for k in range(4): idle.begin(k, noises[k])
    for k in range(4): idle.finish(k, float(k), learn=False)
    assert not idle.W.any() and not idle.batch


def test_round_rules():
    group = ParallelSession(True)
    group.begin(0, np.zeros(8))
    with pytest.raises(ValueError): group.begin(0, np.zeros(8))
    group.finish(0, 1.)
    with pytest.raises(ValueError): group.begin(0, np.zeros(8))
    with pytest.raises(ValueError): group.requested_code(0)
    with pytest.raises(ValueError): group.begin(4, np.zeros(8))


def test_snapshot_loads_for_evaluation_without_noise():
    group = ParallelSession(True); slots = [Slot(group, k) for k in range(4)]
    for s in slots: s.begin(np.ones(8))
    for s, value in zip(slots, [1., 0., .5, 2.]): s.finish(value)
    assert slots[2].updates == 1 and slots[0].digest() == group.digest()
    evaluator = Slot(ParallelSession(False, replicas=1), 0)
    evaluator.load_evaluation_mean(group.snapshot())
    evaluator.begin()
    f = np.array([1., 0., 0., 1.])
    assert np.allclose(evaluator.choose(f), group.W.numpy() @ f)
    evaluator.finish(1., learn=False)
    assert evaluator.digest() == group.digest()
    with pytest.raises(ValueError): ParallelSession(True).load_evaluation_mean(group.snapshot())
    bad = group.snapshot(); bad['W'] = bad['W'] + 1e-3
    with pytest.raises(ValueError): ParallelSession(False).load_evaluation_mean(bad)


def test_scripted_parameters():
    assert not scripted('A')['W'].any()
    c = Slot(ParallelSession(False, replicas=1), 0); c.load_evaluation_mean(scripted('C')); c.begin()
    assert np.allclose(c.choose(at(3.15)), [1.5, -1.5], atol=.15) and not c.choose(at(.2)).any()
    assert np.abs(c.choose(at(2.1))).max() < .15
    d = scripted('D')['W'].numpy()
    assert (d[:, 2] == [1.5, -1.5]).all() and (d[:, 3] == [1.5, -1.5]).all() and not d[:, :2].any()


def test_exploration_table_shape_and_seed():
    table = exploration_table()
    assert table.shape == (60, 400, 8) and np.array_equal(table, exploration_table())
