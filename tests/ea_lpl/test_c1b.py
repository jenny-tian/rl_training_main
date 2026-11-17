import numpy as np
import torch

from rl_training.ea_lpl.c1b import FeatureHistory, Session, digest, exploration_table, returns_to_go, rewards


def test_features_pad_reset_and_use_only_registered_channels():
    history = FeatureHistory(2)
    history.append([[.1, .2], [.8, 0]], [0.4, -.4], [5., 10.])
    value = history.features(np.zeros((2, 2)))
    assert np.allclose(value[0], [.2, .4, 0., .5, .5, 0., 0., 0.], atol=1.e-6)
    assert value[1, -1] == 1.
    before = history.data[1].copy()
    history.reset([0])
    assert np.array_equal(history.data[1], before)
    history.append([[0, 0], [.8, 0]], [0, -.4], [0, 10.])
    assert np.count_nonzero(history.features(np.zeros((2, 2)))[0]) == 0


def test_returns_account_for_terminal_reward_at_real_control_step():
    values = np.zeros(12); values[10] = 2.; values[11] = -1.
    result = returns_to_go(values)
    assert np.isclose(result[10], 1.01)
    assert np.isclose(result[0], .99**10 * 1.01)
    assert rewards([.03], [-.03], np.zeros((1, 12)), np.ones(12), [True], [False])[0] == -.99


def test_independent_updated_session_changes_only_its_parameters():
    frozen, updated, other = Session(False), Session(True), Session(True)
    initial = digest(other.model)
    noise = exploration_table()[0, 0, 0]
    feature = np.ones(8, np.float32)
    assert np.array_equal(frozen.choose(feature, noise), updated.choose(feature, noise))
    for instance in (frozen, updated):
        instance.rewards = [1.] * 10
        instance.finish(allow_update=True)
    assert digest(frozen.model) == digest(other.model) == initial
    assert digest(updated.model) != initial
    assert updated.updates == 1 and frozen.updates == 0
    assert sum(p.numel() for p in updated.model.parameters()) == 18
    assert updated.features[0].grad_fn is None and updated.codes[0].grad_fn is None


def test_smoke_and_incomplete_episodes_do_not_update_and_reset_clears_buffers():
    session = Session(True)
    session.choose(np.zeros(8, np.float32), [1., -1.])
    session.rewards = [1., 2.]
    session.finish(allow_update=False)
    assert session.updates == 0 and digest(session.model) == session.initial_digest
    session.begin()
    assert not session.rewards and not session.decision_steps


def test_baseline_is_previous_history_and_noise_is_reproducible():
    session = Session(True)
    session.baseline = 3.
    session.choose(np.zeros(8, np.float32), [1., -1.])
    session.rewards = [1.]
    stats = session.finish(allow_update=True)
    assert stats['old_baseline'] == 3. and np.isclose(stats['new_baseline'], 2.8)
    assert np.array_equal(exploration_table(), exploration_table())
