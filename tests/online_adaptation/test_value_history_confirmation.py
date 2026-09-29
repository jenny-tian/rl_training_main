import copy

import numpy as np
import pytest
import torch

from rl_training.online_adaptation.value_diagnostics import (
    append_past_velocity_integral, make_history_value_network, make_value_network,
)
from scripts.reinforcement_learning.online_adaptation.confirm_value_history import (
    checked_path, supported_by_confirmation,
)


def test_history_is_exclusive_causal_and_uses_only_measured_velocity():
    x = np.zeros((5, 48), dtype=np.float32)
    x[:, 45:47] = [[1, 2], [3, -1], [4, 5], [6, 7], [8, 9]]
    original = x.copy()
    augmented = append_past_velocity_integral(x)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(augmented[:, :48], x)
    np.testing.assert_allclose(augmented[:, 48:], np.array([[0, 0], [1, 2], [4, 1], [8, 6], [14, 13]]) / 600)
    x[2:, 45:47] = 1000
    # Changing the current/future velocity cannot affect history through t=2.
    np.testing.assert_array_equal(append_past_velocity_integral(x)[:3, 48:], augmented[:3, 48:])
    x = original.copy()
    x[:, :45] = -500
    x[:, 47] = 123
    np.testing.assert_array_equal(append_past_velocity_integral(x)[:, 48:], augmented[:, 48:])


def test_each_episode_resets_prefix_and_preserves_elapsed_column():
    first, second = np.ones((3, 48), dtype=np.float32), np.full((2, 48), 2, dtype=np.float32)
    first[:, 47], second[:, 47] = np.arange(3) / 2000, np.arange(2) / 2000
    batch = np.concatenate([append_past_velocity_integral(first), append_past_velocity_integral(second)])
    np.testing.assert_array_equal(batch[[0, 3], 48:], np.zeros((2, 2)))
    np.testing.assert_allclose(batch[4, 48:], [2 / 600, 2 / 600])
    np.testing.assert_array_equal(batch[:, 47], np.concatenate([first[:, 47], second[:, 47]]))


@pytest.mark.parametrize('shape', [(0, 48), (2, 47), (2, 50), (48,)])
def test_history_rejects_incompatible_episode_schema(shape):
    with pytest.raises(ValueError, match='episode observations'):
        append_past_velocity_integral(np.zeros(shape))


@pytest.mark.parametrize('kwargs', [{'step_dt': 0}, {'distance_scale': -1}, {'step_dt': np.nan}])
def test_history_rejects_invalid_scales(kwargs):
    with pytest.raises(ValueError, match='finite and positive'):
        append_past_velocity_integral(np.ones((2, 48)), **kwargs)


def test_zero_extension_preserves_predictions_without_mutating_initial_state():
    torch.manual_seed(23)
    base = make_value_network(48)
    state = base.state_dict()
    saved = copy.deepcopy(state)
    candidate = make_history_value_network(state)
    x = torch.randn(32, 48)
    augmented = torch.cat((x, 100 * torch.randn(32, 2)), dim=1)
    with torch.no_grad():
        torch.testing.assert_close(candidate(augmented), base(x), atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(candidate[0].weight[:, 48:]) == 0
    for key, value in saved.items():
        assert torch.equal(value, state[key])
        if key != '0.weight':
            assert torch.equal(value, candidate.state_dict()[key])
    with torch.no_grad():
        candidate[0].weight.zero_()
    assert torch.equal(base[0].weight, saved['0.weight'])


def test_confirmation_requires_all_comparators_and_both_error_weightings():
    def metric(rmse, equal, ev=.5):
        return {'pooled': {'rmse': rmse, 'explained_variance': ev}, 'episode_equal_mean_rmse': equal}
    metrics = {name: metric(3, 3) for name in ('original_critic', 'constant', 'time_only')}
    metrics['history_critic'] = metric(2, 2)
    assert supported_by_confirmation(metrics)
    metrics['time_only'] = metric(1, 3)
    assert not supported_by_confirmation(metrics)
    metrics['time_only'] = metric(3, 1)
    assert not supported_by_confirmation(metrics)
    metrics['time_only'] = metric(3, 3)
    for ev in (None, 0, -.1):
        metrics['history_critic'] = metric(2, 2, ev)
        assert not supported_by_confirmation(metrics)


def test_confirmation_rejects_modified_locked_inputs(tmp_path):
    from rl_training.online_adaptation.provenance import sha256_file
    path = tmp_path / 'bundle.pt'
    path.write_bytes(b'locked')
    record = {'path': str(path), 'sha256': sha256_file(path)}
    assert checked_path(record) == path
    path.write_bytes(b'modified')
    with pytest.raises(ValueError, match='locked input changed'):
        checked_path(record)
