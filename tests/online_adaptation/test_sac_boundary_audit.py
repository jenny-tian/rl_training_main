import numpy as np

from docs.analysis.audit_sac_boundaries import aligned_replay_fields


def test_boundary_audit_preserves_terminal_observation_and_truncation_history():
    arrays = {
        'history': np.array([[[[1., 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]),
        'next_frame': np.array([[[9., 10.]], [[11., 12.]]]),
        'learning_terminated': np.array([True, False]),
        'learning_truncated': np.array([False, True]),
        'bootstrap_mask': np.array([0., 1.]),
    }
    for name in ('current_frame', 'base_action', 'executed_residual_action',
                 'requested_residual_action', 'next_base_action'):
        arrays[name] = np.zeros((2, 1, 2))
    arrays['reward'] = np.array([-8., 2.])
    arrays['episode_id'] = np.array([1, 2])
    fields = aligned_replay_fields(arrays, [{'replay_safety_eligible':True}]*2, .99)
    # Task terminal: history cleared, actual terminal next observation retained.
    np.testing.assert_array_equal(fields['next_history'][0], [[0., 0.], [9., 10.]])
    # Physical/collector truncation: bootstrap from the unreset history.
    np.testing.assert_array_equal(fields['next_history'][1], [[7., 8.], [11., 12.]])
    np.testing.assert_array_equal(fields['reward'], [[-8.], [2.]])
    np.testing.assert_array_equal(fields['bootstrap_mask'], [[0.], [1.]])
