import numpy as np
import pytest

from docs.analysis.diagnose_lite3_recovery_objective import collect_episodes, reconstruct, return_ordering, temporal_weights


def fixture():
    shape = (2, 1)
    state = np.zeros((*shape, 13))
    state[..., 3] = 1
    state[:, 0, 1] = [.4, .8]
    state[..., 8] = .1
    origin = np.zeros((*shape, 3))
    actions = np.full((*shape, 12), .1)
    target_yaw = np.clip(np.arctan2(-state[..., 1], 1), -np.deg2rad(20), np.deg2rad(20))
    d = dict(heading_root_state=state, velocity_root_state=state.copy(),
             heading_origins=origin, velocity_origins=origin.copy(), pre_root_state=state.copy(),
             episode_steps=np.array([[11], [12]]), recovery_case=np.full(shape, 26),
             recovery_anchor_mask=np.array([[True], [False]]), emitted_action=actions,
             recovery_anchor_target=np.zeros_like(actions), recovery_anchor_applied_cost=np.array([[.01], [0]]),
             recovery_anchor_reward=np.array([[-.02], [0]]),
             heading_reward=-(1-np.cos(target_yaw))*.02,
             velocity_reward=np.full(shape, -8*.25**2*.02),
             task_success=np.zeros(shape, bool), task_failure=np.array([[False], [True]]),
             task_deadline=np.zeros(shape, bool), done=np.array([[False], [True]]),
             returned_root_state=np.zeros_like(state))
    d['reward'] = d['recovery_anchor_reward']+d['heading_reward']+d['velocity_reward']+np.array([[0.], [-2.]])+.123
    p = dict(recovery_anchor=dict(early_steps=12, weight=100),
             route_training=dict(heading_lookahead_m=1, heading_maximum_deg=20, heading_weight=-1),
             lateral_velocity=dict(gain_per_second=.5, maximum_m_s=.15, weight=-8),
             recovery_course_lateral_weight=-8)
    return d, p


def test_reward_accounting_uses_terminal_pose_and_preserves_unallocated_terms():
    d, p = fixture()
    result = reconstruct(d, p)
    np.testing.assert_allclose(result['unallocated_reward'], .123)
    np.testing.assert_array_equal(result['abs_body_y_m'][:, 0], [.4, .8])
    np.testing.assert_allclose(result['body_lateral_penalty_magnitude_lower_bound'][:, 0], [.0256, .1024])
    np.testing.assert_array_equal(result['anchor_active_fraction'][:, 0], [1, 0])
    np.testing.assert_array_equal(result['terminal_reward'][:, 0], [0, -2])


@pytest.mark.parametrize('defect', ['anchor_after_window', 'wrong_velocity_charge', 'nonfinite', 'overlapping_outcomes'])
def test_reward_accounting_rejects_corrupt_components(defect):
    d, p = fixture()
    if defect == 'anchor_after_window':
        d['recovery_anchor_reward'][1, 0] = -.01
    elif defect == 'wrong_velocity_charge':
        d['velocity_reward'] *= 50
    elif defect == 'nonfinite':
        d['reward'][0, 0] = np.nan
    else:
        d['task_success'][1, 0] = True
    with pytest.raises(AssertionError):
        reconstruct(d, p)


def test_episode_accounting_preserves_failure_terminal_peak_and_censored_tail():
    age = np.array([[0], [1], [0], [1], [2], [0]])
    done = np.array([[0], [1], [0], [0], [1], [0]], bool)
    success = np.array([[0], [0], [0], [0], [1], [0]], bool)
    failure = done & ~success
    y = np.array([[.3], [.9], [.2], [.4], [.5], [.1]])
    reward = np.ones_like(y)
    eps = collect_episodes(age, done, success, failure, np.zeros_like(done), y, reward, .5)
    assert [e['outcome'] for e in eps] == ['failure', 'success', 'censored']
    assert [e['controls'] for e in eps] == [2, 3, 1]
    assert eps[0]['peak_body_y_m'] == .9 and eps[0]['peak_age_s'] == .04
    assert [e['discounted_reward_sum'] for e in eps] == [1.5, 1.75, 1]
    assert eps[1]['start_control'] == 2 and eps[2]['start_control'] == 5
    age[2] = 2
    with pytest.raises(AssertionError):
        collect_episodes(age, done, success, failure, np.zeros_like(done), y, reward, .5)


def test_gae_trace_is_separate_from_discount_and_stops_at_rollout_boundary():
    result = temporal_weights(.99, .95, 32)
    assert result['offsets']['31']['fraction_rollout_positions_with_direct_reward'] == 1/32
    assert result['offsets']['32']['fraction_rollout_positions_with_direct_reward'] == 0
    assert result['offsets']['50']['fraction_rollout_positions_with_direct_reward'] == 0
    assert result['offsets']['50']['return_discount'] == pytest.approx(.6050060671)
    assert result['offsets']['50']['nominal_untruncated_gae_trace'] == pytest.approx((.99*.95)**50)
    assert result['gae_trace_efold_seconds'] < result['discount_efold_seconds']


def test_terminal_rescore_uses_last_transition_discount_and_excludes_censoring():
    episodes = [dict(slot=0, start_control=0, controls=2, outcome='failure', discounted_reward_sum=-4.),
                dict(slot=0, start_control=2, controls=5, outcome='success', discounted_reward_sum=-5.),
                dict(slot=1, start_control=3, controls=3, outcome='censored', discounted_reward_sum=-100.)]
    result = return_ordering(episodes, .5)['windows']['all']
    assert result['failure']['rows'] == result['success']['rows'] == 1
    assert result['comparable_slots'] == result['slots_success_mean_below_failure'] == 1
    assert result['fixed_trace_failure_rescore_minus5']['mean'] == -5.5
