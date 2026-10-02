from copy import deepcopy
from docs.analysis.audit_learning_replication import qualify, retention_pass


def fixture():
    s = {'candidate_completion_rate': 1., 'base_completion_rate': .6,
         'candidate_completions': 5, 'retention_pass': True, 'evaluation_safe': True}
    return {str(k): {'complete': True, 'sessions': [deepcopy(s), deepcopy(s)]} for k in (101,107)}


def test_success_requires_improvement_and_does_not_claim_goal_complete():
    t = fixture()
    assert qualify(t)['stable_development_qualified']
    assert not qualify(t)['goal_complete']
    for terrain in t.values():
        for s in terrain['sessions']:
            s['base_completion_rate'] = 1.
    assert not qualify(t)['engineering_qualified']


def test_missing_trials_or_unsafe_session_prevents_qualification():
    for key in ('complete', 'evaluation_safe', 'retention_pass'):
        t = fixture()
        if key == 'complete':
            t['101'][key] = False
        else:
            t['101']['sessions'][0][key] = False
        assert not qualify(t)['engineering_qualified']


def test_direction_must_replicate_and_nine_of_ten_is_not_eight():
    t = fixture()
    for terrain in t.values():
        terrain['sessions'][0].update(candidate_completion_rate=.8, candidate_completions=4)
        terrain['sessions'][1].update(candidate_completion_rate=.8, candidate_completions=4)
    assert qualify(t)['engineering_qualified']
    assert not qualify(t)['stable_development_qualified']


def test_retention_cannot_hide_forward_regression_in_xy_average():
    base = {'fell': False, 'safety_violation': False, 'flat_retention': {
        'comparison_eligible': True, 'steady_velocity_xy_mae': .03,
        'steady_velocity_x_mae': .02, 'steady_velocity_y_mae': .04}}
    candidate = deepcopy(base)
    candidate['flat_retention'].update(steady_velocity_xy_mae=.025, steady_velocity_x_mae=.04)
    assert not retention_pass([candidate], [base])
    assert not retention_pass([], [])
