from copy import deepcopy
from docs.analysis.audit_c0_confirmation import assess


def child(progress):
    task=[dict(phase='evaluation',status='complete',valid_complete_episode=True,progress_m=p,
               completion=p>=8,fell=False,safety_violation=False) for p in progress]
    flat=[dict(phase='retention',status='complete',fell=False,safety_violation=False,
               flat_retention={'comparison_eligible':True,'steady_velocity_xy_mae':.02,
                               'steady_velocity_x_mae':.02,'steady_velocity_y_mae':.02}) for _ in range(2)]
    return {'trials':task+flat,'all_expected_trials_complete':True}


def test_confirmation_requires_both_new_task_gain_and_flat_retention():
    candidate,base=child([8,8,8]),child([5,5,5])
    assert assess(candidate,base,[1,0])['confirmation_pass']
    assert not assess(candidate,base,[1,0])['stable_gait_qualified']
    candidate['trials'][3]['flat_retention']['steady_velocity_x_mae']=.06
    assert not assess(candidate,base,[1,0])['confirmation_pass']
    assert not assess(child([4,4,4]),base,[1,0])['confirmation_pass']


def test_incomplete_retention_and_missing_attempts_cannot_pass():
    candidate,base=child([8,8,8]),child([5,5,5])
    candidate['trials'][-1]['status']='aborted'
    assert not assess(candidate,base,[1,0])['confirmation_pass']
    candidate=child([8,8,8]);candidate['all_expected_trials_complete']=False
    assert not assess(candidate,base,[1,0])['confirmation_pass']
