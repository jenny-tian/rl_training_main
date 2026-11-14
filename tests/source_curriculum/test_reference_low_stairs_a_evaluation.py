"""Guard completed-model identity, paired regressions, and the agreed screen."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))

from reference_low_stairs_a_analyze import CASES, WINDOWS, OUTCOMES, screen, stairs_heights
from reference_low_stairs_a_evaluate import BASELINE, MODEL, verify_run, wait_training, same_configuration


def inputs():
    heights,groups=[],[]
    for profile in ('nominal','training_randomized'):
        for direction in ('up','down'):
            report=json.loads((BASELINE/f'{direction}_{profile}'/'report.json').read_text())
            for policy in ('baseline','candidate'):
                heights.extend(dict(policy=policy,**r) for r in stairs_heights(report))
                groups.extend(dict(policy=policy,profile=profile,direction=direction,
                    **{k:v for k,v in g.items() if k!='outcomes'},
                    **{k:g['outcomes'].get(k,0) for k in OUTCOMES}) for g in report['groups'])
    pairs=[dict(profile=profile,case=case,window=window,n=20,baseline_mae=.2,candidate_mae=.2)
           for profile in ('nominal','training_randomized') for case in CASES for window in WINDOWS]
    failures=[dict(profile='nominal',case='turn_left',new_failures=0)]
    return heights,groups,pairs,failures


def passing_inputs():
    heights,groups,pairs,failures=inputs()
    for row in heights:
        if (row['policy'],row['profile'],row['direction'],row['height'])==('candidate','nominal','up',.05):
            row['orientation_failure']=11;row['success']=32
    for row in groups:
        if (row['policy'],row['profile'],row['direction'],row['height'])==('candidate','nominal','up',.05):
            row['success']={.32:8,.4:10,.6:14}[row['speed']]
    return heights,groups,pairs,failures


def test_baseline_cannot_pass_the_improvement_screen_by_surviving():
    result=screen(*inputs())
    assert result['status']=='failed'
    failed={r['name'] for r in result['checks'] if not r['passed']}
    assert failed=={'nominal_up_5cm_0.32_success','nominal_up_5cm_0.4_success','nominal_up_5cm_orientation_failures'}


def test_exact_thresholds_and_no_regressions_pass():
    assert screen(*passing_inputs())['status']=='passed'


def test_a_regression_in_a_late_turn_window_blocks_recommendation():
    values=passing_inputs()
    values[2][-1]['candidate_mae']=.221
    result=screen(*values)
    assert result['status']=='failed' and len(result['turn_error_regressions'])==1


def test_new_paired_failure_is_not_hidden_by_equal_total_failure_counts():
    values=passing_inputs()
    values[3][0].update(new_failures=1,baseline_failures=1,candidate_failures=1,recovered=1)
    assert screen(*values)['status']=='failed'


def test_missing_complete_turn_windows_cannot_pass():
    values=passing_inputs();values[2].pop()
    result=screen(*values)
    assert result['status']=='failed' and len(result['turn_missing_windows'])==1


def test_randomized_regression_is_reported_separately():
    values=passing_inputs()
    row=next(r for r in values[1] if r['policy']=='candidate' and r['profile']=='training_randomized')
    row['success']-=1
    result=screen(*values)
    assert len(result['random_decreases_require_review'])==1
    assert result['random_decreases_require_review'][0]['success_delta']==-1


def test_wait_does_not_start_any_training(monkeypatch):
    import reference_low_stairs_a_evaluate as module
    monkeypatch.setattr(module,'read',lambda _:dict(status='running'))
    with pytest.raises(RuntimeError,match='has not completed'):
        wait_training(False)


def test_common_configuration_only_excludes_unsampled_policy_metadata():
    a=dict(model='old',action_mode='mean',action_std=[.5],reward_scoring='reference_frozen_common',configuration={'dt':.005})
    b=copy.deepcopy(a);b.update(model='new',action_std=[.4])
    assert same_configuration(a,b)
    b['configuration']['dt']=.01
    assert not same_configuration(a,b)
    b['configuration']['dt']=.005
    a['action_mode']=b['action_mode']='sampled'
    assert not same_configuration(a,b)


@pytest.mark.parametrize('field,value',[
    ('model','wrong_model'),('model_sha256','b'*64),('actor_unchanged',False),('probe',True),
    ('action_mode','sample'),('first_episodes_only',False),('height_set','original'),
])
def test_evaluation_reuse_rejects_wrong_identity_or_protocol(field,value,tmp_path):
    report=json.loads((BASELINE/'up_nominal/report.json').read_text())
    report.update(model=MODEL,model_sha256='a'*64)
    execution=dict(status='completed',exit_code=0,physical_gpu=1)
    config=dict(reward_scoring='reference_frozen_common')
    for name,obj in (('report',report),('execution',execution),('config',config)):
        (tmp_path/f'{name}.json').write_text(json.dumps(obj))
    verify_run(tmp_path,'stairs','nominal','a'*64,'up')
    report[field]=value
    (tmp_path/'report.json').write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        verify_run(tmp_path,'stairs','nominal','a'*64,'up')
