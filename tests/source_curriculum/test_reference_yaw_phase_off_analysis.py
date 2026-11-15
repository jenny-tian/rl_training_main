import copy

from reference_low_stairs_yaw_phase_off_analyze import pair_motion,approved_screen,CASES,WINDOWS


def test_motion_pairing_uses_identical_repeats_and_reports_missing_group():
    rows=[dict(policy=p,profile='nominal',case='fast_left',window='steady',repeat=i,value=float(i))
          for p in ('parent','phaseoff') for i in range(3)]
    rows=[r for r in rows if not (r['policy']=='phaseoff' and r['repeat']==2)]
    row,=pair_motion(rows,'parent')
    assert row['n']==2 and row['baseline_value']==row['candidate_value']==.5
    rows=[r for r in rows if r['policy']=='parent']
    row,=pair_motion(rows,'parent')
    assert row['n']==0 and row['candidate_value'] is None


def fixture():
    tables={k:[] for k in ('stairs_groups','stairs_heights','turn_pairs','turn_failures','turn_records')}
    for p in ('parent','zero','phaseoff'):
        for profile in ('nominal','training_randomized'):
            for direction in ('up','down'):
                for height in (0.,.02,.03,.05):
                    row=dict(policy=p,profile=profile,direction=direction,height=height,success=60,orientation_failure=0)
                    tables['stairs_heights'].append(row)
                    for speed in (.32,.4,.6):tables['stairs_groups'].append(dict(row,speed=speed,success=20))
    for profile in ('nominal','training_randomized'):
        for case in CASES:
            for window in WINDOWS:
                tables['turn_pairs'].append(dict(baseline_policy='parent',profile=profile,case=case,window=window,
                                                n=20,baseline_mae=.1,candidate_mae=.1))
    tables['turn_failures']=[dict(policy='phaseoff',new_vs_parent=0)]
    tables['turn_records']=[dict(policy='phaseoff',profile='training_randomized',env_id=57,outcome='survived',duration_s=20)]
    return tables


def test_screen_keeps_stronger_upstairs_and_recovery_gates():
    tables=fixture();assert approved_screen(tables)['status']=='passed'
    row=next(r for r in tables['stairs_groups'] if (r['policy'],r['profile'],r['direction'],r['height'],r['speed'])==
             ('phaseoff','nominal','up',.05,.32))
    row['success']=14
    result=approved_screen(tables)
    assert result['status']=='failed' and result['original_parent_screen']['engineering_checks_passed']
    assert any(not r['passed'] and r['name']=='zero_up_5cm_0.32_preserved' for r in result['additional_checks'])
    tables=fixture();tables['turn_records'][0].update(outcome='orientation_failure',duration_s=.82)
    assert approved_screen(tables)['status']=='failed'


def test_random_regression_requires_review_and_missing_turn_window_fails():
    tables=fixture()
    next(r for r in tables['stairs_groups'] if r['policy']=='phaseoff' and r['profile']=='training_randomized')['success']=19
    assert approved_screen(tables)['status']=='requires_review'
    tables=fixture();tables['turn_pairs'].pop()
    result=approved_screen(tables)
    assert result['status']=='failed' and result['original_parent_screen']['turn_missing_windows']
