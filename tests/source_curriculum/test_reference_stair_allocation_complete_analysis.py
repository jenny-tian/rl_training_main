from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_stair_allocation_complete_analyze import outcome_pairs, paired_turn_windows


def test_equal_success_totals_keep_both_new_failure_and_recovered_case():
    old=[dict(env_id=0,outcome='success',duration_s=10.),dict(env_id=1,outcome='timeout',duration_s=20.)]
    new=[dict(env_id=0,outcome='orientation_failure',duration_s=8.),dict(env_id=1,outcome='success',duration_s=12.)]
    pairs=outcome_pairs(old,new,'stairs')
    assert pairs[0]['lost_success'] and pairs[0]['new_orientation_failure']
    assert pairs[1]['gained_success']


def test_turn_pairs_retain_lost_windows_and_exempt_only_low_tracking():
    def row(case,i,complete,mae):
        return dict(case=case,env_id=i,window='steady',complete_window=complete,yaw_actual=0.,yaw_mae=mae)
    old=[row('fast_left',0,True,.2),row('fast_left',1,True,.4),row('low_left',2,True,.1)]
    new=[row('fast_left',0,False,9.),row('fast_left',1,True,.5),row('low_left',2,True,.5)]
    pairs=paired_turn_windows(old,new,'nominal')
    fast=next(r for r in pairs if r['case']=='fast_left' and r['window']=='steady')
    assert fast['paired_complete']==1 and fast['lost_complete']==1 and fast['before_mae']==.4
    assert fast['descriptive_regression']
    low=next(r for r in pairs if r['case']=='low_left' and r['window']=='steady')
    assert low['after_mae']==.5 and not low['tracking_in_scope'] and not low['descriptive_regression']
    empty=next(r for r in pairs if r['case']=='fast_left' and r['window']=='tail')
    assert empty['paired_complete']==0 and empty['mae_delta'] is None
