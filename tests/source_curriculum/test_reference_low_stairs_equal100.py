"""检查分析口径：三方存活交集、停滞暴露和共同截断。"""
import numpy as np

from reference_low_stairs_equal100_analyze import PRIMARY, common_turns, stagnation, paired_contact


def test_common_windows_exclude_each_early_failure():
    windows = {}
    for index, policy in enumerate(PRIMARY):
        windows[policy] = [dict(case='fast_left', window='steady', repeat=i,
            complete_window=i!=index, yaw_command=.8, yaw_actual=i/10, yaw_mae=1-i/10) for i in range(4)]
    row = next(r for r in common_turns(windows,'nominal') if r['case']=='fast_left' and r['window']=='steady')
    assert row['n'] == 1
    assert all(abs(row[f'{p}_yaw_actual']-.3)<1e-8 for p in PRIMARY)


def test_missing_windows_keep_zero_denominator():
    rows = common_turns({p:[] for p in PRIMARY},'nominal')
    assert len(rows) == 48 and all(r['n']==0 for r in rows)
    assert all(r['parent10000_yaw_mae'] is None for r in rows)


def test_stagnation_rejects_short_stop_and_masks_post_tilt():
    x = np.full(200,1.)
    before = np.zeros(200,bool); before[50:74] = True
    assert stagnation(x,before)['first_stagnation_window_end_s'] is None
    before[74] = True
    assert stagnation(x,before)['first_stagnation_window_end_s'] == 1.02
    assert stagnation(x,before)['longest_slow_window_run_s'] == .5


def test_forward_progress_does_not_count_as_stagnation():
    x = np.arange(200)*.02*.2
    assert stagnation(x,np.ones(200,bool))['first_stagnation_window_end_s'] is None


def test_contact_pairing_requires_common_approach_and_pre_tilt():
    series = {}
    for j,p in enumerate(PRIMARY):
        root = np.ones((100,3)); root[:10+j,0] = .5
        series[p] = {180:dict(root=root,cutoff=80-j*10,collision=np.ones((100,4),bool),
                             wall=np.zeros((100,4),bool),speed=.32)}
    row = paired_contact(series,'nominal',False)[0]
    assert row['exposure_s'] == (60-12)*.02
    assert all(row[f'{p}_collision_foot_seconds']==4*row['exposure_s'] for p in PRIMARY)


def test_contact_pairing_handles_initial_tilt_without_nan():
    series = {p:{180:dict(root=np.ones((10,3)),cutoff=0,collision=np.zeros((10,4),bool),
                         wall=np.zeros((10,4),bool),speed=.32)} for p in PRIMARY}
    row = paired_contact(series,'training_randomized',False)[0]
    assert row['exposure_s'] == 0
    assert row['parent10000_collision_foot_seconds'] == 0
