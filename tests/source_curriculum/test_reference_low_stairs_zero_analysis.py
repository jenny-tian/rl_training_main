import copy

import numpy as np
import pytest

from reference_low_stairs_zero_preservation_analyze import (
    POLICIES, CASES, WINDOWS, DT, common_windows, matched_contacts, paired_records)


def test_common_windows_excludes_early_failures_and_keeps_missing_windows():
    windows={p:[dict(case='turn_left',window='steady',repeat=i,complete_window=True,
                    yaw_command=.5,yaw_actual=float(i),yaw_mae=float(i)) for i in range(3)]
             for p in POLICIES}
    windows['zero10100'][2].update(complete_window=False,yaw_actual=999.,yaw_mae=999.)
    windows['guarded10100'].pop(0)
    rows=common_windows(windows,'nominal')
    assert len(rows)==len(CASES)*len(WINDOWS)
    row=next(r for r in rows if (r['case'],r['window'])==('turn_left','steady'))
    assert row['n']==1 and row['starts']==20
    assert all(row[f'{p}_yaw_actual']==1. for p in POLICIES)
    empty=next(r for r in rows if r['window']=='tail')
    assert empty['n']==0 and all(empty[f'{p}_yaw_mae'] is None for p in POLICIES)


@pytest.mark.parametrize('no_overlap',[False,True])
def test_contacts_use_common_pre_tilt_time_and_both_approach_positions(no_overlap):
    a=dict(root=np.array([[1.,0.],[1.,0.],[1.,0.],[1.,0.]]),cutoff=3,speed=.4,
           collision=np.ones((4,4),bool),wall=np.zeros((4,4),bool))
    b=copy.deepcopy(a);b['cutoff']=2;b['root'][0,0]=0.
    b['collision'][:,2:]=False
    if no_overlap:b['root'][:,0]=0.
    row,=matched_contacts({'guarded10100':{7:a},'zero10100':{7:b}},'nominal',False)
    assert row['exposure_s']==(0. if no_overlap else DT)
    assert row['guarded10100_collision_foot_seconds']==(0. if no_overlap else 4*DT)
    assert row['zero10100_collision_foot_seconds']==(0. if no_overlap else 2*DT)
    assert row['guarded10100_wall_foot_seconds']==0.


@pytest.mark.parametrize('kind,key,value',[
    ('turn','case','turn_right'),('turn','repeat',3),
    ('stairs','height',.03),('stairs','speed',.6),('stairs','repeat',3),('stairs','column',3),
])
def test_pairing_rejects_changed_case_despite_equal_environment_ids(kind,key,value):
    before={'records':[dict(env_id=0,case='turn_left',repeat=0,height=.05,speed=.4,column=2)]}
    after=copy.deepcopy(before)
    paired_records(before,after,kind)
    after['records'][0][key]=value
    with pytest.raises(AssertionError):paired_records(before,after,kind)


@pytest.mark.parametrize('duplicate',[False,True])
def test_pairing_rejects_missing_or_duplicate_environment_ids(duplicate):
    before={'records':[dict(env_id=0,case='turn_left',repeat=0)]}
    after=copy.deepcopy(before)
    if duplicate:after['records']*=2
    else:after['records']=[]
    with pytest.raises(AssertionError):paired_records(before,after,'turn')
