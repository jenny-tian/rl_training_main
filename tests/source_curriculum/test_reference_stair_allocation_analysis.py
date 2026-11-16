from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
import reference_stair_allocation_analyze as analysis
from reference_stair_allocation_analyze import crossing_events, foot_metrics, detail_windows


def fixture():
    feet = np.zeros((7,1,4,3),dtype=np.float32)
    feet[:,0,:,0] = np.array([.9,.95,1.05,1.1,1.12,1.14,1.16])[:,None]
    feet[:,0,:,2] = np.array([.10,.11,.13,.072,.072,.072,.072])[:,None]
    forces = np.zeros_like(feet); forces[3:,:,:,2] = 10.
    data = dict(feet=feet,forces=forces,active=np.ones((7,1),bool),
        torque=np.ones((7,1,12)),joint_vel=np.ones((7,1,12)))
    record = dict(env_id=0,height=.05,speed=.4,outcome='success',column=0)
    config = dict(geometry=[dict(fronts=[1.],tops=[.05])])
    return record,data,config


def test_crossing_interpolation_and_sustained_landing_use_distinct_times():
    record,data,config = fixture()
    rows = crossing_events(record,data,config)
    assert len(rows) == 4
    for r in rows:
        assert r['crossing_time_s'] == pytest.approx(.05)
        assert r['interpolated_bottom_above_top_m'] == pytest.approx(.048,abs=1e-7)
        assert r['stable_landing_end_s'] == pytest.approx(.12)
    metrics = foot_metrics(record,data,config)
    assert metrics['HL_max_stably_supported_level'] == 1
    assert metrics['mean_absolute_joint_power_w'] == 12.
    assert metrics['absolute_joint_work_j'] == pytest.approx(1.68)


def test_short_contact_and_post_terminal_crossing_do_not_count():
    record,data,config = fixture()
    data['forces'][4:] = 0
    assert all(r['stable_landing_end_s'] is None for r in crossing_events(record,data,config))
    data['active'][2:] = False
    assert crossing_events(record,data,config) == []


def test_square_stair_corner_crossing_excluded_from_x_face_metric():
    record,data,config = fixture()
    data['feet'][:,:,:,1] = 1.2
    assert crossing_events(record,data,config) == []


def test_focus_windows_exclude_terminal_tail_and_pair_only_pre_tilt_exposure(monkeypatch):
    config = dict(geometry=[dict(fronts=[1.],tops=[.05])])
    report = dict(records=[dict(env_id=0,column=0,outcome='stuck')])
    monkeypatch.setattr(analysis,'read',lambda p: config if p.name=='config.json' else report)
    traces = {}
    for policy,n,tilt in [('zero10100',350,200),('A',300,None),('B',400,250)]:
        feet = np.zeros((400,1,4,3)); feet[...,0]=.98; feet[...,2]=.03
        forces = np.zeros_like(feet); forces[...,0]=-10.; forces[...,2]=10.
        root = np.zeros((400,1,3)); root[100:,:,0]=1.
        gravity = np.zeros((400,1,3)); gravity[...,2]=-1.
        if tilt is not None: gravity[tilt:,:,2]=-.5
        velocity = np.zeros((400,1,6)); velocity[n:]=99.
        action = np.zeros((400,1,12)); action[n:]=99.
        command = np.zeros((400,1,3)); command[...,0]=.4
        traces[policy] = dict(active=(np.arange(400)<n)[:,None],feet=feet,forces=forces,
            root=root,gravity=gravity,velocity=velocity,action=action,command=command)
    paths = {p:Path(p) for p in traces}
    windows,common = detail_windows('test',paths,traces,[0])
    assert len(windows)==6
    for row in windows:
        assert row['mean_body_vx_m_s']==0.
        assert row['mean_action_temporal_std']==0.
        # A vertical-face contact must not be counted as legal support.
        assert row['four_feet_supported_fraction']==0.
    zero = next(r for r in windows if r['policy']=='zero10100' and r['window']=='last_5s')
    assert zero['window_start_s']==2. and zero['window_end_s']==7.
    zero_pair = next(r for r in common if r['comparator']=='zero10100')
    assert zero_pair['common_exposure_s']==2.
    assert zero_pair['before_wall_foot_seconds']==zero_pair['B_wall_foot_seconds']==8.
    a_pair = next(r for r in common if r['comparator']=='A')
    assert a_pair['common_exposure_s']==3.
