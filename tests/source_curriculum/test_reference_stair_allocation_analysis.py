from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_stair_allocation_analyze import crossing_events, foot_metrics


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
