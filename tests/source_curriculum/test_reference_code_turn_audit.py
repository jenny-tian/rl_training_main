"""Regression checks for diagnostic interpretation, without modifying policy behavior."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_reward_code_audit import audit
from reference_turn_diagnose import CASES,schedule
from reference_turn_analyze import analyze


@pytest.fixture(scope='module')
def probes(): return audit()


def test_actual_standing_gate_penalizes_both_yaw_directions(probes):
    p=probes['standstill']['weighted_penalties']
    assert p[0]==p[1]==p[2]<0 and p[3]==0


def test_actual_static_and_short_step_reward_loopholes(probes):
    assert probes['static_gait']['ratio']>.6
    assert probes['static_phase']['mean_weighted_rate']>.9
    assert probes['air_time']['landing_rates'][-1]<0
    assert probes['air_time']['yaw_landing_rates'][1]==probes['air_time']['yaw_landing_rates'][2]<0
    assert probes['air_time']['no_liftoff_rates'][-1]==0


def test_actual_sampler_has_heading_yaw_and_reintroduces_low_xy(probes):
    p=probes['command_sampling']
    assert p['pure_yaw_heading_controlled']==p['pure_yaw']>0
    assert p['constant_yaw_nonheading']==0
    assert p['low_nonzero_xy_after_special_overrides']>0


def test_command_transitions_and_left_right_symmetry():
    assert schedule('turn_left',1.99)[2]==0
    assert schedule('turn_left',2.)[2]==.5
    assert schedule('stop_left',7.98)[2]==.5 and schedule('stop_left',8.)[2]==0
    assert schedule('reverse_right',7.98)[2]==-.5 and schedule('reverse_right',8.)[2]==.5
    for left,right in [('low_left','low_right'),('fast_left','fast_right'),('arc_left','arc_right')]:
        assert schedule(left,5)[2]==-schedule(right,5)[2]


def test_failed_turn_cannot_inflate_complete_window_tracking(tmp_path):
    # One of two starts per case fails at 5 s; its post-reset values must be excluded.
    n=len(CASES)*2; steps=1000; active=np.ones((steps,n),bool); active[250:,0]=False
    done=np.zeros_like(active); done[249,0]=True
    records=[dict(env_id=i,case=case,repeat=r,outcome='orientation_failure' if i==0 else 'survived',
                  duration_s=5. if i==0 else 20.) for i,(case,r) in enumerate((c,r) for c in CASES for r in range(2))]
    report=dict(status='passed',probe=False,actor_unchanged=True,model='update10000',profile='nominal',
                step_dt=.02,records=records,reward_names=['stand_still','feet_air_time_ang_z_lite3','phase_foot_trajectory_exp'])
    (tmp_path/'report.json').write_text(json.dumps(report)); (tmp_path/'execution.json').write_text('{"status":"completed"}')
    velocity=np.zeros((steps,n,6)); velocity[250:,0,5]=999
    np.savez_compressed(tmp_path/'trace.npz',active=active,terminated=done,velocity=velocity,
        command=np.zeros((steps,n,3)),root=np.zeros((steps,n,3)),heading=np.zeros((steps,n)),
        heading_error=np.zeros((steps,n)),reward_rates=np.zeros((steps,n,3)))
    _,_,_,groups=analyze(tmp_path)
    g=next(g for g in groups if g['case']=='standing' and g['window']=='steady')
    assert g['starts']==2 and g['orientation_failures']==1 and g['complete_windows']==1
    assert g['yaw_mae']==0
