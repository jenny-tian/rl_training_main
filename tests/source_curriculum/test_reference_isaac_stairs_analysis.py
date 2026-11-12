"""Ensure completed episodes cannot absorb automatic-reset trajectory samples."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/tools'))
from reference_isaac_stairs_analyze import analyze


def make_run(path, active):
    report=dict(status='passed',probe=False,profile='nominal',steps=4,episodes=1,
                step_dt=1.,reward_names=['tracking'],records=[dict(env_id=0,height=.1,
                speed=.4,repeat=0,column=0,outcome='orientation_failure',duration_s=2.,best_risers=0)])
    config=dict(geometry=[dict(fronts=[1.2],tops=[.1])])
    for name,value in [('report',report),('config',config),('execution',dict(status='completed'))]:
        (path/f'{name}.json').write_text(json.dumps(value))
    root=np.zeros((4,1,3)); root[:,0,0]=[.8,1.,100.,100.]
    feet=np.zeros((4,1,4,3)); feet[:,:,:,0]=1.18; feet[:,:,:,2]=.022
    velocity=np.zeros((4,1,6)); velocity[:,0,0]=[.2,.4,100.,100.]
    rewards=np.array([2.,4.,1000.,1000.]).reshape(4,1,1)
    gravity=np.zeros((4,1,3)); gravity[:,:,2]=-1.
    np.savez_compressed(path/'trace.npz',active=np.array(active,bool).reshape(4,1),
        root=root,feet=feet,velocity=velocity,reward_rates=rewards,gravity=gravity,
        forces=np.zeros((4,1,4,3)),action=np.ones((4,1,12)),scan_mean=np.zeros((4,1)))


def test_analysis_excludes_reset_samples_and_keeps_window_masks_independent(tmp_path):
    make_run(tmp_path,[True,True,False,False])
    _,_,data,episodes,windows,_=analyze(tmp_path)
    assert episodes[0]['return_rate']==3.
    assert episodes[0]['mean_forward_velocity']==pytest.approx(.3)
    assert episodes[0]['last5_progress_m']==pytest.approx(.2)
    assert data['active'][:,0].tolist()==[True,True,False,False]
    last=next(row for row in windows if row['window']=='last5')
    assert last['samples']==2 and last['reward_tracking']==3.


def test_analysis_rejects_noncontiguous_first_episode_mask(tmp_path):
    make_run(tmp_path,[True,False,True,False])
    with pytest.raises(AssertionError):
        analyze(tmp_path)
