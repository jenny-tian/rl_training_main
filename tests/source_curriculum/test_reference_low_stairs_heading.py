"""反馈边界、准备时序和起点配对的 CPU 检查。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reference_low_stairs_heading_probe import CASES, attach_feedback, feedback
from reference_low_stairs_heading_analyze import transition_rows, grouped, common_windows
from reference_low_stairs_zero_preservation_analyze import paired_records


def test_wrap_sign_limit_and_forward_gate():
    heading = torch.tensor([.2, -.2, 3., -3., 2*torch.pi+.2, .5, -.5])
    forward = torch.tensor([.4, .4, .4, .4, .4, 0., -.4])
    torch.testing.assert_close(feedback(heading, forward), torch.tensor([-.1, .1, -.8, .8, -.1, 0., 0.]))


def test_preparation_and_command_update_order():
    command = SimpleNamespace(vel_command_b=torch.zeros(1, 3))
    heading = torch.tensor([.4])
    calls = []
    def original_compute(dt):
        calls.append(dt)
        command.vel_command_b[:] = torch.tensor([[.4, 0., 0.]])
    manager = SimpleNamespace(compute=original_compute, get_term=lambda name:command)
    base = SimpleNamespace(command_manager=manager,
        scene={'robot':SimpleNamespace(data=SimpleNamespace(heading_w=heading))})
    def step(action):
        manager.compute(.02)
        return command.vel_command_b.clone()
    base.step = step
    attach_feedback(base)
    for _ in range(50):
        assert base.step(None)[0, 2] == 0.
    assert manager.compute is original_compute
    for _ in range(3):
        torch.testing.assert_close(base.step(None)[0, 2], torch.tensor(-.2))
    assert len(calls) == 53


def test_only_approved_new_groups():
    assert list(CASES.values()) == [('nominal', 'down'), ('training_randomized', 'up'), ('training_randomized', 'down')]
    assert 'heading_up_nominal' not in CASES


@pytest.mark.parametrize('change', ['repeat', 'missing', 'duplicate'])
def test_reject_unpaired_start(change):
    a = dict(env_id=0, height=.05, speed=.32, repeat=0, column=3)
    before, after = {'records':[a]}, {'records':[dict(a)]}
    if change == 'repeat':
        after['records'][0]['repeat'] = 1
    elif change == 'missing':
        after['records'] = []
    else:
        after['records'].append(dict(a))
    with pytest.raises(AssertionError):
        paired_records(before, after, 'stairs')


def test_equal_total_does_not_hide_new_failure():
    def record(i, outcome):
        return dict(env_id=i, height=.05, speed=.32, repeat=i, column=3,
            outcome=outcome, duration_s=20., best_risers=5)
    before={'records':[record(0,'success'), record(1,'lateral_exit')]}
    after={'records':[record(0,'orientation_failure'), record(1,'success')]}
    rows=transition_rows(before,after,'nominal','up')
    group=grouped(rows,('profile','direction'))[0]
    assert group['before_success']==group['after_success']==1
    assert group['gained_success']==group['lost_success']==group['new_orientation_failure']==1
    assert rows[0]['before']=='success' and rows[0]['after']=='orientation_failure'


def test_incomplete_common_window_is_explicitly_retained():
    pair=dict(profile='nominal',direction='up',env_id=0,height=.05,speed=.32,repeat=0)
    quat=np.zeros((1000,1,4));quat[...,0]=1.
    old=dict(active=np.ones((1000,1),bool),quat=quat,root=np.zeros((1000,1,3)),velocity=np.zeros((1000,1,6)))
    new={k:v.copy() for k,v in old.items()};new['active'][100:,0]=False
    rows=common_windows([pair],old,new)
    assert len(rows)==4 and all(not r['complete'] for r in rows)
    assert all(r['before_yaw_abs_mean_deg'] is None and r['after_yaw_abs_mean_deg'] is None for r in rows)
