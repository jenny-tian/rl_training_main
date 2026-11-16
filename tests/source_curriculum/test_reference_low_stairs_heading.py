"""反馈边界、准备时序和起点配对的 CPU 检查。"""
from types import SimpleNamespace

import pytest
import torch

from reference_low_stairs_heading_probe import CASES, attach_feedback, feedback
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
