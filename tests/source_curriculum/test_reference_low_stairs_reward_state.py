"""检查旁路解释与实际状态更新一致，覆盖支撑清除及落脚拒绝。"""
import pytest
import torch

from reference_low_stairs_reward_observer import RewardObserver


def case():
    return dict(feet=torch.zeros(1,4,3),bottom=torch.zeros(1,4),ground=torch.zeros(1,4),
        obstacle=torch.full((1,4),.05),force=torch.zeros(1,4,3),foot_velocity=torch.zeros(1,4,3),
        direction=torch.tensor([[1.,0.]]),speed=torch.tensor([.32]),enabled=torch.tensor([True]),
        upright=torch.tensor([True]),support=torch.zeros(1,4,dtype=torch.bool),root_xy=torch.zeros(1,2),
        velocity=torch.zeros(1,3),upcoming=torch.ones(1,4,dtype=torch.bool))


def walls():
    return dict(wall=torch.zeros(1,4,dtype=torch.bool),distance=torch.full((1,4,3),float('inf')),
        normal=torch.full((1,4,3,3),float('inf')))


def test_low_landing_release_is_observed_and_never_awarded():
    o=RewardObserver(1,'cpu');x=case();w=walls()
    a=o.observe(x,w,.02,1);assert a['clear_start'].all()
    x['support'][:]=True
    a=o.observe(x,w,.02,2)
    assert a['clearance_rate'].eq(0.).all() and not a['clear_release_support'].any()
    a=o.observe(x,w,.02,3)
    assert a['clear_release_lower'].all() and not a['clear_after_active'].any()
    assert not a['land_awarded'].any()


def test_wall_support_does_not_release_clearance_target():
    o=RewardObserver(1,'cpu');x=case();w=walls();x['support'][:]=True;w['wall'][:]=True
    for tick in range(4):
        a=o.observe(x,w,.02,tick)
        assert a['clear_after_active'].all() and not a['clear_release_support'].any()


@pytest.mark.parametrize('body_forward',[True,False])
def test_real_landing_dwell_and_body_progress_guard(body_forward):
    o=RewardObserver(1,'cpu');x=case();w=walls();x['support'][:]=True
    o.observe(x,w,.02,0)
    x['support'][:]=False;x['bottom'][:]=.08;x['feet'][...,0]=.1
    x['root_xy'][...,0]=.04 if body_forward else .0
    o.observe(x,w,.02,1);o.observe(x,w,.02,2)
    x['support'][:]=True;x['ground'][:]=.05;x['bottom'][:]=.05
    a=o.observe(x,w,.02,3)
    assert a['land_qualifies'].all().item()==body_forward
    assert a['land_rule_body_forward'].all().item()==body_forward
    assert not a['land_awarded'].any()
    o.observe(x,w,.02,4);a=o.observe(x,w,.02,5)
    assert a['land_awarded'].all().item()==body_forward
    assert a['landing_rate'].item()==(50. if body_forward else 0.)


def test_clearance_sufficient_height_has_zero_cost_and_duplicate_tick_rejected():
    o=RewardObserver(1,'cpu');x=case();x['bottom'][:]=.075;w=walls()
    a=o.observe(x,w,.02,1);assert a['clearance_rate'].eq(0.).all()
    with pytest.raises(AssertionError):o.observe(x,w,.02,1)
    o.reset();a=o.observe(x,w,.02,1);assert not a['land_awarded'].any()
