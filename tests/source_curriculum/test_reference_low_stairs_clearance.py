"""覆盖抬脚高度、停脚、碰阶、真实支撑与重置，不依赖仿真器。"""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/tools'))
from reference_low_stairs_clearance import ClearanceState, height_cost, wall_contact
from reference_low_stairs_rewards import clearance_cost


def case(count=1):
    return dict(bottom=torch.zeros(count, 4), obstacle=torch.full((count, 4), .05),
        upcoming=torch.ones(count, 4, dtype=torch.bool), support=torch.zeros(count, 4, dtype=torch.bool),
        wall=torch.zeros(count, 4, dtype=torch.bool), direction=torch.tensor([[1., 0.]]).repeat(count, 1),
        enabled=torch.ones(count, dtype=torch.bool), upright=torch.ones(count, dtype=torch.bool), dt=.02)


def test_height_has_no_low_lift_plateau_and_never_rewards_hovering():
    bottom = torch.tensor([0., .01, .02, .025, .03, .05, .075, .10])
    actual = height_cost(bottom, torch.full_like(bottom, .05))
    assert torch.all(actual[:6] > actual[1:7])
    assert actual[6:].eq(0.).all() and actual.ge(0.).all() and actual.lt(1.).all()
    assert actual[0].item() == pytest.approx(.69230769)
    force = torch.zeros(4, 4, 3)
    velocity = force.clone(); velocity[..., 0] = .5
    old = clearance_cost(bottom[:4, None].expand(-1, 4), torch.full((4, 4), .05),
        force, velocity, torch.tensor([[1., 0.]]).repeat(4, 1), torch.ones(4, 4, dtype=torch.bool))
    assert old.eq(1.).all()


def test_stationary_and_wall_contact_do_not_erase_an_active_swing():
    x = case(); state = ClearanceState(1, 'cpu')
    initial = state.update(**x, tick=1).clone()
    # 不再看得到前方台阶，目标仍应保留；速度和接触力不作为乘数。
    x['upcoming'][:] = False; x['obstacle'][:] = 0.
    assert torch.equal(state.update(**x, tick=2), initial)
    x['support'][:] = True; x['wall'][:] = True
    for tick in range(3, 30):
        assert torch.equal(state.update(**x, tick=tick), initial)
    assert state.active.all()


def test_wall_contact_can_start_scoring_without_a_preceding_air_frame():
    x = case(); x['support'][:] = True; x['wall'][0, 3] = True
    state = ClearanceState(1, 'cpu')
    result = state.update(**x, tick=1)
    assert result[0, :3].eq(0.).all() and result[0, 3] > 0.


@pytest.mark.parametrize('height', [0., .05])
def test_real_lower_or_upper_landing_is_immediately_free_and_then_releases(height):
    x = case(); state = ClearanceState(1, 'cpu'); state.update(**x, tick=1)
    x['support'][:] = True; x['bottom'][:] = height
    assert state.update(**x, tick=2).eq(0.).all()
    assert state.active.all()
    assert state.update(**x, tick=3).eq(0.).all() and not state.active.any()


def test_valid_loaded_stance_is_not_a_lift_request():
    x = case(); x['support'][:] = True; state = ClearanceState(1, 'cpu')
    for tick in range(5):
        assert state.update(**x, tick=tick).eq(0.).all() and not state.active.any()


def test_brief_support_keeps_target_and_same_tick_does_not_advance_dwell():
    x = case(); state = ClearanceState(1, 'cpu'); state.update(**x, tick=1)
    x['support'][:] = True
    state.update(**x, tick=2)
    for _ in range(5):
        state.update(**x, tick=2)
    assert state.active.all() and torch.allclose(state.dwell, torch.full((1, 4), .02))
    x['support'][:] = False; x['upcoming'][:] = False
    assert state.update(**x, tick=3).gt(0.).all()


@pytest.mark.parametrize('reason', ['non_stair_or_stop_or_yaw', 'tilt', 'reverse'])
def test_command_scope_and_orientation_reset_a_previous_target(reason):
    x = case(); state = ClearanceState(1, 'cpu'); state.update(**x, tick=1)
    x['upcoming'][:] = False
    if reason == 'non_stair_or_stop_or_yaw': x['enabled'][:] = False
    elif reason == 'tilt': x['upright'][:] = False
    else: x['direction'] *= -1
    assert state.update(**x, tick=2).eq(0.).all() and not state.active.any()


def test_selective_reset_and_same_tick_recompute_do_not_touch_other_environments():
    x = case(2); state = ClearanceState(2, 'cpu'); state.update(**x, tick=1)
    state.reset(torch.tensor([0]))
    assert not state.active[0].any() and state.active[1].all()
    x['upcoming'][:] = False
    state.update(**x, tick=1)
    assert state.cost[0].eq(0.).all() and state.cost[1].gt(0.).all()
    state.reset()
    assert state.cost.eq(0.).all() and not state.active.any()


def test_four_legs_and_world_rotation_use_the_same_rules():
    x = case(); x['bottom'][0] = torch.tensor([.0, .01, .02, .03])
    a = ClearanceState(1, 'cpu').update(**x, tick=1)
    permutation = torch.tensor([3, 2, 0, 1])
    x['bottom'] = x['bottom'][:, permutation]; x['direction'][:] = torch.tensor([0., 1.])
    b = ClearanceState(1, 'cpu').update(**x, tick=1)
    assert torch.equal(a[:, permutation], b)


def wall_case():
    force = torch.zeros(1, 4, 3); force[..., 0] = -20.
    distance = torch.full((1, 4, 3), float('inf'))
    normal = torch.full((1, 4, 3, 3), float('inf'))
    distance[:, :, 2] = .015; normal[:, :, 2] = torch.tensor([-1., 0., 0.])
    return dict(force=force, direction=torch.tensor([[1., 0.]]), distance=distance, normal=normal)


def test_lower_sphere_rays_detect_edge_when_center_ray_misses_and_rotate_equally():
    x = wall_case(); assert wall_contact(**x).all()
    x['direction'][:] = torch.tensor([0., 1.])
    x['force'][..., 1] = x['force'][..., 0]; x['force'][..., 0] = 0.
    x['normal'][:, :, 2] = torch.tensor([0., -1., 0.])
    assert wall_contact(**x).all()


@pytest.mark.parametrize('reason', ['far', 'floor', 'weak_force', 'away', 'miss'])
def test_wall_geometry_and_force_are_both_required(reason):
    x = wall_case()
    if reason == 'far': x['distance'][:] = .035
    elif reason == 'floor': x['normal'][:] = torch.tensor([0., 0., 1.])
    elif reason == 'weak_force': x['force'] *= .1
    elif reason == 'away': x['force'] *= -1
    else: x['distance'][:] = float('inf')
    assert not wall_contact(**x).any()


def test_landing_dwell_uses_seconds():
    for dt in (.01, .02):
        x = case(); x['dt'] = dt; state = ClearanceState(1, 'cpu'); state.update(**x, tick=0)
        x['support'][:] = True
        for tick in range(1, round(.04 / dt)):
            state.update(**x, tick=tick); assert state.active.all()
        state.update(**x, tick=round(.04 / dt)); assert not state.active.any()


@pytest.mark.parametrize('height', [.02, .03, .05])
def test_entire_training_height_range_distinguishes_low_lifts_and_world_height(height):
    bottom = torch.tensor([0., .005, .01, .015])
    target = torch.full_like(bottom, height)
    cost = height_cost(bottom, target)
    assert (cost[:-1] > cost[1:]).all()
    assert torch.allclose(cost, height_cost(bottom + 10., target + 10.), atol=2e-5)


@pytest.mark.parametrize('flags', [['--train'], ['--max-iterations', '10'], ['--gpu', '0'],
                                  ['--variant', 'riser_contact']])
def test_validator_cannot_accept_training_or_another_variant(flags, monkeypatch, tmp_path):
    from reference_low_stairs_clearance_validate import main
    monkeypatch.setattr(sys, 'argv', ['validate', '--variant', 'candidate', '--output', str(tmp_path), *flags])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_wall_ray_batch_shape_and_sample_cache(monkeypatch):
    import reference_low_stairs_clearance as module
    x = dict(feet=torch.zeros(2, 4, 3), direction=torch.tensor([[1., 0.], [0., 0.]]),
             enabled=torch.tensor([True, False]), force=torch.zeros(2, 4, 3))
    env = NS(num_envs=2, common_step_counter=10, cfg=NS(scene=NS(terrain=NS(prim_path='ground'))))
    calls = []
    def raycast(starts, directions, mesh, **kwargs):
        assert starts.shape == directions.shape == (2, 12, 3)
        assert directions.norm(dim=-1).eq(1.).all()
        calls.append(1)
        return None, torch.full((2, 12), float('inf')), torch.full((2, 12, 3), float('inf')), None
    monkeypatch.setattr(module, 'sample', lambda env: x)
    monkeypatch.setitem(sys.modules, 'isaaclab.sensors', NS(RayCaster=NS(meshes={'ground': object()})))
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.warp', NS(raycast_mesh=raycast))
    a = module.clearance_sample(env); b = module.clearance_sample(env)
    assert a is b and len(calls) == 1 and not a['wall'].any()
    assert a['distance'].shape == (2, 4, 3)
    env.common_step_counter += 1
    module.clearance_sample(env)
    assert len(calls) == 2
