from types import SimpleNamespace

import pytest
import torch

from rl_training.policy_route import lateral_velocity_values, route_lateral_velocity_cost, route_lateral_velocity_trace


def test_centerline_target_saturates_and_penalizes_outward_velocity_more():
    origins = torch.tensor([[10., 20., 0.]]).repeat(5, 1)
    pos = origins.clone()
    pos[:, 1] += torch.tensor([0., .1, -.1, 2., -2.])
    vel = torch.zeros_like(pos)
    target, cost = lateral_velocity_values(pos, vel, origins, gain_per_second=.5, maximum_m_s=.15)
    torch.testing.assert_close(target, torch.tensor([0., -.05, .05, -.15, .15]))
    assert cost[0] == 0
    vel[:, 1] = target
    _, inward = lateral_velocity_values(pos, vel, origins, gain_per_second=.5, maximum_m_s=.15)
    vel[:, 1] *= -1
    _, outward = lateral_velocity_values(pos, vel, origins, gain_per_second=.5, maximum_m_s=.15)
    assert torch.count_nonzero(inward) == 0 and (outward[1:] > cost[1:]).all()
    mirrored_pos = 2*origins-pos
    mirrored_vel = -vel
    mirrored_target, mirrored_cost = lateral_velocity_values(mirrored_pos, mirrored_vel, origins, gain_per_second=.5, maximum_m_s=.15)
    torch.testing.assert_close(mirrored_target, -target)
    torch.testing.assert_close(mirrored_cost, outward)


@pytest.mark.parametrize('gain,maximum', [(0., .15), (-1., .15), (float('nan'), .15), (.5, 0.), (.5, -1.), (.5, float('inf'))])
def test_invalid_return_scale_is_rejected(gain, maximum):
    with pytest.raises(ValueError):
        lateral_velocity_values(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3), gain_per_second=gain, maximum_m_s=maximum)


def test_world_velocity_snapshot_survives_reset_and_checks_actual_dt_charge():
    # Nonzero yaw must not rotate world velocity into the body frame.
    root = torch.tensor([[0., .4, .3, .70710678, 0., 0., .70710678, .3, .2, 0., 0., 0., 0.]])
    class Scene(dict):
        pass
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(root_state_w=root)))
    scene.env_origins = torch.zeros(1, 3)
    env = SimpleNamespace(scene=scene, step_dt=.02)
    cost = route_lateral_velocity_cost(env, .5, .15)
    torch.testing.assert_close(cost, torch.tensor([.35**2]))
    env.reward_manager = SimpleNamespace(active_terms=['course_lateral_velocity'], _step_reward=-4*cost[:, None],
        get_term_cfg=lambda _: SimpleNamespace(weight=-4.))
    root.zero_()
    scene.env_origins.add_(10.)
    result = route_lateral_velocity_trace(env)
    assert result['velocity_root_state'][0, 1] == .4 and not result['velocity_origins'].any()
    torch.testing.assert_close(result['velocity_reward'], -4*cost*.02)
    env.reward_manager._step_reward *= .02
    with pytest.raises(AssertionError):
        route_lateral_velocity_trace(env)
    env._route_velocity_snapshot = None
    with pytest.raises(RuntimeError):
        route_lateral_velocity_trace(env)
