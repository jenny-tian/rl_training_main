"""Explicit simulator-localized route inputs for the full Lite3 neural policy."""
import math
import torch

ROUTE_SCHEMA = 'sim-localization-v1'


def route_features(root_position, root_quaternion, origins):
    """Normalized cross-track error, sine yaw and cosine yaw (world +X course)."""
    w, x, y, z = root_quaternion.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    lateral = ((root_position[..., 1] - origins[..., 1]) / 1.65).clamp(-2, 2)
    return torch.stack((lateral, yaw.sin(), yaw.cos()), dim=-1)


def commands_with_route(env, command_name):
    robot = env.scene['robot']
    route = route_features(robot.data.root_pos_w, robot.data.root_quat_w, env.scene.env_origins)
    return torch.cat((env.command_manager.get_command(command_name), route), dim=-1)


def configure_route_observations(cfg):
    for name in ('policy', 'critic'):
        term = getattr(cfg.observations, name).velocity_commands
        term.func = commands_with_route
        term.params = {'command_name': 'base_velocity'}


def lookahead_heading_values(root_position, root_quaternion, origins, *, lookahead_m, maximum_heading_deg):
    """Training reward target points toward the centerline, with a bounded turn."""
    if not math.isfinite(lookahead_m) or lookahead_m <= 0:
        raise ValueError('positive finite heading lookahead required')
    if not math.isfinite(maximum_heading_deg) or not 0 < maximum_heading_deg < 90:
        raise ValueError('heading limit must be finite and between0 and90 degrees')
    features = route_features(root_position, root_quaternion, origins)
    yaw = torch.atan2(features[..., 1], features[..., 2])
    lateral = root_position[..., 1] - origins[..., 1]
    bound = math.radians(maximum_heading_deg)
    target = torch.atan2(-lateral, torch.full_like(lateral, lookahead_m)).clamp(-bound, bound)
    return target, 1 - torch.cos(yaw-target), 1 - features[..., 2]


def route_heading_cost(env, lookahead_m=None, maximum_heading_deg=20., record=False):
    robot = env.scene['robot']
    if lookahead_m is None:
        return 1 - route_features(robot.data.root_pos_w, robot.data.root_quat_w, env.scene.env_origins)[..., 2]
    target, cost, old_cost = lookahead_heading_values(robot.data.root_pos_w, robot.data.root_quat_w,
        env.scene.env_origins, lookahead_m=lookahead_m, maximum_heading_deg=maximum_heading_deg)
    if record:
        env._route_heading_snapshot = dict(heading_root_state=robot.data.root_state_w.detach().clone(),
            heading_origins=env.scene.env_origins.detach().clone(), heading_target=target.detach().clone(),
            heading_cost=cost.detach().clone(), heading_zero_target_cost=old_cost.detach().clone())
    return cost


def route_heading_trace(env):
    """Read the reward-time snapshot before reset, and verify its actual dt charge."""
    snapshot = env._route_heading_snapshot
    if snapshot is None:
        raise RuntimeError('heading reward did not record this control')
    index = env.reward_manager.active_terms.index('course_heading')
    component = env.reward_manager._step_reward[:, index] * env.step_dt
    weight = env.reward_manager.get_term_cfg('course_heading').weight
    torch.testing.assert_close(component, weight*snapshot['heading_cost']*env.step_dt, atol=1e-6, rtol=1e-5)
    return dict(snapshot, heading_reward=component.detach().clone())


def lateral_velocity_values(root_position, root_linear_velocity, origins, *, gain_per_second, maximum_m_s):
    """World-Y velocity towards the world-X centerline, independent of body yaw."""
    if not math.isfinite(gain_per_second) or gain_per_second <= 0:
        raise ValueError('positive finite lateral return gain required')
    if not math.isfinite(maximum_m_s) or maximum_m_s <= 0:
        raise ValueError('positive finite lateral speed limit required')
    lateral = root_position[..., 1] - origins[..., 1]
    target = (-gain_per_second*lateral).clamp(-maximum_m_s, maximum_m_s)
    return target, (root_linear_velocity[..., 1]-target).square()


def route_lateral_velocity_cost(env, gain_per_second, maximum_m_s):
    robot = env.scene['robot']
    state = robot.data.root_state_w
    target, cost = lateral_velocity_values(state[..., :3], state[..., 7:10], env.scene.env_origins,
        gain_per_second=gain_per_second, maximum_m_s=maximum_m_s)
    env._route_velocity_snapshot = dict(velocity_root_state=state.detach().clone(),
        velocity_origins=env.scene.env_origins.detach().clone(), velocity_target=target.detach().clone(),
        velocity_cost=cost.detach().clone())
    return cost


def route_lateral_velocity_trace(env):
    snapshot = env._route_velocity_snapshot
    if snapshot is None:
        raise RuntimeError('lateral velocity reward did not record this control')
    index = env.reward_manager.active_terms.index('course_lateral_velocity')
    component = env.reward_manager._step_reward[:, index] * env.step_dt
    weight = env.reward_manager.get_term_cfg('course_lateral_velocity').weight
    torch.testing.assert_close(component, weight*snapshot['velocity_cost']*env.step_dt, atol=1e-6, rtol=1e-5)
    return dict(snapshot, velocity_reward=component.detach().clone())


def configure_lateral_velocity_training(cfg, options):
    from isaaclab.managers import RewardTermCfg
    gain, maximum, weight = (options[key] for key in ('gain_per_second', 'maximum_m_s', 'weight'))
    if not all(math.isfinite(v) for v in (gain, maximum, weight)) or gain <= 0 or maximum <= 0 or weight >= 0:
        raise ValueError('finite positive gain/speed and negative velocity weight required')
    cfg.rewards.course_lateral_velocity = RewardTermCfg(func=route_lateral_velocity_cost, weight=weight,
        params=dict(gain_per_second=gain, maximum_m_s=maximum))


def expand_actor_route_inputs(payload):
    """Insert zero route columns after the original command; never mutate the source."""
    source = payload.get('actor_state_dict', {})
    weight = source.get('mlp.0.weight')
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[1] != 45:
        raise ValueError('Route expansion requires a canonical45D MLP actor')
    if not torch.isfinite(weight).all():
        raise ValueError('Nonfinite original actor weight')
    expanded = torch.cat((weight[:, :9], weight.new_zeros(weight.shape[0], 3), weight[:, 9:]), dim=1)
    return dict(payload, actor_state_dict=dict(source, **{'mlp.0.weight': expanded}))


def configure_route_training(cfg, options):
    from isaaclab.managers import RewardTermCfg
    configure_route_observations(cfg)
    cfg.rewards.track_lin_vel_xy_exp.params['std'] = options['tracking_std']
    cfg.rewards.course_heading = RewardTermCfg(func=route_heading_cost, weight=options['heading_weight'])
    if 'heading_lookahead_m' in options:
        # Validate before any physical step; the pure helper also checks callers.
        length, limit = options['heading_lookahead_m'], options['heading_maximum_deg']
        if not math.isfinite(length) or length <= 0 or not math.isfinite(limit) or not 0 < limit < 90:
            raise ValueError('invalid route heading geometry')
        cfg.rewards.course_heading.params = dict(lookahead_m=length, maximum_heading_deg=limit, record=True)
