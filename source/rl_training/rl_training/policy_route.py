"""Explicit simulator-localized route inputs for the full Lite3 neural policy."""
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


def route_heading_cost(env):
    robot = env.scene['robot']
    return 1 - route_features(robot.data.root_pos_w, robot.data.root_quat_w, env.scene.env_origins)[..., 2]


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
