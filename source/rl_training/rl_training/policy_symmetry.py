"""Sagittal reflection for canonical Lite3 proprioception and joint commands."""
import torch


def mirror_action(action):
    if action.shape[-1] != 12:
        raise ValueError('Lite3 reflection requires 12 canonical joint coordinates')
    permutation = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], device=action.device)
    signs = action.new_tensor([-1, 1, 1] * 4)
    return action[..., permutation] * signs


def mirror_observation(obs, *, is_critic=None):
    if is_critic is None:
        if obs.shape[-1] not in (45, 48):
            raise ValueError('Explicit observation role required for expanded route inputs')
        is_critic = obs.shape[-1] == 48
    offset = 3 if is_critic else 0
    route_dim = obs.shape[-1] - (45 + offset)
    if route_dim not in (0, 3):
        raise ValueError('Lite3 reflection requires canonical proprio or route schema')
    result = obs.clone()
    if offset:
        result[..., :3] *= obs.new_tensor([1, -1, 1])  # polar linear velocity
    result[..., offset:offset+3] *= obs.new_tensor([-1, 1, -1])  # axial angular velocity
    result[..., offset+3:offset+6] *= obs.new_tensor([1, -1, 1])  # polar gravity
    result[..., offset+6:offset+9] *= obs.new_tensor([1, -1, -1])  # vx, vy, yaw rate
    if route_dim:
        result[..., offset+9:offset+12] *= obs.new_tensor([-1, -1, 1])
    for start in (offset+route_dim+9, offset+route_dim+21, offset+route_dim+33):
        result[..., start:start+12] = mirror_action(obs[..., start:start+12])
    return result


def augment_lite3_mirror(obs=None, actions=None, env=None):
    """Original batch first, then its reflection; supports the RSL mirror-loss API."""
    if obs is not None:
        if set(obs.keys()) != {'policy', 'critic'}:
            raise ValueError('Unexpected observation groups for Lite3 mirror loss')
        mirrored = obs.clone()
        for name in ('policy', 'critic'):
            mirrored[name] = mirror_observation(obs[name], is_critic=name=='critic')
        obs = torch.cat([obs, mirrored], dim=0)
    if actions is not None:
        actions = torch.cat([actions, mirror_action(actions)], dim=0)
    return obs, actions
