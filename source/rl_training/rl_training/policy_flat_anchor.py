"""Frozen original-policy targets for on-policy flat training states only."""
import torch
from torch import nn
from torch.nn import functional as F
from rl_training.lite3_joint_conventions import clip_lite3_policy_action


class FrozenFlatMean(nn.Module):
    def __init__(self, payload):
        super().__init__()
        state = payload['actor_state_dict']
        self.observation_dim = state['mlp.0.weight'].shape[1]
        if self.observation_dim not in (45, 48, 49):
            raise ValueError('Expected a canonical45 or route48 actor')
        previous = self.observation_dim
        for index in (0, 2, 4, 6):
            weight, bias = state[f'mlp.{index}.weight'], state[f'mlp.{index}.bias']
            if weight.ndim != 2 or weight.shape[1] != previous or bias.shape != (weight.shape[0],):
                raise ValueError('Incompatible original flat actor layout')
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                raise ValueError('Nonfinite original flat actor')
            self.register_buffer(f'weight_{index}', weight.detach().clone())
            self.register_buffer(f'bias_{index}', bias.detach().clone())
            previous = weight.shape[0]
        if previous != 12:
            raise ValueError('Original actor must output12 joint commands')

    @torch.no_grad()
    def forward(self, observation):
        if observation.shape[-1] == 49 and self.observation_dim != 49:
            from rl_training.policy_route_history import strip_route_history
            observation = strip_route_history(observation)
        if observation.shape[-1] == 48 and self.observation_dim == 45:
            observation = torch.cat((observation[..., :9], observation[..., 12:]), dim=-1)
        if observation.shape[-1] != self.observation_dim:
            raise ValueError('Expected actor45 or route-actor48 observation')
        value = observation
        for index in (0, 2, 4, 6):
            value = F.linear(value, getattr(self, f'weight_{index}'), getattr(self, f'bias_{index}'))
            if index != 6:
                value = F.elu(value)
        return clip_lite3_policy_action(value)


def flat_anchor_cost(emitted_action, target, flat_mask):
    if emitted_action.shape != target.shape or emitted_action.shape[-1] != 12:
        raise ValueError('Anchor action shapes differ')
    if flat_mask.dtype != torch.bool or flat_mask.shape != emitted_action.shape[:-1]:
        raise ValueError('Anchor requires an explicit flat-environment boolean mask')
    return torch.where(flat_mask, (emitted_action-target).square().mean(-1), 0.)


def flat_anchor_reward(env):
    # The collector must populate both from pre-action state before physics.
    cost = flat_anchor_cost(env.action_manager.action, env._flat_anchor_target, env._flat_anchor_mask)
    env._flat_anchor_applied_cost = cost.detach().clone()
    return cost


def configure_flat_anchor(cfg, weight):
    from isaaclab.managers import RewardTermCfg
    import math
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('Anchor weight must be finite and positive')
    generator = cfg.scene.terrain.terrain_generator
    if generator.num_cols != 8 or list(generator.sub_terrains)[0] != 'flat' or generator.sub_terrains['flat'].proportion != .25:
        raise ValueError('Anchor supports only the registered first-two-flat-column layout')
    cfg.rewards.flat_policy_anchor = RewardTermCfg(func=flat_anchor_reward, weight=-weight)


def prepare_flat_anchor(env, teacher, observation):
    mask = env.scene.terrain.terrain_types < 2
    target = observation.new_zeros((observation.shape[0], 12))
    target[mask] = teacher(observation[mask])
    env._flat_anchor_target = target
    env._flat_anchor_mask = mask.clone()
    return target, mask
