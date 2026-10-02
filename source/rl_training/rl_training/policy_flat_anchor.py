"""Frozen original-policy targets for on-policy flat training states only."""
import torch
from torch import nn
from torch.nn import functional as F
from rl_training.lite3_joint_conventions import clip_lite3_policy_action


class FrozenFlatMean(nn.Module):
    def __init__(self, payload):
        super().__init__()
        state = payload['actor_state_dict']
        previous = 45
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
        if observation.shape[-1] == 48:
            observation = torch.cat((observation[..., :9], observation[..., 12:]), dim=-1)
        if observation.shape[-1] != 45:
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
    return flat_anchor_cost(env.action_manager.action, env._flat_anchor_target, env._flat_anchor_mask)
