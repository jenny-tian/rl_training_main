"""Current-memory shadow response; no simulator stepping or live-state mutation."""
import torch

from .current_safety import CurrentModelSafetyFilter
from .shadow import request_from_codes


def rollout_mean_action(limits, snapshot, base_actions, states, request):
    if not states or len(base_actions) != len(states):
        raise ValueError('need one base action per recorded state')
    shadow = CurrentModelSafetyFilter.from_snapshot(limits, snapshot)
    total = torch.zeros_like(base_actions[0])
    for action, state in zip(base_actions, states):
        total += shadow.project(action, request, state).action
    return total / len(states)


def finite_difference_jacobian(limits, snapshot, base_actions, states, v, u_ref, delta):
    if delta <= 0:
        raise ValueError('delta must be positive')
    columns = []
    for i in range(u_ref.shape[1]):
        step = torch.zeros_like(u_ref); step[:, i] = delta
        plus = rollout_mean_action(limits, snapshot, base_actions, states, request_from_codes(v, u_ref + step))
        minus = rollout_mean_action(limits, snapshot, base_actions, states, request_from_codes(v, u_ref - step))
        columns.append((plus-minus) / (2*delta))
    return torch.stack(columns, dim=-1)
