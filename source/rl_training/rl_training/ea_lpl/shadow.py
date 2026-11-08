"""Side-effect-free SafetyFilter evaluations for execution-sensitivity estimates.

Every evaluation starts from a copied pre-decision snapshot; the live filter
is never called. A response over several steps holds one residual request
while replaying recorded base actions and safety states (open loop).
"""
from __future__ import annotations

import torch

from rl_training.online_adaptation.safety import SafetyFilter


def filter_from_snapshot(limits, snapshot):
    shadow = SafetyFilter(limits)
    shadow._previous_residual = snapshot["filter_previous_residual_action"].clone()
    shadow._paused = snapshot["filter_paused"].clone()
    shadow._stable_steps = snapshot["filter_stable_steps"].clone()
    shadow._contact_steps = snapshot["filter_contact_steps"].clone()
    return shadow


def request_from_codes(v, u):
    return torch.einsum("njk,nk->nj", v, torch.tanh(u))


def rollout_mean_action(limits, snapshot, base_actions, states, request):
    if not states or len(base_actions) != len(states):
        raise ValueError("need one base action per recorded state")
    shadow = filter_from_snapshot(limits, snapshot)
    total = torch.zeros_like(base_actions[0])
    for base, state in zip(base_actions, states):
        total = total + shadow.project(base, request, state).action
    return total / len(states)


def finite_difference_jacobian(limits, snapshot, base_actions, states, v, u_ref, delta):
    columns = []
    for i in range(u_ref.shape[1]):
        step = torch.zeros_like(u_ref)
        step[:, i] = delta
        plus = rollout_mean_action(limits, snapshot, base_actions, states, request_from_codes(v, u_ref + step))
        minus = rollout_mean_action(limits, snapshot, base_actions, states, request_from_codes(v, u_ref - step))
        columns.append((plus - minus) / (2 * delta))
    return torch.stack(columns, dim=-1)


def parameterization_jacobian(v, u_ref, delta):
    """The same finite difference without the filter: only the tanh parameterization."""
    scale = (torch.tanh(u_ref + delta) - torch.tanh(u_ref - delta)) / (2 * delta)
    return v * scale[:, None, :]


def sensitivity(j):
    return j.transpose(-1, -2) @ j


def relative_deviation(m, m0):
    return torch.linalg.matrix_norm(m - m0) / torch.linalg.matrix_norm(m0).clamp_min(1e-12)
