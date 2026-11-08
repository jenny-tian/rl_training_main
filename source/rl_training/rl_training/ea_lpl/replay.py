"""Open-loop replay of recorded safety states for execution-response diagnostics."""
from __future__ import annotations

import numpy as np
import torch

from rl_training.online_adaptation.safety import SafetyFilter
from rl_training.online_adaptation.types import SafetyState

from .shadow import (finite_difference_jacobian, parameterization_jacobian, relative_deviation,
                     request_from_codes, rollout_mean_action, sensitivity)


def state_at(rec, t):
    n = rec["base_action"].shape[1]
    return SafetyState(
        projected_gravity_xy=torch.as_tensor(rec["gravity_xy"][t]), base_height=torch.as_tensor(rec["base_height"][t]),
        joint_pos=torch.zeros(n, 12), joint_vel=torch.as_tensor(rec["joint_vel"][t]),
        joint_lower=torch.as_tensor(rec["joint_lower"]), joint_upper=torch.as_tensor(rec["joint_upper"]),
        default_joint_pos=torch.as_tensor(rec["default_joint_pos"]), action_scale=torch.as_tensor(rec["action_scale"]),
        nonfoot_contact=torch.as_tensor(rec["nonfoot_contact"][t]),
        control_timed_out=torch.as_tensor(rec["control_timed_out"][t]))


def replay_fidelity(rec, limits):
    """Replay the recorded constant request from zero memory; compare with the executed actions."""
    request = torch.as_tensor(rec["request"])
    end = np.asarray(rec["episode_end"])
    live = SafetyFilter(limits)
    worst = 0.0
    for t in range(int(end.max())):
        action = live.project(torch.as_tensor(rec["base_action"][t]), request, state_at(rec, t)).action
        alive = torch.as_tensor(t < end)
        diff = (action - torch.as_tensor(rec["executed_action"][t])).abs().amax(-1)[alive]
        if len(diff):
            worst = max(worst, float(diff.max()))
    return worst


def fixed_code_execution_stats(rec, *, ramp_steps=10, tilt_guard=0.55):
    base, executed, request = rec["base_action"], rec["executed_action"], rec["request"]
    steps = base.shape[0]
    window = (np.arange(steps)[:, None] >= ramp_steps) & (np.arange(steps)[:, None] < np.asarray(rec["episode_end"])[None, :])
    mismatch = np.abs(executed - base - request[None]).max(-1) > 1e-6
    guard = rec["nonfoot_contact"][..., 0] | (np.linalg.norm(rec["gravity_xy"], axis=-1) > tilt_guard)
    return dict(p_proj=float(mismatch[window].mean()), p_guard=float(guard[window].mean()),
                samples=int(window.sum()))


def switching_diagnostics(rec, v, limits, *, sigma0=0.6, interval=10, seed=0, threshold=0.3):
    steps, n = rec["base_action"].shape[:2]
    end = torch.as_tensor(np.asarray(rec["episode_end"]))
    vb = v.to(torch.float32).expand(n, 12, 2)
    generator = torch.Generator().manual_seed(seed)
    live = SafetyFilter(limits)
    u_prev = torch.zeros(n, 2)
    single, macro, errors = [], [], []
    activations = samples = decisions = 0
    for k in range(steps // interval):
        t0 = k * interval
        alive = (t0 + interval) <= end
        if not bool(alive.any()):
            break
        times = range(t0, t0 + interval)
        states = [state_at(rec, t) for t in times]
        bases = [torch.as_tensor(rec["base_action"][t]) for t in times]
        snapshot = live.temporal_snapshot(bases[0])
        u_k = sigma0 * torch.randn(n, 2, generator=generator)
        if k > 0:
            zero = torch.zeros(n, 2)
            m_a = sensitivity(finite_difference_jacobian(limits, snapshot, bases[:1], states[:1], vb, zero, 0.05))
            m_a0 = sensitivity(parameterization_jacobian(vb, zero, 0.05))
            j_b = finite_difference_jacobian(limits, snapshot, bases, states, vb, u_prev, sigma0)
            m_b0 = sensitivity(parameterization_jacobian(vb, u_prev, sigma0))
            actual = (rollout_mean_action(limits, snapshot, bases, states, request_from_codes(vb, u_k))
                      - rollout_mean_action(limits, snapshot, bases, states, request_from_codes(vb, u_prev)))
            predicted = torch.einsum("njk,nk->nj", j_b, u_k - u_prev)
            error = (actual - predicted).norm(dim=-1) / actual.norm(dim=-1).clamp_min(1e-9)
            single += relative_deviation(m_a, m_a0)[alive].tolist()
            macro += relative_deviation(sensitivity(j_b), m_b0)[alive].tolist()
            errors += error[alive].tolist()
            decisions += int(alive.sum())
        request = request_from_codes(vb, u_k)
        for base, state in zip(bases, states):
            accepted = live.project(base, request, state).action - base
            activations += int(((accepted - request).abs().amax(-1) > 1e-6)[alive].sum())
            samples += int(alive.sum())
        u_prev = u_k
    return dict(decisions=decisions,
                informative_fraction_single=float(np.mean(np.asarray(single) >= threshold)) if single else 0.0,
                informative_fraction_macro=float(np.mean(np.asarray(macro) >= threshold)) if macro else 0.0,
                median_prediction_error=float(np.median(errors)) if errors else float("nan"),
                projection_activation=activations / max(samples, 1))
