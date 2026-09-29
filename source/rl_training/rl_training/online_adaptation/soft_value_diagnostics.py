"""Read-only fixed-policy soft Bellman diagnostics; no return calibration claim."""

from dataclasses import fields

import torch

from .sac import compute_sac_target
from .types import TransitionBatch


@torch.no_grad()
def fixed_soft_rows(actor, critic, target_critic, batch, *, alpha, residual_gain,
                    draws=16, batch_size=128, seed=23):
    """CPU diagnostics with repeatable action draws and preserved caller RNG.

    Log density is in normalized tanh coordinates. Limits and gain transform
    the requested action passed to Q, not the entropy coordinates. Q receives
    replay requests before safety projection. Zero-request target Q is a
    coverage reference, not a ground-truth value.
    """
    if draws < 2 or batch_size < 1 or not 0 <= residual_gain <= 1 or not 0 < alpha < float('inf'):
        raise ValueError('invalid diagnostic sampling contract')
    n = len(batch.reward)
    if not n:
        raise ValueError('nonempty replay batch required')
    for field in fields(batch):
        value = getattr(batch, field.name)
        if value.device.type != 'cpu' or len(value) != n or not torch.isfinite(value).all():
            raise ValueError('finite CPU replay fields with matching rows required')
    if (not torch.isin(batch.bootstrap_mask, torch.tensor([0., 1.])).all()
            or not ((batch.discount >= 0) & (batch.discount <= 1)).all()):
        raise ValueError('binary bootstrap mask and bounded discounts required')
    pieces = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for start in range(0, n, batch_size):
            b = TransitionBatch(**{f.name: getattr(batch, f.name)[start:start+batch_size] for f in fields(batch)})
            q1, q2 = critic(b.history, b.current_frame, b.base_action, b.requested_residual_action)
            zero1, zero2 = target_critic(b.next_history, b.next_frame, b.next_base_action,
                                       torch.zeros_like(b.requested_residual_action))
            targets, entropies, gaps, magnitudes = [], [], [], []
            for _ in range(draws):
                action, log_prob, _ = actor.sample(b.next_history, b.next_frame, b.next_base_action)
                request = action * residual_gain
                next1, next2 = target_critic(b.next_history, b.next_frame, b.next_base_action, request)
                entropy = -alpha * log_prob
                target = compute_sac_target(reward=b.reward, discount=b.discount,
                                            bootstrap_mask=b.bootstrap_mask,
                                            next_q=torch.minimum(next1, next2), alpha_log_prob=-entropy)
                targets.append(target)
                entropies.append(entropy)
                gaps.append(torch.minimum(next1, next2) - torch.minimum(zero1, zero2))
                magnitudes.append(request.square().mean(dim=-1, keepdim=True))
            targets, entropies = torch.stack(targets), torch.stack(entropies)
            pieces.append({'q1': q1, 'q2': q2, 'target_mean': targets.mean(0),
                           'target_sample_std': targets.std(0, unbiased=True),
                           'entropy_bonus': entropies.mean(0),
                           'effective_entropy_bonus': b.bootstrap_mask * b.discount * entropies.mean(0),
                           'sampled_minus_zero_next_q': torch.stack(gaps).mean(0),
                           'sampled_request_rms': torch.stack(magnitudes).mean(0).sqrt()})
    result = {key: torch.cat([p[key] for p in pieces]) for key in pieces[0]}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError('non-finite fixed-policy diagnostic')
    return result


def summarize_soft_rows(rows, mask=None):
    if mask is None:
        mask = torch.ones(len(rows['q1']), dtype=torch.bool)
    selected = {k: v[mask.reshape(-1)] for k, v in rows.items()}
    if not len(selected['q1']):
        return {'samples': 0}
    q = torch.minimum(selected['q1'], selected['q2']).double()
    target = selected['target_mean'].double()
    error = q - target
    return {'samples': len(q), 'mean_min_q': float(q.mean()), 'mean_soft_target': float(target.mean()),
            'target_variance_across_rows': float(target.var(unbiased=False)),
            'bellman_rmse': float(error.square().mean().sqrt()), 'bellman_bias_q_minus_target': float(error.mean()),
            'mean_abs_bellman_error': float(error.abs().mean()),
            'mean_twin_q_gap': float((selected['q1']-selected['q2']).abs().mean()),
            'mean_target_sample_std': float(selected['target_sample_std'].mean()),
            'mean_entropy_bonus': float(selected['entropy_bonus'].mean()),
            'mean_effective_entropy_bonus': float(selected['effective_entropy_bonus'].mean()),
            'mean_sampled_minus_zero_next_q': float(selected['sampled_minus_zero_next_q'].mean()),
            'mean_abs_sampled_minus_zero_next_q': float(selected['sampled_minus_zero_next_q'].abs().mean()),
            'mean_sampled_request_rms': float(selected['sampled_request_rms'].mean())}
