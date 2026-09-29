"""Complete-episode, reward-only value diagnostics (not SAC soft-Q targets)."""

import numpy as np
import torch
from torch import nn


def make_value_network(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 128), nn.ELU(),
                         nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))


def append_past_velocity_integral(observations, *, step_dt=.02, distance_scale=12.):
    """Append an exclusive body-velocity prefix for ONE complete episode.

    Input columns 45:47 are measured body XY velocity; column 47 remains
    elapsed time. New columns 48:50 start at zero on every call. Rotating body
    axes are not corrected, so this is a travel proxy, not world odometry.
    No reward, terminal, simulator position or future measurement is used.
    """
    x = np.asarray(observations, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 48 or not len(x) or not np.isfinite(x).all():
        raise ValueError("history requires finite nonempty episode observations [T, 48]")
    if not np.isfinite([step_dt, distance_scale]).all() or step_dt <= 0 or distance_scale <= 0:
        raise ValueError("history integration scales must be finite and positive")
    history = np.zeros((len(x), 2), dtype=np.float64)
    history[1:] = np.cumsum(x[:-1, 45:47], axis=0, dtype=np.float64) * (step_dt / distance_scale)
    result = np.concatenate((x, history.astype(np.float32)), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("history integration produced non-finite observations")
    return result


def make_history_value_network(initial_state):
    """Zero-extend a 48D initial critic; never modify the supplied state."""
    original = make_value_network(48)
    original.load_state_dict(initial_state, strict=True)
    state = {k: v.detach().clone() for k, v in original.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("initial critic must be finite")
    weight = state['0.weight']
    state['0.weight'] = torch.cat((weight, weight.new_zeros((128, 2))), dim=1)
    model = make_value_network(50)
    model.load_state_dict(state, strict=True)
    return model


def complete_episode_returns(rewards, terminated, truncated, collector_cutoff, *, gamma):
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if not len(rewards) or not np.isfinite(rewards).all() or not 0 < gamma <= 1:
        raise ValueError("finite rewards and a valid discount are required")
    endings = []
    for value in (terminated, truncated, collector_cutoff):
        value = np.asarray(value).reshape(-1)
        if len(value) != len(rewards) or not np.isin(value, [0, 1]).all():
            raise ValueError("episode boundary vectors must be binary and match rewards")
        endings.append(value.astype(bool))
    term, trunc, cutoff = endings
    if not term[-1] or term[:-1].any() or trunc.any() or cutoff.any():
        raise ValueError("full task terminal required; never relabel collector cutoff as MC")
    result = np.empty_like(rewards)
    tail = 0.
    for i in range(len(rewards) - 1, -1, -1):
        tail = rewards[i] + gamma * tail
        result[i] = tail
    return result


def value_metrics(target, prediction):
    y, p = (np.asarray(x, dtype=np.float64).reshape(-1) for x in (target, prediction))
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("finite predictions must match targets")
    if not len(y):
        return {"samples": 0}
    error = p - y
    variance = float(y.var())
    interpretable = variance > max(1e-12, 1e-8 * float(np.mean(y * y)))
    mse = float(np.mean(error * error))
    return {"samples": len(y), "target_mean": float(y.mean()), "target_variance": variance,
            "rmse": mse ** .5, "bias": float(error.mean()),
            "normalized_rmse": (mse / variance) ** .5 if interpretable else None,
            "explained_variance": 1. - float(error.var()) / variance if interpretable else None,
            "r_squared": 1. - mse / variance if interpretable else None,
            "variance_interpretable": interpretable}


def time_features(observations, degree=3):
    t = np.asarray(observations)[:, -1].astype(np.float64)
    return np.stack([t ** i for i in range(degree + 1)], axis=1)


def fit_value_network(model, observations, targets, *, epochs, batch_size, learning_rate,
                      seed, value_loss_scale=.25, max_grad_norm=1.):
    """Only training arrays enter this function; no validation-based selection."""
    x = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(targets), dtype=torch.float32).reshape(-1, 1)
    if len(x) != len(y) or not len(x) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("invalid value training data")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("invalid value fitting budget")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    curve = []
    steps = 0
    for epoch in range(epochs):
        permutation = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            idx = permutation[start:start + batch_size]
            loss = value_loss_scale * (model(x[idx]) - y[idx]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite critic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                rmse = float((model(x) - y).square().mean().sqrt())
            curve.append({"epoch": epoch + 1, "training_rmse": rmse})
    return {"optimizer_steps": steps, "training_curve": curve, "optimizer": optimizer.state_dict()}
