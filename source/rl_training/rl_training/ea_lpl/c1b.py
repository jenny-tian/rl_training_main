"""Small independent state-conditioned sessions; no simulator or base optimizer."""
from __future__ import annotations

import hashlib

import numpy as np
import torch


class FeatureHistory:
    def __init__(self, count):
        self.data = np.zeros((count, 25, 4), np.float32)
        self.ready = np.zeros(count, bool)

    def reset(self, ids):
        self.ready[ids] = False
        self.data[ids] = 0

    def append(self, gravity_xy, yaw_rate, joint_velocity_mean):
        frame = np.column_stack((gravity_xy, yaw_rate, joint_velocity_mean)).astype(np.float32)
        if not np.isfinite(frame).all():
            raise ValueError('nonfinite proprioceptive frame')
        self.data[:, :-1] = self.data[:, 1:].copy()
        self.data[:, -1] = frame
        self.data[~self.ready] = frame[~self.ready, None, :]
        self.ready[:] = True

    def features(self, previous_code):
        norm = np.linalg.norm(self.data[:, :, :2], axis=-1)
        mean = self.data.mean(1)
        value = np.column_stack((mean[:, 0] / .5, mean[:, 1] / .5,
            norm.std(1) / .2, mean[:, 2] / .8, mean[:, 3] / 10.,
            np.tanh(previous_code), (norm > .55).mean(1)))
        return np.clip(value, -3, 3).astype(np.float32)


def exploration_table():
    # Namespace does not alter the table: keys are session/episode/decision.
    rng = np.random.default_rng(2026111701)
    return rng.standard_normal((8, 64, 100, 2)).astype(np.float32)


def digest(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def rewards(dx, dy, learned_effect, residual_limits, posture, lateral):
    return (np.clip(dx, -.02, .02) - .5 * np.abs(np.clip(dy, -.02, .02))
        - .001 * np.mean((learned_effect / residual_limits)**2, axis=-1)
        - np.asarray(posture, float) - .5 * np.asarray(lateral, float))


def returns_to_go(values, gamma=.99):
    out = np.zeros(len(values), np.float64)
    carry = 0.
    for t in range(len(values)-1, -1, -1):
        carry = float(values[t]) + gamma * carry
        out[t] = carry
    return out


class Session:
    def __init__(self, updated):
        self.model = torch.nn.Linear(8, 2)
        with torch.no_grad():
            self.model.weight.zero_(); self.model.bias.zero_()
        self.updated = bool(updated)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.01) if updated else None
        self.baseline = 0.
        self.updates = 0
        self.initial_digest = digest(self.model)
        self.begin()

    def begin(self):
        self.features, self.codes, self.decision_steps, self.rewards = [], [], [], []
        self.episode_digest = digest(self.model)

    def choose(self, features, noise=None):
        features = torch.as_tensor(features, dtype=torch.float32).detach().clone()
        with torch.no_grad():
            mean = self.model(features)
            code = mean if noise is None else mean + .6 * torch.as_tensor(noise)
        if not torch.isfinite(code).all():
            raise ValueError('nonfinite requested code')
        self.features.append(features)
        self.codes.append(code.clone())
        self.decision_steps.append(len(self.rewards))
        return code.numpy().copy()

    def finish(self, *, allow_update):
        if not self.rewards or not self.codes:
            raise ValueError('empty complete episode')
        if digest(self.model) != self.episode_digest:
            raise ValueError('parameters changed inside an episode')
        returns = returns_to_go(self.rewards)
        old_baseline = self.baseline
        stats = dict(discounted_return=float(returns[0]), return_sum=float(sum(self.rewards)),
                     old_baseline=old_baseline, decisions=len(self.codes), updated=False)
        if self.updated and allow_update:
            x, action = torch.stack(self.features), torch.stack(self.codes)
            advantages = torch.tensor(returns[self.decision_steps] - old_baseline, dtype=torch.float32)
            distribution = torch.distributions.Normal(self.model(x), .6)
            loss = -(distribution.log_prob(action).sum(-1) * advantages).sum() / 100.
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all()
                                               for p in self.model.parameters()):
                raise ValueError('nonfinite adapter loss or gradient')
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError('nonfinite adapter parameters')
            self.updates += 1
            stats.update(updated=True, loss=float(loss.detach()), gradient_norm=float(norm))
        self.baseline = .9 * old_baseline + .1 * float(returns[0])
        if not self.updated and digest(self.model) != self.initial_digest:
            raise ValueError('Frozen adapter changed')
        stats.update(updates=self.updates, new_baseline=self.baseline, parameter_sha256=digest(self.model))
        return stats

    def snapshot(self):
        return dict(state_dict={k: v.detach().clone() for k, v in self.model.state_dict().items()},
                    optimizer=None if self.optimizer is None else self.optimizer.state_dict(),
                    baseline=self.baseline, updates=self.updates, parameter_sha256=digest(self.model))
