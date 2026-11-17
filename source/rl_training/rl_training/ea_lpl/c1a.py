"""Episode-level bias learning; no simulator, actor or C1-b state features."""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch


def episode_return(progress_m, completed, safety_ending):
    """Original C1-a score, including its literal 9 m normalization.

    Call with final forward displacement, not maximum progress. Bounds exits
    are separate from safety endings in the original episode schema.
    """
    if not math.isfinite(float(progress_m)):
        raise ValueError('nonfinite progress')
    if completed and safety_ending:
        raise ValueError('success and safety ending are mutually exclusive')
    return float(np.clip(progress_m, 0., 9.) / 9. + bool(completed) - bool(safety_ending))


def loo_advantages(returns):
    values = torch.as_tensor(returns, dtype=torch.float64).detach().clone()
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError('finite returns from at least two episodes required')
    return values - (values.sum() - values) / (len(values) - 1)


def exploration_table(pairs=60, max_episodes=400, seed=2026111702):
    return np.random.default_rng(seed).standard_normal((pairs, max_episodes, 2))


class Session:
    """Independent on-policy batches of four episodes under the same mean.

    Sample once with begin(), use requested_code() throughout the episode,
    then finish() only on a genuine complete episode. Incomplete episodes and
    incomplete batches are saved but never force an optimizer update.
    """

    def __init__(self, updated):
        self.updated = bool(updated)
        self.b = torch.zeros(2, dtype=torch.float64)
        self.m = torch.zeros_like(self.b)
        self.v = torch.zeros_like(self.b)
        self.updates = 0
        self.batch = []
        self.active = None

    def digest(self):
        return hashlib.sha256(self.b.numpy().tobytes()).hexdigest()

    def begin(self, noise=None):
        if self.active is not None:
            raise ValueError('episode still active')
        noise = torch.zeros(2, dtype=torch.float64) if noise is None else torch.as_tensor(noise, dtype=torch.float64).detach().clone()
        if noise.shape != (2,) or not torch.isfinite(noise).all():
            raise ValueError('noise must be a finite 2-vector')
        if not torch.isfinite(self.b).all():
            raise ValueError('nonfinite mean')
        code = self.b + .6 * noise
        self.active = dict(mean=self.b.clone(), code=code.clone(), version=self.updates)
        return self.requested_code()

    def requested_code(self):
        if self.active is None:
            raise ValueError('no active episode')
        if self.updates != self.active['version'] or not torch.equal(self.b, self.active['mean']):
            raise ValueError('parameters changed inside episode')
        return self.active['code'].numpy().copy()

    def finish(self, value, *, learn=True):
        code = self.requested_code()
        if not math.isfinite(float(value)):
            raise ValueError('nonfinite return')
        entry = dict(self.active, episode_return=float(value))
        old_digest = self.digest()
        stats = dict(episode_return=float(value), code=code.tolist(), mean=self.b.tolist(),
                     version=self.updates, updated=False)
        # Validate before touching buffers, including restored/corrupted batches.
        if learn:
            candidate = [*self.batch, entry]
            if len(candidate) > 4 or any(e['version'] != self.updates or not torch.equal(e['mean'], self.b) for e in candidate):
                raise ValueError('batch must use one current parameter version')
            if len(candidate) == 4:
                actions = torch.stack([e['code'] for e in candidate])
                adv = loo_advantages([e['episode_return'] for e in candidate])
                gradient = -(adv[:, None] * (actions - self.b)).mean(0) / .6**2
                if not torch.isfinite(gradient).all():
                    raise ValueError('nonfinite gradient')
                stats.update(gradient=gradient.tolist(), advantages=adv.tolist())
                if self.updated:
                    t = self.updates + 1
                    m = .9 * self.m + .1 * gradient
                    v = .999 * self.v + .001 * gradient.square()
                    b = (self.b - .3 * (m / (1 - .9**t)) / ((v / (1 - .999**t)).sqrt() + 1.e-8)).clamp(-2.5, 2.5)
                    if not all(torch.isfinite(x).all() for x in (b, m, v)):
                        raise ValueError('nonfinite Adam state')
                    self.b, self.m, self.v, self.updates = b, m, v, t
                    stats['updated'] = True
                self.batch = []
            else:
                self.batch = candidate
        self.active = None
        if not self.updated and self.digest() != old_digest:
            raise ValueError('Frozen parameters changed')
        stats.update(updates=self.updates, pending_complete_episodes=len(self.batch),
                     parameter_sha256=self.digest(), b_after=self.b.tolist())
        return stats

    def snapshot(self):
        return copy.deepcopy(dict(b=self.b, m=self.m, v=self.v, updates=self.updates,
                                  batch=self.batch, active=self.active, updated=self.updated,
                                  parameter_sha256=self.digest()))

    def load_evaluation_mean(self, snapshot):
        if self.updated or self.active is not None or self.batch:
            raise ValueError('evaluation load requires an empty frozen session')
        b = torch.as_tensor(snapshot['b'], dtype=torch.float64).detach().clone()
        if b.shape != (2,) or not torch.isfinite(b).all() or (b.abs() > 2.5).any():
            raise ValueError('invalid evaluation mean')
        self.b = b
        if self.digest() != snapshot['parameter_sha256']:
            raise ValueError('checkpoint mean hash mismatch')
