"""C1-a-M4：一个会话的多个并行环境共用一个均值，凑满一轮后用普通梯度步更新。

与 c1a.Session 的区别：每轮的回合来自并行环境而不是先后的回合；更新不经过 Adam；
探索标准差是 1.0。梯度的定义（留一基线）不变。方案见 docs/lite3-online-c1a-m4-20261118/PLAN.md。
"""
from __future__ import annotations

import copy
import hashlib
import math

import torch

from .c1a import loo_advantages

REPLICAS, SIGMA, LEARNING_RATE, BOUND = 4, 1., .6, 2.5


class ParallelSession:
    def __init__(self, updated, *, replicas=REPLICAS, sigma=SIGMA, learning_rate=LEARNING_RATE):
        self.updated, self.replicas, self.sigma, self.learning_rate = bool(updated), replicas, sigma, learning_rate
        self.b = torch.zeros(2, dtype=torch.float64)
        self.updates = 0
        self.active = {}
        self.batch = []

    def digest(self):
        return hashlib.sha256(self.b.numpy().tobytes()).hexdigest()

    def begin(self, replica, noise=None):
        if replica in self.active or not 0 <= replica < self.replicas:
            raise ValueError('episode still active')
        if any(e['replica'] == replica for e in self.batch):
            raise ValueError('replica already finished this round')
        if any(e['version'] != self.updates for e in [*self.active.values(), *self.batch]):
            raise ValueError('round must use one parameter version')
        noise = torch.zeros(2, dtype=torch.float64) if noise is None else torch.as_tensor(noise, dtype=torch.float64).detach().clone()
        if noise.shape != (2,) or not torch.isfinite(noise).all() or not torch.isfinite(self.b).all():
            raise ValueError('finite 2-vectors required')
        self.active[replica] = dict(replica=replica, mean=self.b.clone(), code=self.b + self.sigma * noise, version=self.updates)
        return self.requested_code(replica)

    def requested_code(self, replica):
        entry = self.active.get(replica)
        if entry is None:
            raise ValueError('no active episode')
        if self.updates != entry['version'] or not torch.equal(self.b, entry['mean']):
            raise ValueError('parameters changed inside episode')
        return entry['code'].numpy().copy()

    def finish(self, replica, value, *, learn=True):
        code = self.requested_code(replica)
        if not math.isfinite(float(value)):
            raise ValueError('nonfinite return')
        entry = dict(self.active.pop(replica), episode_return=float(value))
        old_digest = self.digest()
        stats = dict(episode_return=float(value), code=code.tolist(), mean=self.b.tolist(), version=self.updates,
                     updated=False)
        if learn:
            self.batch.append(entry)
            if len(self.batch) == self.replicas:
                if self.active:
                    raise ValueError('round complete while an episode is active')
                actions = torch.stack([e['code'] for e in self.batch])
                advantages = loo_advantages([e['episode_return'] for e in self.batch])
                gradient = -(advantages[:, None] * (actions - self.b)).mean(0) / self.sigma**2
                if not torch.isfinite(gradient).all():
                    raise ValueError('nonfinite gradient')
                stats.update(gradient=gradient.tolist(), advantages=advantages.tolist())
                if self.updated:
                    self.b = (self.b - self.learning_rate * gradient).clamp(-BOUND, BOUND)
                    self.updates += 1
                    stats['updated'] = True
                self.batch = []
        if not self.updated and self.digest() != old_digest:
            raise ValueError('Frozen parameters changed')
        stats.update(updates=self.updates, pending_complete_episodes=len(self.batch),
                     parameter_sha256=self.digest(), b_after=self.b.tolist())
        return stats

    def snapshot(self):
        return copy.deepcopy(dict(b=self.b, updates=self.updates, batch=self.batch, active=self.active,
                                  updated=self.updated, parameter_sha256=self.digest()))


class Slot:
    """一个并行环境看到的接口，与 c1a.Session 的调用方式相同。"""

    def __init__(self, group, replica):
        self.group, self.replica = group, replica

    updates = property(lambda self: self.group.updates)
    batch = property(lambda self: self.group.batch)

    def digest(self): return self.group.digest()
    def begin(self, noise=None): return self.group.begin(self.replica, noise)
    def requested_code(self): return self.group.requested_code(self.replica)
    def finish(self, value, *, learn=True): return self.group.finish(self.replica, value, learn=learn)
    def snapshot(self): return self.group.snapshot()
