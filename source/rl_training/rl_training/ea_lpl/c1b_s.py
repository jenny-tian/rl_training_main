"""C1-b-S：在固定姿态偏置和摆动缩腿两组动作基上学 4 个幅度，探索加在参数上，一个回合抽一次。

w = [前腿固定, 后腿固定, 前腿摆动, 后腿摆动]。每条腿的残差是 sag 方向乘以
clip(tanh(w_固定) + tanh(w_摆动) * 是否摆动, -1, 1) * 起伏特征。更新规则与 c1b_h 相同。
方案见 docs/lite3-online-c1b-swing-20261118/PLAN.md。
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch

from .basis import LEGS
from .c1a import loo_advantages

REPLICAS, SIGMA, LEARNING_RATE, BOUND = 4, .7, .6, 2.5
PARAMETERS = ('front_static', 'rear_static', 'front_swing', 'rear_swing')
HIPY, KNEE = -.10, .20  # sag 动作基的一列
REAR = torch.tensor([float(leg.startswith('H')) for leg in LEGS])


def exploration_table(pairs=60, max_episodes=400, seed=2026111802):
    return np.random.default_rng(seed).standard_normal((pairs, max_episodes, len(PARAMETERS)))


def residual_request(weights, stance, rough):
    """weights (n, 4)，stance (n, 4) 布尔（FL FR HL HR 是否着地），rough (n,) 起伏特征。返回 (n, 12)。"""
    stance = torch.as_tensor(stance)
    z = torch.tanh(torch.as_tensor(weights, dtype=torch.float32, device=stance.device))
    rough = torch.as_tensor(rough, dtype=torch.float32, device=stance.device)
    if stance.dtype != torch.bool or stance.ndim != 2 or stance.shape[1] != 4 or z.shape != (len(stance), 4) or rough.shape != (len(stance),):
        raise ValueError('weights (n, 4), stance (n, 4) bool and rough (n,) required')
    if (rough < 0).any() or (rough > 1).any():
        raise ValueError('rough must be in [0, 1]')
    rear = REAR.to(stance.device)
    static = z[:, :1] * (1 - rear) + z[:, 1:2] * rear
    swing = (z[:, 2:3] * (1 - rear) + z[:, 3:4] * rear) * (~stance).float()
    amount = (static + swing).clamp(-1, 1) * rough[:, None]
    request = torch.zeros(len(stance), 12, device=stance.device)
    request[:, 1::3], request[:, 2::3] = HIPY * amount, KNEE * amount
    return request


def _digest(w):
    return hashlib.sha256(w.numpy().tobytes()).hexdigest()


class ParallelSession:
    def __init__(self, updated, *, replicas=REPLICAS, sigma=SIGMA, learning_rate=LEARNING_RATE):
        self.updated, self.replicas, self.sigma, self.learning_rate = bool(updated), replicas, sigma, learning_rate
        self.w = torch.zeros(len(PARAMETERS), dtype=torch.float64)
        self.updates = 0
        self.active = {}
        self.batch = []

    def digest(self):
        return _digest(self.w)

    def begin(self, replica, noise=None):
        if replica in self.active or not 0 <= replica < self.replicas:
            raise ValueError('episode still active')
        if any(e['replica'] == replica for e in self.batch):
            raise ValueError('replica already finished this round')
        if any(e['version'] != self.updates for e in [*self.active.values(), *self.batch]):
            raise ValueError('round must use one parameter version')
        noise = torch.zeros_like(self.w) if noise is None else torch.as_tensor(noise, dtype=torch.float64).detach().clone()
        if noise.shape != self.w.shape or not torch.isfinite(noise).all() or not torch.isfinite(self.w).all():
            raise ValueError('finite 4-vectors required')
        self.active[replica] = dict(replica=replica, mean=self.w.clone(), noise=noise, code=self.w + self.sigma * noise, version=self.updates)
        return self.requested_code(replica)

    def requested_code(self, replica):
        entry = self.active.get(replica)
        if entry is None:
            raise ValueError('no active episode')
        if self.updates != entry['version'] or not torch.equal(self.w, entry['mean']):
            raise ValueError('parameters changed inside episode')
        return entry['code'].numpy().copy()

    def finish(self, replica, value, *, learn=True):
        code = self.requested_code(replica)
        if not math.isfinite(float(value)):
            raise ValueError('nonfinite return')
        entry = dict(self.active.pop(replica), episode_return=float(value))
        old_digest = self.digest()
        stats = dict(episode_return=float(value), code=code.tolist(), mean=self.w.tolist(), version=self.updates, updated=False)
        if learn:
            self.batch.append(entry)
            if len(self.batch) == self.replicas:
                if self.active:
                    raise ValueError('round complete while an episode is active')
                advantages = loo_advantages([e['episode_return'] for e in self.batch])
                gradient = -(advantages[:, None] * torch.stack([e['noise'] for e in self.batch])).mean(0)
                if not torch.isfinite(gradient).all():
                    raise ValueError('nonfinite gradient')
                stats.update(gradient=gradient.tolist(), advantages=advantages.tolist())
                if self.updated:
                    self.w = (self.w - self.learning_rate * self.sigma * gradient).clamp(-BOUND, BOUND)
                    self.updates += 1
                    stats['updated'] = True
                self.batch = []
        if not self.updated and self.digest() != old_digest:
            raise ValueError('Frozen parameters changed')
        stats.update(updates=self.updates, pending_complete_episodes=len(self.batch),
                     parameter_sha256=self.digest(), b_after=self.w.tolist())
        return stats

    def load_evaluation_mean(self, snapshot):
        if self.updated or self.active or self.batch:
            raise ValueError('evaluation load requires an empty frozen session')
        w = torch.as_tensor(snapshot['w'], dtype=torch.float64).detach().clone()
        if w.shape != self.w.shape or not torch.isfinite(w).all() or (w.abs() > BOUND).any():
            raise ValueError('invalid evaluation parameters')
        self.w = w
        if self.digest() != snapshot['parameter_sha256']:
            raise ValueError('checkpoint parameter hash mismatch')

    def snapshot(self):
        return copy.deepcopy(dict(w=self.w, updates=self.updates, batch=self.batch, active=self.active,
                                  updated=self.updated, parameter_sha256=self.digest()))


class Slot:
    def __init__(self, group, replica):
        self.group, self.replica = group, replica

    updates = property(lambda self: self.group.updates)
    batch = property(lambda self: self.group.batch)

    def digest(self): return self.group.digest()
    def begin(self, noise=None): return self.group.begin(self.replica, noise)
    def requested_code(self): return self.group.requested_code(self.replica)
    def finish(self, value, *, learn=True): return self.group.finish(self.replica, value, learn=learn)
    def load_evaluation_mean(self, snapshot): return self.group.load_evaluation_mean(snapshot)
    def snapshot(self): return self.group.snapshot()
