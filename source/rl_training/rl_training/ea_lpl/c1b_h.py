"""C1-b-H：码由地形高度特征决定，每 10 步重新计算；探索加在参数上，一个回合抽一次。

码 = W f，W 是 2 x 4 矩阵，f = [起伏, 刚上台阶, 台阶中段, 台阶顶部]，平地上 f = 0。
更新规则与 c1a_m4 相同：一个会话的并行环境共用一个 W，凑满一轮后做一次普通梯度步。
方案见 docs/lite3-online-c1b-height-20261118/PLAN.md。
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch

from .c1a import loo_advantages

REPLICAS, SIGMA, LEARNING_RATE, BOUND, INTERVAL = 4, .7, .6, 2.5, 10
FEATURES = ('rough', 'entry', 'middle', 'top')
SHAPE = (2, len(FEATURES))
RISE, RISE_FLOOR, ROUGH_FLOOR, ROUGH_SPAN = .06, .005, .004, .008
SCRIPTS = dict(A={}, B=dict(rough=(1.5, -1.5)), C=dict(top=(1.5, -1.5)),
               D=dict(middle=(1.5, -1.5), top=(1.5, -1.5)))


def exploration_table(pairs=60, max_episodes=400, seed=2026111801):
    return np.random.default_rng(seed).standard_normal((pairs, max_episodes, SHAPE[0] * SHAPE[1]))


def scripted(name):
    """脚本化检查用的固定 W，按 load_evaluation_mean 接受的格式返回。"""
    w = torch.zeros(SHAPE, dtype=torch.float64)
    for feature, column in SCRIPTS[name].items():
        w[:, FEATURES.index(feature)] = torch.tensor(column, dtype=torch.float64)
    return dict(W=w, parameter_sha256=_digest(w))


def _digest(w):
    return hashlib.sha256(w.numpy().tobytes()).hexdigest()


class TerrainFeatures:
    """由机身下方的高度射线算出 4 个特征。xy 是射线在机身偏航坐标系里的水平位置。"""

    def __init__(self, xy):
        xy = torch.as_tensor(xy).detach().double()
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError('ray positions must be (rays, 2)')
        self.device = xy.device
        xs = torch.unique(torch.round(xy[:, 0] * 1000) / 1000)
        lane = xy[:, 1].abs() <= .2 + 1e-6
        columns = [torch.nonzero(lane & ((xy[:, 0] - x).abs() < 1e-3))[:, 0] for x in xs]
        if len({len(c) for c in columns}) != 1 or len(columns[0]) == 0:
            raise ValueError('rays do not form a regular grid')
        self.columns = torch.stack(columns)
        between = lambda low, high: (xs >= low - 1e-6) & (xs <= high + 1e-6)
        self.front, self.middle, self.back = between(.3, .6), between(-.1, .1), between(-.6, -.3)
        if not (self.front.any() and self.middle.any() and self.back.any()):
            raise ValueError('ray grid does not cover the three height windows')
        design = torch.cat((xy, torch.ones_like(xy[:, :1])), 1)
        self.design, self.solve = design, torch.linalg.pinv(design)
        self.missing = 0

    def __call__(self, heights):
        z = torch.as_tensor(heights, device=self.device).detach().double()
        if z.ndim != 2 or z.shape[1] != len(self.design):
            raise ValueError('heights must be (environments, rays)')
        finite = torch.isfinite(z)
        if not finite.all():
            if not finite.any(1).all():
                raise ValueError('no ray hit the ground')
            self.missing += int((~finite).sum())
            fill = torch.stack([row[ok].median() for row, ok in zip(z, finite)])
            z = torch.where(finite, z, fill[:, None])
        profile = z[:, self.columns].median(-1).values
        front, middle, back = (profile[:, m].mean(1) for m in (self.front, self.middle, self.back))
        up = ((front - middle - RISE_FLOOR) / (RISE - RISE_FLOOR)).clamp(0, 1)
        behind = ((middle - back - RISE_FLOOR) / (RISE - RISE_FLOOR)).clamp(0, 1)
        residual = z - (z @ self.solve.T) @ self.design.T
        rough = ((residual.pow(2).mean(1).sqrt() - ROUGH_FLOOR) / ROUGH_SPAN).clamp(0, 1)
        # 三个台阶特征都乘上起伏，平整斜坡上前后也有高度差，但起伏为零。
        return torch.stack((rough, rough * up * (1 - behind), rough * up * behind, rough * (1 - up) * behind), 1)


class ParallelSession:
    def __init__(self, updated, *, replicas=REPLICAS, sigma=SIGMA, learning_rate=LEARNING_RATE):
        self.updated, self.replicas, self.sigma, self.learning_rate = bool(updated), replicas, sigma, learning_rate
        self.W = torch.zeros(SHAPE, dtype=torch.float64)
        self.updates = 0
        self.active = {}
        self.batch = []

    def digest(self):
        return _digest(self.W)

    def begin(self, replica, noise=None):
        if replica in self.active or not 0 <= replica < self.replicas:
            raise ValueError('episode still active')
        if any(e['replica'] == replica for e in self.batch):
            raise ValueError('replica already finished this round')
        if any(e['version'] != self.updates for e in [*self.active.values(), *self.batch]):
            raise ValueError('round must use one parameter version')
        noise = torch.zeros(SHAPE, dtype=torch.float64) if noise is None else torch.as_tensor(noise, dtype=torch.float64).detach().clone().reshape(SHAPE)
        if not torch.isfinite(noise).all() or not torch.isfinite(self.W).all():
            raise ValueError('finite parameters required')
        self.active[replica] = dict(replica=replica, mean=self.W.clone(), noise=noise, weights=self.W + self.sigma * noise,
                                    version=self.updates, code=torch.zeros(2, dtype=torch.float64), decisions=0)
        return self.requested_code(replica)

    def _entry(self, replica):
        entry = self.active.get(replica)
        if entry is None:
            raise ValueError('no active episode')
        if self.updates != entry['version'] or not torch.equal(self.W, entry['mean']):
            raise ValueError('parameters changed inside episode')
        return entry

    def choose(self, replica, features):
        entry = self._entry(replica)
        features = torch.as_tensor(features, dtype=torch.float64).detach().clone()
        if features.shape != (SHAPE[1],) or not torch.isfinite(features).all() or (features < 0).any() or (features > 1).any():
            raise ValueError('features must be four numbers in [0, 1]')
        entry['code'] = entry['weights'] @ features
        entry['decisions'] += 1
        return entry['code'].numpy().copy()

    def requested_code(self, replica):
        return self._entry(replica)['code'].numpy().copy()

    def finish(self, replica, value, *, learn=True):
        entry = self._entry(replica)
        if not math.isfinite(float(value)):
            raise ValueError('nonfinite return')
        del self.active[replica]
        entry = dict(entry, episode_return=float(value))
        old_digest = self.digest()
        stats = dict(episode_return=float(value), code=entry['weights'].flatten().tolist(), mean=self.W.flatten().tolist(),
                     version=self.updates, decisions=entry['decisions'], updated=False)
        if learn:
            self.batch.append(entry)
            if len(self.batch) == self.replicas:
                if self.active:
                    raise ValueError('round complete while an episode is active')
                advantages = loo_advantages([e['episode_return'] for e in self.batch])
                noises = torch.stack([e['noise'] for e in self.batch])
                gradient = -(advantages[:, None, None] * noises).mean(0)
                if not torch.isfinite(gradient).all():
                    raise ValueError('nonfinite gradient')
                stats.update(gradient=gradient.flatten().tolist(), advantages=advantages.tolist())
                if self.updated:
                    self.W = (self.W - self.learning_rate * self.sigma * gradient).clamp(-BOUND, BOUND)
                    self.updates += 1
                    stats['updated'] = True
                self.batch = []
        if not self.updated and self.digest() != old_digest:
            raise ValueError('Frozen parameters changed')
        stats.update(updates=self.updates, pending_complete_episodes=len(self.batch),
                     parameter_sha256=self.digest(), b_after=self.W.flatten().tolist())
        return stats

    def load_evaluation_mean(self, snapshot):
        if self.updated or self.active or self.batch:
            raise ValueError('evaluation load requires an empty frozen session')
        w = torch.as_tensor(snapshot['W'], dtype=torch.float64).detach().clone()
        if w.shape != SHAPE or not torch.isfinite(w).all() or (w.abs() > BOUND).any():
            raise ValueError('invalid evaluation parameters')
        self.W = w
        if self.digest() != snapshot['parameter_sha256']:
            raise ValueError('checkpoint parameter hash mismatch')

    def snapshot(self):
        return copy.deepcopy(dict(W=self.W, updates=self.updates, batch=self.batch, active=self.active,
                                  updated=self.updated, parameter_sha256=self.digest()))


class Slot:
    """一个并行环境看到的接口，与 c1a_m4.Slot 相同，另有 choose。"""

    def __init__(self, group, replica):
        self.group, self.replica = group, replica

    updates = property(lambda self: self.group.updates)
    batch = property(lambda self: self.group.batch)

    def digest(self): return self.group.digest()
    def begin(self, noise=None): return self.group.begin(self.replica, noise)
    def choose(self, features): return self.group.choose(self.replica, features)
    def requested_code(self): return self.group.requested_code(self.replica)
    def finish(self, value, *, learn=True): return self.group.finish(self.replica, value, learn=learn)
    def load_evaluation_mean(self, snapshot): return self.group.load_evaluation_mean(snapshot)
    def snapshot(self): return self.group.snapshot()
