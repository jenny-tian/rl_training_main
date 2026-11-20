"""C1-b-S-EA：与 c1b_s.ParallelSession 的唯一区别是两个摆动参数的探索标准差按执行响应调整。

标准差只用已经结束的一轮回合计算，在下一轮开始前固定。标准差等于 SIGMA0 时与 c1b_s 完全相同。
求解公式和系数与 c1a_ea 相同，只把 Σ0 换成 0.7²I。方案见 docs/lite3-online-c1b-ea-20261120/PLAN.md。
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .c1a import loo_advantages
from .c1b_s import BOUND, HIPY, KNEE, PARAMETERS, SIGMA, ParallelSession

SIGMA0 = SIGMA
VARIANCE_MIN, VARIANCE_MAX = .2**2, 1.**2
LATENT_BUDGET = EXECUTION_BUDGET = 2 * SIGMA0**2
BETA, LAMBDA_LOW, LAMBDA_REG = 1., .5, 1.
LOW_RESPONSE, ISOTROPY, CONDITION_MAX, REQUEST_MIN = .25, .9, 1.e4, 1.e-3
GRID = np.linspace(VARIANCE_MIN, VARIANCE_MAX, 97)  # 步长 0.01
SWING = [PARAMETERS.index('front_swing'), PARAMETERS.index('rear_swing')]


def execution_sums(request, executed):
    """request、executed：(n, 12) 的请求残差和安全过滤后实际执行的残差。返回两个 (n, 2)：
    每条腿在 sag 方向上的系数，前腿两条、后腿两条分别求 Σ 执行×请求 和 Σ 请求²。"""
    request, executed = torch.as_tensor(request).double(), torch.as_tensor(executed).double()
    if request.ndim != 2 or request.shape[1] != 12 or executed.shape != request.shape:
        raise ValueError('request and executed must both be (n, 12)')
    coefficient = lambda x: (HIPY * x[:, 1::3] + KNEE * x[:, 2::3]) / (HIPY**2 + KNEE**2)  # (n, 4)：FL FR HL HR
    r, e = coefficient(request), coefficient(executed)
    pair = lambda x: torch.stack((x[:, :2].sum(1), x[:, 2:].sum(1)), 1)
    return pair(e * r).cpu().numpy(), pair(r * r).cpu().numpy()


def response(ratio, mean):
    """(执行比例 × tanh 在均值处的斜率)²；前腿和后腿的摆动残差作用在不同的腿上，响应矩阵是对角的。"""
    ratio, mean = np.asarray(ratio, np.float64), np.asarray(mean, np.float64)
    if ratio.shape != (2,) or mean.shape != (2,) or not (np.isfinite(ratio).all() and np.isfinite(mean).all()):
        raise ValueError('finite 2-vectors required')
    return (ratio * (1. - np.tanh(mean)**2))**2


def ea_variance(eigenvalues):
    m = np.asarray(eigenvalues, np.float64)
    base = np.full(2, SIGMA0**2)
    if m.shape != (2,) or not np.isfinite(m).all() or (m < 0).any(): return base, 'fallback_invalid'
    if m.max() < 1.e-6: return base, 'fallback_no_response'
    if m.min() >= ISOTROPY * m.max(): return base, 'fallback_isotropic'
    if m.max() > CONDITION_MAX * m.min(): return base, 'fallback_condition'
    first, second = np.meshgrid(GRID, GRID, indexing='ij')
    low = (m < LOW_RESPONSE).astype(np.float64)
    objective = (BETA * (m[0] * first + m[1] * second) - LAMBDA_LOW * (low[0] * first + low[1] * second)
                 - LAMBDA_REG * ((first - base[0])**2 + (second - base[1])**2))
    feasible = (first + second <= LATENT_BUDGET + 1.e-12) & (m[0] * first + m[1] * second <= EXECUTION_BUDGET + 1.e-12)
    if not feasible.any(): return base, 'fallback_infeasible'
    objective[~feasible] = -np.inf
    i, j = np.unravel_index(np.argmax(objective), objective.shape)
    return np.array([GRID[i], GRID[j]]), 'solved'


class ParallelSessionEA(ParallelSession):
    def __init__(self, updated, **kwargs):
        super().__init__(updated, active=('front_swing', 'rear_swing'), **kwargs)
        self.std = self.mask * SIGMA0
        self.execution = []
        self.status = 'initial'

    def begin(self, replica, noise=None):
        super().begin(replica, noise)
        entry = self.active[replica]
        entry['code'] = entry['mean'] + self.std * entry['noise']
        entry['std'] = self.std.clone()
        return self.requested_code(replica)

    def finish(self, replica, value, *, learn=True, executed_requested=None, requested_squared=None, steps=0):
        """executed_requested、requested_squared：本回合各步 执行系数×请求系数、请求系数² 的和（前腿、后腿各一个）。"""
        code = self.requested_code(replica)
        if not math.isfinite(float(value)): raise ValueError('nonfinite return')
        if not torch.equal(self.active[replica]['std'], self.std): raise ValueError('covariance changed inside round')
        entry = dict(self.active.pop(replica), episode_return=float(value))
        old_digest = self.digest()
        stats = dict(episode_return=float(value), code=code.tolist(), mean=self.w.tolist(), version=self.updates,
                     updated=False, std=self.std.tolist())
        if learn:
            sums = np.asarray([executed_requested, requested_squared], np.float64)
            if sums.shape != (2, 2) or not np.isfinite(sums).all() or steps <= 0: raise ValueError('execution statistics required')
            self.batch.append(entry); self.execution.append((sums, int(steps)))
            if len(self.batch) == self.replicas:
                if self.active: raise ValueError('round complete while an episode is active')
                advantages = loo_advantages([e['episode_return'] for e in self.batch])
                gradient = -(advantages[:, None] * torch.stack([e['noise'] for e in self.batch])).mean(0)
                if not torch.isfinite(gradient).all(): raise ValueError('nonfinite gradient')
                stats.update(gradient=gradient.tolist(), advantages=advantages.tolist())
                if self.updated:
                    self.w = (self.w - self.learning_rate * self.std * gradient).clamp(-BOUND, BOUND)
                    self.updates += 1
                    stats['updated'] = True
                self.batch = []
                total = sum(s for s, _ in self.execution); count = sum(n for _, n in self.execution)
                self.execution = []
                if (total[1] / count < REQUEST_MIN).any():
                    variance, self.status, ratio, eigenvalues = np.full(2, SIGMA0**2), 'fallback_small_request', [None, None], [None, None]
                else:
                    ratio = total[0] / total[1]
                    eigenvalues = response(ratio, self.w[SWING].numpy())
                    variance, self.status = ea_variance(eigenvalues)
                    ratio, eigenvalues = ratio.tolist(), eigenvalues.tolist()
                self.std = self.std.clone()
                self.std[SWING] = torch.as_tensor(np.sqrt(variance), dtype=torch.float64)
                stats.update(execution_ratio=ratio, response=eigenvalues, covariance_status=self.status, std_after=self.std.tolist())
        if not self.updated and self.digest() != old_digest: raise ValueError('Frozen parameters changed')
        stats.update(updates=self.updates, pending_complete_episodes=len(self.batch),
                     parameter_sha256=self.digest(), b_after=self.w.tolist())
        return stats

    def snapshot(self):
        return copy.deepcopy(dict(super().snapshot(), std=self.std, covariance_status=self.status))


class SlotEA:
    def __init__(self, group, replica):
        self.group, self.replica = group, replica

    updates = property(lambda self: self.group.updates)
    batch = property(lambda self: self.group.batch)

    def digest(self): return self.group.digest()
    def begin(self, noise=None): return self.group.begin(self.replica, noise)
    def requested_code(self): return self.group.requested_code(self.replica)
    def finish(self, value, *, learn=True, **execution): return self.group.finish(self.replica, value, learn=learn, **execution)
    def snapshot(self): return self.group.snapshot()
