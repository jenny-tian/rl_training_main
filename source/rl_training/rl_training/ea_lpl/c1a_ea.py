"""C1-a-EA：回合级偏置学习，探索协方差按执行响应调整。

与 c1a.Session 的唯一区别是探索标准差。均值更新规则、批大小、学习率都不变。
协方差只用已经结束的回合计算，在下一批回合开始前固定，梯度把它当作给定量。
规则见 docs/lite3-online-c1a-ea-20261117/PLAN.md。
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .c1a import Session, loo_advantages

SIGMA0 = .6
VARIANCE_MIN, VARIANCE_MAX = .2**2, 1.**2
LATENT_BUDGET = 2 * SIGMA0**2
EXECUTION_BUDGET = 2 * SIGMA0**2  # tr(M0 Σ0)，M0 为零码、无过滤时的响应（单位阵）
BETA, LAMBDA_LOW, LAMBDA_REG = 1., .5, 1.
LOW_RESPONSE = .25  # 响应特征值低于它（等效增益低于 0.5）的方向记为低响应
ISOTROPY = .9  # 最小与最大特征值之比不低于它时视为各向同性
CONDITION_MAX = 1.e4
REQUEST_MIN = 1.e-3  # 一批回合内请求量的均方值低于它时无法估计执行比例
GRID = np.linspace(VARIANCE_MIN, VARIANCE_MAX, 97)  # 步长 0.01


def response(ratio, mean):
    """对角响应矩阵的特征值：(执行比例 × tanh 在均值处的斜率)²。sag 的两列作用在不同关节上，正交且等长。"""
    ratio, mean = np.asarray(ratio, np.float64), np.asarray(mean, np.float64)
    if ratio.shape != (2,) or mean.shape != (2,) or not (np.isfinite(ratio).all() and np.isfinite(mean).all()):
        raise ValueError('finite 2-vectors required')
    return (ratio * (1. - np.tanh(mean)**2))**2


def ea_variance(eigenvalues):
    """返回 (两个方向的方差, 说明)。任何回退情形都返回 Σ0。"""
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


class SessionEA(Session):
    """finish() 额外接收本回合的执行统计；每凑满 4 个回合重新计算一次探索标准差。"""

    def __init__(self, updated):
        super().__init__(updated)
        self.std = torch.full((2,), SIGMA0, dtype=torch.float64)
        self.execution = []
        self.status = 'initial'

    def begin(self, noise=None):
        code = super().begin(noise)
        if noise is not None:
            self.active['code'] = self.active['mean'] + self.std * torch.as_tensor(noise, dtype=torch.float64)
            code = self.requested_code()
        self.active['std'] = self.std.clone()
        return code

    def finish(self, value, *, learn=True, executed_requested=None, requested_squared=None, steps=0):
        """executed_requested、requested_squared：本回合各步 执行系数×请求系数、请求系数² 的和（每个码维一个）。"""
        code = self.requested_code()
        if not math.isfinite(float(value)): raise ValueError('nonfinite return')
        std = self.active['std']
        if not torch.equal(std, self.std): raise ValueError('covariance changed inside episode')
        stats = dict(episode_return=float(value), code=code.tolist(), mean=self.b.tolist(), version=self.updates,
                     updated=False, std=std.tolist())
        if learn:
            sums = np.asarray([executed_requested, requested_squared], np.float64)
            if sums.shape != (2, 2) or not np.isfinite(sums).all() or steps <= 0: raise ValueError('execution statistics required')
            entry = dict(self.active, episode_return=float(value))
            candidate = [*self.batch, entry]
            if len(candidate) > 4 or any(e['version'] != self.updates or not torch.equal(e['mean'], self.b)
                                         or not torch.equal(e['std'], self.std) for e in candidate):
                raise ValueError('batch must use one current parameter version and covariance')
            self.execution.append((sums, int(steps)))
            if len(candidate) == 4:
                actions = torch.stack([e['code'] for e in candidate])
                adv = loo_advantages([e['episode_return'] for e in candidate])
                gradient = -(adv[:, None] * (actions - self.b)).mean(0) / self.std**2
                if not torch.isfinite(gradient).all(): raise ValueError('nonfinite gradient')
                stats.update(gradient=gradient.tolist(), advantages=adv.tolist())
                if self.updated:
                    t = self.updates + 1
                    m = .9 * self.m + .1 * gradient
                    v = .999 * self.v + .001 * gradient.square()
                    b = (self.b - .3 * (m / (1 - .9**t)) / ((v / (1 - .999**t)).sqrt() + 1.e-8)).clamp(-2.5, 2.5)
                    if not all(torch.isfinite(x).all() for x in (b, m, v)): raise ValueError('nonfinite Adam state')
                    self.b, self.m, self.v, self.updates = b, m, v, t
                    stats['updated'] = True
                self.batch = []
                total = sum(s for s, _ in self.execution); count = sum(n for _, n in self.execution)
                self.execution = []
                if (total[1] / count < REQUEST_MIN).any():
                    variance, self.status, ratio, eigenvalues = np.full(2, SIGMA0**2), 'fallback_small_request', [None, None], [None, None]
                else:
                    ratio = total[0] / total[1]
                    eigenvalues = response(ratio, self.b.numpy())
                    variance, self.status = ea_variance(eigenvalues)
                    ratio, eigenvalues = ratio.tolist(), eigenvalues.tolist()
                self.std = torch.as_tensor(np.sqrt(variance), dtype=torch.float64)
                stats.update(execution_ratio=ratio, response=eigenvalues, covariance_status=self.status,
                             std_after=self.std.tolist())
            else:
                self.batch = candidate
        self.active = None
        if not self.updated and self.updates: raise ValueError('Frozen parameters changed')
        stats.update(updates=self.updates, pending_complete_episodes=len(self.batch),
                     parameter_sha256=self.digest(), b_after=self.b.tolist())
        return stats

    def snapshot(self):
        return dict(super().snapshot(), std=self.std.clone(), covariance_status=self.status,
                    execution=[(s.tolist(), n) for s, n in self.execution])
