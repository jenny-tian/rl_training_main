"""SARA 式在线残差（arXiv 2603.07775）的对照实现：冻结策略上加一条在线残差通道，用关节跟踪误差更新，不用强化学习。

论文最新版把更新规则放在没有公开的附录里，这里按第一版正文的公式写，超参数是按本项目的量级定的，不是原实现的复现。
  特征   h = tanh(V x)，V 固定；φ = 快滤波(h) − 慢滤波(h)
  残差   a = (W_fast + W_slow) φ
  更新   W ← (1 − 衰减) W + 学习率 · e φᵀ / (1 + |φ|²)，一个会话的并行环境取平均；快权重的学习率在机身倾斜大时放大
  门控   u = γ (β ⊙ a)；与基础动作符号相反的分量乘 KAPPA；逐关节上限和总幅度上限
一个会话（copy）一份权重，会话里的每个并行环境各有自己的滤波状态和门控状态。方案见 docs/lite3-online-baselines-20261125/PLAN.md。
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

FEATURES, ALPHA_FAST, ALPHA_SLOW = 2500, .5, .05
SETTINGS = dict(sa1=dict(eta_fast=.02, eta_slow=.002), sa2=dict(eta_fast=.10, eta_slow=.010))
DECAY_FAST, DECAY_SLOW = .01, .0005
TILT_THRESHOLD, BOOST_DECAY, BOOST_STEP, BOOST_MAX = .15, .98, .1, 3.
KAPPA, NORM_BOUND = .25, .5
GAMMA_MIN, GAMMA_GAIN, GAMMA_MAX, J_NOMINAL, J_SMOOTH, XI = .2, .8, 1., .88, .02, .05
BETA_MIN, BETA_MAX, BETA_STEP, ERROR_THRESHOLD = 1., 2., .01, 1.5
DT = .02


def tracking_error(target, position, velocity, scale):
    """e = [(q目标 − q) − DT · q̇] / 动作比例；正的误差表示关节还没到目标。"""
    return ((target - position) - DT * velocity) / scale


def gate(residual, nominal, gamma, beta, limits):
    """residual、nominal (n, 12)；gamma (n,)；beta (n, 12)；limits (12,)。返回门控后的残差。"""
    u = gamma[:, None] * beta * residual
    u = torch.where(torch.sign(nominal) * torch.sign(u) < 0, KAPPA * u, u)
    u = torch.maximum(torch.minimum(u, limits), -limits)
    norm = torch.linalg.vector_norm(u, dim=1, keepdim=True)
    return u * (NORM_BOUND / norm.clamp(min=NORM_BOUND))


class Sara:
    def __init__(self, variant, copies, replicas, observations, limits, *, learn, seed, device='cpu'):
        if variant not in SETTINGS:
            raise ValueError('unknown SARA setting')
        self.setting, self.copies, self.replicas, self.learn = SETTINGS[variant], copies, replicas, bool(learn)
        generator = torch.Generator().manual_seed(seed)
        self.V = (torch.randn(FEATURES, observations, generator=generator) / observations**.5).to(device)
        self.W_fast, self.W_slow = (torch.zeros(copies, 12, FEATURES, device=device) for _ in range(2))
        self.limits = torch.as_tensor(limits, dtype=torch.float32, device=device)
        n = copies * replicas
        self.trace_fast, self.trace_slow = (torch.zeros(n, FEATURES, device=device) for _ in range(2))
        self.boost = torch.zeros(n, device=device)
        self.performance, self.minimum = (torch.full((n,), J_NOMINAL, device=device) for _ in range(2))
        self.beta = torch.full((n, 12), BETA_MIN, device=device)
        self.gamma = torch.full((n,), GAMMA_MIN, device=device)
        self.updates = np.zeros(copies, int)
        self.phi = None

    def reset(self, rows):
        """一次尝试开始时调用：滤波和门控状态清零，权重保留。"""
        self.trace_fast[rows] = 0; self.trace_slow[rows] = 0; self.boost[rows] = 0
        self.performance[rows] = J_NOMINAL; self.minimum[rows] = J_NOMINAL
        self.beta[rows] = BETA_MIN; self.gamma[rows] = GAMMA_MIN

    def act(self, observation, nominal, active):
        """observation (n, m)，nominal (n, 12) 基础动作，active (n,) 是否在策略阶段。返回 (n, 12) 残差请求。"""
        h = torch.tanh(observation @ self.V.T)
        self.trace_fast = torch.where(active[:, None], (1 - ALPHA_FAST) * self.trace_fast + ALPHA_FAST * h, self.trace_fast)
        self.trace_slow = torch.where(active[:, None], (1 - ALPHA_SLOW) * self.trace_slow + ALPHA_SLOW * h, self.trace_slow)
        self.phi = self.trace_fast - self.trace_slow
        phi = self.phi.view(self.copies, self.replicas, FEATURES)
        residual = torch.einsum('cjp,crp->crj', self.W_fast + self.W_slow, phi).reshape(-1, 12)
        return gate(residual, nominal, self.gamma, self.beta, self.limits) * active[:, None]

    def observe(self, error, tilt, performance, active):
        """环境走完一步之后调用。error (n, 12) 跟踪误差；tilt (n,) 俯仰与横滚重力分量之和；performance (n,) 前进速度比；active (n,)。"""
        if self.phi is None:
            raise ValueError('act must be called first')
        on = active.float()
        self.boost = torch.where(active, (BOOST_DECAY * self.boost + BOOST_STEP * (tilt > TILT_THRESHOLD)).clamp(0, BOOST_MAX), self.boost)
        self.performance = torch.where(active, (1 - J_SMOOTH) * self.performance + J_SMOOTH * performance, self.performance)
        self.minimum = torch.minimum(self.minimum, self.performance)
        self.gamma = (GAMMA_MIN + GAMMA_GAIN * (J_NOMINAL - self.performance) / (J_NOMINAL - self.minimum + XI)).clamp(0, GAMMA_MAX)
        self.beta = torch.where(active[:, None], (self.beta + BETA_STEP * (error.abs() > ERROR_THRESHOLD)).clamp(BETA_MIN, BETA_MAX), self.beta)
        if not self.learn: return
        count = on.view(self.copies, self.replicas).sum(1)
        used = count > 0
        if not used.any(): return
        phi = self.phi / (1 + (self.phi**2).sum(1, keepdim=True))
        weight = on / count.repeat_interleave(self.replicas).clamp(min=1)
        shape = (self.copies, self.replicas)
        step_slow = torch.einsum('crj,crp->cjp', (error * weight[:, None]).view(*shape, 12), phi.view(*shape, FEATURES))
        step_fast = torch.einsum('crj,crp->cjp', (error * (weight * (1 + self.boost))[:, None]).view(*shape, 12), phi.view(*shape, FEATURES))
        keep = used[:, None, None]
        self.W_fast = torch.where(keep, (1 - DECAY_FAST) * self.W_fast + self.setting['eta_fast'] * step_fast, self.W_fast)
        self.W_slow = torch.where(keep, (1 - DECAY_SLOW) * self.W_slow + self.setting['eta_slow'] * step_slow, self.W_slow)
        if not (torch.isfinite(self.W_fast).all() and torch.isfinite(self.W_slow).all()):
            raise ValueError('nonfinite SARA weights')
        self.updates[used.cpu().numpy()] += 1

    def snapshot(self, copy):
        return dict(mode='sara', fast=self.W_fast[copy].cpu().clone(), slow=self.W_slow[copy].cpu().clone(), updates=int(self.updates[copy]))

    def load(self, copy, saved):
        assert saved['mode'] == 'sara'
        self.W_fast[copy] = saved['fast'].to(self.W_fast); self.W_slow[copy] = saved['slow'].to(self.W_slow)

    def digest(self, copy):
        return hashlib.sha256(self.W_fast[copy].cpu().numpy().tobytes() + self.W_slow[copy].cpu().numpy().tobytes()).hexdigest()


class Group:
    """采集循环里一个会话的一组。Frozen 组没有残差通道，sara 和 copy 都是 None。"""

    def __init__(self, sara, copy):
        self.sara, self.copy, self.batch = sara, copy, []

    updates = property(lambda self: 0 if self.sara is None else int(self.sara.updates[self.copy]))

    def digest(self): return 'frozen' if self.sara is None else self.sara.digest(self.copy)
    def snapshot(self): return dict(mode='frozen') if self.sara is None else self.sara.snapshot(self.copy)

    def load_evaluation_mean(self, saved):
        if self.sara is None: assert saved['mode'] == 'frozen'
        else: self.sara.load(self.copy, saved)


class Slot:
    def __init__(self, group, replica):
        self.group, self.replica = group, replica

    updates = property(lambda self: self.group.updates)
    batch = property(lambda self: self.group.batch)

    def digest(self): return self.group.digest()
    def begin(self, noise=None): return np.zeros(2)
    def requested_code(self): return np.zeros(2)
    def finish(self, value, *, learn=True, **_): return {}
    def load_evaluation_mean(self, saved): return self.group.load_evaluation_mean(saved)
    def snapshot(self): return self.group.snapshot()
