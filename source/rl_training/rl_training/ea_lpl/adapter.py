"""S1 的小模型：基础策略冻结，每个会话一份约 5000 个权重的适配器，在线更新。方案见 docs/lite3-online-adapter-20261126/PLAN.md。

结构
  film      小网络由输入给出 γ、β，调制基础 actor 最后一个隐层：h′ = (1 + 0.5 tanh γ) ⊙ h + 0.5 tanh β；
            输出端 a = a₀ + D a₀ + d（SHOES，arXiv 2610.16457）。
  residual  小网络的输出直接加在基础动作上（对照）。
两种结构的输出层初始为 0，所以开始时与基础策略逐步相同。

更新
  a  PPO：自定义奖励，损失里加对基础动作的偏离惩罚；critic 从基础策略的复制，先校准再更新适配器。
  b  有方向的误差：适配器输出的残差回归到 directed_target，每个控制步更新一次。
  c  a 和 b 同时进行：b 每个控制步更新，学习率再乘 beta(PPO 更新次数)；a 每批更新。校准阶段两者都不更新适配器。
  s  b 加两处改动（docs/lite3-online-adapter2-20261126/PLAN.md）：学习率按 DECAY_STEPS_S 衰减；作用在机器人上、保存和评估用的
     是训练权重的指数滑动平均（系数 EMA）。
环境按 [copies, replicas] 的顺序排列，与 finetune.Tuner 相同。
"""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .c1b_s import HIPY, KNEE
from .finetune import CLIP, DESIRED_KL, EPOCHS, GAMMA, LAM, MAX_GRAD_NORM, MINIBATCHES, StackedAdam, StackedMLP, clip_per_copy, gae

FILM_HIDDEN, RESIDUAL_HIDDEN, FILM_SCALE = 16, 64, .5
DT, TRACK_SPEED, TRACK_WIDTH, FREE_WEIGHT, CORRECTION_WEIGHT, FAILURE_PENALTY = .02, .4, .0625, .5, 2., 1.
STD, LEARNING_RATE, DEVIATION, VALUE, CALIBRATION, HORIZON, MINIMUM = .1, 1e-3, .5, 1., 1000, 250, 64
TRACK_GAIN, SLOW, DECAY_STEPS, WEIGHT_DECAY = .2, .02, 5000, 1e-4
BETA_START, BETA_HALF = 1., 10
RESIDUAL_LIMITS = [.24, .20, .20] * 4
DECAY_STEPS_S, EMA = 1000, .996  # 更新方式 s：学习率的衰减步数；滑动平均的系数（时间常数 250 步，约一次尝试）
LR_LOW, LR_HIGH = 1e-6, 1e-2  # 探索标准差只有 0.1，同样的 KL 对应的步长比基础训练小得多，下限比基础训练的 1e-5 低一档


class StackedAdapter(nn.Module):
    def __init__(self, actor_state, copies, kind, *, extra, generator=None):
        super().__init__()
        if kind not in ('film', 'residual'):
            raise ValueError('unknown adapter structure')
        self.kind, self.copies = kind, copies
        layers = sorted({int(k.split('.')[1]) for k in actor_state if k.startswith('mlp.')})
        for n, k in enumerate(layers):  # 冻结的基础 actor：buffer，不是参数
            self.register_buffer(f'W{n}', actor_state[f'mlp.{k}.weight'].detach().float().cpu().clone())
            self.register_buffer(f'b{n}', actor_state[f'mlp.{k}.bias'].detach().float().cpu().clone())
        self.depth = len(layers)
        observations, hidden, actions = self.W0.shape[1], getattr(self, f'W{self.depth - 1}').shape[1], getattr(self, f'W{self.depth - 1}').shape[0]
        inputs = observations + extra
        first = lambda rows: nn.Parameter(torch.randn(copies, rows, inputs, generator=generator) / math.sqrt(inputs))
        zeros = lambda *shape: nn.Parameter(torch.zeros(copies, *shape))
        if kind == 'film':
            self.G1, self.g1 = first(FILM_HIDDEN), zeros(FILM_HIDDEN)
            self.G2, self.g2 = zeros(2 * hidden, FILM_HIDDEN), zeros(2 * hidden)
            self.D, self.d = zeros(actions, actions), zeros(actions)
        else:
            self.R1, self.r1 = first(RESIDUAL_HIDDEN), zeros(RESIDUAL_HIDDEN)
            self.R2, self.r2 = zeros(actions, RESIDUAL_HIDDEN), zeros(actions)

    @staticmethod
    def _linear(x, weight, bias):
        return torch.baddbmm(bias[:, None, :], x, weight.transpose(1, 2))

    def forward(self, obs, extra):
        """obs (copies, replicas, 观测)，extra (copies, replicas, 其余输入)。返回（适配后的动作，基础动作）。"""
        with torch.no_grad():
            h = obs
            for n in range(self.depth - 1):
                h = nn.functional.elu(h @ getattr(self, f'W{n}').T + getattr(self, f'b{n}'))
            W, b = getattr(self, f'W{self.depth - 1}'), getattr(self, f'b{self.depth - 1}')
            base = h @ W.T + b
        x = torch.cat((obs, extra), -1)
        if self.kind == 'residual':
            return base + self._linear(torch.tanh(self._linear(x, self.R1, self.r1)), self.R2, self.r2), base
        gamma, shift = self._linear(torch.tanh(self._linear(x, self.G1, self.g1)), self.G2, self.g2).chunk(2, -1)
        a0 = ((1 + FILM_SCALE * torch.tanh(gamma)) * h + FILM_SCALE * torch.tanh(shift)) @ W.T + b
        return a0 + self._linear(a0, self.D, self.d), base

    def export(self, copy):
        return {name: p[copy].detach().float().cpu().clone() for name, p in self.named_parameters()}

    def load(self, copy, saved):
        parameters = dict(self.named_parameters())
        assert set(saved) == set(parameters)
        with torch.no_grad():
            for name, value in saved.items(): parameters[name][copy] = value.to(parameters[name])


def reward(forward_speed, swing, stalled, correction, failed):
    """每步奖励。forward_speed (n,)；swing、stalled (n, 4) 布尔；correction (n, 12) 请求的残差减执行的残差；failed (n,) 姿态终止。"""
    track = torch.exp(-(forward_speed - TRACK_SPEED)**2 / TRACK_WIDTH)
    legs = swing.sum(1)
    free = torch.where(legs > 0, 1 - stalled.sum(1) / legs.clamp(min=1), torch.ones_like(track))
    return DT * (track + FREE_WEIGHT * free - CORRECTION_WEIGHT * (correction**2).sum(1)) - FAILURE_PENALTY * failed.float()


def directed_target(error, stalled, limits=None):
    """error (n, 12) 去掉慢变部分的跟踪误差；stalled (n, 4) 这一步里受阻的摆动腿；limits 是逐关节的残差幅度上限，缺省为 RESIDUAL_LIMITS。返回目标残差 (n, 12)。"""
    target = TRACK_GAIN * error
    lift = stalled.to(error)
    target[:, 1::3] = target[:, 1::3] + HIPY * lift
    target[:, 2::3] = target[:, 2::3] + KNEE * lift
    limits = torch.as_tensor(RESIDUAL_LIMITS if limits is None else limits, dtype=target.dtype, device=target.device)
    return torch.maximum(torch.minimum(target, limits), -limits)  # 超过残差上限的部分会被安全过滤截掉，不作为目标


def adapt_lr(lr, kl, desired=DESIRED_KL):
    lr = torch.where(kl > 2 * desired, (lr / 1.5).clamp(min=LR_LOW), lr)
    return torch.where((kl < desired / 2) & (kl > 0), (lr * 1.5).clamp(max=LR_HIGH), lr)


def beta(updates):
    return BETA_START * .5 ** (updates / BETA_HALF)


class Learner:
    def __init__(self, actor_state, critic_state, *, kind, update, copies, replicas, learn, seed, device, extra,
                 horizon=HORIZON, minimum=MINIMUM, calibration=CALIBRATION, limits=None):
        if update not in ('a', 'b', 'c', 's'):
            raise ValueError('unknown update')
        generator = torch.Generator().manual_seed(seed)
        self.kind, self.update, self.copies, self.replicas, self.learn = kind, update, copies, replicas, bool(learn)
        self.horizon, self.minimum, self.calibration = horizon, minimum, calibration
        self.ppo = update in ('a', 'c')
        self.adapter = StackedAdapter(actor_state, copies, kind, extra=extra, generator=generator).to(device)
        # 更新方式 s：训练的是 adapter，作用在机器人上、保存和评估用的是它的滑动平均 average；其余更新方式两者是同一个网络。
        self.averaged = update == 's'
        self.average = deepcopy(self.adapter).requires_grad_(False) if self.averaged else self.adapter
        self.limits = [float(x) for x in (RESIDUAL_LIMITS if limits is None else limits)]
        self.log_std = nn.Parameter(torch.full((copies, self.adapter.d.shape[1] if kind == 'film' else self.adapter.r2.shape[1]), math.log(STD), device=device))
        self.actor_parameters = [*self.adapter.parameters()] + ([self.log_std] if self.ppo else [])
        self.critic = StackedMLP(critic_state, copies, 'full').to(device) if self.ppo and self.learn else None
        lr = torch.full((copies,), LEARNING_RATE, device=device)
        self.optimizer = StackedAdam(self.actor_parameters, lr) if self.learn and self.ppo else None
        self.directed = StackedAdam(list(self.adapter.parameters()), lr.clone()) if self.learn and update in ('b', 'c', 's') else None
        self.directed_loss = []
        self.critic_optimizer = StackedAdam(list(self.critic.parameters()), lr) if self.critic is not None else None
        self.generator = torch.Generator(device=device).manual_seed(seed + 1)
        n = copies * replicas
        self.slow, self.fresh = torch.zeros(n, 12, device=device), torch.ones(n, dtype=torch.bool, device=device)
        self.updates, self.critic_updates = np.zeros(copies, int), np.zeros(copies, int)
        self.steps, self.rounds, self.buffer, self.pending, self.log = 0, 0, [], None, []

    def _shape(self, x): return x.view(self.copies, self.replicas, *x.shape[1:])

    def learning_rate(self):
        """有方向误差的学习率，随训练步数减小。"""
        return LEARNING_RATE / (1 + self.steps / (DECAY_STEPS_S if self.averaged else DECAY_STEPS))

    def reset(self, rows):
        """一次尝试开始时调用：跟踪误差的慢滤波重新开始。"""
        self.fresh[rows] = True

    @staticmethod
    def _logp(action, mean, log_std):
        log_std = log_std[:, None, :]
        return (-.5 * ((action - mean) / log_std.exp())**2 - log_std - .5 * math.log(2 * math.pi)).sum(-1)

    @torch.no_grad()
    def act(self, obs, critic_obs, extra, active):
        obs, extra = self._shape(obs.float()), self._shape(extra.float())
        mean, base = self.average(obs, extra)
        action = mean
        if self.learn and self.ppo and self.steps >= self.calibration:
            action = mean + self.log_std.exp()[:, None, :] * torch.randn(mean.shape, generator=self.generator, device=mean.device)
        if self.learn:
            self.pending = dict(obs=obs, extra=extra, action=action, mean=mean, base=base, valid=self._shape(active).clone())
            if self.ppo:
                cobs = self._shape(critic_obs.float())
                self.pending.update(cobs=cobs, logp=self._logp(action, mean, self.log_std), value=self.critic(cobs).squeeze(-1))
        return action.reshape(self.copies * self.replicas, -1)

    @torch.no_grad()
    def target(self, error, stalled, active):
        """目标残差 (n, 12)。跟踪误差减去它自己的慢滤波；每次尝试的第一步慢滤波取当时的误差。"""
        error = error.float()
        start = self.fresh & active
        self.slow = torch.where(start[:, None], error, torch.where(active[:, None], self.slow + SLOW * (error - self.slow), self.slow))
        self.fresh = self.fresh & ~active
        return directed_target(error - self.slow, stalled, self.limits) * active[:, None]

    def observe(self, reward, next_critic_obs, done, failed, error, stalled):
        """环境走完一步之后调用。reward (n,)；done、failed (n,) 这一步里回合结束、姿态终止；error (n, 12) 跟踪误差；stalled (n, 4)。"""
        if not self.learn: return None
        if self.pending is None:
            raise ValueError('act must be called first')
        row, self.pending = self.pending, None
        row['target'] = self._shape(self.target(error, stalled, row['valid'].reshape(-1)))
        self.steps += 1
        if not self.ppo: return self._supervised(row, 1.)
        if self.update == 'c' and self.steps > self.calibration: self._supervised(row, beta(self.rounds))
        with torch.no_grad():
            row.update(reward=self._shape(reward.float()), next_value=self.critic(self._shape(next_critic_obs.float())).squeeze(-1),
                       done=self._shape(done), failed=self._shape(failed))
        self.buffer.append(row)
        return self._ppo() if len(self.buffer) >= self.horizon else None

    def _decay(self, active, lr):
        with torch.no_grad():
            for p in self.adapter.parameters():
                p.mul_(torch.where(active, 1 - lr * WEIGHT_DECAY, torch.ones_like(lr)).view((-1,) + (1,) * (p.ndim - 1)))

    def _supervised(self, row, scale):
        weight = row['valid'].float()
        count = weight.sum(1)
        active = count > 0
        if not active.any(): return None
        mean, base = self.adapter(row['obs'], row['extra'])
        loss = (((mean - base - row['target'])**2).sum(-1) * weight).sum(1) / count.clamp(min=1)
        self.directed.zero_grad()
        (loss * active).sum().backward()
        clip_per_copy(self.directed.params, MAX_GRAD_NORM)
        self.directed.lr.fill_(self.learning_rate() * scale)
        self.directed.step(active)
        self._decay(active, self.directed.lr)
        if self.averaged:
            with torch.no_grad():
                for slow, fast in zip(self.average.parameters(), self.adapter.parameters()):
                    on = active.view((-1,) + (1,) * (fast.ndim - 1))
                    slow.copy_(torch.where(on, EMA * slow + (1 - EMA) * fast, slow))
        if not all(torch.isfinite(p).all() for p in self.adapter.parameters()):
            raise ValueError('nonfinite adapter weights')
        if self.ppo: self.directed_loss.append(float(loss.detach()[active].mean()))
        else: self.updates[active.cpu().numpy()] += 1
        return dict(loss=loss.detach())

    def _ppo(self):
        rows, self.buffer = self.buffer, []
        S, N = self.copies, len(rows) * self.replicas
        data = {k: torch.stack([row[k] for row in rows]) for k in rows[0]}  # [T, S, B, ...]
        data['returns'], data['advantage'] = gae(data['reward'], data['value'], data['next_value'], data['done'], data['failed'], data['valid'])
        data = {k: x.transpose(0, 1).reshape(S, N, *x.shape[3:]) for k, x in data.items()}
        weight = data['valid'].float()
        count = weight.sum(1)
        active = count >= self.minimum
        centre = (data['advantage'] * weight).sum(1, keepdim=True) / count.clamp(min=1)[:, None]
        spread = (((data['advantage'] - centre)**2 * weight).sum(1, keepdim=True) / count.clamp(min=1)[:, None]).sqrt()
        data['advantage'] = (data['advantage'] - centre) / (spread + 1e-8)
        data['weight'] = weight
        calibrating = self.steps <= self.calibration  # 这一批数据全部来自校准阶段：只更新 critic
        with torch.no_grad(): data['start'] = self.adapter(data['obs'], data['extra'])[0]  # KL 相对这次更新开始时的策略；c 里采集期间均值还在被 b 改变
        old_sigma = self.log_std.detach().exp()[:, None, :].clone()
        entry = dict(valid=count.int().tolist(), mean_reward=((data['reward'] * weight).sum(1) / count.clamp(min=1)).tolist(), calibrating=calibrating)
        for _ in range(EPOCHS):
            order = torch.argsort(torch.rand(S, N, generator=self.generator, device=weight.device), dim=1)
            for chunk in order.chunk(MINIBATCHES, dim=1):
                b = {k: torch.gather(x, 1, chunk.view(*chunk.shape, *(1,) * (x.ndim - 2)).expand(-1, -1, *x.shape[2:])) for k, x in data.items()}
                n = b['weight'].sum(1).clamp(min=1)
                on = active & (b['weight'].sum(1) > 0)
                value = self.critic(b['cobs']).squeeze(-1)
                clipped = b['value'] + (value - b['value']).clamp(-CLIP, CLIP)
                value_loss = (torch.maximum((value - b['returns'])**2, (clipped - b['returns'])**2) * b['weight']).sum(1) / n
                self.critic_optimizer.zero_grad()
                (VALUE * value_loss * active).sum().backward()
                clip_per_copy(list(self.critic.parameters()), MAX_GRAD_NORM)
                self.critic_optimizer.step(on)
                if calibrating: continue
                mean, base = self.adapter(b['obs'], b['extra'])
                sigma = self.log_std.exp()[:, None, :]
                with torch.no_grad():
                    kl = (torch.log(sigma / old_sigma) + (old_sigma**2 + (b['start'] - mean)**2) / (2 * sigma**2) - .5).sum(-1)
                    kl = (kl * b['weight']).sum(1) / n
                    self.optimizer.lr = torch.where(active, adapt_lr(self.optimizer.lr, kl), self.optimizer.lr)
                    self.critic_optimizer.lr = self.optimizer.lr
                ratio = (self._logp(b['action'], mean, self.log_std) - b['logp']).exp()
                surrogate = -torch.minimum(b['advantage'] * ratio, b['advantage'] * ratio.clamp(1 - CLIP, 1 + CLIP))
                deviation = ((mean - base)**2).sum(-1)
                loss = ((surrogate + DEVIATION * deviation) * b['weight']).sum(1) / n
                self.optimizer.zero_grad()
                (loss * active).sum().backward()
                clip_per_copy(self.actor_parameters, MAX_GRAD_NORM)
                self.optimizer.step(on)
        used = active.cpu().numpy()
        self.critic_updates[used] += 1
        entry.update(value_loss=value_loss.tolist(), lr=self.optimizer.lr.tolist(), std=self.log_std.detach().exp().mean(1).tolist())
        if not calibrating:
            self.updates[used] += 1; self.rounds += 1
            entry.update(kl=kl.tolist(), deviation=((deviation * b['weight']).sum(1) / n).tolist())
            if self.update == 'c':
                entry.update(supervised=float(np.mean(self.directed_loss)) if self.directed_loss else None, beta=beta(self.rounds - 1)); self.directed_loss = []
            if not all(torch.isfinite(p).all() for p in self.actor_parameters):
                raise ValueError('nonfinite adapter weights')
        self.log.append(entry)
        return entry

    def snapshot(self, copy):
        return dict(mode='adapter', kind=self.kind, update=self.update, adapter=self.average.export(copy),
                    log_std=self.log_std[copy].detach().float().cpu().clone(), updates=int(self.updates[copy]))

    def load(self, copy, saved):
        assert saved['mode'] == 'adapter' and saved['kind'] == self.kind and saved['update'] == self.update
        self.adapter.load(copy, saved['adapter'])
        if self.averaged: self.average.load(copy, saved['adapter'])
        with torch.no_grad(): self.log_std[copy] = saved['log_std'].to(self.log_std)

    def digest(self, copy):
        parts = [*self.average.export(copy).values(), self.log_std[copy].detach().cpu()]
        return hashlib.sha256(b''.join(p.numpy().tobytes() for p in parts)).hexdigest()
