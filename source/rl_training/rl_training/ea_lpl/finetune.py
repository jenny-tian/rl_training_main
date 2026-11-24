"""直接微调基础策略的对照：全参数微调和秩 1 LoRA，每个会话一份互相独立的网络，一起前向和更新。

PPO 的设置与训练基础策略时相同（见 agents/rsl_rl_ppo_cfg.py），按 KL 调学习率的规则也相同。
LoRA 的做法取自 SLowRL（arXiv 2603.17092）：原权重冻结，每个全连接层并联 (alpha / rank) * B A x，在激活函数之前相加，B 初始为 0。
方案见 docs/lite3-online-finetune-20261124/PLAN.md。
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn

GAMMA, LAM, CLIP, ENTROPY, VALUE, EPOCHS, MINIBATCHES, MAX_GRAD_NORM, DESIRED_KL = .99, .95, .2, .01, 1., 5, 4, 1., .01
LR_LOW, LR_HIGH = 1e-5, 1e-2


class StackedMLP(nn.Module):
    """copies 份 MLP。full：每份有自己的全部权重和偏置；lora：权重和偏置共享并冻结，每份只有各层的低秩支路。"""

    def __init__(self, state, copies, mode, *, rank=1, alpha=1., generator=None):
        super().__init__()
        assert mode in ('full', 'lora')
        self.mode, self.copies, self.scale = mode, copies, alpha / rank
        self.layers = sorted({int(k.split('.')[1]) for k in state if k.startswith('mlp.')})
        weights = [state[f'mlp.{k}.weight'].detach().float().cpu() for k in self.layers]
        biases = [state[f'mlp.{k}.bias'].detach().float().cpu() for k in self.layers]
        if mode == 'full':
            self.W = nn.ParameterList([nn.Parameter(w.expand(copies, -1, -1).clone()) for w in weights])
            self.b = nn.ParameterList([nn.Parameter(b.expand(copies, -1).clone()) for b in biases])
        else:
            for k, (w, b) in enumerate(zip(weights, biases)):
                self.register_buffer(f'W0_{k}', w.clone()); self.register_buffer(f'b0_{k}', b.clone())
            self.A = nn.ParameterList([nn.Parameter(torch.randn(copies, rank, w.shape[1], generator=generator) / math.sqrt(w.shape[1])) for w in weights])
            self.B = nn.ParameterList([nn.Parameter(torch.zeros(copies, w.shape[0], rank)) for w in weights])

    def forward(self, x):
        for k in range(len(self.layers)):
            if self.mode == 'full':
                x = torch.baddbmm(self.b[k][:, None, :], x, self.W[k].transpose(1, 2))
            else:
                branch = torch.bmm(torch.bmm(x, self.A[k].transpose(1, 2)), self.B[k].transpose(1, 2))
                x = x @ getattr(self, f'W0_{k}').T + getattr(self, f'b0_{k}') + self.scale * branch
            if k < len(self.layers) - 1: x = nn.functional.elu(x)
        return x

    def export(self, copy):
        return {name: p[copy].detach().float().cpu().clone() for name, p in self.named_parameters()}

    def load(self, copy, saved):
        parameters = dict(self.named_parameters())
        assert set(saved) == set(parameters)
        with torch.no_grad():
            for name, value in saved.items(): parameters[name][copy] = value.to(parameters[name])


class StackedAdam:
    """与 torch.optim.Adam 相同的更新；每份有自己的学习率和步数，step 只更新 active 的那些份。"""

    def __init__(self, params, lr, betas=(.9, .999), eps=1e-8):
        self.params, self.lr, self.betas, self.eps = list(params), lr.clone().float(), betas, eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = torch.zeros(len(lr), device=self.params[0].device)

    def zero_grad(self):
        for p in self.params: p.grad = None

    @torch.no_grad()
    def step(self, active):
        b1, b2 = self.betas
        self.t += active
        t = self.t.clamp(min=1)
        size = self.lr.to(t) / (1 - b1**t)
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None: continue
            shape = (-1,) + (1,) * (p.ndim - 1)
            on = active.view(shape)
            m.copy_(torch.where(on, b1 * m + (1 - b1) * p.grad, m))
            v.copy_(torch.where(on, b2 * v + (1 - b2) * p.grad**2, v))
            step = size.view(shape) * m / (v.sqrt() / (1 - b2**t).sqrt().view(shape) + self.eps)
            p.sub_(torch.where(on, step, torch.zeros_like(step)))


@torch.no_grad()
def clip_per_copy(params, max_norm):
    params = [p for p in params if p.grad is not None]
    norm = torch.sqrt(sum((p.grad**2).flatten(1).sum(1) for p in params))
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.)
    for p in params: p.grad.mul_(scale.view((-1,) + (1,) * (p.ndim - 1)))


def adapt_lr(lr, kl, desired=DESIRED_KL):
    lr = torch.where(kl > 2 * desired, (lr / 1.5).clamp(min=LR_LOW), lr)
    return torch.where((kl < desired / 2) & (kl > 0), (lr * 1.5).clamp(max=LR_HIGH), lr)


def gae(reward, value, next_value, done, failed, valid, gamma=GAMMA, lam=LAM):
    """第 0 维是时间。failed：姿态终止，不自举；其他结束方式（done 且不是 failed）用下一状态的价值自举。valid 为假的步不参与。"""
    advantage, following, following_valid = torch.zeros_like(value), torch.zeros_like(value[0]), torch.zeros_like(valid[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * next_value[t] * (~failed[t]) - value[t]
        following = (delta + gamma * lam * ((~done[t]) & following_valid) * following) * valid[t]
        advantage[t], following_valid = following, valid[t]
    return advantage + value * valid, advantage


class Tuner:
    """copies 份 actor 和 critic，每份对应 replicas 个环境。环境按 [copies, replicas] 的顺序排列。"""

    def __init__(self, actor_state, critic_state, *, copies, replicas, mode, learn, seed, device, lr, horizon=250, minimum=64):
        generator = torch.Generator().manual_seed(seed)
        self.copies, self.replicas, self.mode, self.learn, self.horizon, self.minimum = copies, replicas, mode, learn, horizon, minimum
        self.actor = StackedMLP(actor_state, copies, mode, generator=generator).to(device)
        self.critic = StackedMLP(critic_state, copies, mode, generator=generator).to(device)
        self.log_std = nn.Parameter(actor_state['distribution.log_std_param'].detach().float().expand(copies, -1).clone().to(device))
        self.parameters = [p for p in (*self.actor.parameters(), *self.critic.parameters(), self.log_std)]
        self.optimizer = StackedAdam(self.parameters, torch.full((copies,), float(lr), device=device)) if learn else None
        self.generator = torch.Generator(device=device).manual_seed(seed + 1)
        self.updates = np.zeros(copies, int)
        self.buffer, self.pending, self.log = [], None, []

    def _shape(self, x): return x.view(self.copies, self.replicas, *x.shape[1:])

    @torch.no_grad()
    def act(self, obs, critic_obs, active, sample):
        obs, critic_obs = self._shape(obs.float()), self._shape(critic_obs.float())
        mean = self.actor(obs)
        action = mean
        if sample:
            sigma = self.log_std.exp()[:, None, :]
            action = mean + sigma * torch.randn(mean.shape, generator=self.generator, device=mean.device)
        if self.learn:
            self.pending = dict(obs=obs, cobs=critic_obs, action=action, mean=mean, logp=self._logp(action, mean, self.log_std),
                                value=self.critic(critic_obs).squeeze(-1), valid=self._shape(active).clone())
        return action.reshape(self.copies * self.replicas, -1)

    @staticmethod
    def _logp(action, mean, log_std):
        log_std = log_std[:, None, :]
        return (-.5 * ((action - mean) / log_std.exp())**2 - log_std - .5 * math.log(2 * math.pi)).sum(-1)

    @torch.no_grad()
    def record(self, reward, next_critic_obs, done, failed):
        if not self.learn: return
        row, self.pending = self.pending, None
        row.update(reward=self._shape(reward.float()), next_value=self.critic(self._shape(next_critic_obs.float())).squeeze(-1),
                   done=self._shape(done), failed=self._shape(failed))
        self.buffer.append(row)

    def ready(self): return self.learn and len(self.buffer) >= self.horizon

    def update(self):
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
        old_sigma = self.log_std.detach().exp()[:, None, :].clone()
        entry = dict(valid=count.int().tolist(), mean_reward=((data['reward'] * weight).sum(1) / count.clamp(min=1)).tolist())
        for _ in range(EPOCHS):
            order = torch.argsort(torch.rand(S, N, generator=self.generator, device=weight.device), dim=1)
            for chunk in order.chunk(MINIBATCHES, dim=1):
                b = {k: torch.gather(x, 1, chunk.view(*chunk.shape, *(1,) * (x.ndim - 2)).expand(-1, -1, *x.shape[2:])) for k, x in data.items()}
                n = b['weight'].sum(1).clamp(min=1)
                mean, value = self.actor(b['obs']), self.critic(b['cobs']).squeeze(-1)
                sigma = self.log_std.exp()[:, None, :]
                with torch.no_grad():
                    kl = (torch.log(sigma / old_sigma) + (old_sigma**2 + (b['mean'] - mean)**2) / (2 * sigma**2) - .5).sum(-1)
                    kl = (kl * b['weight']).sum(1) / n
                    self.optimizer.lr = torch.where(active, adapt_lr(self.optimizer.lr, kl), self.optimizer.lr)
                ratio = (self._logp(b['action'], mean, self.log_std) - b['logp']).exp()
                surrogate = -torch.minimum(b['advantage'] * ratio, b['advantage'] * ratio.clamp(1 - CLIP, 1 + CLIP))
                clipped = b['value'] + (value - b['value']).clamp(-CLIP, CLIP)
                value_loss = torch.maximum((value - b['returns'])**2, (clipped - b['returns'])**2)
                entropy = (.5 + .5 * math.log(2 * math.pi) + self.log_std).sum(-1)
                loss = ((surrogate + VALUE * value_loss) * b['weight']).sum(1) / n - ENTROPY * entropy
                self.optimizer.zero_grad()
                (loss * active).sum().backward()
                clip_per_copy(self.parameters, MAX_GRAD_NORM)
                self.optimizer.step(active & (b['weight'].sum(1) > 0))
        self.updates[active.cpu().numpy()] += 1
        entry.update(kl=kl.tolist(), lr=self.optimizer.lr.tolist(), value_loss=((value_loss * b['weight']).sum(1) / n).tolist(),
                     std=self.log_std.detach().exp().mean(1).tolist())
        self.log.append(entry)
        return entry

    def snapshot(self, copy):
        return dict(mode=self.mode, actor=self.actor.export(copy), log_std=self.log_std[copy].detach().float().cpu().clone(), updates=int(self.updates[copy]))

    def load(self, copy, saved):
        assert saved['mode'] == self.mode
        self.actor.load(copy, saved['actor'])
        with torch.no_grad(): self.log_std[copy] = saved['log_std'].to(self.log_std)

    def digest(self, copy):
        parts = [*self.actor.export(copy).values(), *self.critic.export(copy).values(), self.log_std[copy].detach().cpu()]
        return hashlib.sha256(b''.join(p.numpy().tobytes() for p in parts)).hexdigest()


class Group:
    """采集循环里一个会话的一组（Frozen 或 Updated）。Frozen 组没有网络，tuner 和 copy 都是 None。"""

    def __init__(self, tuner, copy):
        self.tuner, self.copy, self.batch = tuner, copy, []

    updates = property(lambda self: 0 if self.tuner is None else int(self.tuner.updates[self.copy]))

    def digest(self): return 'frozen' if self.tuner is None else self.tuner.digest(self.copy)
    def snapshot(self): return dict(mode='frozen') if self.tuner is None else self.tuner.snapshot(self.copy)

    def load_evaluation_mean(self, saved):
        if self.tuner is None: assert saved['mode'] == 'frozen'
        else: self.tuner.load(self.copy, saved)


class Slot:
    """采集循环按环境调用的接口。微调不用码，残差请求由采集循环直接从微调后的动作算出。"""

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
