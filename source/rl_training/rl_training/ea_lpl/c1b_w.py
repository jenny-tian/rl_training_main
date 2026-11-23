"""C1-b-W：与 C1-b-S 相同的 4 个幅度，扰动和更新放到回合内部，每 100 个策略步（2 秒）一个窗口。

一个会话的并行环境同时开始回合、共用窗口边界。每个窗口每个环境抽一次参数扰动；窗口结束时用各环境的
窗口反馈做留一法优势并更新。前腿参数的反馈是前腿摆动未受阻比例加前进量，后腿参数用后腿的。
方案见 docs/lite3-online-window-20261121/PLAN.md。
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch

from .c1b_s import BOUND, LEARNING_RATE, PARAMETERS, REPLICAS, SIGMA

WINDOW, SETTLE, MINIMUM_STEPS, ROUGH_MINIMUM = 100, 10, 25, .5
STALL_SPEED, STEP_DISTANCE, POSTURE_PENALTY = .10, .008, 3.
GROUP = torch.tensor([0, 1, 0, 1])  # 每个参数用哪一组腿的反馈：0 前腿，1 后腿
MAX_WINDOWS = 160


def exploration_table(pairs=60, max_windows=MAX_WINDOWS, replicas=REPLICAS, seed=2026112101):
    return np.random.default_rng(seed).standard_normal((pairs, max_windows, replicas, len(PARAMETERS)))


def window_feedback(swing, stalled, steps, distance, posture):
    """swing、stalled：(2,) 前后腿的摆动步数与受阻步数；steps：窗口内存活的步数；distance：窗口内前进距离。返回 (2,)。"""
    swing, stalled = np.asarray(swing, float), np.asarray(stalled, float)
    if steps <= 0 or (stalled > swing).any() or (swing < 0).any():
        raise ValueError('invalid window counts')
    unimpeded = np.where(swing > 0, 1 - stalled / np.maximum(swing, 1), 0.)
    progress = float(np.clip(distance / (STEP_DISTANCE * steps), -1., 1.))
    return unimpeded + progress - POSTURE_PENALTY * bool(posture)


def _digest(w):
    return hashlib.sha256(w.numpy().tobytes()).hexdigest()


class WindowSession:
    def __init__(self, updated, noise, *, replicas=REPLICAS, sigma=SIGMA, learning_rate=LEARNING_RATE, active=None, initial=None):
        self.updated, self.replicas, self.sigma, self.learning_rate = bool(updated), replicas, sigma, learning_rate
        self.noise = torch.as_tensor(np.asarray(noise), dtype=torch.float64)
        if self.noise.ndim != 3 or self.noise.shape[1:] != (replicas, len(PARAMETERS)) or not torch.isfinite(self.noise).all():
            raise ValueError('noise must be (windows, replicas, 4)')
        if active is not None and (not active or set(active) - set(PARAMETERS)):
            raise ValueError('unknown or empty active parameters')
        self.mask = torch.tensor([float(active is None or name in active) for name in PARAMETERS], dtype=torch.float64)
        self.w = torch.zeros(len(PARAMETERS), dtype=torch.float64) if initial is None else torch.as_tensor(initial, dtype=torch.float64).detach().clone()
        if self.w.shape != (len(PARAMETERS),) or not torch.isfinite(self.w).all() or (self.w.abs() > BOUND).any():
            raise ValueError('invalid initial parameters')
        self.initial_digest = self.digest()
        self.updates = self.windows = self.clock = self.episode = 0
        self.running, self.started, self.log = set(), set(), []
        self.batch = []  # 与其他学习器的接口一致；窗口更新没有跨回合的待处理数据
        self._open()

    def digest(self):
        return _digest(self.w)

    def _open(self):
        if self.windows >= len(self.noise):
            raise ValueError('noise table exhausted')
        self.codes = self.w + self.sigma * self.noise[self.windows] * self.mask
        self.acc = [dict(steps=0, swing=np.zeros(2), stalled=np.zeros(2), rough=0., distance=0., alive=0, posture=False) for _ in range(self.replicas)]

    def begin(self, replica, noise=None):
        if noise is not None:
            raise ValueError('the window learner draws its own noise')
        if not 0 <= replica < self.replicas or replica in self.started:
            raise ValueError('episode still active')
        if self.clock:
            raise ValueError('replicas of one session must start together')
        self.started.add(replica); self.running.add(replica)
        return self.requested_code(replica)

    def requested_code(self, replica):
        if replica not in self.running:
            raise ValueError('no active episode')
        return self.codes[replica].numpy().copy()

    def observe(self, replica, stance, foot_velocity_x, rough, distance):
        """策略步之后调用一次。stance：这一步开始时 4 条腿是否着地；foot_velocity_x：这一步里足端的前向速度，
        回合第一步没有上一帧时传 None；distance：这一步机身前进的距离。"""
        if replica not in self.running:
            raise ValueError('no active episode')
        a = self.acc[replica]
        a['alive'] += 1; a['distance'] += float(distance)
        if self.clock % WINDOW < SETTLE or foot_velocity_x is None: return
        swing = ~np.asarray(stance, bool); stalled = swing & (np.asarray(foot_velocity_x, float) < STALL_SPEED)
        a['steps'] += 1; a['rough'] += float(rough)
        a['swing'] += swing.reshape(2, 2).sum(1); a['stalled'] += stalled.reshape(2, 2).sum(1)

    def finish(self, replica, value, *, learn=True, posture=False):
        code = self.requested_code(replica)
        if not math.isfinite(float(value)):
            raise ValueError('nonfinite return')
        self.running.discard(replica)
        self.acc[replica]['posture'] = bool(posture)
        return dict(episode_return=float(value), code=code.tolist(), mean=self.w.tolist(), version=self.updates, updated=False,
                    updates=self.updates, pending_complete_episodes=0, parameter_sha256=self.digest(), b_after=self.w.tolist(),
                    window=self.windows)

    def tick(self, learn=True):
        """每个控制步、在本会话所有环境的 observe 和 finish 之后调用一次。窗口结束时返回这次窗口的记录。"""
        if not self.started: return None
        self.clock += 1
        done = len(self.started) == self.replicas and not self.running
        if self.clock % WINDOW and not done: return None
        if done and len(self.started) != self.replicas:
            raise ValueError('episode ended before every replica started')
        # 姿态终止的环境总是参与，否则刚换窗口就摔倒的扣分会丢掉。
        used = [k for k, a in enumerate(self.acc) if (a['posture'] and a['alive']) or (a['steps'] >= MINIMUM_STEPS and a['rough'] / a['steps'] >= ROUGH_MINIMUM)]
        entry = dict(window=self.windows, episode=self.episode, clock=self.clock, used=used, mean=self.w.tolist(), updated=False,
                     alive=[a['alive'] for a in self.acc], posture=[a['posture'] for a in self.acc])
        if used:
            feedback = torch.tensor(np.stack([window_feedback(a['swing'], a['stalled'], a['alive'], a['distance'], a['posture'])
                                              for a in (self.acc[k] for k in used)]), dtype=torch.float64)  # (环境, 2)
            entry.update(feedback=feedback.tolist(), noise=(self.noise[self.windows, used] * self.mask).tolist())
        if learn and len(used) >= 2:
            advantages = feedback - (feedback.sum(0) - feedback) / (len(used) - 1)
            step = (advantages[:, GROUP] * self.noise[self.windows, used] * self.mask).mean(0)
            if not torch.isfinite(step).all():
                raise ValueError('nonfinite update')
            entry.update(step=step.tolist())
            if self.updated:
                self.w = (self.w + self.learning_rate * self.sigma * step).clamp(-BOUND, BOUND)
                self.updates += 1; entry['updated'] = True
        if not self.updated and self.digest() != self.initial_digest:
            raise ValueError('Frozen parameters changed')
        entry['after'] = self.w.tolist()
        self.log.append(entry)
        self.windows += 1
        if done:
            self.clock, self.started = 0, set(); self.episode += 1
        self._open()
        return entry

    def snapshot(self):
        return copy.deepcopy(dict(w=self.w, updates=self.updates, windows=self.windows, updated=self.updated, parameter_sha256=self.digest()))


class Slot:
    def __init__(self, group, replica):
        self.group, self.replica = group, replica

    updates = property(lambda self: self.group.updates)
    batch = property(lambda self: self.group.batch)

    def digest(self): return self.group.digest()
    def begin(self, noise=None): return self.group.begin(self.replica, noise)
    def requested_code(self): return self.group.requested_code(self.replica)
    def observe(self, *args): return self.group.observe(self.replica, *args)
    def finish(self, value, *, learn=True, posture=False, **extra): return self.group.finish(self.replica, value, learn=learn, posture=posture, **extra)
    def snapshot(self): return self.group.snapshot()
