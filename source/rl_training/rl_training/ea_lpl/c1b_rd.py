"""C1-b-RD：不用参数库。第一个回合按摆动受阻比例把两个摆动幅度学到位（与 C1-b-R 相同），
回合结束后用学到的幅度生成几种动作形式作为候选，之后按 C1-b-D 的方式持续选择。

候选（w = [前腿固定, 后腿固定, 前腿摆动, 后腿摆动]，sf、sr 是第一个回合学到的摆动幅度参数）：
  0  只有摆动：[0, 0, sf, sr]
  1  摆动加固定偏置（前腿缩、后腿伸）：[STATIC, -STATIC, sf, sr]
  2  固定偏置加减半的摆动：[STATIC, -STATIC, sf / 2, sr / 2]
  3  全 0
方案见 docs/lite3-online-hybrid-20261124/PLAN.md。
"""
from __future__ import annotations

import numpy as np
import torch

from .c1b_d import SelectSession
from .c1b_r import GAIN, SWING, TARGET
from .c1b_s import BOUND, PARAMETERS, REPLICAS
from .c1b_w import WindowSession

STATIC = .8


def forms(w):
    sf, sr = float(w[2]), float(w[3])
    return torch.tensor([[0., 0., sf, sr], [STATIC, -STATIC, sf, sr], [STATIC, -STATIC, sf / 2, sr / 2], [0., 0., 0., 0.]], dtype=torch.float64)


class StallSelectSession(SelectSession):
    def __init__(self, updated, noise, *, seed, **kwargs):
        self.warm = True
        super().__init__(updated, noise, torch.zeros(kwargs.get('replicas', REPLICAS), len(PARAMETERS), dtype=torch.float64), seed=seed, **kwargs)
        self.warm_updates = 0

    def _open(self):
        super()._open()
        if self.warm: self.codes = self.w.repeat(self.replicas, 1)

    def tick(self, learn=True):
        if not self.warm: return super().tick(learn)
        acc, episode = self.acc, self.episode
        entry = WindowSession.tick(self, False)
        if entry is None: return None
        entry['warm'] = True
        if learn and entry['used']:
            swing = np.sum([acc[k]['swing'] for k in entry['used']], 0); stalled = np.sum([acc[k]['stalled'] for k in entry['used']], 0)
            if (swing > 0).any():
                ratio = np.where(swing > 0, stalled / np.maximum(swing, 1), TARGET)
                entry.update(stall=ratio.tolist())
                if self.updated:
                    for g, k in enumerate(SWING):
                        self.w[k] = float(np.clip(self.w[k] + GAIN * (ratio[g] - TARGET), 0., BOUND))
                    self.updates += 1; self.warm_updates += 1; entry['updated'] = True
        if not self.updated and self.digest() != self.initial_digest:
            raise ValueError('Frozen parameters changed')
        entry['after'] = self.w.tolist()
        if self.episode != episode:  # 第一个回合结束：生成候选，转入持续选择
            self.arms = list(forms(self.w))
            self.warm, self.outcomes, self.scores = False, {}, [[] for _ in range(self.replicas)]
            self.assignment = self._assign()
        self._open()
        return entry

    @property
    def selection(self):
        chosen = SelectSession.selection.fget(self)
        return chosen and dict(chosen, warm_updates=self.warm_updates)
