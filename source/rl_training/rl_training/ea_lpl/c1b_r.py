"""C1-b-R：用摆动受阻比例直接修正两个摆动幅度，不加扰动。

每个 2 秒窗口结束时，把本会话各并行环境的摆动步数和受阻步数合在一起，分前腿、后腿算受阻比例；
这组腿的摆动幅度参数加上 GAIN * (受阻比例 - TARGET)，限制在 [0, BOUND]。窗口是否可用的条件与 C1-b-W 相同。
两个固定偏置不学，恒为 0。方案见 docs/lite3-online-stall-20261123/PLAN.md。
"""
from __future__ import annotations

import numpy as np
import torch

from .c1b_s import BOUND
from .c1b_w import WindowSession

GAIN, TARGET = 1.5, .30
SWING = (2, 3)  # 前腿、后腿摆动幅度在参数里的位置


class StallSession(WindowSession):
    def __init__(self, updated, noise, **kwargs):
        super().__init__(updated, noise, active=['front_swing', 'rear_swing'], **kwargs)

    def _open(self):
        super()._open()
        self.codes = self.w.repeat(self.replicas, 1)  # 不加扰动

    def tick(self, learn=True):
        acc = self.acc
        entry = super().tick(False)
        if entry is None: return None
        swing = np.sum([acc[k]['swing'] for k in entry['used']], 0) if entry['used'] else np.zeros(2)
        stalled = np.sum([acc[k]['stalled'] for k in entry['used']], 0) if entry['used'] else np.zeros(2)
        entry.update(swing=np.asarray(swing).tolist(), stalled=np.asarray(stalled).tolist())
        if learn and entry['used'] and (swing > 0).any():
            ratio = np.where(swing > 0, stalled / np.maximum(swing, 1), TARGET)
            entry.update(stall=ratio.tolist(), step=(GAIN * (ratio - TARGET)).tolist())
            if self.updated:
                for g, k in enumerate(SWING):
                    self.w[k] = float(np.clip(self.w[k] + GAIN * (ratio[g] - TARGET), 0., BOUND))
                self.updates += 1; entry['updated'] = True
                self.codes = self.w.repeat(self.replicas, 1)
        if not self.updated and self.digest() != self.initial_digest:
            raise ValueError('Frozen parameters changed')
        entry['after'] = self.w.tolist()
        return entry
