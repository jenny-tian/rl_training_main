"""C1-b-L：先从参数库里选一组，再按 C1-b-W 继续修正。

前 SELECT_ROUNDS 个回合不加扰动，每个并行环境用库里的一组参数，每个回合轮换一位；一组参数的得分是
它所有窗口反馈（前后腿平均）的平均，得分最高的作为起点。方案见 docs/lite3-online-library-20261122/PLAN.md。
"""
from __future__ import annotations

import numpy as np
import torch

from .c1b_s import BOUND, PARAMETERS
from .c1b_w import WindowSession

SELECT_ROUNDS = 2


class LibrarySession(WindowSession):
    def __init__(self, updated, noise, library, **kwargs):
        self.library = torch.as_tensor(np.asarray(library), dtype=torch.float64).detach().clone()
        if self.library.ndim != 2 or self.library.shape[1] != len(PARAMETERS) or not torch.isfinite(self.library).all() or (self.library.abs() > BOUND).any():
            raise ValueError('library must be (entries, 4) within bounds')
        self.round, self.scores, self.selection = 0, [[] for _ in self.library], None
        super().__init__(updated, noise, **kwargs)
        if len(self.library) != self.replicas:
            raise ValueError('one library entry per replica required')

    selecting = property(lambda self: self.round < SELECT_ROUNDS)

    def entry(self, replica):
        return (replica + self.round) % len(self.library)

    def _open(self):
        super()._open()
        if self.selecting:
            self.codes = torch.stack([self.library[self.entry(k)] for k in range(self.replicas)])

    def tick(self, learn=True):
        if not self.selecting: return super().tick(learn)
        episode = self.episode
        closed_round = self.round
        entry = super().tick(False)
        if entry is None: return None
        for row, k in zip(entry.get('feedback', []), entry['used']):
            self.scores[(k + closed_round) % len(self.library)].append(float(np.mean(row)))
        entry['selecting'] = True
        if self.episode != episode:
            self.round += 1
            if not self.selecting:
                means = [float(np.mean(s)) if s else float('-inf') for s in self.scores]
                best = int(np.argmax(means))
                self.selection = dict(entry=best, scores=means, windows=[len(s) for s in self.scores], applied=self.updated)
                if self.updated: self.w = self.library[best].clone()
            self._open()  # 换成下一轮的参数；选择结束后是新起点加扰动
        return entry

    def snapshot(self):
        return dict(super().snapshot(), selection=self.selection)
