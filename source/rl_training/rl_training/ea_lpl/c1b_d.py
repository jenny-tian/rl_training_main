"""C1-b-D：整个会话在一组候选参数里持续选择，不加扰动，也不做梯度更新。

候选开始时是参数库（最后一组是全 0）。前 ROTATE_ROUNDS 个回合每个并行环境用一组并轮换，与 C1-b-L 的选择阶段相同；
之后每个回合开始时，每个环境按各组的 Beta(1 + 成功次数, 1 + 未成功次数) 各抽一个数，用抽到最大的一组。
一次尝试的得分是成功记 1，加 SCORE_WINDOW 倍的该次尝试窗口平均反馈（只用来分出并列）；平均得分最高的一组是当前参数。
expand=True 时，第 EXPAND_AFTER 个回合结束后加入由领先和第二的两组生成的候选。方案见 docs/lite3-online-select-20261123/PLAN.md。
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .c1b_s import BOUND, PARAMETERS, REPLICAS
from .c1b_w import WindowSession

ROTATE_ROUNDS, EXPAND_AFTER, SCORE_WINDOW = 2, 4, .01
SWING_SCALES = (1.2, .8)
LIMIT = math.tanh(BOUND)


def candidates(leader, second):
    """leader、second：w 空间的两组参数。返回 tanh 空间里两组的中点，以及领先一组的两个摆动幅度乘 SWING_SCALES，都换回 w 空间。"""
    a, b = torch.tanh(leader), torch.tanh(second)
    swing = torch.tensor([0., 0., 1., 1.], dtype=torch.float64)
    rows = [(a + b) / 2] + [a * (1 + (scale - 1) * swing) for scale in SWING_SCALES]
    return [torch.atanh(row.clamp(-LIMIT, LIMIT)).clamp(-BOUND, BOUND) for row in rows]


class SelectSession(WindowSession):
    def __init__(self, updated, noise, library, *, seed, expand=False, **kwargs):
        library = torch.as_tensor(np.asarray(library), dtype=torch.float64).detach().clone()
        if library.ndim != 2 or library.shape[1] != len(PARAMETERS) or not torch.isfinite(library).all() or (library.abs() > BOUND).any():
            raise ValueError('library must be (entries, 4) within bounds')
        if len(library) != kwargs.get('replicas', REPLICAS):
            raise ValueError('one library entry per replica required')
        self.arms, self.expand, self.rng = list(library), bool(expand), np.random.default_rng(seed)
        self.tries, self.wins, self.total = [0] * len(library), [0] * len(library), [0.] * len(library)
        self.round, self.leader, self.history, self.added = 0, len(library) - 1, [], []
        self.assignment = [k % len(library) for k in range(len(library))]
        self.scores, self.outcomes = [[] for _ in library], {}
        super().__init__(updated, noise, **kwargs)

    def means(self):
        return [t / n if n else None for t, n in zip(self.total, self.tries)]

    def _assign(self):
        if self.round < ROTATE_ROUNDS:
            return [(k + self.round) % self.replicas for k in range(self.replicas)]
        wins, tries = np.asarray(self.wins, float), np.asarray(self.tries, float)
        return [int(np.argmax(self.rng.beta(1 + wins, 1 + tries - wins))) for _ in range(self.replicas)]

    def _open(self):
        super()._open()
        self.codes = torch.stack([self.arms[a] for a in self.assignment])

    def finish(self, replica, value, *, learn=True, posture=False, success=False):
        entry = super().finish(replica, value, learn=learn, posture=posture)
        self.outcomes[replica] = bool(success)
        return dict(entry, candidate=self.assignment[replica], leader=self.leader)  # 记录里的 arm 已经用来表示 Frozen / Updated

    def tick(self, learn=True):
        episode, assignment = self.episode, list(self.assignment)
        entry = super().tick(False)
        if entry is None: return None
        for row, k in zip(entry.get('feedback', []), entry['used']):
            self.scores[k].append(float(np.mean(row)))
        entry['arms'] = assignment
        if self.episode != episode:
            if set(self.outcomes) != set(range(self.replicas)):
                raise ValueError('episode closed without every outcome')
            for k, arm in enumerate(assignment):
                self.tries[arm] += 1; self.wins[arm] += self.outcomes[k]
                self.total[arm] += self.outcomes[k] + SCORE_WINDOW * (float(np.mean(self.scores[k])) if self.scores[k] else 0.)
            self.round += 1
            means = self.means()
            # 得分完全相同时先取全 0 的一组（库的最后一组），与 C1-b-L 没有可用得分时的做法一致。
            order = sorted((k for k, m in enumerate(means) if m is not None), key=lambda k: (-means[k], k != self.replicas - 1))
            self.leader = order[0]
            if self.expand and self.round == EXPAND_AFTER and len(order) > 1:
                for row in candidates(self.arms[order[0]], self.arms[order[1]]):
                    if all((row - arm).abs().max() > 1e-6 for arm in self.arms):
                        self.added.append(len(self.arms)); self.arms.append(row)
                        self.tries.append(0); self.wins.append(0); self.total.append(0.)
            if self.updated:
                self.w = self.arms[self.leader].clone(); self.updates += 1
            self.history.append(dict(round=self.round, arms=assignment, success=[self.outcomes[k] for k in range(self.replicas)],
                                     windows=[len(s) for s in self.scores], leader=self.leader, tries=list(self.tries), wins=list(self.wins)))
            self.scores, self.outcomes = [[] for _ in range(self.replicas)], {}
            self.assignment = self._assign()
            self._open()  # 换成下一个回合的参数
        return entry

    @property
    def selection(self):
        if not self.round: return None
        return dict(entry=self.leader, scores=self.means(), tries=list(self.tries), wins=list(self.wins), rounds=self.round, added=list(self.added),
                    arms=[arm.tolist() for arm in self.arms], history=self.history, applied=self.updated)

    def snapshot(self):
        chosen = self.selection
        return dict(super().snapshot(), selection=chosen and {k: v for k, v in chosen.items() if k != 'history'})
