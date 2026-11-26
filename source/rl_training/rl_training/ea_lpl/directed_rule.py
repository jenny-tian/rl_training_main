"""不学习的对照：把有方向误差的目标残差直接作为下一步的残差请求。方案见 docs/lite3-online-adapter2-20261126/PLAN.md 第 12 节。

小模型（adapter.py 的更新方式 b、s）回归的目标是 directed_target：0.2 × 去掉慢变部分的关节跟踪误差，加受阻摆动腿上的抬腿方向。
这里不经过网络：走完第 t 步后算出的目标残差，原样作为第 t+1 步的请求。它比小模型晚一步，也不随状态泛化；
用来回答小模型的增益里有多少是这条反馈规则本身带来的。慢滤波的做法与 adapter.Learner.target 相同，有测试核对两者逐步一致。
"""
from __future__ import annotations

import torch

from .adapter import SLOW, directed_target


class DirectedRule:
    def __init__(self, environments, limits=None, device='cpu'):
        self.n, self.limits = environments, limits
        self.slow = torch.zeros(environments, 12, device=device)
        self.fresh = torch.ones(environments, dtype=torch.bool, device=device)
        self.request = torch.zeros(environments, 12, device=device)

    def reset(self, rows):
        """一次尝试开始时调用：请求清零，跟踪误差的慢滤波重新开始。"""
        self.fresh[rows] = True
        self.request[rows] = 0

    def act(self, active):
        """这一步的残差请求 (n, 12)：上一步结束时算出的目标残差；不在策略阶段的环境是 0。"""
        return self.request * active[:, None]

    def observe(self, error, stalled, active):
        """环境走完一步之后调用。error (n, 12) 跟踪误差；stalled (n, 4) 这一步里受阻的摆动腿；active (n,) 是否在策略阶段。"""
        if error.shape != (self.n, 12) or stalled.shape != (self.n, 4) or stalled.dtype != torch.bool or active.shape != (self.n,) or active.dtype != torch.bool:
            raise ValueError('error (n, 12), stalled (n, 4) bool and active (n,) bool required')
        error = error.float()
        start = self.fresh & active
        self.slow = torch.where(start[:, None], error, torch.where(active[:, None], self.slow + SLOW * (error - self.slow), self.slow))
        self.fresh = self.fresh & ~active
        self.request = directed_target(error - self.slow, stalled, self.limits) * active[:, None]
