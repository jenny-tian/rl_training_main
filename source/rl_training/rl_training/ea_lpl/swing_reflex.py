"""单次触发的抬腿反射，只用于对照评估，不学习。

一条腿离地后的前 EARLY 步内足端前向速度低于 STALL_SPEED，就触发：这条腿的摆动残差请求取满幅。
  same  保持到这次摆动结束
  next  保持到这次摆动结束，并且这条腿的下一次摆动全程使用
请求的形式与 c1b_s.residual_request 的摆动部分相同（sag 方向，乘起伏特征）。方案见 docs/lite3-online-baselines-20261125/PLAN.md。
"""
from __future__ import annotations

import torch

from .c1b_s import HIPY, KNEE
from .c1b_w import STALL_SPEED

RULES = ('same', 'next')
EARLY, DT = 2, .02


class SwingReflex:
    def __init__(self, rule, environments, device='cpu'):
        if rule not in RULES:
            raise ValueError('unknown reflex rule')
        self.rule = rule
        self.age = torch.zeros(environments, 4, dtype=torch.long, device=device)  # 已连续摆动的步数
        self.active = torch.zeros(environments, 4, dtype=torch.bool, device=device)  # 这次摆动是否带反射
        self.pending = torch.zeros_like(self.active)  # 下一次摆动是否带反射
        self.swing = torch.zeros_like(self.active)
        self.feet = None
        self.triggers = torch.zeros(environments, dtype=torch.long, device=device)

    def reset(self, ids):
        self.age[ids] = 0; self.active[ids] = False; self.pending[ids] = False; self.swing[ids] = False
        if self.feet is not None: self.feet[ids] = float('nan')

    def step(self, stance, feet_x, rough):
        """每个控制步在算动作之前调用一次。stance (n, 4) 布尔；feet_x (n, 4) 足端前向位置；rough (n,)。返回 (n, 12) 残差请求。"""
        if stance.dtype != torch.bool or stance.shape != self.age.shape or feet_x.shape != stance.shape or rough.shape != stance.shape[:1]:
            raise ValueError('stance (n, 4) bool, feet (n, 4) and rough (n,) required')
        feet_x = feet_x.to(self.age.device, torch.float32)
        speed = torch.full_like(feet_x, float('inf')) if self.feet is None else torch.nan_to_num((feet_x - self.feet) / DT, nan=float('inf'))
        # 上一步里这条腿在摆动、属于离地后的前 EARLY 步、足端几乎没有前进
        trigger = self.swing & (self.age < EARLY) & (speed < STALL_SPEED)
        self.triggers += trigger.sum(1)
        swing = ~stance
        started = swing & ~self.swing
        self.active = torch.where(started, self.pending, self.active)  # 新的一次摆动：用掉留给它的反射
        self.pending = self.pending & ~started
        self.active = (self.active | trigger) & swing
        if self.rule == 'next': self.pending = self.pending | trigger
        self.age = torch.where(swing & self.swing, self.age + 1, torch.zeros_like(self.age))
        self.swing, self.feet = swing.clone(), feet_x.clone()
        amount = self.active.float() * rough.to(self.age.device, torch.float32)[:, None]
        request = torch.zeros(len(stance), 12, device=self.age.device)
        request[:, 1::3], request[:, 2::3] = HIPY * amount, KNEE * amount
        return request
