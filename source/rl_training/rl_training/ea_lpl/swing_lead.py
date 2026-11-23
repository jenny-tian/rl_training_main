"""提前起抬：在腿离地之前就开始请求摆动残差，让安全过滤的每步变化上限来得及把残差升上去。只用于脚本化检查。

规则决定一条着地的腿什么时候算作即将离地；被标记的腿在算残差时按摆动处理（c1b_s.residual_request 的 stance 置为 False）。
  none     不提前（现有做法）
  knee     基础策略这条腿的膝关节动作比上一步增加超过 KNEE_RISE
  partner  另一对角的两条腿刚同时着地不超过 PARTNER_STEPS 步，且这条腿已着地至少 MIN_STANCE 步
  age      这条腿已着地至少 AGE_STEPS 步
方案见 docs/lite3-online-lead-20261123/PLAN.md。
"""
from __future__ import annotations

import torch

RULES = ('none', 'knee', 'partner', 'age')
KNEE_RISE, PARTNER_STEPS, MIN_STANCE, AGE_STEPS = .15, 3, 3, 8
DIAGONAL = torch.tensor([[1, 2], [0, 3], [0, 3], [1, 2]])  # FL FR HL HR 各自的另一对角


class SwingLead:
    def __init__(self, rule, environments, device='cpu'):
        if rule not in RULES:
            raise ValueError('unknown lead rule')
        self.rule = rule
        self.age = torch.zeros(environments, 4, dtype=torch.long, device=device)  # 每条腿已连续着地的步数
        self.pair = torch.zeros(environments, 4, dtype=torch.long, device=device)  # 另一对角已同时着地的步数
        self.knee = None

    def reset(self, ids):
        self.age[ids] = 0; self.pair[ids] = 0
        if self.knee is not None: self.knee[ids] = float('nan')

    def step(self, stance, base_action):
        """stance (n, 4) 布尔；base_action (n, 12) 裁剪后的基础动作。每个控制步调用一次，返回算残差用的 stance。"""
        if stance.dtype != torch.bool or stance.shape != self.age.shape or base_action.shape != (len(stance), 12):
            raise ValueError('stance (n, 4) bool and base action (n, 12) required')
        self.age = torch.where(stance, self.age + 1, torch.zeros_like(self.age))
        both = stance[:, DIAGONAL.to(stance.device)].all(2)
        self.pair = torch.where(both, self.pair + 1, torch.zeros_like(self.pair))
        knee = base_action[:, 2::3].detach().clone()
        rise = torch.zeros_like(knee) if self.knee is None else torch.nan_to_num(knee - self.knee, nan=0.)
        self.knee = knee
        if self.rule == 'none': lead = torch.zeros_like(stance)
        elif self.rule == 'knee': lead = rise > KNEE_RISE
        elif self.rule == 'partner': lead = (self.pair >= 1) & (self.pair <= PARTNER_STEPS) & (self.age >= MIN_STANCE)
        else: lead = self.age >= AGE_STEPS
        return stance & ~lead
