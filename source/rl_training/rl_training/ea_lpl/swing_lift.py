"""手工设定的摆动抬脚残差，只用于训练前的脚本化检查，不涉及学习。

腿在摆动（脚不受力）时，这条腿的 HipY、Knee 加上 sag 方向的残差（腿缩短），幅度是残差上限的 scale 倍；
着地时为零。整体乘上地形起伏特征，平地上为零。方案见 docs/lite3-online-swing-probe-20261118/PLAN.md。
"""
from __future__ import annotations

import torch

from .basis import LEGS

GROUPS = dict(R=dict(legs=('HL', 'HR'), front_static=0.), Q=dict(legs=LEGS, front_static=0.),
              S=dict(legs=('HL', 'HR'), front_static=1.))
HIPY, KNEE = -.10, .20  # 与 sag 动作基的一列相同
FRONT_STATIC = .905  # tanh(1.5)，即固定码 +1.5 的幅度；不随 scale 放大


def swing_request(stance, rough, groups, scale):
    """stance: (n, 4) 布尔，FL FR HL HR 是否着地；rough: (n,) 起伏特征；groups: 每个环境的组名或 None。"""
    stance = torch.as_tensor(stance)
    rough = torch.as_tensor(rough, device=stance.device, dtype=torch.float32)
    if stance.dtype != torch.bool or stance.shape != (len(groups), 4) or rough.shape != (len(groups),):
        raise ValueError('stance must be (n, 4) bool and rough (n,)')
    if not 0 < scale <= 4 or (rough < 0).any() or (rough > 1).any():
        raise ValueError('scale in (0, 4] and rough in [0, 1] required')
    request = torch.zeros(len(groups), 12, device=stance.device)
    for i, name in enumerate(groups):
        if name is None: continue
        group = GROUPS[name]
        for k, leg in enumerate(LEGS):
            amount = scale * float(leg in group['legs']) * float(not stance[i, k])
            if leg in ('FL', 'FR'): amount = amount + FRONT_STATIC * group['front_static']
            request[i, 3 * k + 1] = HIPY * amount * rough[i]
            request[i, 3 * k + 2] = KNEE * amount * rough[i]
    return request
