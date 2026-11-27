"""上阶最后一级的脚本化检查：手工设定的动作，只做评估，不学习。方案见 docs/lite3-online-s2-20261127/PLAN.md 第 5 节。

离线检查（S2 方案第 1 节）：机器人停在最后一级时前腿已在顶上，后腿面对最后一个立面的摆动只有 7% 跨过去；这时安全过滤常因
机身离地高度低于下限而暂停，残差被清零。这里的动作是着地的腿伸腿（抬腿缩腿方向的反向：髋 +、膝 −），乘地形高度特征里的
顶部特征（前方平、后方低），所以只在楼梯顶部起作用；摆动期的抬腿不变，由受阻修正给出。

h15 两组不改动作形式，只把仿真里安全过滤的机身高度下限从 0.20 m 降到 0.15 m，用来看暂停本身占多大作用，只是诊断。
"""
from __future__ import annotations

import torch

from .c1b_s import HIPY, KNEE

MIN_BASE_HEIGHT, LOWERED_BASE_HEIGHT = .20, .15
# 名字 -> (伸腿的幅度（相对满幅），哪些腿：FL FR HL HR，是否降低机身高度下限)
PROBES = dict(
    none=(0., (0, 0, 0, 0), False),
    hind_half=(.5, (0, 0, 1, 1), False),
    hind_full=(1., (0, 0, 1, 1), False),
    all_half=(.5, (1, 1, 1, 1), False),
    h15=(0., (0, 0, 0, 0), True),
    h15_hind_half=(.5, (0, 0, 1, 1), True))


def _probe(name):
    if name not in PROBES:
        raise ValueError(f'unknown top probe: {name}')
    return PROBES[name]


def min_base_height(name):
    """这一组在仿真里使用的安全过滤机身高度下限。"""
    return LOWERED_BASE_HEIGHT if _probe(name)[2] else MIN_BASE_HEIGHT


def request(name, stance, top):
    """stance (n, 4) 布尔，四条腿是否着地；top (n,) 顶部特征，0 到 1。返回残差请求 (n, 12)。"""
    amplitude, legs, _ = _probe(name)
    stance = torch.as_tensor(stance)
    top = torch.as_tensor(top, dtype=torch.float32, device=stance.device)
    if stance.dtype != torch.bool or stance.ndim != 2 or stance.shape[1] != 4 or top.shape != (len(stance),):
        raise ValueError('stance (n, 4) bool and top (n,) required')
    if (top < 0).any() or (top > 1).any():
        raise ValueError('top feature must be in [0, 1]')
    amount = amplitude * torch.tensor(legs, dtype=torch.float32, device=stance.device) * stance.float() * top[:, None]
    result = torch.zeros(len(stance), 12, device=stance.device)
    result[:, 1::3], result[:, 2::3] = -HIPY * amount, -KNEE * amount
    return result
