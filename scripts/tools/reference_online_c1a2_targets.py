"""在线入口里各目标地形的参数和射线核对中不依赖仿真的部分。

上阶、箱体的高度表仍在 reference_online_c1a2_env.py 里；这里是 S1 之后加进来的目标：坡面（坡度和起伏幅度固定）和下行台阶。
地形参数与 S0 筛选入口 reference_frozen_rebaseline.terrain_spec 相同，有测试核对。方案见 docs/lite3-online-adapter2-20261126/PLAN.md。
"""
from __future__ import annotations

import numpy as np

# 目标名 -> 训练地图里的子地形（source）和要固定的参数。名字沿用历史：除了粗糙坡面，现在也包括光滑上坡和粗糙地面。
# 粗糙上坡 x11 是 S1 的目标；粗糙下坡 x10 取训练范围的上限（坡度 0.40）。
# 粗糙地形的起伏是随机生成的，同一组参数在不同的地图里是不同的地块：在线入口的地图只有一个地块，坡度 0.34 在这个地块上
# 冻结策略已有 57/64，所以目标改用 0.40（12/64）。见 docs/lite3-online-adapter2-20261126/PLAN.md 第 11 节。
SLOPE_TARGETS = dict(
    slope_rough_up_x11=dict(source='slope_rough_up', slope_range=(.44, .44), noise_range=(.01, .07)),
    slope_rough_down_x10=dict(source='slope_rough_down', slope_range=(.40, .40), noise_range=(.01, .06)),
    # S2 的候选（docs/lite3-online-s2-20261127/PLAN.md）
    slope_rough_up_x12=dict(source='slope_rough_up', slope_range=(.48, .48), noise_range=(.01, .07)),
    slope_rough_down_x11=dict(source='slope_rough_down', slope_range=(.44, .44), noise_range=(.01, .07)),
    slope_rough_down_x12=dict(source='slope_rough_down', slope_range=(.48, .48), noise_range=(.01, .07)),
    slope_up_55=dict(source='slope_up', slope_range=(.55, .55)), slope_up_57=dict(source='slope_up', slope_range=(.57, .57)),
    slope_up_60=dict(source='slope_up', slope_range=(.60, .60)),
    rough_09=dict(source='rough', noise_range=(.01, .09)), rough_10=dict(source='rough', noise_range=(.01, .10)), rough_11=dict(source='rough', noise_range=(.01, .11)))
# 训练地图里这些参数的范围；改之前核对，保证候选是在筛选时的那张训练地图上改出来的。
TRAINING_RANGE = dict(slope_range=(0., .4), noise_range=(.01, .06))
# 下行台阶：训练范围是 2 到 5 cm。
DOWN_STAIR_HEIGHT = dict(stairs_down_085=.085, stairs_down_09=.09, stairs_down_10=.10)
MINIMUM_RISERS = 5


def stairs_target(name):
    """按台阶处理的目标：终点在最后一个立面之后，成功条件看脚是否都过了最后一个立面。"""
    return name.startswith('stairs')


def stair_profile(x, z, height, tolerance=.002):
    """由中心线的射线采样得到立面位置和每个立面之后的高度。height 是每级的高度，负数表示下行。

    x 从地块中心沿前进方向递增，z 是相对地块原点的高度。立面数少于 5 个，或者任何一级的高度与 height 相差超过 tolerance 时报错；
    方向相反的地形（上行的按下行核对）立面数为 0，同样报错。
    """
    x, z = np.asarray(x, float), np.asarray(z, float)
    if x.shape != z.shape or x.ndim != 1 or not np.isfinite(z).all() or height == 0:
        raise ValueError('invalid stair samples')
    sign = 1. if height > 0 else -1.
    edges = np.flatnonzero(np.diff(z) * sign > abs(height) / 2) + 1
    fronts, tops = ((x[edges] + x[edges - 1]) / 2).tolist(), z[edges].tolist()
    if len(fronts) < MINIMUM_RISERS:
        raise ValueError(f'{len(fronts)} risers of {height:+.3f} m found, at least {MINIMUM_RISERS} expected')
    steps = np.diff([float(z[0])] + tops)
    if not np.allclose(steps, height, atol=tolerance):
        raise ValueError(f'riser heights {np.round(steps, 4).tolist()} do not match {height:+.3f} m')
    return fronts, tops


def target_sub_terrains(target, captured):
    """把目标变成地图里唯一的子地形。captured 是训练地图各子地形的配置（会被就地修改）；不归这里处理的目标返回 None。"""
    if target in SLOPE_TARGETS:
        spec = SLOPE_TARGETS[target]
        sub = captured[spec['source']]
        for field, value in spec.items():
            if field == 'source': continue
            if tuple(getattr(sub, field)) != TRAINING_RANGE[field]:
                raise ValueError(f'{spec["source"]} no longer has the training range the targets were screened against')
            setattr(sub, field, value)
    elif target in DOWN_STAIR_HEIGHT:
        sub = captured['stairs_down']
        sub.step_height_range = (DOWN_STAIR_HEIGHT[target],) * 2
    else:
        return None
    sub.proportion = 1.
    return {target: sub}
