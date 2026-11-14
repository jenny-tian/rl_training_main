"""四足共用的台阶摆脚净空规则，只作为离线训练的特权奖励输入。

保留历史奖励源码。本模块不改变策略输入，也不提供训练入口。
"""
from __future__ import annotations

import torch

from reference_low_stairs_rewards import FOOT_RADIUS, MARGIN, sample

HEIGHT_SCALE = .05
LANDING_DWELL = .04
WALL_OFFSETS = (0., -FOOT_RADIUS / 2., -FOOT_RADIUS + .003)


def height_cost(bottom, target):
    """有界、连续的净空不足惩罚；正常高度范围内没有截断平台。"""
    deficit = (target + MARGIN - bottom).clamp_min(0.)
    squared = deficit.square()
    return squared / (HEIGHT_SCALE ** 2 + squared)


def wall_contact(force, direction, distance, normal):
    """几何与反向力联合判定，输出逐足接触代理，不声称恢复接触点。

    三条水平射线覆盖足球中心及下部。各高度使用对应的球截面半径，
    防止把较远的台阶与地面摩擦误判成碰阶。射线未命中允许为无穷大。
    """
    valid = torch.isfinite(distance) & torch.isfinite(normal).all(-1)
    normals = torch.where(valid[..., None], normal, 0.)
    offsets = distance.new_tensor(WALL_OFFSETS)
    reach = (FOOT_RADIUS ** 2 - offsets.square()).sqrt() + .008
    face = (valid & (distance >= 0.) & (distance <= reach)
            & (normals[..., 2].abs() < .2)
            & ((normals[..., :2] * direction[:, None, None]).sum(-1).abs() > .5))
    opposing_force = -(force[..., :2] * direction[:, None]).sum(-1)
    return face.any(-1) & (opposing_force > 5.)


def clearance_sample(env):
    cached = getattr(env, '_persistent_clearance_sample', None)
    if cached is not None and cached['tick'] == env.common_step_counter:
        return cached
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh
    x = sample(env)
    starts = x['feet'][:, :, None].expand(-1, -1, len(WALL_OFFSETS), -1).clone()
    starts[..., 2] += starts.new_tensor(WALL_OFFSETS)
    direction = torch.where(x['enabled'][:, None], x['direction'],
                            starts.new_tensor([1., 0.]))
    rays = torch.zeros_like(starts)
    rays[..., :2] = direction[:, None, None]
    _, distances, normals, _ = raycast_mesh(
        starts.reshape(env.num_envs, -1, 3), rays.reshape(env.num_envs, -1, 3),
        RayCaster.meshes[env.cfg.scene.terrain.prim_path], max_dist=.04,
        return_distance=True, return_normal=True)
    distances = distances.reshape(*starts.shape[:-1])
    normals = normals.reshape_as(starts)
    wall = wall_contact(x['force'], direction, distances, normals)
    cached = dict(tick=env.common_step_counter, wall=wall,
                  distance=distances, normal=normals)
    env._persistent_clearance_sample = cached
    return cached


class ClearanceState:
    """从摆脚开始保留目标，停脚或碰阶不会自行清除。

    合法支撑立即免罚，连续支撑 40 ms 后结束这一摆脚目标。短暂接触
    不会丢失目标；由几何和反向力识别的碰阶不算合法支撑。
    停止平移、离开台阶任务、姿态超限或命令反向会清除目标。
    """
    def __init__(self, count, device):
        shape = (count, 4)
        self.active = torch.zeros(shape, dtype=torch.bool, device=device)
        self.target = torch.zeros(shape, device=device)
        self.direction = torch.zeros((*shape, 2), device=device)
        self.dwell = torch.zeros(shape, device=device)
        self.cost = torch.zeros(shape, device=device)
        self.scored = torch.zeros_like(self.active)
        self.last_tick = torch.full((count,), -1, dtype=torch.long, device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for name in ('active', 'target', 'direction', 'dwell', 'cost', 'scored'):
            getattr(self, name)[ids] = 0
        self.last_tick[ids] = -1

    def update(self, *, bottom, obstacle, upcoming, support, wall, direction,
               enabled, upright, dt, tick):
        if dt <= 0:
            raise ValueError('控制周期必须大于零')
        fresh = self.last_tick.ne(tick)[:, None]
        valid = (enabled & upright)[:, None]
        aligned = (self.direction * direction[:, None]).sum(-1) > .5
        active = self.active & valid & aligned
        supported = support & ~wall
        start = valid & upcoming & ~supported & ~active
        active |= start
        target = torch.where(start, obstacle, self.target)
        chosen_direction = torch.where(start[..., None], direction[:, None], self.direction)
        dwell = torch.where(active & supported, self.dwell + dt, 0.)
        active &= dwell < LANDING_DWELL - 1e-6
        scored = active & ~supported
        cost = height_cost(bottom, target) * scored
        for name, value in (('active', active), ('target', target), ('dwell', dwell),
                            ('scored', scored), ('cost', cost)):
            setattr(self, name, torch.where(fresh, value, getattr(self, name)))
        self.direction = torch.where(fresh[..., None], chosen_direction, self.direction)
        self.last_tick[:] = tick
        return self.cost
