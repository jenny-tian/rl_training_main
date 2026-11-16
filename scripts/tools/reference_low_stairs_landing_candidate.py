"""独立落脚奖励取舍候选，仅用于离线重计分，未接入训练或部署。

复制历史 update 以保持历史源码和哈希不变；唯一数值差异是峰值裕量
15 mm 改为 5 mm，测试以 AST 比较锁定这个差异。持续净空目标仍为 25 mm。
"""
import torch

from reference_low_stairs_rewards import LandingState


class LandingPeak5mmState(LandingState):
    def update(self, *, feet, bottom, ground, support, root_xy, direction, enabled, upright, dt):
        first = support & ~self.initialized
        self.highest_credited_z = torch.where(first, ground, self.highest_credited_z)
        self.initialized |= first
        self.peak_bottom = torch.where(self.was_supported, bottom, torch.maximum(self.peak_bottom, bottom))
        airborne = ~support
        self.air_time = torch.where(airborne, self.air_time + dt, self.air_time)
        displacement = ((feet[..., :2] - self.support_xy) * self.direction).sum(-1)
        body_progress = ((root_xy[:, None] - self.support_root_xy) * self.direction).sum(-1)
        touchdown = support & ~self.was_supported & (self.air_time >= .04 - 1e-6) & ~first
        candidate = (touchdown & self.initialized & enabled[:, None] & upright[:, None]
                  & (support.sum(-1) >= 2)[:, None]
                  & (ground > self.support_z + .008)
                  & (ground > self.highest_credited_z + .008)
                  & (displacement >= .04) & (body_progress >= .01)
                  & (self.peak_bottom >= ground + .005)
                  & ((bottom - ground).abs() <= .025))
        self.pending_z = torch.where(candidate, ground, self.pending_z)
        self.pending = ((self.pending | candidate) & support & enabled[:, None] & upright[:, None]
                        & (support.sum(-1) >= 2)[:, None] & ((ground-self.pending_z).abs()<.005))
        self.dwell = torch.where(self.pending, self.dwell + dt, 0.)
        landed = self.pending & (self.dwell >= .06 - 1e-6)
        self.pending &= ~landed
        self.dwell = torch.where(landed, 0., self.dwell)
        self.highest_credited_z = torch.where(landed, ground, self.highest_credited_z)
        self.support_z = torch.where(support, ground, self.support_z)
        self.support_xy = torch.where(support[..., None], feet[..., :2], self.support_xy)
        self.support_root_xy = torch.where(support[..., None], root_xy[:, None], self.support_root_xy)
        self.direction = torch.where(support[..., None], direction[:, None], self.direction)
        self.air_time = torch.where(support, 0., self.air_time)
        self.was_supported = support.clone()
        return landed.float().mean(-1) / dt
