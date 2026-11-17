"""在已验证的原生重置环境上加航向闭环和箱体目标；被复用的文件不改动。"""
import numpy as np
import torch

import reference_online_c1b_env_repaired as environment_module
import reference_online_c1b_env_reset_api as reset_module
from reference_online_c1a2 import protocol, TARGET


class Environment(reset_module.Environment):
    def __init__(self, args, specs):
        resolve = environment_module.resolve_lite3_asset

        def resolve_with_target(cfg):
            # 基类在保存 config.json 和创建环境之前调用这里，所以记录下来的配置就是实际使用的地形。
            if TARGET == 'boxes_06' and args.kind != 'retention':
                from lite3_terrain_reuse.heldout_boxes import HeldoutBoxesTrackCfg
                sub = HeldoutBoxesTrackCfg(flat_length=3., lane_width=2., grid_width=.4, grid_height_range=[.06, .06], seed=101)
                sub.proportion = 1.
                cfg.scene.terrain.terrain_generator.sub_terrains = {'boxes_06': sub}
            return resolve(cfg)

        original = environment_module.protocol, reset_module.protocol
        environment_module.protocol = reset_module.protocol = protocol
        environment_module.resolve_lite3_asset = resolve_with_target
        try:
            super().__init__(args, specs)
        finally:
            environment_module.resolve_lite3_asset = resolve
            environment_module.protocol, reset_module.protocol = original
        self.feedback_error = 0.

    def reset(self, ids, episodes):
        original = environment_module.protocol, reset_module.protocol
        environment_module.protocol = reset_module.protocol = protocol
        try:
            return super().reset(ids, episodes)
        finally:
            environment_module.protocol, reset_module.protocol = original

    def set_command(self, next_step=False):
        super().set_command(next_step)
        forward = self.command.vel_command_b[:, 0]
        heading = self.robot.data.heading_w
        error = torch.remainder(-heading + torch.pi, 2 * torch.pi) - torch.pi
        wz = torch.where(forward > 0, (.5 * error).clamp(-.8, .8), torch.zeros_like(error))
        self.command.vel_command_b[:, 2] = wz
        h = heading.double().cpu().numpy()
        expect = np.clip(.5 * np.arctan2(np.sin(-h), np.cos(-h)), -.8, .8) * (forward.cpu().numpy() > 0)
        self.feedback_error = max(getattr(self, 'feedback_error', 0.), float(np.abs(wz.double().cpu().numpy() - expect).max()))

    def audit_geometry(self, path):
        result = super().audit_geometry(path)
        for key, value in result.items():
            if key.startswith('boxes'):
                assert value['centerline_max_z'] > .01
                value['goal_x'] = 5.5
        return result

    def frame(self):
        result = super().frame()
        result['body_velocity'] = self.robot.data.root_lin_vel_b[:, :2].cpu().numpy().copy()
        return result
