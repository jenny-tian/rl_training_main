"""在已验证的原生重置环境上加航向闭环和箱体目标；被复用的文件不改动。"""
import copy
import sys

import numpy as np
import torch

import reference_online_c1b_env_repaired as environment_module
import reference_online_c1b_env_reset_api as reset_module
from reference_online_c1a2 import protocol, TARGET


STAIR_HEIGHT = dict(stairs_up_065=.065, stairs_up_07=.07)
BOX_HEIGHT = dict(boxes_06=.06, boxes_07=.07)


class Environment(reset_module.Environment):
    def __init__(self, args, specs):
        resolve = environment_module.resolve_lite3_asset

        def resolve_with_target(cfg):
            # 基类在保存 config.json 和创建环境之前调用这里，所以记录下来的配置就是实际使用的地形。
            if TARGET in BOX_HEIGHT and args.kind != 'retention':
                from lite3_terrain_reuse.heldout_boxes import HeldoutBoxesTrackCfg
                sub = HeldoutBoxesTrackCfg(flat_length=3., lane_width=2., grid_width=.4, grid_height_range=[BOX_HEIGHT[TARGET]] * 2, seed=101)
                sub.proportion = 1.
                cfg.scene.terrain.terrain_generator.sub_terrains = {TARGET: sub}
            if TARGET in STAIR_HEIGHT and args.kind != 'retention':
                cfg.scene.terrain.terrain_generator.sub_terrains['stairs_06'].step_height_range = (STAIR_HEIGHT[TARGET],) * 2
            if TARGET == 'slope_rough_up_x11' and args.kind != 'retention':
                # 与 reference_frozen_rebaseline 的 x11 相同：坡度参数 0.44，粗糙幅度 0.01 到 0.07 m。
                sub = captured['slope_rough_up']
                assert tuple(sub.slope_range) == (0., .4) and tuple(sub.noise_range) == (.01, .06)
                sub.slope_range, sub.noise_range, sub.proportion = (.44, .44), (.01, .07), 1.
                cfg.scene.terrain.terrain_generator.sub_terrains = {'slope_rough_up_x11': sub}
            return resolve(cfg)

        # 基类在换成台阶地形之前调用 replace_terrain；在这里记下六类地形的原始子地形配置。
        # reference_six_terrain_cfg 要在参考配置加载之后才能导入，所以挂在 load_reference 后面。
        captured, load, patched = {}, environment_module.load_reference, {}

        def load_and_patch():
            result = load()
            import reference_six_terrain_cfg
            replace = patched['replace'] = reference_six_terrain_cfg.replace_terrain

            def replace_and_capture(reference):
                cfg = replace(reference)
                captured.update(copy.deepcopy(cfg.scene.terrain.terrain_generator.sub_terrains))
                return cfg
            reference_six_terrain_cfg.replace_terrain = replace_and_capture
            return result

        original = environment_module.protocol, reset_module.protocol
        environment_module.protocol = reset_module.protocol = protocol
        environment_module.resolve_lite3_asset = resolve_with_target
        environment_module.load_reference = load_and_patch
        try:
            super().__init__(args, specs)
        finally:
            environment_module.load_reference = load
            if patched: sys.modules['reference_six_terrain_cfg'].replace_terrain = patched['replace']
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
        if TARGET in STAIR_HEIGHT and self.args.kind != 'retention':
            # 基类按 6 cm 核对台阶高度；这里用同样的做法按该目标的高度核对。
            height = STAIR_HEIGHT[TARGET]
            from isaaclab.sensors import RayCaster
            from isaaclab.utils.warp import raycast_mesh
            origin = self.terrain.env_origins[0]
            xs = torch.linspace(0., 3.8, 3401, device=self.device)
            starts = origin + torch.stack((xs, torch.zeros_like(xs), torch.full_like(xs, 20.)), -1)
            directions = torch.zeros_like(starts); directions[:, 2] = -1
            hits = raycast_mesh(starts, directions, RayCaster.meshes[path])[0]
            assert torch.isfinite(hits).all()
            z, x = (hits[:, 2] - origin[2]).cpu().numpy(), xs.cpu().numpy()
            edges = np.flatnonzero(np.diff(z) > height / 2) + 1
            fronts, tops = ((x[edges] + x[edges - 1]) / 2).tolist(), z[edges].tolist()
            assert len(fronts) >= 5 and np.allclose(np.diff([0.] + tops), height, atol=.002)
            return {f'{TARGET}:0': dict(fronts=fronts, tops=tops, goal_x=fronts[-1] + .3,
                                        centerline_min_z=float(z.min()), centerline_max_z=float(z.max()))}
        result = super().audit_geometry(path)
        for key, value in result.items():
            if key.startswith('boxes'):
                assert .01 < value['centerline_max_z'] <= BOX_HEIGHT[TARGET] + .002
                value['goal_x'] = 5.5
        return result

    def frame(self):
        result = super().frame()
        result['body_velocity'] = self.robot.data.root_lin_vel_b[:, :2].cpu().numpy().copy()
        return result
