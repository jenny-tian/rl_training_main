"""待审查的能力保持组件；不自动接入奖励、PPO 或训练入口。"""
from __future__ import annotations

import torch

GROUPS = ('standing', 'standing_recovery', 'forward', 'left', 'right')
ACTION_SCALES = (.125, .25, .25) * 4


def capability_groups(command, tilt_radians):
    """来自父模型成功回合的状态分组，恢复姿态单列，防止被平稳站立淹没。"""
    if command.shape[-1] != 3 or command.shape[:-1] != tilt_radians.shape:
        raise ValueError('命令和倾角维度不一致')
    groups = torch.full(tilt_radians.shape, -1, dtype=torch.long, device=command.device)
    still_xy = command[..., :2].norm(dim=-1) < .1
    still_yaw = command[..., 2].abs() < .1
    standing = still_xy & still_yaw
    groups[standing & (tilt_radians <= torch.pi / 18)] = 0
    groups[standing & (tilt_radians > torch.pi / 18)] = 1
    groups[(command[..., 0] >= .1) & (command[..., 1].abs() < .1) & still_yaw] = 2
    groups[still_xy & (command[..., 2] > .1)] = 3
    groups[still_xy & (command[..., 2] < -.1)] = 4
    return groups


def joint_target_preservation_loss(prediction, teacher, groups, reference_radians=.1):
    """在真实父模型观测上限制关节目标偏移，各类等权；不限制台阶摆脚。

    这是独立候选辅助损失，尚无 PPO 集成或权重。父模型输出停止梯度。
    使用固定参照角度归一化，不把 HipX 的动作缩放当作其他关节的缩放。
    """
    if prediction.shape != teacher.shape or prediction.ndim != 2 or prediction.shape[1] != 12:
        raise ValueError('动作应为相同的 N×12 张量')
    if groups.shape != prediction.shape[:1] or reference_radians <= 0:
        raise ValueError('无效的分组或角度尺度')
    if not torch.isfinite(prediction).all() or not torch.isfinite(teacher).all():
        raise ValueError('动作包含非有限值')
    if ((groups < -1) | (groups >= len(GROUPS))).any():
        raise ValueError('未知的能力分组')
    difference = (prediction - teacher.detach()) * prediction.new_tensor(ACTION_SCALES)
    costs = (difference / reference_radians).square().mean(-1)
    terms = [costs[groups == group].mean() for group in range(len(GROUPS)) if (groups == group).any()]
    return torch.stack(terms).mean() if terms else prediction.sum() * 0.


def split_heading_turn_samples(term, env_ids, choice, magnitude):
    """仅把一半纯航向转动样本改为恒定角速度，左右概率相同。

    choice∈[0,1)：前 1/2 保留航向，随后 1/4 左转、1/4 右转。
    magnitude∈[0,1] 映射至 0.2–0.8 rad/s。调用者负责提供独立随机数。
    必须在原生 _resample_command 之后、_update_command 之前调用。
    """
    if env_ids.ndim != 1 or choice.shape != env_ids.shape or magnitude.shape != env_ids.shape:
        raise ValueError('重采样索引和随机数尺寸不一致')
    if not torch.isfinite(choice).all() or not torch.isfinite(magnitude).all():
        raise ValueError('随机数包含非有限值')
    if ((choice < 0) | (choice >= 1) | (magnitude < 0) | (magnitude > 1)).any():
        raise ValueError('随机数不在约定范围')
    if len(torch.unique(env_ids)) != len(env_ids):
        raise ValueError('重采样索引不能重复')
    eligible = (~term.is_standing_env[env_ids] & term.is_heading_env[env_ids]
                & (term.vel_command_b[env_ids, :2].norm(dim=-1) < 1e-8))
    selected = eligible & (choice >= .5)
    ids = env_ids[selected]
    rate = .2 + .6 * magnitude[selected]
    term.vel_command_b[ids, 2] = torch.where(choice[selected] < .75, rate, -rate)
    term.is_heading_env[ids] = False
    return ids
