"""独立恢复约束候选；未接入 PPO、推理或部署，不包含训练入口。"""
from __future__ import annotations

import torch

from reference_low_stairs_preservation import ACTION_SCALES


def recovery_mask(command, tilt_radians):
    """只选择零命令、10–30 度倾斜的恢复状态；倾角由训练真值提供。"""
    if command.shape[-1]!=3 or command.shape[:-1]!=tilt_radians.shape:
        raise ValueError('命令与倾角形状不一致')
    if not torch.isfinite(command).all() or not torch.isfinite(tilt_radians).all():
        raise ValueError('恢复门控输入必须有限')
    zero=(command[...,:2].norm(dim=-1)<=1e-5) & (command[...,2].abs()<=1e-5)
    return zero & (tilt_radians>torch.pi/18) & (tilt_radians<torch.pi/6)


def recovery_loss(prediction, teacher, command, tilt_radians):
    """仅对命中门控的样本计算关节目标偏移，教师无梯度。

    门控限制损失的样本范围，不能保证共享网络其他状态的输出不变。
    系数、独立训练样本和 PPO 接入均留待后续批准的试验。
    """
    if prediction.shape!=teacher.shape or prediction.ndim!=2 or prediction.shape[1]!=12:
        raise ValueError('动作应为相同 N×12 张量')
    if command.shape!=(prediction.shape[0],3):
        raise ValueError('命令应为 N×3 张量')
    if not torch.isfinite(prediction).all() or not torch.isfinite(teacher).all():
        raise ValueError('动作必须有限')
    selected=recovery_mask(command,tilt_radians)
    delta=(prediction-teacher.detach())*prediction.new_tensor(ACTION_SCALES)/.1
    return delta[selected].square().mean() if selected.any() else prediction.sum()*0.
