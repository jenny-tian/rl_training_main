"""Isaac Lab MDP functions for the dedicated flat motion-imitation task."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .runtime import (
    get_motion_manager,
    reference_offsets,
    reference_time,
    require_imitation_state,
    sample_reference,
)


def motion_reference_phase(env: Any, reference_path: str) -> torch.Tensor:
    manager = get_motion_manager(env, reference_path)
    phase = reference_time(env)[:, None] / manager.duration
    return torch.cat((torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)), dim=-1)


def motion_reference_joint_targets(
    env: Any,
    asset_cfg: Any,
    reference_path: str,
    future_steps: Sequence[int] = (1, 2, 10, 30),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    default = asset.data.default_joint_pos[:, joint_ids]
    targets = [sample_reference(env, reference_path, step).joint_pos - default for step in future_steps]
    return torch.cat(targets, dim=1)


def reset_from_motion_reference(
    env: Any,
    env_ids: torch.Tensor,
    asset_cfg: Any,
    reference_path: str,
    reference_state_probability: float = 0.9,
) -> None:
    """Reset each environment at an independently sampled reference frame."""
    if not 0.0 <= reference_state_probability <= 1.0:
        raise ValueError("reference_state_probability must be in [0, 1]")
    if env_ids.numel() == 0:
        return

    manager = get_motion_manager(env, reference_path)
    offsets = reference_offsets(env)
    offsets[env_ids] = torch.rand(env_ids.numel(), device=env.device) * manager.duration
    use_reference = torch.rand(env_ids.numel(), device=env.device) < reference_state_probability
    offsets[env_ids[~use_reference]] = 0.0
    ref = manager.sample(offsets[env_ids])
    require_imitation_state(ref)

    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_pos = asset.data.default_joint_pos[env_ids][:, joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids][:, joint_ids].clone()
    joint_pos[use_reference] = ref.joint_pos[use_reference]
    joint_vel[use_reference] = ref.joint_vel[use_reference]

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 7:] = 0.0
    if torch.any(use_reference):
        from isaaclab.utils import math as math_utils

        quat = root_state[use_reference, 3:7]
        root_state[use_reference, 7:10] = math_utils.quat_apply(quat, ref.base_lin_vel_b[use_reference])
        root_state[use_reference, 10:13] = math_utils.quat_apply(quat, ref.base_ang_vel_b[use_reference])

    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)


def motion_imitation_time_out(
    env: Any,
    start_steps: int = 20,
    end_steps: int = 600,
    curriculum_steps: int = 7_500,
) -> torch.Tensor:
    """Cubic short-to-long episode curriculum used by motion imitation."""
    if start_steps <= 0 or end_steps < start_steps or curriculum_steps <= 0:
        raise ValueError("invalid motion-imitation episode curriculum")
    progress = min(float(env.common_step_counter) / float(curriculum_steps), 1.0) ** 3
    current_limit = round((1.0 - progress) * start_steps + progress * end_steps)
    return env.episode_length_buf >= current_limit
