"""Motion-prior rewards with no hard import on the Isaac Sim USD runtime."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .manager import MotionPriorManager, MotionPriorSample, prior_weight
from .runtime import reference_time, require_imitation_state


LOGGER = logging.getLogger(__name__)


def _zero(env: Any) -> torch.Tensor:
    return torch.zeros(int(env.num_envs), device=env.device, dtype=torch.float32)


def _manager(env: Any, reference_path: str | None) -> MotionPriorManager | None:
    if not reference_path:
        return None
    cache = getattr(env, "_motion_prior_managers", None)
    if cache is None:
        cache = {}
        env._motion_prior_managers = cache
    key = str(reference_path)
    if key not in cache:
        try:
            cache[key] = MotionPriorManager(key, device=env.device)
        except (OSError, ValueError, RuntimeError) as exc:
            warned = getattr(env, "_motion_prior_warning_paths", set())
            if key not in warned:
                LOGGER.warning("motion prior disabled for invalid reference %s: %s", key, exc)
                warned.add(key)
                env._motion_prior_warning_paths = warned
            return None
    return cache[key]


def _scale(env: Any, mode: str, prior_weight_value: float, decay_steps: int) -> float:
    step = getattr(env, "common_step_counter", 0)
    return prior_weight(mode, step, decay_steps, prior_weight_value)


def _sample(
    env: Any,
    reference_path: str | None,
    mode: str,
    prior_weight_value: float,
    decay_steps: int,
) -> tuple[MotionPriorManager, MotionPriorSample, float] | None:
    scale = _scale(env, mode, prior_weight_value, decay_steps)
    if scale == 0.0:
        return None
    manager = _manager(env, reference_path)
    if manager is None:
        return None
    time_s = reference_time(env)
    return manager, manager.sample(time_s), scale


def _ids(cfg: Any, attribute: str, size: int) -> list[int] | slice:
    values = getattr(cfg, attribute, None)
    if values is None or len(values) == 0:
        return slice(0, size)
    return values


def _current_joint_pos(env: Any, asset_cfg: Any) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    value = asset.data.joint_pos
    return value[:, _ids(asset_cfg, "joint_ids", value.shape[1])]


def _current_joint_vel(env: Any, asset_cfg: Any) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    value = asset.data.joint_vel
    return value[:, _ids(asset_cfg, "joint_ids", value.shape[1])]


def _root_quat_per_body(root_quat: torch.Tensor, body_vectors: torch.Tensor) -> torch.Tensor:
    """Broadcast one root quaternion to each selected body vector."""
    return root_quat[:, None, :].expand(-1, body_vectors.shape[1], -1)


def _current_foot_pos_b(env: Any, asset_cfg: Any) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    if hasattr(asset.data, "body_pos_b"):
        value = asset.data.body_pos_b
        return value[:, _ids(asset_cfg, "body_ids", value.shape[1])]
    body_pos_w = getattr(asset.data, "body_link_pos_w", None)
    if body_pos_w is None:
        body_pos_w = asset.data.body_pos_w
    root_pos_w = getattr(asset.data, "root_link_pos_w", None)
    if root_pos_w is None:
        root_pos_w = asset.data.root_pos_w
    body_ids = _ids(asset_cfg, "body_ids", body_pos_w.shape[1])
    rel_pos_w = body_pos_w[:, body_ids] - root_pos_w[:, None, :]
    from isaaclab.utils import math as math_utils

    root_quat = getattr(asset.data, "root_link_quat_w", None)
    if root_quat is None:
        root_quat = asset.data.root_quat_w
    return math_utils.quat_apply_inverse(_root_quat_per_body(root_quat, rel_pos_w), rel_pos_w)


def _current_contact(env: Any, sensor_cfg: Any) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name] if hasattr(env.scene, "sensors") else env.scene[sensor_cfg.name]
    if hasattr(sensor.data, "contact"):
        value = sensor.data.contact
        return value[:, _ids(sensor_cfg, "body_ids", value.shape[1])].float()
    forces = sensor.data.net_forces_w
    body_ids = _ids(sensor_cfg, "body_ids", forces.shape[1])
    return torch.linalg.norm(forces[:, body_ids, :], dim=-1).gt(1.0).float()


def _exp_error(error: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-torch.mean(torch.square(error), dim=tuple(range(1, error.ndim))) / max(float(std) ** 2, 1e-8))


def _paper_exp_error(error: torch.Tensor, error_scale: float) -> torch.Tensor:
    """Exponential tracking form used by Motion Priors Reimagined Table 1."""
    squared_sum = torch.square(error).reshape(error.shape[0], -1).sum(dim=1)
    return torch.exp(-float(error_scale) * squared_sum)


def _weighted_exp_error(
    error: torch.Tensor,
    std: float,
    channel_weights: tuple[float, ...] | list[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Exponentiated error with optional per-channel weighting.

    The unweighted path is kept identical to ``_exp_error``.  For foot
    positions, channels are feet and the final coordinate dimension is
    averaged first; for joint positions, each channel is one joint.
    Normalizing by the sum of weights preserves the reward scale while
    preventing the front legs from dominating the hind-leg prior.
    """
    if channel_weights is None:
        return _exp_error(error, std)
    if error.ndim < 2:
        raise ValueError("weighted error must have at least one channel dimension")
    weights = torch.as_tensor(channel_weights, dtype=error.dtype, device=error.device).flatten()
    if weights.numel() != error.shape[1]:
        raise ValueError(
            f"channel_weights must contain {error.shape[1]} values, got {weights.numel()}"
        )
    if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
        raise ValueError("channel_weights must be finite and positive")
    per_channel = torch.square(error).reshape(error.shape[0], error.shape[1], -1).mean(dim=-1)
    mean_error = (per_channel * weights[None, :]).sum(dim=1) / weights.sum()
    return torch.exp(-mean_error / max(float(std) ** 2, 1e-8))


def motion_prior_joint_pos_exp(
    env: Any,
    asset_cfg: Any,
    reference_path: str | None,
    std: float = 0.25,
    mode: str = "off",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
    joint_weights: tuple[float, ...] | list[float] | None = None,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    _, ref, scale = sampled
    return _weighted_exp_error(_current_joint_pos(env, asset_cfg) - ref.joint_pos, std, joint_weights) * scale


def motion_prior_foot_pos_exp(
    env: Any,
    asset_cfg: Any,
    reference_path: str | None,
    std: float = 0.04,
    mode: str = "off",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
    foot_weights: tuple[float, ...] | list[float] | None = None,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    _, ref, scale = sampled
    return _weighted_exp_error(_current_foot_pos_b(env, asset_cfg) - ref.foot_pos_b, std, foot_weights) * scale


def motion_prior_contact_phase(
    env: Any,
    sensor_cfg: Any,
    reference_path: str | None,
    mode: str = "off",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
    foot_weights: tuple[float, ...] | list[float] | None = None,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    _, ref, scale = sampled
    error = 1.0 - torch.abs(_current_contact(env, sensor_cfg) - ref.contact)
    if foot_weights is None:
        value = error.mean(dim=1)
    else:
        weights = torch.as_tensor(foot_weights, dtype=error.dtype, device=error.device).flatten()
        if weights.numel() != error.shape[1] or not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
            raise ValueError("foot_weights must contain positive finite values for each foot")
        value = (error * weights[None, :]).sum(dim=1) / weights.sum()
    return value * scale


def motion_prior_swing_clearance(
    env: Any,
    asset_cfg: Any,
    reference_path: str | None,
    std: float = 0.04,
    mode: str = "off",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
    foot_weights: tuple[float, ...] | list[float] | None = None,
    relative_height: bool = False,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    _, ref, scale = sampled
    actual = _current_foot_pos_b(env, asset_cfg)
    swing = ref.contact < 0.5
    actual_z = actual[..., 2]
    reference_z = ref.foot_pos_b[..., 2]
    if relative_height:
        # Remove morphology and base-height offsets.  At least one stance foot
        # is present in each reference frame, so the lowest foot is a robust
        # local ground proxy for both the retarget and the simulated robot.
        actual_z = actual_z - actual_z.min(dim=1, keepdim=True).values
        reference_z = reference_z - reference_z.min(dim=1, keepdim=True).values
    error = torch.square(actual_z - reference_z)
    masked = torch.where(swing, error, torch.zeros_like(error))
    if foot_weights is None:
        denominator = swing.float().sum(dim=1).clamp_min(1.0)
        mean_error = masked.sum(dim=1) / denominator
    else:
        weights = torch.as_tensor(foot_weights, dtype=error.dtype, device=error.device).flatten()
        if weights.numel() != error.shape[1] or not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
            raise ValueError("foot_weights must contain positive finite values for each foot")
        mean_error = (masked * weights[None, :]).sum(dim=1) / (weights[None, :] * swing).sum(dim=1).clamp_min(1.0)
    return torch.exp(-mean_error / max(float(std) ** 2, 1e-8)) * scale


def motion_prior_symmetry_exp(
    env: Any,
    asset_cfg: Any,
    reference_path: str | None,
    std: float = 0.04,
    mode: str = "off",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    _, ref, scale = sampled
    actual = _current_foot_pos_b(env, asset_cfg)
    pairs = ((0, 3), (1, 2))
    errors = []
    for left, right in pairs:
        errors.append((actual[:, left] - actual[:, right]) - (ref.foot_pos_b[:, left] - ref.foot_pos_b[:, right]))
    return _exp_error(torch.stack(errors, dim=1), std) * scale


def motion_prior_joint_smoothness(
    env: Any,
    asset_cfg: Any,
    reference_path: str | None,
    std: float = 0.5,
    mode: str = "off",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    manager, ref, scale = sampled
    time_s = reference_time(env)
    next_ref = manager.sample(time_s + float(env.step_dt)).joint_pos
    ref_vel = (next_ref - ref.joint_pos) / max(float(env.step_dt), 1e-6)
    return _exp_error(_current_joint_vel(env, asset_cfg) - ref_vel, std) * scale


def motion_reference_base_lin_vel_exp(
    env: Any,
    asset_cfg: Any,
    reference_path: str,
    error_scale: float = 1.0,
    mode: str = "on",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    _, ref, scale = sampled
    require_imitation_state(ref)
    asset = env.scene[asset_cfg.name]
    return _paper_exp_error(asset.data.root_lin_vel_b - ref.base_lin_vel_b, error_scale) * scale


def motion_reference_base_ang_vel_exp(
    env: Any,
    asset_cfg: Any,
    reference_path: str,
    error_scale: float = 0.8,
    mode: str = "on",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    _, ref, scale = sampled
    require_imitation_state(ref)
    asset = env.scene[asset_cfg.name]
    return _paper_exp_error(asset.data.root_ang_vel_b - ref.base_ang_vel_b, error_scale) * scale


def motion_reference_joint_pos_exp(
    env: Any,
    asset_cfg: Any,
    reference_path: str,
    error_scale: float = 2.0,
    mode: str = "on",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    _, ref, scale = sampled
    return _paper_exp_error(_current_joint_pos(env, asset_cfg) - ref.joint_pos, error_scale) * scale


def motion_reference_projected_gravity_exp(
    env: Any,
    asset_cfg: Any,
    reference_path: str,
    error_scale: float = 3.0,
    mode: str = "on",
    prior_weight_value: float = 1.0,
    decay_steps: int = 100_000,
) -> torch.Tensor:
    sampled = _sample(env, reference_path, mode, prior_weight_value, decay_steps)
    if sampled is None:
        return _zero(env)
    _, ref, scale = sampled
    require_imitation_state(ref)
    asset = env.scene[asset_cfg.name]
    return _paper_exp_error(asset.data.projected_gravity_b - ref.projected_gravity_b, error_scale) * scale
