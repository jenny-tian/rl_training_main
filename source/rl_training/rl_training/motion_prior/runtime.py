"""Runtime state shared by motion-reference observations, rewards, and resets."""

from __future__ import annotations

from typing import Any

import torch

from .manager import MotionPriorManager, MotionPriorSample


def get_motion_manager(env: Any, reference_path: str) -> MotionPriorManager:
    cache = getattr(env, "_motion_prior_managers", None)
    if cache is None:
        cache = {}
        env._motion_prior_managers = cache
    key = str(reference_path)
    if key not in cache:
        cache[key] = MotionPriorManager(key, device=env.device)
    return cache[key]


def reference_offsets(env: Any) -> torch.Tensor:
    offsets = getattr(env, "_motion_reference_offsets", None)
    if offsets is None or offsets.shape != (int(env.num_envs),):
        offsets = torch.zeros(int(env.num_envs), device=env.device, dtype=torch.float32)
        env._motion_reference_offsets = offsets
    return offsets


def reference_time(env: Any, future_steps: int | float = 0) -> torch.Tensor:
    elapsed = env.episode_length_buf.to(device=env.device, dtype=torch.float32) * float(env.step_dt)
    return reference_offsets(env) + elapsed + float(future_steps) * float(env.step_dt)


def sample_reference(env: Any, reference_path: str, future_steps: int | float = 0) -> MotionPriorSample:
    manager = get_motion_manager(env, reference_path)
    return manager.sample(reference_time(env, future_steps))


def require_imitation_state(sample: MotionPriorSample) -> None:
    missing = [
        name
        for name in ("joint_vel", "base_lin_vel_b", "base_ang_vel_b", "projected_gravity_b")
        if getattr(sample, name) is None
    ]
    if missing:
        raise ValueError(f"motion imitation reference is missing: {', '.join(missing)}")
