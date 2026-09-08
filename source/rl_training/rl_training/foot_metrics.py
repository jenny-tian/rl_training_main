"""Pure Torch gait metrics shared by training and offline tests."""

from __future__ import annotations

import torch


def relative_foot_clearance(foot_pos_w: torch.Tensor) -> torch.Tensor:
    """Return each foot's height above the lowest foot in its environment."""
    if foot_pos_w.ndim != 3 or foot_pos_w.shape[1:] != (4, 3):
        raise ValueError("foot_pos_w must have shape (num_envs, 4, 3)")
    foot_z = foot_pos_w[..., 2]
    return (foot_z - foot_z.min(dim=1, keepdim=True).values).clamp_min(0.0)
