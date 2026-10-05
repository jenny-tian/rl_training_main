"""Predeclared training-only velocity impulses, with no random sampling."""
import math

import torch


def scheduled_recovery_impulse(step, active, terrain_types, *, steps, roll_velocity_rad_s, lateral_velocity_m_s):
    """Perturb half the rough slots, alternating sign at each scheduled step.

    Columns are world-frame COM vx,vy,vz,wx,wy,wz increments. This is a
    simulator training intervention, never a deployed controller or free rollout.
    """
    if active.ndim != 1 or active.dtype != torch.bool or terrain_types.shape != active.shape:
        raise ValueError("matching activity and terrain vectors required")
    if not steps or any(type(s) is not int or s < 0 for s in steps) or sorted(set(steps)) != list(steps):
        raise ValueError("strictly increasing nonnegative impulse steps required")
    if not all(math.isfinite(v) and v > 0 for v in (roll_velocity_rad_s, lateral_velocity_m_s)):
        raise ValueError("positive finite impulse magnitudes required")
    delta = torch.zeros((len(active), 6), device=active.device)
    if step not in steps:
        return delta
    index = torch.arange(len(active), device=active.device)
    selected = active & (terrain_types >= 2) & (index % 2 == 0)
    sign = torch.where((index // 2 + list(steps).index(step)) % 2 == 0, 1., -1.)
    delta[:, 1] = torch.where(selected, -sign*lateral_velocity_m_s, 0.)
    delta[:, 3] = torch.where(selected, sign*roll_velocity_rad_s, 0.)
    return delta
