"""Torch-only outcome, command and statistics logic for the Source curriculum."""
from __future__ import annotations

import math

import torch

from .rows import COLUMN_SPECS, NUM_COLUMNS, OMNI_COLUMNS

TILE_FAMILY_NAMES = ("flat", "multi_scale_rough", "slope", "multi_step_stairs", "slope_plus_rough",
                     "friction_variation")
FAMILY_NAMES = TILE_FAMILY_NAMES + ("omni_flat",)
OMNI_FAMILY_ID = 6
OUTCOME_NAMES = ("completion", "failure", "timeout")


def column_family_ids(omni_columns=OMNI_COLUMNS, device="cpu"):
    ids = [TILE_FAMILY_NAMES.index(spec["family"]) for _, spec in COLUMN_SPECS]
    for column in omni_columns:
        ids[column] = OMNI_FAMILY_ID
    return torch.tensor(ids, dtype=torch.long, device=device)


def wrap_to_pi(angle):
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def apply_tile_commands(vel_command_b, heading_target, is_heading_env, is_standing_env, ids, *,
                        heading, lin_vel_x, stiffness, ang_range, generator=None):
    """Overwrite the command of tile envs `ids` in place. `heading` is heading_w[ids]."""
    if ids.numel() == 0:
        return
    low, high = float(lin_vel_x[0]), float(lin_vel_x[1])
    if not low <= high:
        raise ValueError("lin_vel_x must be (low, high) with low <= high")
    u = torch.rand(ids.numel(), device=vel_command_b.device, generator=generator)
    vel_command_b[ids, 0] = low + (high - low) * u
    vel_command_b[ids, 1] = 0.0
    heading_target[ids] = 0.0
    is_heading_env[ids] = True
    is_standing_env[ids] = False
    vel_command_b[ids, 2] = torch.clamp(stiffness * wrap_to_pi(0.0 - heading),
                                        min=float(ang_range[0]), max=float(ang_range[1]))


def resolve_outcomes(completed, failed, timed_out):
    failed = failed.bool()
    completed = completed.bool() & ~failed
    timed_out = timed_out.bool() & ~failed & ~completed
    return completed, failed, timed_out


def curriculum_moves(completed, failed, timed_out, progress, commanded_distance):
    completed, failed, timed_out = resolve_outcomes(completed, failed, timed_out)
    move_up = completed
    move_down = failed | (timed_out & (progress < 0.5 * commanded_distance))
    return move_up, move_down


def omni_levels(count, *, num_rows, device, generator=None):
    if num_rows < 3:
        raise ValueError("num_rows must be at least 3")
    return torch.randint(1, num_rows - 1, (count,), device=device, generator=generator)


def omni_truncation_bounds(*, tile_size=(12.0, 4.0), num_rows=10, flat_region_columns=(7, 14), inset_m=1.0):
    """World (x_min, x_max, y_min, y_max) of the flat region shrunk by inset_m (no border)."""
    size_x, size_y = tile_size
    x_half = num_rows * size_x / 2.0
    y_origin = -NUM_COLUMNS * size_y / 2.0
    first, last = flat_region_columns
    return (-x_half + inset_m, x_half - inset_m,
            y_origin + first * size_y + inset_m, y_origin + (last + 1) * size_y - inset_m)


def outside_bounds(xy, bounds):
    x, y = xy[:, 0], xy[:, 1]
    inside = (x >= bounds[0]) & (x <= bounds[1]) & (y >= bounds[2]) & (y <= bounds[3])
    return ~inside  # NaN compares False, so NaN is outside


def safety_unsafe(projected_gravity_xy, base_height, joint_vel, nonfoot_contact, control_timed_out,
                  contact_steps, limits):
    """Same formula as SourceEpisodeLifecycle; returns (unsafe, next_contact_steps)."""
    n = contact_steps.shape[0]
    contact = nonfoot_contact.reshape(n, -1).bool().any(-1)
    next_steps = torch.where(contact, contact_steps + 1, torch.zeros_like(contact_steps))
    gxy = projected_gravity_xy.reshape(n, -1)
    height = base_height.reshape(n, -1)[:, 0]
    jv = joint_vel.reshape(n, -1)
    finite = torch.isfinite(gxy).all(-1) & torch.isfinite(height) & torch.isfinite(jv).all(-1)
    unsafe = ((gxy.norm(dim=-1) > limits.max_projected_gravity_xy)
              | (height < limits.min_base_height)
              | (jv.abs().amax(-1) > limits.max_joint_velocity)
              | (next_steps > limits.nonfoot_contact_grace_steps)
              | control_timed_out.reshape(n, -1).bool().any(-1))
    return unsafe | ~finite, next_steps


class FamilyStats:
    """EMA of completion / failure / timeout per family, plus level summaries."""

    def __init__(self, device, decay=0.995):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.rates = torch.zeros(len(FAMILY_NAMES), 3, device=device)
        self.seen = torch.zeros(len(FAMILY_NAMES), dtype=torch.bool, device=device)

    def update(self, family_ids, completed, failed, timed_out):
        completed, failed, timed_out = resolve_outcomes(completed, failed, timed_out)
        outcome = torch.stack([completed, failed, timed_out], dim=-1).float()
        has = outcome.sum(-1) > 0
        ids, outcome = family_ids[has], outcome[has]
        for family in ids.unique().tolist():
            batch = outcome[ids == family]
            weight = self.decay ** batch.shape[0]
            mean = batch.mean(0)
            if self.seen[family]:
                self.rates[family] = weight * self.rates[family] + (1.0 - weight) * mean
            else:
                self.rates[family] = mean
                self.seen[family] = True

    def report(self, levels, env_family_ids, top_row):
        out = {}
        for family, name in enumerate(FAMILY_NAMES):
            mask = env_family_ids == family
            if mask.any():
                out[f"{name}/mean_level"] = float(levels[mask].float().mean())
                out[f"{name}/top_row_fraction"] = float((levels[mask] == top_row).float().mean())
            for k, outcome in enumerate(OUTCOME_NAMES):
                out[f"{name}/{outcome}_ema"] = float(self.rates[family, k])
        return out
