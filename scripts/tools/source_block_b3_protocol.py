"""Frozen B3 command schedules and first-episode statistics; no simulator dependency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261113
REPEATS = 32
CASES = (
    "turn_low_left", "turn_low_right", "turn_left", "turn_right",
    "stop_left", "stop_right", "ramp_left", "ramp_right",
    "reverse_left", "reverse_right", "heading_left", "heading_right",
    "lateral_left", "lateral_right", "forward", "standing",
)
WINDOWS = {"all": (0, 20), "settle": (0, 2), "onset": (2, 4),
           "steady": (4, 8), "transition": (8, 10), "late": (10, 14), "tail": (14, 20)}
FIELDS = ("cmd_x", "cmd_y", "cmd_yaw", "vel_x", "vel_y", "vel_yaw",
          "heading_error", "heading_mode", "standing_mode", "action_delta_sq",
          "base_height", "gravity_xy", "pos_x", "pos_y", "pos_z", "scan_invalid")


def schedule(case: str, t: float):
    """Return (vx, vy, wz, heading_offset). Commands apply at the interval start."""
    if case not in CASES:
        raise ValueError(case)
    sign = -1.0 if case.endswith("right") else 1.0
    if t < 2 or case == "standing":
        return (0.0, 0.0, 0.0, None)
    if case.startswith("heading"):
        return (0.0, 0.0, 0.0, sign * 1.2)
    if case.startswith("lateral"):
        return (0.0, sign * .3 if t < 8 else 0.0, 0.0, None)
    if case == "forward":
        return (.25 if t < 8 else 0.0, 0.0, 0.0, None)
    speed = .15 if case.startswith("turn_low") else .5
    if case.startswith("stop") and t >= 8:
        speed = 0.0
    if case.startswith("ramp"):
        speed *= np.clip((10 - t) / 2, 0, 1)
    if case.startswith("reverse") and t >= 8:
        sign *= -1
    return (0.0, 0.0, float(sign * speed), None)


def models():
    from source_block_b2_protocol import checkpoint_for
    result = []
    for name in ("flat1998", "terrain3497"):
        path, sha = checkpoint_for(name)
        result.append(dict(name=name, checkpoint=str(path), sha256=sha, updates=0))
    audit = json.loads((ROOT / "docs/lite3-block-b2-initialization-20261111/model_audit.json").read_text())
    for row in audit:
        result.append(dict(name=f"{row['start']}_seed{row['seed']}",
                           checkpoint=row["checkpoint"], sha256=row["sha256"], updates=100))
    for row in result:
        if hashlib.sha256(Path(row["checkpoint"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"checkpoint mismatch: {row['name']}")
    return result


def first_episode_masks(ended):
    """Terminal samples are kept separately; metric samples never include reset states."""
    previous = np.concatenate((np.zeros_like(ended[:1]), np.cumsum(ended[:-1], axis=0)), axis=0)
    first = previous == 0
    return first, first & ~ended


def categories(command):
    x, y, yaw = np.moveaxis(command, -1, 0)
    pure = (abs(x) < 1e-6) & (abs(y) < 1e-6)
    return {"standing": pure & (abs(yaw) < 1e-6),
            "turn_low": pure & (abs(yaw) >= 1e-6) & (abs(yaw) < .2),
            "turn_high": pure & (abs(yaw) >= .2),
            "forward": (abs(x) >= 1e-6) & (abs(y) < 1e-6) & (abs(yaw) < 1e-6),
            "lateral": (abs(x) < 1e-6) & (abs(y) >= 1e-6) & (abs(yaw) < 1e-6),
            "mixed": ~pure & ((abs(yaw) >= 1e-6) | ((abs(x) >= 1e-6) & (abs(y) >= 1e-6)))}


def summarize(trace, ended, terminated, groups, reward_names, dt, mode):
    first, valid = first_episode_masks(ended)
    times = np.arange(len(trace)) * dt
    result = []
    for group, env_mask in groups.items():
        selections = WINDOWS if mode == "controlled" else {"all": (0, len(trace) * dt)}
        for window, (start, stop) in selections.items():
            mask = valid & env_mask[None, :] & ((times >= start) & (times < stop))[:, None]
            choices = {"all": np.ones_like(mask)} if mode == "controlled" else {
                "all": np.ones_like(mask), **categories(trace[..., :3])}
            for category, category_mask in choices.items():
                use = mask & category_mask
                data = trace[use]
                n = len(data)
                if not n:
                    continue
                error = data[:, 3:6] - data[:, :3]
                row = dict(group=group, window=window, category=category, samples=n,
                           envs_with_samples=int(use.any(axis=0).sum()), seconds=n * dt,
                           rms_xy=float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=-1)))),
                           rms_yaw=float(np.sqrt(np.mean(error[:, 2] ** 2))),
                           mean_command=data[:, :3].mean(0).tolist(), mean_velocity=data[:, 3:6].mean(0).tolist(),
                           mean_abs_yaw=float(abs(data[:, 5]).mean()),
                           heading_abs_error=float(abs(data[data[:, 7] > .5, 6]).mean()) if (data[:, 7] > .5).any() else None,
                           action_delta_sq=float(data[:, 9].mean()), base_height=float(data[:, 10].mean()),
                           gravity_xy=float(data[:, 11].mean()), scan_invalid_fraction=float(data[:, 15].mean()),
                           reward_rates=dict(zip(reward_names, data[:, len(FIELDS):len(FIELDS)+len(reward_names)].mean(0).tolist())))
                result.append(row)
    episodes = {}
    for group, env_mask in groups.items():
        episodes[group] = dict(starts=int(env_mask.sum()),
            failures=int((first & terminated & env_mask[None]).sum()),
            truncations=int((first & ended & ~terminated & env_mask[None]).sum()),
            censored=int((~ended.any(axis=0) & env_mask).sum()))
    return dict(rows=result, episodes=episodes)
