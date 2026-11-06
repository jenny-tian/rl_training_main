"""Deterministic perturbed initial states keyed by (seed, namespace, slot, episode).

The offsets do not consume the simulator's global random stream, so two arms
with the same keys start from identical states.
"""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_RESET_RANGES = {"x": (-0.2, 0.2), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}
NAMESPACES = {"nominal": 0, "p0_eval": 1, "p1_grid": 2, "p1_confirm": 3,
              "p2_train": 4, "p2_eval": 5, "p2_flat": 6, "sealed": 9}


def initial_state_offsets(table_seed, namespace, slots, episodes, ranges=DEFAULT_RESET_RANGES):
    if len(slots) != len(episodes):
        raise ValueError("slots and episodes must have the same length")
    bounds = (ranges["x"], ranges["y"], ranges["yaw"])
    rows = []
    for slot, episode in zip(slots, episodes):
        key = [int(table_seed), int(namespace), int(slot), int(episode)]
        if min(key) < 0:
            raise ValueError("reset keys must be nonnegative")
        draw = np.random.default_rng(key).random(3)
        rows.append([low + (high - low) * value for (low, high), value in zip(bounds, draw)])
    return np.asarray(rows, dtype=np.float64).reshape(len(slots), 3)


def _yaw_quat_wxyz(yaw):
    half = 0.5 * yaw
    zero = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)


def _quat_mul_wxyz(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), dim=-1)


def reset_root_state_from_table(env, env_ids, table_seed, namespace=0, slots=None, ranges=None, asset_cfg=None):
    """Isaac Lab reset event: default root state + env origin + tabulated (x, y, yaw) offset, zero velocity."""
    asset = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    default = asset.data.default_root_state
    ids = torch.as_tensor(env_ids, device=default.device, dtype=torch.long).reshape(-1)
    id_list = ids.cpu().tolist()
    counters = getattr(env, "_ea_reset_episode_counter", None)
    if counters is None:
        counters = torch.zeros(env.num_envs, dtype=torch.long)
        env._ea_reset_episode_counter = counters
    slot_list = [i if slots is None else int(slots[i]) for i in id_list]
    episodes = counters[id_list].tolist()
    counters[id_list] += 1
    offsets = torch.as_tensor(
        initial_state_offsets(table_seed, namespace, slot_list, episodes, ranges or DEFAULT_RESET_RANGES),
        dtype=default.dtype, device=default.device)
    root = default[ids].clone()
    positions = root[:, 0:3] + env.scene.env_origins[ids]
    positions[:, 0] += offsets[:, 0]
    positions[:, 1] += offsets[:, 1]
    orientations = _quat_mul_wxyz(root[:, 3:7], _yaw_quat_wxyz(offsets[:, 2]))
    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=ids)
    asset.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:13]), env_ids=ids)
    env._ea_last_reset_keys = {i: (int(namespace), s, e) for i, s, e in zip(id_list, slot_list, episodes)}
