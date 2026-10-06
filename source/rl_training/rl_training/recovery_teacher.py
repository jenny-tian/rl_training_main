"""Kinematic recovery resets for separately budgeted PPO teacher training."""
from pathlib import Path

import numpy as np
import torch

from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES
from rl_training.online_adaptation.provenance import sha256_file


def bank_indices(terrain_types, env_ids, reset_counts, bank_terrain, validation, *, split, mixed):
    if split not in ("train", "validation"):
        raise ValueError("explicit recovery bank split required")
    result = torch.full_like(env_ids, -1)
    for category, mask in ((101, (terrain_types >= 2) & (terrain_types < 5)), (107, terrain_types >= 5)):
        pool = torch.where((bank_terrain == category) & (validation == (split == "validation")))[0]
        if not len(pool):
            raise ValueError("empty terrain/split recovery pool")
        selected = mask
        if mixed:
            selected = selected & ((env_ids+reset_counts) % 2 == 0)
        position = env_ids[selected]+reset_counts[selected]
        # Undo the parity selection, otherwise an even-sized pool would expose
        # only its even indices during mixed training.
        if mixed:
            position = position//2
        result[selected] = pool[position % len(pool)]
    return result


def dedicated_recovery_mask(terrain_types, per_terrain):
    """Choose fixed recovery slots from the entire scene, not the reset subset."""
    if type(per_terrain) is not int or per_terrain < 1:
        raise ValueError("positive dedicated slot count required")
    mask=torch.zeros_like(terrain_types,dtype=torch.bool)
    for category in ((terrain_types>=2)&(terrain_types<5),terrain_types>=5):
        pool=torch.where(category)[0]
        if len(pool)<per_terrain:raise ValueError("insufficient rough slots for recovery allocation")
        mask[pool[:per_terrain]]=True
    return mask


def reset_recovery_states(env, env_ids, *, bank_path, bank_sha256, split, mixed, dedicated_per_terrain=None):
    if dedicated_per_terrain is not None and (split!='train' or mixed):
        raise ValueError("dedicated recovery requires unmixed training resets")
    if not hasattr(env, "_recovery_bank"):
        if sha256_file(bank_path) != bank_sha256:
            raise ValueError("recovery bank hash mismatch")
        with np.load(bank_path, allow_pickle=False) as data:
            env._recovery_bank = {k: torch.as_tensor(data[k].copy(), device=env.device) for k in data.files}
        env._recovery_case = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        env._recovery_reset_count = torch.zeros_like(env._recovery_case)
        env._recovery_reset_rows = 0
        env._recovery_dedicated_mask = (dedicated_recovery_mask(env.scene.terrain.terrain_types,dedicated_per_terrain)
            if dedicated_per_terrain is not None else None)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    bank = env._recovery_bank
    cases = bank_indices(env.scene.terrain.terrain_types[ids], ids, env._recovery_reset_count[ids],
        bank["terrain"], bank["validation"], split=split, mixed=mixed)
    if env._recovery_dedicated_mask is not None:
        cases[~env._recovery_dedicated_mask[ids]]=-1
    env._recovery_reset_count[ids] += 1
    env._recovery_case[ids] = cases
    chosen = ids[cases >= 0]; cases = cases[cases >= 0]
    if not len(chosen):
        return
    robot = env.scene["robot"]
    root = bank["root"][cases].clone()
    root[:,:3] += env.scene.env_origins[chosen]
    joints, names = robot.find_joints(list(LITE3_POLICY_JOINT_NAMES), preserve_order=True)
    if tuple(names) != LITE3_POLICY_JOINT_NAMES:
        raise ValueError("recovery joint convention differs")
    robot.write_root_state_to_sim(root, env_ids=chosen)
    robot.write_joint_state_to_sim(bank["joint_pos"][cases], bank["joint_vel"][cases],
                                  joint_ids=joints, env_ids=chosen)
    env._recovery_reset_rows += len(chosen)


def configure_recovery_resets(cfg, options):
    from isaaclab.managers import EventTermCfg
    path = Path(options["bank_path"])
    if sha256_file(path) != options["bank_sha256"]:
        raise ValueError("recovery bank hash mismatch")
    cfg.events.recovery_state = EventTermCfg(func=reset_recovery_states, mode="reset", params=dict(options))


def check_recovery_event_order(env):
    # Root and joint reset events must run before our kinematic overwrite.
    if env.event_manager.active_terms["reset"][-1] != "recovery_state":
        raise ValueError("recovery state event must be the last reset event")
