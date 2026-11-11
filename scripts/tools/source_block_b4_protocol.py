"""Pre-registered single-reward trial; no automatic training extension."""
from __future__ import annotations

import json
from pathlib import Path

from source_block_b2_protocol import Q4_LIMITS, ROOT

RAW = ROOT / "logs/stage1/block_b4"
OUT = ROOT / "docs/lite3-block-b4-turngate-20261111"
TASKS = {"control": "SourceBlock-B2-Deeprobotics-Lite3-v0",
         "turngate": "SourceBlock-TurnGate-Deeprobotics-Lite3-v0"}
SEEDS = (42, 43)
UPDATES, NUM_ENVS, STEPS = 25, 4096, 24


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [dict(path=path, before=sorted(a), after=sorted(b))]
        return [d for key in a for d in differences(a[key], b[key], f"{path}/{key}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}/{i}")]
    return [] if a == b else [dict(path=path, before=a, after=b)]


def config_audit(seed, num_envs):
    """Call only after AppLauncher. Compare actual registered config values."""
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    configs, agents = {}, {}
    for branch, task in TASKS.items():
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
        cfg.seed = seed
        configs[branch] = cfg
        agents[branch] = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    serialized = {k: json.loads(json.dumps(v.to_dict(), default=str)) for k, v in configs.items()}
    changes = differences(serialized["control"], serialized["turngate"])
    if [d["path"] for d in changes] != ["/rewards/stand_still/func"]:
        raise ValueError(f"trial has extra environment differences: {changes}")
    if differences(agents["control"].to_dict(), agents["turngate"].to_dict()):
        raise ValueError("trial algorithm configurations differ")
    old, new = (configs[k].rewards.stand_still for k in TASKS)
    if (old.func.__name__ != "stand_still_joint_deviation_l1"
            or new.func.__name__ != "stand_still_without_cmd"
            or new.weight != -.2 or new.params["command_threshold"] != .1):
        raise ValueError("unexpected reward intervention")
    agent = agents["control"]
    if agent.algorithm.learning_rate != .0003844 or agent.num_steps_per_env != STEPS:
        raise ValueError("training LR or rollout budget changed")
    return configs, agents, dict(status="passed", differences=changes,
                                configurations=serialized, agent=agent.to_dict())


def q4_failures(flat):
    if flat.get("status") != "passed" or flat.get("completed_steps") != 3000:
        raise ValueError("Q4 execution incomplete")
    return [name for name, limit in Q4_LIMITS.items() if flat["metrics"][name] > limit]


def candidate_gate(candidate_flat, candidate_block, control_block):
    if (candidate_block["initial_state_hashes_by_column"]
            != control_block["initial_state_hashes_by_column"]):
        raise ValueError("unpaired stair evaluation")
    reasons = [f"Q4/{name}" for name in q4_failures(candidate_flat)]
    up, down = (candidate_block["columns"][i]["complete"] for i in (4, 5))
    if up < 21:
        reasons.append("upstairs_below_21_of_32")
    for c in (4, 5):
        if candidate_block["columns"][c]["complete"] < control_block["columns"][c]["complete"] - 2:
            reasons.append(f"column_{c}_more_than_2_below_control")
    return dict(eligible_for_discussion=not reasons, reasons=reasons, upstairs=up, downstairs=down,
                automatic_extension=False)
