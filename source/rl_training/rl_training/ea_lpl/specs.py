"""Run specifications for the EA-LPL runner and stage-1 qualification."""
from __future__ import annotations

import math
from pathlib import Path
import re

from .basis import BASIS_NAMES

SPEC_SCHEMA = "ea_lpl_run_spec_v1"
NUM_ENVS = 120
LEVELS = ("L1", "L2", "L3")
EQUAL8 = tuple(column for column in range(15) for _ in range(8))
P0_TABLE_SEED = 20261106
REQUIRED = {"schema", "label", "mode", "level", "heading_hold", "columns", "reset_table_seed", "namespace",
            "slots", "episodes_per_env", "max_episode_steps", "codes", "wall_limit_s"}


def _int_list(values, *, length, low, high=None):
    return (isinstance(values, list) and len(values) == length
            and all(type(v) is int and v >= low and (high is None or v < high) for v in values))


def validate_spec(spec):
    missing = REQUIRED - set(spec)
    if missing:
        raise ValueError(f"spec is missing {sorted(missing)}")
    if spec["schema"] != SPEC_SCHEMA or spec["mode"] != "fixed":
        raise ValueError("unknown schema or mode")
    if spec["level"] not in LEVELS or type(spec["heading_hold"]) is not bool:
        raise ValueError("level must be L1/L2/L3 and heading_hold boolean")
    if "actor" in spec:
        actor = spec["actor"]
        if (not isinstance(actor, dict) or set(actor) != {"path", "sha256"}
                or not isinstance(actor["path"], str) or not Path(actor["path"]).is_absolute()
                or not isinstance(actor["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", actor["sha256"]) is None):
            raise ValueError("actor must define an absolute path and 64 lowercase hex sha256")
    if spec["columns"] is not None and not _int_list(spec["columns"], length=NUM_ENVS, low=0, high=15):
        raise ValueError("columns must list 120 registered column indices")
    if type(spec["episodes_per_env"]) is not int or spec["episodes_per_env"] < 1:
        raise ValueError("episodes_per_env must be a positive integer")
    if spec["episodes_per_env"] > 1 and spec["columns"] is None:
        raise ValueError("multiple episodes per environment require fixed columns")
    if spec["max_episode_steps"] != 2000:
        raise ValueError("the episode limit is fixed at 2000 control steps")
    seed = spec["reset_table_seed"]
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("reset_table_seed must be a nonnegative integer or null")
    if type(spec["namespace"]) is not int or not 0 <= spec["namespace"] < 9:
        raise ValueError("namespace must be in 0..8; 9 is reserved for sealed runs")
    if spec["slots"] is not None and (seed is None or not _int_list(spec["slots"], length=NUM_ENVS, low=0)):
        raise ValueError("slots need a reset table seed and 120 nonnegative integers")
    codes = spec["codes"]
    if codes is not None:
        if set(codes) != {"basis", "u", "code_name", "record_replay"}:
            raise ValueError("codes must define basis, u, code_name and record_replay")
        if spec["columns"] is None:
            raise ValueError("residual codes require fixed columns")
        if len(codes["basis"]) != NUM_ENVS or any(b not in BASIS_NAMES for b in codes["basis"]):
            raise ValueError("codes.basis must name one registered basis per environment")
        if len(codes["u"]) != NUM_ENVS or any(
                len(u) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3 for x in u)
                for u in codes["u"]):
            raise ValueError("codes.u must hold 120 finite pairs with |u| <= 3")
        if len(codes["code_name"]) != NUM_ENVS or type(codes["record_replay"]) is not bool:
            raise ValueError("codes.code_name must have 120 entries and record_replay must be boolean")
    if (type(spec["wall_limit_s"]) not in (int, float) or not math.isfinite(spec["wall_limit_s"])
            or spec["wall_limit_s"] <= 0):
        raise ValueError("wall_limit_s must be positive")
    return spec


def p0_specs():
    common = dict(schema=SPEC_SCHEMA, mode="fixed", episodes_per_env=1, max_episode_steps=2000,
                  codes=None, wall_limit_s=900)
    perturbed = dict(columns=list(EQUAL8), reset_table_seed=P0_TABLE_SEED, namespace=1, slots=list(range(NUM_ENVS)))
    return [
        dict(common, label="p0_e0n_l1", level="L1", heading_hold=False, columns=None,
             reset_table_seed=None, namespace=0, slots=None),
        dict(common, label="p0_e0p_l1", level="L1", heading_hold=False, **perturbed),
        dict(common, label="p0_e1p_l1", level="L1", heading_hold=True, **perturbed),
        dict(common, label="p0_e1p_l2", level="L2", heading_hold=True, **perturbed),
        dict(common, label="p0_e0p_l2", level="L2", heading_hold=False, **perturbed),
    ]


def spec_actor(spec, *, default):
    """Return (path, sha256) for the actor of a validated spec."""
    actor = spec.get("actor")
    if actor is None:
        return default
    return actor["path"], actor["sha256"]


def stage1_specs(policies, *, kind):
    """Build paired qualification, preview or extension runs for named actors."""
    if kind not in ("qualify", "preview", "extend", "targeted", "targeted_preview"):
        raise ValueError(f"unknown stage-1 spec kind: {kind!r}")
    levels = ("L3",) if kind in ("preview", "targeted_preview") else ("L1", "L2")
    prefix = {"qualify": "s1q", "preview": "s1p", "extend": "s1x", "targeted": "s1t",
              "targeted_preview": "s1tp"}[kind]
    offset = {"extend": 480, "targeted": 1080, "targeted_preview": 1080}.get(kind, 0)
    specs = []
    for name, actor in policies.items():
        for level in levels:
            for k in range(4):
                spec = dict(schema=SPEC_SCHEMA, label=f"{prefix}_{name}_{level.lower()}_{k}", mode="fixed",
                            level=level, heading_hold=True, columns=list(EQUAL8),
                            reset_table_seed=P0_TABLE_SEED, namespace=4,
                            slots=list(range(offset + NUM_ENVS * k, offset + NUM_ENVS * (k + 1))),
                            episodes_per_env=1, max_episode_steps=2000, codes=None, wall_limit_s=900,
                            actor=actor.copy() if isinstance(actor, dict) else actor)
                validate_spec(spec)
                specs.append(spec)
    return specs
