"""P1 code grid, run packing and the registered headroom rules."""
from __future__ import annotations

import numpy as np  # noqa: F401  (used by the headroom rules added in later tasks)

from .basis import BASIS_NAMES
from .specs import NUM_ENVS, SPEC_SCHEMA

GRID_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0)
P1_TABLE_SEED = 20261106
GRID_NAMESPACE = 2
CONFIRM_NAMESPACE = 3


def grid_codes():
    codes = [dict(code_name="zero", basis="sag", u=[0.0, 0.0])]
    for name in BASIS_NAMES:
        for a in GRID_VALUES:
            for b in GRID_VALUES:
                if (a, b) != (0.0, 0.0):
                    codes.append(dict(code_name=f"{name}:{a:+.0f},{b:+.0f}", basis=name, u=[a, b]))
    return codes


def grid_rows(targets, *, starts=8):
    return [dict(level=t["level"], column=t["column"], terrain=t["terrain"], slot=slot, **code)
            for t in targets for code in grid_codes() for slot in range(starts)]


def pack_specs(rows, *, label_prefix, namespace, wall_limit_s=900):
    """One run per 120 rows of the same level; padding rows run the zero code on fresh slots and are ignored."""
    specs = []
    for level in sorted({r["level"] for r in rows}):
        group = [r for r in rows if r["level"] == level]
        for start in range(0, len(group), NUM_ENVS):
            chunk = group[start:start + NUM_ENVS]
            chunk = chunk + [dict(chunk[0], code_name="pad", basis="sag", u=[0.0, 0.0], slot=1000 + i)
                             for i in range(NUM_ENVS - len(chunk))]
            specs.append(dict(
                schema=SPEC_SCHEMA, label=f"{label_prefix}_{level.lower()}_{start // NUM_ENVS:02d}", mode="fixed",
                level=level, heading_hold=True, columns=[r["column"] for r in chunk], reset_table_seed=P1_TABLE_SEED,
                namespace=namespace, slots=[r["slot"] for r in chunk], episodes_per_env=1, max_episode_steps=2000,
                codes=dict(basis=[r["basis"] for r in chunk], u=[list(map(float, r["u"])) for r in chunk],
                           code_name=[r["code_name"] for r in chunk], record_replay=True),
                wall_limit_s=wall_limit_s))
    return specs
