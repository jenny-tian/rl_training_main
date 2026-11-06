"""Run specifications for the EA-LPL runner (P0 subset; later phases extend validate_spec)."""
from __future__ import annotations

SPEC_SCHEMA = "ea_lpl_run_spec_v1"
NUM_ENVS = 120
LEVELS = ("L1", "L2")
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
        raise ValueError("level must be L1/L2 and heading_hold boolean")
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
    if spec["codes"] is not None:
        raise ValueError("residual codes are added in P1")
    if type(spec["wall_limit_s"]) not in (int, float) or spec["wall_limit_s"] <= 0:
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
