"""P1 code grid, run packing and the registered headroom rules."""
from __future__ import annotations

import numpy as np

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


def collect_rows(results):
    """results: list of result.json dicts from code runs; returns one row per environment's first episode."""
    rows = []
    for report in results:
        case = report["case"]
        spec_codes = case["codes"]
        for record in case["records"]:
            if record["episode_index"] != 0:
                continue
            i = record["env_id"]
            rows.append(dict(level=report["level"], column=record["column"], terrain=record["terrain"],
                             code_name=spec_codes["code_name"][i], basis=spec_codes["basis"][i],
                             u=spec_codes["u"][i], slot=report["slots"][i], completed=bool(record["completed"]),
                             safety_ending=bool(record["safety_ending"]),
                             forward_progress_m=record["forward_progress_m"]))
    return rows


def screen_table(rows):
    groups = {}
    for r in rows:
        if r["code_name"] != "pad":
            groups.setdefault((r["level"], r["column"], r["code_name"]), []).append(r)
    table = []
    for (level, column, name), group in sorted(groups.items()):
        table.append(dict(level=level, column=column, terrain=group[0]["terrain"], code_name=name,
                          basis=group[0]["basis"], u=group[0]["u"], n=len(group),
                          completed=sum(g["completed"] for g in group), safety=sum(g["safety_ending"] for g in group),
                          progress_mean_m=float(np.mean([g["forward_progress_m"] for g in group]))))
    return table


def confirmation_candidates(table, *, min_gain=2, top=3):
    zero = {(r["level"], r["column"]): r for r in table if r["code_name"] == "zero"}
    picks = []
    for key, z in sorted(zero.items()):
        for name in BASIS_NAMES:
            rows = [r for r in table if (r["level"], r["column"]) == key and r["basis"] == name
                    and r["code_name"] != "zero" and r["completed"] - z["completed"] >= min_gain]
            rows.sort(key=lambda r: (-r["completed"], -r["progress_mean_m"], r["code_name"]))
            picks += rows[:top]
    return picks


def confirmation_rows(candidates, targets, *, starts=16):
    rows = [dict(level=t["level"], column=t["column"], terrain=t["terrain"], code_name="zero", basis="sag",
                 u=[0.0, 0.0], slot=s) for t in targets for s in range(starts)]
    rows += [dict(level=c["level"], column=c["column"], terrain=c["terrain"], code_name=c["code_name"],
                  basis=c["basis"], u=c["u"], slot=s) for c in candidates for s in range(starts)]
    return rows


def confirmation_result(table, *, min_gain=4):
    zero = {(r["level"], r["column"]): r for r in table if r["code_name"] == "zero"}
    result = []
    for r in table:
        if r["code_name"] == "zero":
            continue
        z = zero[(r["level"], r["column"])]
        gain = r["completed"] - z["completed"]
        result.append(dict(r, zero_completed=z["completed"], gain=gain,
                           confirmed=gain >= min_gain and r["safety"] <= z["safety"] + 1))
    return result


def choose_basis(confirmed, *, starts=16, max_targets=2):
    best = {}
    for r in confirmed:
        if not r["confirmed"]:
            continue
        key = (r["basis"], r["level"], r["column"])
        if key not in best or (r["gain"], r["progress_mean_m"]) > (best[key]["gain"], best[key]["progress_mean_m"]):
            best[key] = r
    scores = {name: sum(r["gain"] / starts for (b, _, _), r in best.items() if b == name) for name in BASIS_NAMES}
    top = max(scores.values())
    if top <= 0:
        return None
    chosen = "sag" if scores["sag"] == top else next(name for name in BASIS_NAMES if scores[name] == top)
    rows = sorted((r for (b, _, _), r in best.items() if b == chosen), key=lambda r: (-r["gain"], -r["progress_mean_m"]))
    return dict(basis=chosen, scores=scores, targets=[
        dict(level=r["level"], column=r["column"], terrain=r["terrain"], oracle_u=r["u"], oracle_code=r["code_name"],
             confirmed_gain=r["gain"] / starts) for r in rows[:max_targets]])


def ea_decision(c1a, c1b, *, basis_name):
    stats = c1b[basis_name]
    return dict(c1a_go=c1a["p_proj"] >= 0.05 or c1a["p_guard"] >= 0.05,
                c1b_go=stats["informative_fraction_macro"] >= 0.20 and stats["median_prediction_error"] <= 0.30)
