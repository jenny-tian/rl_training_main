"""Exact binomial intervals, paired tests and the registered P0 target rule (no scipy needed)."""
from __future__ import annotations

import math

import numpy as np


def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def _bisect(condition):
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if condition(middle):
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def clopper_pearson(k, n, alpha=0.05):
    if type(n) is not int or n <= 0 or type(k) is not int or not 0 <= k <= n:
        raise ValueError("need integers 0 <= k <= n, n > 0")
    lower = 0.0 if k == 0 else _bisect(lambda p: 1 - _binom_cdf(k - 1, n, p) >= alpha / 2)
    upper = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) <= alpha / 2)
    return lower, upper


def condition_table(records, *, level):
    rows = []
    for column in sorted({r["column"] for r in records}):
        group = [r for r in records if r["column"] == column]
        n, k = len(group), sum(bool(r["completed"]) for r in group)
        low, high = clopper_pearson(k, n)
        distinct = {(r["steps"], round(r["terminal_lateral_m"], 6), round(r["forward_progress_m"], 6)) for r in group}
        rows.append(dict(
            level=level, column=column, terrain=group[0]["terrain"], n=n, completed=k, rate=k / n,
            ci_low=low, ci_high=high, bounds=sum(r["bounds_ending"] for r in group),
            safety=sum(r["safety_ending"] for r in group), timeouts=sum(r["timeout"] for r in group),
            progress_mean_m=float(np.mean([r["forward_progress_m"] for r in group])),
            lateral_peak_mean_m=float(np.mean([r["lateral_peak_m"] for r in group])),
            terminal_abs_yaw_mean_deg=float(np.mean([abs(r["terminal_yaw_deg"]) for r in group])),
            distinct_trajectories=len(distinct)))
    return rows


def paired_mcnemar(records_a, records_b):
    a = {r["env_id"]: bool(r["completed"]) for r in records_a if r["episode_index"] == 0}
    b = {r["env_id"]: bool(r["completed"]) for r in records_b if r["episode_index"] == 0}
    if set(a) != set(b):
        raise ValueError("paired runs cover different environments")
    only_a = sum(a[i] and not b[i] for i in a)
    only_b = sum(b[i] and not a[i] for i in a)
    m = only_a + only_b
    p_value = 1.0 if m == 0 else min(1.0, 2 * _binom_cdf(min(only_a, only_b), m, 0.5))
    return dict(pairs=len(a), both=sum(a[i] and b[i] for i in a), only_first=only_a, only_second=only_b,
                p_value=p_value)


def _mirror(terrain):
    if terrain.endswith("cross_slope_left"):
        return terrain[: -len("left")] + "right"
    if terrain.endswith("cross_slope_right"):
        return terrain[: -len("right")] + "left"
    return None


def select_target_conditions(rows, *, max_success=0.6, max_safety=0.10, min_progress_when_zero=4.0):
    """Research plan section 5, P0 rule. Rows come from condition_table of heading-hold runs."""
    eligible = []
    for row in rows:
        if row["terrain"] == "flat" or row["rate"] > max_success or row["safety"] / row["n"] > max_safety:
            continue
        if row["completed"] == 0 and row["progress_mean_m"] < min_progress_when_zero:
            continue
        eligible.append(dict(level=row["level"], column=row["column"], terrain=row["terrain"], rate=row["rate"],
                             progress_mean_m=row["progress_mean_m"],
                             mechanism="lateral" if "cross_slope" in row["terrain"] else "sagittal"))
    eligible.sort(key=lambda c: (0 if 0.1 <= c["rate"] <= max_success else 1, abs(c["rate"] - 0.3),
                                 -c["progress_mean_m"], c["level"], c["column"]))
    chosen = []
    lateral = next((c for c in eligible if c["mechanism"] == "lateral"), None)
    if lateral is not None:
        chosen.append(lateral)
        twin = next((c for c in eligible if c["level"] == lateral["level"]
                     and c["terrain"] == _mirror(lateral["terrain"])), None)
        if twin is not None:
            chosen.append(twin)
    sagittal = next((c for c in eligible if c["mechanism"] == "sagittal"), None)
    if sagittal is not None:
        chosen.append(sagittal)
    return eligible, chosen
