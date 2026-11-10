"""CPU-only stage-1 qualification rules for explicitly declared final actors."""
from __future__ import annotations

from collections import Counter
import math

from .summary import clopper_pearson

FAMILIES = ("flat", "multi_scale_rough", "slope", "multi_step_stairs", "slope_plus_rough", "friction_variation")
DIRECTIONS = ("uphill", "downhill", "cross_slope_left", "cross_slope_right")
CONDITIONS = ("flat", "multi_scale_rough", *(f"slope_{d}" for d in DIRECTIONS),
              "stairs_ascending", "stairs_descending", *(f"slope_rough_{d}" for d in DIRECTIONS),
              "friction_left_right_bands", "friction_front_back_bands", "friction_alternating_regions")
LEVELS = ("L1", "L2")
RETENTION_KEYS = ("rms_xy_error", "rms_yaw_error", "falls_per_episode")


def family_of(terrain):
    """Map real evaluator labels, testing slope_rough before slope."""
    if terrain not in CONDITIONS:
        raise ValueError(f"unknown condition label: {terrain!r}")
    if terrain.startswith("slope_rough_"):
        return "slope_plus_rough"
    if terrain.startswith("slope_"):
        return "slope"
    if terrain.startswith("stairs_"):
        return "multi_step_stairs"
    if terrain.startswith("friction_"):
        return "friction_variation"
    return terrain


def _number(value, name, *, minimum=None, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (minimum is not None and value < minimum) or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be a finite number in the allowed range")
    return value


def validate_records(records):
    """Require the complete 15 x 32 design and unique (namespace, slot, episode)."""
    if not isinstance(records, list) or len(records) != 480:
        raise ValueError("each level requires exactly 480 records (15 conditions x 32 starts)")
    keys = set()
    for record in records:
        terrain = record["terrain"]
        family_of(terrain)
        column = record["column"]
        if type(column) is not int or column != CONDITIONS.index(terrain):
            raise ValueError("record column and condition label disagree")
        for name in ("completed", "safety_ending", "bounds_ending", "timeout"):
            if type(record[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        if type(record["steps"]) is not int or not 1 <= record["steps"] <= 2000:
            raise ValueError("steps must be an integer in 1..2000")
        _number(record["projected_fraction"], "projected_fraction", minimum=0, maximum=1)
        for name in ("forward_progress_m", "terminal_lateral_m", "terminal_yaw_deg"):
            _number(record[name], name)
        _number(record["lateral_peak_m"], "lateral_peak_m", minimum=0)
        key = tuple(record[name] for name in ("namespace", "slot", "episode_index"))
        if any(type(v) is not int or v < 0 for v in key) or key[2] != 0:
            raise ValueError("invalid reset-key provenance or non-first episode")
        if key in keys:
            raise ValueError(f"duplicate reset-key provenance: {key}")
        keys.add(key)
    if Counter(r["terrain"] for r in records) != Counter({name: 32 for name in CONDITIONS}):
        raise ValueError("each of the 15 expected conditions requires exactly 32 starts")


def family_table(records):
    rows = []
    for family in FAMILIES:
        group = [r for r in records if family_of(r["terrain"]) == family]
        if not group:
            raise ValueError(f"missing family: {family}")
        n, k = len(group), sum(r["completed"] for r in group)
        low, high = clopper_pearson(k, n)
        steps = sum(r["steps"] for r in group)
        rows.append(dict(family=family, n=n, completed=k, rate=k / n, ci_low=low, ci_high=high,
                         safety=sum(r["safety_ending"] for r in group),
                         projected_fraction=sum(r["projected_fraction"] * r["steps"] for r in group) / steps))
    return rows


def condition_counts(records):
    return {name: (sum(r["completed"] for r in records if r["terrain"] == name),
                   sum(r["terrain"] == name for r in records)) for name in CONDITIONS}


def _family_verdict(tables):
    detail = {level: dict(threshold=threshold, families=tables[level])
              for level, threshold in (("L1", .90), ("L2", .85))}
    return {"pass": all(row["rate"] >= detail[level]["threshold"]
                        for level in LEVELS for row in tables[level]), "detail": detail}


def _retention_values(metrics):
    for key in RETENTION_KEYS:
        _number(metrics[key], key, minimum=0, maximum=1 if key == "falls_per_episode" else None)
    return {key: metrics[key] for key in RETENTION_KEYS}


def verdicts(records_by_level, retention, baseline_retention):
    """Q1-Q4, with malformed/incomplete data represented as explicit failures."""
    result = {}
    try:
        if set(records_by_level) != set(LEVELS):
            raise ValueError("both L1 and L2 are required")
        for level in LEVELS:
            validate_records(records_by_level[level])
        q1 = _family_verdict({level: family_table(records_by_level[level]) for level in LEVELS})
        result["Q1"] = q1
        conditions = {level: {name: dict(completed=k, n=n, minimum_completed=24)
                              for name, (k, n) in condition_counts(records_by_level[level]).items()}
                      for level in LEVELS}
        result["Q2"] = {"pass": all(row["completed"] >= 24 for rows in conditions.values() for row in rows.values()),
                        "detail": conditions}
        safety = {level: dict(safety=sum(r["safety_ending"] for r in records_by_level[level]), n=480, maximum=24)
                  for level in LEVELS}
        result["Q3"] = {"pass": all(row["safety"] <= 24 for row in safety.values()), "detail": safety}
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        for name in ("Q1", "Q2", "Q3"):
            result[name] = {"pass": False, "detail": {"error": str(error)}}
    try:
        actual, baseline = _retention_values(retention), _retention_values(baseline_retention)
        limits = {key: baseline[key] + .01 if key == "falls_per_episode" else baseline[key] * 1.10
                  for key in RETENTION_KEYS}
        result["Q4"] = {"pass": all(actual[key] <= limits[key] for key in RETENTION_KEYS),
                        "detail": dict(actual=actual, baseline=baseline, limits=limits)}
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        result["Q4"] = {"pass": False, "detail": {"error": str(error)}}
    return result


def choose_seed(results):
    """Select only qualifying final actors; results contain seed, verdicts and pooled counts."""
    qualified = []
    for item in results:
        if not all(item.get("verdicts", {}).get(f"Q{i}", {}).get("pass") is True for i in range(1, 5)):
            continue
        seed = item["seed"]
        if type(seed) is not int or seed not in (42, 43):
            raise ValueError("only registered seeds 42 and 43 can be selected")
        rate = _number(item["l2_completion_rate"], "L2 completion rate", minimum=0, maximum=1)
        safety = item["total_safety_endings"]
        if type(safety) is not int or not 0 <= safety <= 960:
            raise ValueError("total safety endings must be an integer in 0..960")
        qualified.append((seed, rate, safety))
    if not qualified:
        return None, "No qualified seed: neither final checkpoint passes all of Q1-Q4."
    seed, _, _ = min(qualified, key=lambda row: (-row[1], row[2], row[0] != 42))
    return seed, "Qualified final checkpoint; ranked by higher L2 pooled completion, fewer total safety endings, then seed 42."
