"""Row table and column layout for the Source curriculum grid.

Row 0 is easier than L1, the L1 anchor row equals L1 exactly, the top row
equals L2 exactly, and values are linear in between. Nothing exceeds L2.
"""
from __future__ import annotations

import math
from numbers import Real

NUM_ROWS = 10
ANCHOR_ROWS = {10: (4, 9), 8: (3, 7)}  # num_rows -> (L1 row, L2 row)
TREAD_DEPTH_M = 0.5
EXIT_FLAT_M = 2.0
FRICTION_RANGE = (0.5, 1.0)
FRICTION_REGIONS = 4

_KEYS = ("rough_peak_valley_m", "slope_angle_deg", "step_count", "step_height_m",
         "slope_rough_peak_valley_m", "slope_rough_angle_deg")
EASY = {"rough_peak_valley_m": 0.005, "slope_angle_deg": 1.0, "step_count": 3, "step_height_m": 0.005,
        "slope_rough_peak_valley_m": 0.0025, "slope_rough_angle_deg": 1.0}
L1 = {"rough_peak_valley_m": 0.02, "slope_angle_deg": 5.0, "step_count": 3, "step_height_m": 0.02,
      "slope_rough_peak_valley_m": 0.01, "slope_rough_angle_deg": 5.0}
L2 = {"rough_peak_valley_m": 0.03, "slope_angle_deg": 8.0, "step_count": 5, "step_height_m": 0.035,
      "slope_rough_peak_valley_m": 0.015, "slope_rough_angle_deg": 8.0}


def _check_num_rows(num_rows):
    if type(num_rows) is not int or num_rows not in ANCHOR_ROWS:
        raise ValueError(f"num_rows must be one of {sorted(ANCHOR_ROWS)}, got {num_rows!r}")


def curriculum_row_parameters(row, num_rows=NUM_ROWS):
    _check_num_rows(num_rows)
    if type(row) is not int or not 0 <= row < num_rows:
        raise ValueError(f"row must be an int in [0, {num_rows}), got {row!r}")
    a1, a2 = ANCHOR_ROWS[num_rows]
    if row <= a1:
        low, high, t = EASY, L1, row / a1
    else:
        low, high, t = L1, L2, (row - a1) / (a2 - a1)
    params = {key: low[key] + t * (high[key] - low[key]) for key in _KEYS}
    params["step_count"] = int(math.floor(params["step_count"] + 0.5))
    params.update(tread_depth_m=TREAD_DEPTH_M, exit_flat_length_m=EXIT_FLAT_M,
                  friction_range=FRICTION_RANGE, friction_region_count=FRICTION_REGIONS)
    return params


def row_from_difficulty(difficulty, num_rows=NUM_ROWS):
    _check_num_rows(num_rows)
    if isinstance(difficulty, bool) or not isinstance(difficulty, Real):
        raise ValueError(f"difficulty must be a real number, got {difficulty!r}")
    value = float(difficulty)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"difficulty must be finite and in [0, 1], got {difficulty!r}")
    return min(num_rows - 1, int(math.floor(value * num_rows + 1e-9)))


def track_overrides(family, params):
    """SourceTerrainTrackCfg field overrides for one family at one row."""
    if family in ("flat", "friction_variation"):
        return {}
    if family == "multi_scale_rough":
        return {"rough_peak_valley_m": params["rough_peak_valley_m"],
                "exit_flat_length_m": params["exit_flat_length_m"]}
    if family == "slope":
        return {"slope_angle_deg": params["slope_angle_deg"],
                "exit_flat_length_m": params["exit_flat_length_m"]}
    if family == "multi_step_stairs":
        return {"step_count": params["step_count"], "step_height_m": params["step_height_m"],
                "tread_depth_m": params["tread_depth_m"]}
    if family == "slope_plus_rough":
        return {"slope_angle_deg": params["slope_rough_angle_deg"],
                "rough_peak_valley_m": params["slope_rough_peak_valley_m"],
                "exit_flat_length_m": params["exit_flat_length_m"]}
    raise ValueError(f"unknown Source family: {family!r}")


COLUMN_SPECS = (
    ("slope_uphill", {"family": "slope", "direction": "uphill"}),
    ("slope_downhill", {"family": "slope", "direction": "downhill"}),
    ("slope_cross_slope_left", {"family": "slope", "direction": "cross_slope_left"}),
    ("slope_cross_slope_right", {"family": "slope", "direction": "cross_slope_right"}),
    ("stairs_ascending", {"family": "multi_step_stairs", "direction": "ascending"}),
    ("stairs_descending", {"family": "multi_step_stairs", "direction": "descending"}),
    ("multi_scale_rough_a", {"family": "multi_scale_rough"}),
    ("friction_left_right_bands", {"family": "friction_variation", "friction_layout": "left_right_bands"}),
    ("friction_front_back_bands", {"family": "friction_variation", "friction_layout": "front_back_bands"}),
    ("omni_flat_0", {"family": "flat"}),
    ("omni_flat_1", {"family": "flat"}),
    ("omni_flat_2", {"family": "flat"}),
    ("omni_flat_3", {"family": "flat"}),
    ("flat", {"family": "flat"}),
    ("friction_alternating_regions", {"family": "friction_variation", "friction_layout": "alternating_regions"}),
    ("multi_scale_rough_b", {"family": "multi_scale_rough"}),
    ("slope_rough_uphill", {"family": "slope_plus_rough", "direction": "uphill"}),
    ("slope_rough_downhill", {"family": "slope_plus_rough", "direction": "downhill"}),
    ("slope_rough_cross_slope_left", {"family": "slope_plus_rough", "direction": "cross_slope_left"}),
    ("slope_rough_cross_slope_right", {"family": "slope_plus_rough", "direction": "cross_slope_right"}),
)
NUM_COLUMNS = len(COLUMN_SPECS)  # 20
OMNI_COLUMNS = (9, 10, 11, 12)
FLAT_REGION_COLUMNS = (7, 14)  # first and last geometrically flat column


def column_fields(column):
    if type(column) is not int or not 0 <= column < NUM_COLUMNS:
        raise ValueError(f"column must be an int in [0, {NUM_COLUMNS}), got {column!r}")
    fields = dict(COLUMN_SPECS[column][1])
    if fields["family"] == "friction_variation":
        fields.update(friction_range=FRICTION_RANGE, friction_region_count=FRICTION_REGIONS)
    return fields
