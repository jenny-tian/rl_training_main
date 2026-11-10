# Stage 1: Source curriculum PPO from model_1998 — implementation plan

> For agentic workers: execute with superpowers:subagent-driven-development. Tasks 1–4 and 6–8 go to implementer subagents; Tasks 5 and 9 run inline by the controller.

**Goal:** continue PPO from model_1998 (actor, critic, optimizer, iteration counter) on a 10×20 Source curriculum grid that tops out at exactly L2, then qualify the result (θ1) on all six Source families at L1 and L2, check flat retention, and preview L3.

**Spec:** `docs/lite3-ea-lpl-research-plan-v2-20261109.md` §5 (S1-1 … S1-11, §5.4–§5.9).

**Architecture:** an Isaac-free core (`source_curriculum` package: row table, column layout, outcome logic, safety formula, stats) carries all the decisions and is unit-tested on CPU. Thin Isaac Lab glue (command term, termination terms, curriculum term, terrain function, env cfg) calls the core. The existing P0 evaluator gains an optional actor checkpoint so qualification reuses it unchanged.

## Global constraints

- Worktree `W=/home/tjx/rl_training_main/.worktrees/lite3-ea-lpl-20261106`, branch `codex/lite3-ea-lpl-20261106`. Start every Bash call with `cd W &&`. Verify the branch before committing.
- Python: `/home/tjx/anaconda3/envs/env_isaaclab/bin/python` with `PYTHONPATH=$PWD/source/rl_training` (the editable install otherwise imports `/home/tjx/rl_training`).
- Test command (the typeguard plugin breaks collection without `-p no:typeguard`):
  `cd W && PYTHONPATH=$PWD/source/rl_training /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -p no:typeguard <paths> -q`
  Baseline: `tests/ea_lpl` 39 passed.
- Test basenames must be unique (no `__init__.py`, no root conftest). Isaac-free tests must not import `rl_training.tasks...terrains` (its `__init__` imports isaaclab); path-load modules under a unique `sys.modules` name.
- Production code raises `ValueError` (never `assert`). Non-finite inputs fail closed.
- Never `git add -A`. `.superpowers/` and `logs/` are never committed. Add explicit paths only.
- Never use bare `git stash`. Never edit `source/` or `scripts/` while a GPU eval that needs a clean tree is running (`ea_lpl_run.py` clean-tree guard).
- GPU: only GPU0/GPU1, only with ≥ 12288 MiB free.
- Interfaces quoted in this plan come from reading the code on 2026-11-09. If a name or signature differs, follow the real code, keep the behaviour, and note the difference in the commit message.
- Commit after each task; the controller pushes after each review passes.

Aliases: `S=source/rl_training/rl_training`, `V=$S/tasks/manager_based/locomotion/velocity`, `C=$V/config/quadruped/deeprobotics_lite3`, `T=$V/terrains`.

## Rulings (fixed; do not reopen)

| ID | Ruling |
|---|---|
| R1 | Completion distance 9.25 m (scanner boundary with margin 0.1 caps root local x at about 9.43–9.53). |
| R2 | The evaluator's unsafe formula becomes a training termination; control timeout is fed as zeros. |
| R3 | Omni envs get a random row in 1–8 at every reset, including the initial one. |
| R4 | Tile envs get their command at resample time (vx U(0.25, 0.5), vy 0, heading target 0, heading loop on, never standing). |
| R5 | Commanded distance D accumulates in `_update_metrics`, is zeroed in command reset, and is read by the curriculum before that reset. |
| R6 | Same step: failure beats completion; completion beats timeout. |
| R7 | Plain `ManagerBasedRLEnv` entry point. |
| R8 | Flat retention runs with action clip 100 and no safety filter, like training. |
| R9 | Non-finite safety inputs or contact forces count as unsafe. |
| R10 | Tile bounds use a new masked term (tile envs only); the inherited `source_terrain_tile_bounds` is set to None. |
| R11 | Timeout-driven level-downs caused by `init_at_random_ep_len` are accepted. |
| R12 | Evaluation geometry seed 30000 (unchanged evaluator). |
| R13 | Scanner footprint margin 0.1; keep `scanner_boundary`'s own raises. |
| R14 | Robot init pos `(0.0, 0.0, DEEPROBOTICS_LITE3_CFG.init_state.pos[2])`. |
| R15 | Per-family outcome EMA decay 0.995. |
| R16 | Safety term uses `Lite3IsaacLabAdapter(env, safety_height_source="base_scan")` and `SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4, device=env.device))`, built lazily on first call. |
| R17 | Stock `MultiMeshRayCaster` for `height_scanner_base`. |
| R18 | Env cfg field `reward_gait_level: float = 1.0`. |
| R19 | Material randomization keeps `body_names=".*"` and 1024 buckets and adds `make_consistent=True`. |

## Review focus (every reviewer checks these)

1. No row exceeds L2 on any parameter; exactly one friction palette across friction columns.
2. At the initial reset nothing moves; omni rows are in 1–8 and tile rows in 0–4.
3. Same-step completion and failure counts as failure (level down).
4. Bounds and completion apply to tile envs only; omni truncation applies to omni envs only.
5. Resume continues at iteration 1998 with lr 3.844e-4; P0 specs behave exactly as before; new checkpoints load through `expected_sha256`.

## File map

| File | Task | Kind |
|---|---|---|
| `$S/source_curriculum/__init__.py`, `rows.py`, `logic.py` | 1 | new, Isaac-free |
| `tests/source_curriculum/test_curriculum_rows.py`, `test_curriculum_logic.py` | 1 | new |
| `$V/mdp/source_curriculum.py`, `$V/mdp/__init__.py` (one import line) | 2 | new / edit |
| `$T/source_curriculum_terrain.py`, `$T/__init__.py` (export) | 3 | new / edit |
| `$C/source_curriculum_env_cfg.py`, `$C/__init__.py`, `$C/agents/rsl_rl_ppo_cfg.py` | 3 | new / edit |
| `scripts/tools/source_curriculum_check.py` | 4 | new, GPU |
| `$S/ea_lpl/specs.py`, `$S/source_scan/frozen_actor.py`, `scripts/tools/ea_lpl_run.py`, `$C/ea_lpl_env_cfg.py`, `scripts/tools/stage1_make_specs.py` + tests | 6 | edit / new |
| `scripts/tools/stage1_flat_retention.py` | 7 | new, GPU |
| `$S/ea_lpl/stage1.py`, `scripts/tools/stage1_qualification.py`, `scripts/tools/stage1_curves.py` + tests | 8 | new |
| `docs/lite3-two-stage-stage1-20261109/` | 9 | new |

---

## Task 1: Isaac-free curriculum core

**Files:** create `$S/source_curriculum/__init__.py`, `$S/source_curriculum/rows.py`, `$S/source_curriculum/logic.py`, `tests/source_curriculum/test_curriculum_rows.py`, `tests/source_curriculum/test_curriculum_logic.py`.

`__init__.py`: one docstring line, `"""Isaac-free Source curriculum core (row table, column layout, outcome logic)."""`. No imports.

### rows.py

```python
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
```

### logic.py

```python
"""Torch-only outcome, command and statistics logic for the Source curriculum."""
from __future__ import annotations

import math

import torch

from .rows import COLUMN_SPECS, NUM_COLUMNS, OMNI_COLUMNS

TILE_FAMILY_NAMES = ("flat", "multi_scale_rough", "slope", "multi_step_stairs", "slope_plus_rough",
                     "friction_variation")
FAMILY_NAMES = TILE_FAMILY_NAMES + ("omni_flat",)
OMNI_FAMILY_ID = 6
OUTCOME_NAMES = ("completion", "failure", "timeout")


def column_family_ids(omni_columns=OMNI_COLUMNS, device="cpu"):
    ids = [TILE_FAMILY_NAMES.index(spec["family"]) for _, spec in COLUMN_SPECS]
    for column in omni_columns:
        ids[column] = OMNI_FAMILY_ID
    return torch.tensor(ids, dtype=torch.long, device=device)


def wrap_to_pi(angle):
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def apply_tile_commands(vel_command_b, heading_target, is_heading_env, is_standing_env, ids, *,
                        heading, lin_vel_x, stiffness, ang_range, generator=None):
    """Overwrite the command of tile envs `ids` in place. `heading` is heading_w[ids]."""
    if ids.numel() == 0:
        return
    low, high = float(lin_vel_x[0]), float(lin_vel_x[1])
    if not low <= high:
        raise ValueError("lin_vel_x must be (low, high) with low <= high")
    u = torch.rand(ids.numel(), device=vel_command_b.device, generator=generator)
    vel_command_b[ids, 0] = low + (high - low) * u
    vel_command_b[ids, 1] = 0.0
    heading_target[ids] = 0.0
    is_heading_env[ids] = True
    is_standing_env[ids] = False
    vel_command_b[ids, 2] = torch.clamp(stiffness * wrap_to_pi(0.0 - heading),
                                        min=float(ang_range[0]), max=float(ang_range[1]))


def resolve_outcomes(completed, failed, timed_out):
    failed = failed.bool()
    completed = completed.bool() & ~failed
    timed_out = timed_out.bool() & ~failed & ~completed
    return completed, failed, timed_out


def curriculum_moves(completed, failed, timed_out, progress, commanded_distance):
    completed, failed, timed_out = resolve_outcomes(completed, failed, timed_out)
    move_up = completed
    move_down = failed | (timed_out & (progress < 0.5 * commanded_distance))
    return move_up, move_down


def omni_levels(count, *, num_rows, device, generator=None):
    if num_rows < 3:
        raise ValueError("num_rows must be at least 3")
    return torch.randint(1, num_rows - 1, (count,), device=device, generator=generator)


def omni_truncation_bounds(*, tile_size=(12.0, 4.0), num_rows=10, flat_region_columns=(7, 14), inset_m=1.0):
    """World (x_min, x_max, y_min, y_max) of the flat region shrunk by inset_m (no border)."""
    size_x, size_y = tile_size
    x_half = num_rows * size_x / 2.0
    y_origin = -NUM_COLUMNS * size_y / 2.0
    first, last = flat_region_columns
    return (-x_half + inset_m, x_half - inset_m,
            y_origin + first * size_y + inset_m, y_origin + (last + 1) * size_y - inset_m)


def outside_bounds(xy, bounds):
    x, y = xy[:, 0], xy[:, 1]
    inside = (x >= bounds[0]) & (x <= bounds[1]) & (y >= bounds[2]) & (y <= bounds[3])
    return ~inside  # NaN compares False, so NaN is outside


def safety_unsafe(projected_gravity_xy, base_height, joint_vel, nonfoot_contact, control_timed_out,
                  contact_steps, limits):
    """Same formula as SourceEpisodeLifecycle; returns (unsafe, next_contact_steps)."""
    n = contact_steps.shape[0]
    contact = nonfoot_contact.reshape(n, -1).bool().any(-1)
    next_steps = torch.where(contact, contact_steps + 1, torch.zeros_like(contact_steps))
    gxy = projected_gravity_xy.reshape(n, -1)
    height = base_height.reshape(n, -1)[:, 0]
    jv = joint_vel.reshape(n, -1)
    finite = torch.isfinite(gxy).all(-1) & torch.isfinite(height) & torch.isfinite(jv).all(-1)
    unsafe = ((gxy.norm(dim=-1) > limits.max_projected_gravity_xy)
              | (height < limits.min_base_height)
              | (jv.abs().amax(-1) > limits.max_joint_velocity)
              | (next_steps > limits.nonfoot_contact_grace_steps)
              | control_timed_out.reshape(n, -1).bool().any(-1))
    return unsafe | ~finite, next_steps


class FamilyStats:
    """EMA of completion / failure / timeout per family, plus level summaries."""

    def __init__(self, device, decay=0.995):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.rates = torch.zeros(len(FAMILY_NAMES), 3, device=device)
        self.seen = torch.zeros(len(FAMILY_NAMES), dtype=torch.bool, device=device)

    def update(self, family_ids, completed, failed, timed_out):
        completed, failed, timed_out = resolve_outcomes(completed, failed, timed_out)
        outcome = torch.stack([completed, failed, timed_out], dim=-1).float()
        has = outcome.sum(-1) > 0
        ids, outcome = family_ids[has], outcome[has]
        for family in ids.unique().tolist():
            batch = outcome[ids == family]
            weight = self.decay ** batch.shape[0]
            mean = batch.mean(0)
            if self.seen[family]:
                self.rates[family] = weight * self.rates[family] + (1.0 - weight) * mean
            else:
                self.rates[family] = mean
                self.seen[family] = True

    def report(self, levels, env_family_ids, top_row):
        out = {}
        for family, name in enumerate(FAMILY_NAMES):
            mask = env_family_ids == family
            if mask.any():
                out[f"{name}/mean_level"] = float(levels[mask].float().mean())
                out[f"{name}/top_row_fraction"] = float((levels[mask] == top_row).float().mean())
            for k, outcome in enumerate(OUTCOME_NAMES):
                out[f"{name}/{outcome}_ema"] = float(self.rates[family, k])
        return out
```

### Tests (write first, see them fail, then implement)

`tests/source_curriculum/test_curriculum_rows.py`:

```python
import ast
import importlib.util
import math
import sys
from pathlib import Path

import pytest

from rl_training.source_curriculum.rows import (
    ANCHOR_ROWS, COLUMN_SPECS, FRICTION_RANGE, L1, L2, OMNI_COLUMNS, column_fields,
    curriculum_row_parameters, row_from_difficulty, track_overrides)

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"


def _level_candidates():
    tree = ast.parse((V / "config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py").read_text())
    for node in ast.walk(tree):
        targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
        if any(getattr(t, "id", None) == "_LEVEL_CANDIDATES" for t in targets if t is not None):
            return ast.literal_eval(node.value)
    raise AssertionError("_LEVEL_CANDIDATES not found")


def test_anchors_match_source_level_candidates():
    cand = _level_candidates()
    for anchor, level in ((L1, "L1"), (L2, "L2")):
        for key, value in anchor.items():
            assert cand[level][key] == pytest.approx(value), (level, key)
    assert tuple(cand["L2"]["friction_range"]) == FRICTION_RANGE
    for num_rows, (a1, a2) in ANCHOR_ROWS.items():
        p1, p2 = curriculum_row_parameters(a1, num_rows), curriculum_row_parameters(a2, num_rows)
        for key in L1:
            assert p1[key] == pytest.approx(L1[key]) and p2[key] == pytest.approx(L2[key])
        assert p2["exit_flat_length_m"] == cand["L2"]["exit_flat_length_m"]
        assert p2["tread_depth_m"] == cand["L2"]["tread_depth_m"]
```

Note for the implementer: open `_LEVEL_CANDIDATES` and use its real key names in this test (the plan's keys are `rough_peak_valley_m`, `slope_angle_deg`, `step_count`, `step_height_m`, `slope_rough_peak_valley_m`, `slope_rough_angle_deg`, `exit_flat_length_m`, `tread_depth_m`, `friction_range`). If the file uses different keys, map them in the test with an explicit dict; do not rename the plan's keys.

```python
def test_spec_row_table():
    rows = [curriculum_row_parameters(r) for r in range(10)]
    assert [p["slope_angle_deg"] for p in rows] == pytest.approx([1, 2, 3, 4, 5, 5.6, 6.2, 6.8, 7.4, 8])
    assert [p["step_count"] for p in rows] == [3, 3, 3, 3, 3, 3, 4, 4, 5, 5]
    assert [100 * p["step_height_m"] for p in rows] == pytest.approx(
        [0.5, 0.875, 1.25, 1.625, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5])
    assert 100 * rows[0]["slope_rough_peak_valley_m"] == pytest.approx(0.25)
    assert 100 * rows[9]["slope_rough_peak_valley_m"] == pytest.approx(1.5)


@pytest.mark.parametrize("num_rows", [8, 10])
def test_monotonic_and_capped_at_l2(num_rows):
    rows = [curriculum_row_parameters(r, num_rows) for r in range(num_rows)]
    for key in L2:
        values = [p[key] for p in rows]
        assert values == sorted(values) and max(values) <= L2[key] + 1e-12
    assert all(type(p["step_count"]) is int for p in rows)
    assert {tuple(p["friction_range"]) for p in rows} == {FRICTION_RANGE}


def test_row_from_difficulty_edges_and_rejections():
    assert row_from_difficulty(0.0) == 0
    assert row_from_difficulty(0.4) == 4
    assert row_from_difficulty(0.39999999999) == 4  # epsilon absorbs float error
    assert row_from_difficulty(0.99) == 9
    assert row_from_difficulty(1.0) == 9
    assert row_from_difficulty(0.5, 8) == 4
    for bad in (True, -0.01, 1.01, math.nan, math.inf, "0.5", None):
        with pytest.raises(ValueError):
            row_from_difficulty(bad)
    with pytest.raises(ValueError):
        row_from_difficulty(0.5, 9)
    for bad in (-1, 10, 1.0, True):
        with pytest.raises(ValueError):
            curriculum_row_parameters(bad)


def test_track_overrides():
    p = curriculum_row_parameters(9)
    assert track_overrides("flat", p) == {} and track_overrides("friction_variation", p) == {}
    assert track_overrides("slope_plus_rough", p) == {
        "slope_angle_deg": 8.0, "rough_peak_valley_m": pytest.approx(0.015), "exit_flat_length_m": 2.0}
    assert track_overrides("multi_step_stairs", p)["step_count"] == 5
    with pytest.raises(ValueError, match="unknown Source family"):
        track_overrides("boxes", p)


def test_column_layout():
    assert len(COLUMN_SPECS) == 20 and OMNI_COLUMNS == (9, 10, 11, 12)
    families = [column_fields(c)["family"] for c in range(20)]
    assert all(f in ("flat", "friction_variation") for f in families[7:15])
    friction = [column_fields(c) for c in range(20) if families[c] == "friction_variation"]
    assert {(f["friction_range"], f["friction_region_count"]) for f in friction} == {(FRICTION_RANGE, 4)}
    assert sorted(set(families)) == sorted(["flat", "multi_scale_rough", "slope", "multi_step_stairs",
                                            "slope_plus_rough", "friction_variation"])
```

Add one geometry test that path-loads `$T/source_terrain_tracks.py` as `source_curriculum_geometry_under_test` (pattern: `importlib.util.spec_from_file_location`, then `sys.modules[name] = module` before `exec_module`). For rows {0, 4, 9} and every column, build the track cfg with `column_fields(c)` plus `track_overrides(family, curriculum_row_parameters(row))` and run the module's validation function (the one at `source_terrain_tracks.py:86–125`; find its name there). Assert it passes for all 60 cells and that columns 7–14 produce zero height everywhere if the module exposes a height-field builder; if it does not, only assert validation passes. If `source_terrain_tracks.py` imports isaaclab at module level, skip this test with `pytest.importorskip("isaaclab")` and say so in the commit message.

`tests/source_curriculum/test_curriculum_logic.py`:

```python
import math
from types import SimpleNamespace

import pytest
import torch

from rl_training.source_curriculum.logic import (
    OMNI_FAMILY_ID, FamilyStats, apply_tile_commands, column_family_ids, curriculum_moves,
    omni_levels, omni_truncation_bounds, outside_bounds, resolve_outcomes, safety_unsafe)

LIMITS = SimpleNamespace(max_projected_gravity_xy=0.65, min_base_height=0.20, max_joint_velocity=30.0,
                         nonfoot_contact_grace_steps=10)
B = lambda *v: torch.tensor(v, dtype=torch.bool)


def test_failure_beats_completion_beats_timeout():
    c, f, t = resolve_outcomes(B(1, 1, 0, 0), B(1, 0, 0, 0), B(1, 1, 1, 0))
    assert c.tolist() == [False, True, False, False]
    assert f.tolist() == [True, False, False, False]
    assert t.tolist() == [False, False, True, False]


def test_moves():
    up, down = curriculum_moves(B(1, 1, 0, 0, 0), B(1, 0, 0, 0, 0), B(0, 0, 1, 1, 0),
                                torch.tensor([9.3, 9.3, 1.0, 6.0, 0.0]), torch.full((5,), 10.0))
    assert up.tolist() == [False, True, False, False, False]
    assert down.tolist() == [True, False, True, False, False]


def test_initial_reset_has_no_moves():
    z = torch.zeros(6, dtype=torch.bool)
    up, down = curriculum_moves(z, z, z, torch.zeros(6), torch.zeros(6))
    assert not up.any() and not down.any()


def test_omni_levels_and_bounds():
    g = torch.Generator().manual_seed(0)
    levels = omni_levels(10000, num_rows=10, device="cpu", generator=g)
    assert levels.min() == 1 and levels.max() == 8
    assert omni_truncation_bounds() == (-59.0, 59.0, -11.0, 19.0)
    xy = torch.tensor([[0.0, 0.0], [59.5, 0.0], [0.0, -11.5], [math.nan, 0.0]])
    assert outside_bounds(xy, omni_truncation_bounds()).tolist() == [False, True, True, True]


def test_family_ids():
    ids = column_family_ids()
    assert ids.shape == (20,) and ids[9:13].eq(OMNI_FAMILY_ID).all() and ids[13] == 0


def test_tile_commands():
    n = 6
    vel = torch.full((n, 3), 7.0); target = torch.full((n,), 2.0)
    is_h = torch.zeros(n, dtype=torch.bool); is_s = torch.ones(n, dtype=torch.bool)
    ids = torch.tensor([1, 3, 4]); heading = torch.tensor([0.1, -3.0, 3.0])
    apply_tile_commands(vel, target, is_h, is_s, ids, heading=heading, lin_vel_x=(0.25, 0.5),
                        stiffness=0.5, ang_range=(-0.8, 0.8), generator=torch.Generator().manual_seed(1))
    assert ((vel[ids, 0] >= 0.25) & (vel[ids, 0] <= 0.5)).all() and vel[ids, 1].eq(0).all()
    assert vel[ids, 2].tolist() == pytest.approx([-0.05, 0.8, -0.8])
    assert is_h[ids].all() and not is_s[ids].any() and target[ids].eq(0).all()
    assert vel[0].tolist() == [7.0, 7.0, 7.0] and is_s[0]  # untouched


def test_safety_formula_and_nonfinite():
    n = 5
    steps = torch.zeros(n, dtype=torch.long)
    gxy = torch.tensor([[0.0, 0.0], [0.6, 0.3], [0.0, 0.0], [0.0, 0.0], [math.nan, 0.0]])
    h = torch.tensor([[0.3], [0.3], [0.19], [0.3], [0.3]])
    jv = torch.zeros(n, 12); jv[3, 5] = 31.0
    contact = torch.zeros(n, 1, dtype=torch.bool); timeout = torch.zeros(n, 1, dtype=torch.bool)
    unsafe, steps = safety_unsafe(gxy, h, jv, contact, timeout, steps, LIMITS)
    assert unsafe.tolist() == [False, True, True, True, True]
    contact[0] = True
    for k in range(11):
        unsafe, steps = safety_unsafe(torch.zeros(n, 2), torch.full((n, 1), 0.3), torch.zeros(n, 12),
                                      contact, timeout, steps, LIMITS)
    assert unsafe[0] and steps[0] == 11 and not unsafe[1]


def test_family_stats():
    stats = FamilyStats("cpu", decay=0.5)
    fam = torch.tensor([0, 0, 2, OMNI_FAMILY_ID])
    stats.update(fam, B(1, 0, 0, 0), B(0, 1, 0, 0), B(0, 0, 0, 1))
    report = stats.report(torch.tensor([9, 3, 4, 5]), fam, top_row=9)
    assert report["flat/completion_ema"] == pytest.approx(0.5)
    assert report["flat/top_row_fraction"] == pytest.approx(0.5)
    assert report["slope/completion_ema"] == 0.0  # no outcome yet
    assert report["omni_flat/timeout_ema"] == 1.0
```

Add a parity test against the real evaluator: find `class SourceEpisodeLifecycle` (`grep -rn "class SourceEpisodeLifecycle" source/`), drive it and `safety_unsafe` with the same 20 random steps of SafetyState-like inputs (seeded), and assert the per-step unsafe flags match. Build the SafetyState from `rl_training.online_adaptation.types.SafetyState` and the limits from `rl_training.online_adaptation.safety.SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4))`. If the lifecycle API needs inputs that cannot be built without Isaac Sim, write the test against its pure step function instead and state which one in the commit.

Run: `cd W && PYTHONPATH=$PWD/source/rl_training /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -p no:typeguard tests/source_curriculum -q` → all pass. Commit `feat(stage1): Isaac-free Source curriculum core`.

---

## Task 2: Isaac Lab terms (`$V/mdp/source_curriculum.py`)

**Files:** create `$V/mdp/source_curriculum.py`; append `from .source_curriculum import *  # noqa: F401, F403` as the last line of `$V/mdp/__init__.py`.

Before coding, open and confirm: `$V/mdp/commands.py` (`UniformThresholdVelocityCommand`, its cfg, `reset`, `_resample_command`, `_update_metrics`), the module that defines `source_terrain_tile_bounds` (`grep -rn "def source_terrain_tile_bounds" $V`), and `$S/online_adaptation/isaaclab_adapter.py` (`Lite3IsaacLabAdapter`, `safety_state`, attribute holding the contact sensor). Import `source_terrain_tile_bounds` from its defining module, not from the `mdp` package (avoids a circular import).

```python
"""Source curriculum terms: tile commands, masked terminations, safety and level updates."""
from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

from rl_training.source_curriculum.logic import (
    FamilyStats, OMNI_FAMILY_ID, apply_tile_commands, column_family_ids, curriculum_moves,
    omni_levels, outside_bounds, safety_unsafe)
from rl_training.source_curriculum.rows import OMNI_COLUMNS

from .commands import UniformThresholdVelocityCommand, UniformThresholdVelocityCommandCfg
from .<defining_module> import source_terrain_tile_bounds  # fill in from grep

__all__ = ["SourceCurriculumVelocityCommand", "SourceCurriculumVelocityCommandCfg",
           "source_curriculum_tile_bounds", "source_curriculum_completion",
           "source_curriculum_omni_truncation", "SourceCurriculumSafety", "SourceCurriculumLevels"]


def _omni_mask(env, omni_columns):
    types = env.scene.terrain.terrain_types
    columns = torch.as_tensor(tuple(omni_columns), device=types.device, dtype=types.dtype)
    return torch.isin(types, columns)


def _env_ids(env_ids, num_envs, device):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


class SourceCurriculumVelocityCommand(UniformThresholdVelocityCommand):
    """Omni envs keep the Flat command mix; tile envs walk straight along +x."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.tile_mask = ~_omni_mask(env, cfg.omni_columns)
        self.commanded_distance = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        self.commanded_distance[slice(None) if env_ids is None else env_ids] = 0.0
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = _env_ids(env_ids, self.num_envs, self.device)
        tile = ids[self.tile_mask[ids]]
        apply_tile_commands(self.vel_command_b, self.heading_target, self.is_heading_env, self.is_standing_env,
                            tile, heading=self.robot.data.heading_w[tile], lin_vel_x=self.cfg.tile_lin_vel_x,
                            stiffness=self.cfg.heading_control_stiffness, ang_range=self.cfg.ranges.ang_vel_z)

    def _update_metrics(self):
        super()._update_metrics()
        self.commanded_distance += self.vel_command_b[:, 0] * self._env.step_dt


@configclass
class SourceCurriculumVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    class_type: type = SourceCurriculumVelocityCommand
    omni_columns: tuple = OMNI_COLUMNS
    tile_lin_vel_x: tuple = (0.25, 0.5)


def source_curriculum_tile_bounds(env, asset_cfg, size, flat_length_m, collision_radius_m, numerical_margin_m,
                                  scanner_footprint_margin_m, omni_columns):
    base = source_terrain_tile_bounds(env, asset_cfg=asset_cfg, size=size, flat_length_m=flat_length_m,
                                      collision_radius_m=collision_radius_m, numerical_margin_m=numerical_margin_m,
                                      scanner_footprint_margin_m=scanner_footprint_margin_m)
    return base & ~_omni_mask(env, omni_columns)


def source_curriculum_completion(env, distance_m, omni_columns):
    robot = env.scene["robot"]
    progress = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return (progress >= distance_m) & ~_omni_mask(env, omni_columns)


def source_curriculum_omni_truncation(env, bounds, omni_columns):
    xy = env.scene["robot"].data.root_pos_w[:, :2]
    return outside_bounds(xy, bounds) & _omni_mask(env, omni_columns)


class SourceCurriculumSafety(ManagerTermBase):
    """Evaluator unsafe formula as a termination (R2, R9, R16)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._adapter = None
        self._limits = None

    def reset(self, env_ids=None):
        self.contact_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env):
        if self._adapter is None:
            from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
            from rl_training.online_adaptation.safety import SafetyLimits
            self._adapter = Lite3IsaacLabAdapter(env, safety_height_source="base_scan")
            self._limits = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4, device=env.device))
        timed_out = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
        state = self._adapter.safety_state(timed_out)
        unsafe, self.contact_steps = safety_unsafe(state.projected_gravity_xy, state.base_height, state.joint_vel,
                                                   state.nonfoot_contact, state.control_timed_out,
                                                   self.contact_steps, self._limits)
        forces = self._adapter.contact_sensor.data.net_forces_w
        return unsafe | ~torch.isfinite(forces).flatten(1).all(-1)


class SourceCurriculumLevels(ManagerTermBase):
    """Moves tile envs by outcome; re-draws omni rows 1..num_rows-2 at every reset (R3, R5, R6)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        omni_columns = cfg.params["omni_columns"]
        self.family_by_column = column_family_ids(omni_columns, device=env.device)
        self.env_family_ids = self.family_by_column[env.scene.terrain.terrain_types.long()]
        self.omni = self.env_family_ids == OMNI_FAMILY_ID
        self.stats = FamilyStats(env.device)

    def __call__(self, env, env_ids, omni_columns):
        terrain = env.scene.terrain
        ids = _env_ids(env_ids, env.num_envs, env.device)
        tile = ids[~self.omni[ids]]
        omni = ids[self.omni[ids]]
        if tile.numel():
            tm = env.termination_manager
            completed = tm.get_term("source_curriculum_completion")[tile]
            failed = tm.terminated[tile]
            timed_out = tm.get_term("time_out")[tile]
            progress = env.scene["robot"].data.root_pos_w[tile, 0] - terrain.env_origins[tile, 0]
            distance = env.command_manager.get_term("base_velocity").commanded_distance[tile]
            up, down = curriculum_moves(completed, failed, timed_out, progress, distance)
            self.stats.update(self.env_family_ids[tile], completed, failed, timed_out)
            terrain.update_env_origins(tile, up, down)
        if omni.numel():
            levels = omni_levels(omni.numel(), num_rows=terrain.max_terrain_level, device=env.device)
            terrain.terrain_levels[omni] = levels
            terrain.env_origins[omni] = terrain.terrain_origins[levels, terrain.terrain_types[omni]]
        return self.stats.report(terrain.terrain_levels, self.env_family_ids, top_row=terrain.max_terrain_level - 1)
```

Notes for the implementer:
- `tm.terminated` holds the non-timeout terms (illegal_contact, bad_orientation_2, safety, tile bounds), which is exactly failure. Completion and omni truncation are `time_out=True` terms, so they are not in `terminated`.
- At the very first `env.reset()` all termination buffers are zero, so tile envs do not move (review focus 2).
- If the termination manager is created after the curriculum manager in this Isaac Lab version, `get_term` still works at call time; do not touch managers in `__init__` besides the terrain.
- Check the real signature of `source_terrain_tile_bounds`; pass every argument by keyword.

**Tests:** these terms need Isaac Sim, so they are covered by Task 4. In this task only run `python -c "import ast; ast.parse(open('$V/mdp/source_curriculum.py').read())"` and the Task 1 tests. Commit `feat(stage1): Source curriculum Isaac Lab terms`.

---

## Task 3: terrain function, env cfg, registration, runner

**Files:** create `$T/source_curriculum_terrain.py`, `$C/source_curriculum_env_cfg.py`; edit `$T/__init__.py` (export the two new names), `$C/__init__.py` (register), `$C/agents/rsl_rl_ppo_cfg.py` (runner).

### `$T/source_curriculum_terrain.py`

```python
"""Source track whose parameters follow the curriculum row of each tile."""
from __future__ import annotations

import copy

from isaaclab.utils import configclass

from rl_training.source_curriculum.rows import (
    NUM_COLUMNS, NUM_ROWS, column_fields, curriculum_row_parameters, row_from_difficulty, track_overrides)

from .source_terrain_isaac import SourceTerrainTrackCfg, source_terrain_track


def source_curriculum_track(difficulty, cfg):
    row = row_from_difficulty(difficulty, cfg.curriculum_num_rows)
    params = curriculum_row_parameters(row, cfg.curriculum_num_rows)
    track = copy.deepcopy(cfg)
    for key, value in track_overrides(cfg.family, params).items():
        setattr(track, key, value)
    return source_terrain_track(difficulty, track)


@configclass
class SourceCurriculumTrackCfg(SourceTerrainTrackCfg):
    function = source_curriculum_track
    curriculum_num_rows: int = NUM_ROWS


def source_curriculum_sub_terrains(num_rows=NUM_ROWS):
    return {f"c{column:02d}": SourceCurriculumTrackCfg(proportion=1.0 / NUM_COLUMNS, curriculum_num_rows=num_rows,
                                                       **column_fields(column))
            for column in range(NUM_COLUMNS)}
```

Check how `source_terrain_track` uses `difficulty` (it seeds geometry from it). Keep passing the original difficulty so per-tile seeds stay distinct. Confirm that Isaac Lab's curriculum generator assigns sub-terrain k to column k when all 20 proportions are 0.05 (`TerrainGenerator._generate_curriculum_terrains`: `np.min(np.where(sub_col / num_cols + 0.001 < proportions)[0])` on the cumulative sum). If `SourceTerrainGenerator` reorders sub-terrains, stop and report.

### `$C/source_curriculum_env_cfg.py`

Copy the import block for `SourceTerrainImporterCfg`, `SourceTerrainGenerator`, `MultiMeshRayCasterCfg`, `patterns`, `sim_utils`, `TerrainGeneratorCfg`, `SceneEntityCfg`, `DoneTerm`, `CurrTerm`, `DEEPROBOTICS_LITE3_CFG` and `LITE3_MAX_PROJECTED_GRAVITY_XY` from `$C/source_terrain_env_cfg.py` / `$C/source_training_env_cfg.py`.

```python
@configclass
class DeeproboticsLite3SourceCurriculumEnvCfg(DeeproboticsLite3FlatEnvCfg):
    reward_gait_level: float = 1.0
    source_omni_columns: tuple = OMNI_COLUMNS
    source_tile_lin_vel_x: tuple = (0.25, 0.5)
    source_completion_distance_m: float = 9.25

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0
        self.scene.terrain = SourceTerrainImporterCfg(
            prim_path="/World/ground", terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                class_type=SourceTerrainGenerator, seed=20_261_109, curriculum=True, size=(12.0, 4.0),
                border_width=0.0, num_rows=NUM_ROWS, num_cols=NUM_COLUMNS, difficulty_range=(0.0, 1.0),
                use_cache=False, sub_terrains=source_curriculum_sub_terrains()),
            max_init_terrain_level=4, parallel_reset_sampler_enabled=False, parallel_reset_sampler_num_envs=None,
            shared_bank_rows=None, collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            debug_vis=False)
        self.scene.height_scanner = None
        self.scene.height_scanner_base = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name, update_period=self.sim.dt,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)), debug_vis=False,
            mesh_prim_paths=[MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground", merge_prim_meshes=True, track_mesh_transforms=False)])
        self.scene.robot.init_state.pos = (0.0, 0.0, DEEPROBOTICS_LITE3_CFG.init_state.pos[2])
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")

        old = self.commands.base_velocity
        if type(old) is not UniformThresholdVelocityCommandCfg:
            raise ValueError(f"expected UniformThresholdVelocityCommandCfg, got {type(old).__name__}")
        new = SourceCurriculumVelocityCommandCfg()
        for name in old.__dataclass_fields__:
            if name != "class_type":
                setattr(new, name, copy.deepcopy(getattr(old, name)))
        new.omni_columns = self.source_omni_columns
        new.tile_lin_vel_x = self.source_tile_lin_vel_x
        if not new.heading_command or tuple(new.ranges.ang_vel_z) != (-0.8, 0.8):
            raise ValueError("Flat command must keep the heading loop and ang_vel_z (-0.8, 0.8)")
        self.commands.base_velocity = new

        omni = self.source_omni_columns
        t = self.terminations
        t.terrain_out_of_bounds = None
        if hasattr(t, "source_terrain_tile_bounds"):
            t.source_terrain_tile_bounds = None
        nonfoot = [name for name in self.link_names if not name.endswith("_FOOT")]
        t.illegal_contact = DoneTerm(func=mdp.startup_aware_illegal_contact, params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=nonfoot), "threshold": 5.0,
            "consecutive_steps": 3})
        t.bad_orientation_2.func = mdp.startup_aware_bad_orientation_2
        t.bad_orientation_2.params["max_xy_norm"] = LITE3_MAX_PROJECTED_GRAVITY_XY
        t.source_curriculum_safety = DoneTerm(func=mdp.SourceCurriculumSafety, time_out=False)
        t.source_curriculum_tile_bounds = DoneTerm(func=mdp.source_curriculum_tile_bounds, time_out=False, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"), "size": (12.0, 4.0), "flat_length_m": 3.0,
            "collision_radius_m": 0.27, "numerical_margin_m": 0.001, "scanner_footprint_margin_m": 0.1,
            "omni_columns": omni})
        t.source_curriculum_completion = DoneTerm(func=mdp.source_curriculum_completion, time_out=True, params={
            "distance_m": self.source_completion_distance_m, "omni_columns": omni})
        t.source_curriculum_omni_truncation = DoneTerm(func=mdp.source_curriculum_omni_truncation, time_out=True,
                                                       params={"bounds": omni_truncation_bounds(), "omni_columns": omni})

        for name in list(vars(self.curriculum)):
            if not name.startswith("_"):
                setattr(self.curriculum, name, None)
        self.curriculum.source_curriculum_levels = CurrTerm(func=mdp.SourceCurriculumLevels,
                                                            params={"omni_columns": omni})

        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.2, 0.2), "y": (-0.1, 0.1), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0),
            "yaw": (-0.2, 0.2)}
        material = self.events.randomize_rigid_body_material
        material.params["static_friction_range"] = (0.5, 1.1)
        material.params["dynamic_friction_range"] = (0.4, 1.0)
        material.params["restitution_range"] = (0.0, 0.0)
        material.params["make_consistent"] = True
        self.disable_zero_weight_rewards()
```

Checks while writing:
- Confirm the Flat cfg's `reward_gait_level` / `gait_level` mechanism (Flat sets `reward_mdp.gait_level = 1.0` at module level, line 61). R18 only adds the field for the record; do not change global state.
- Confirm `self.link_names` and `self.base_link_name` exist on the cfg; otherwise copy how `source_terrain_env_cfg.py` builds the illegal_contact body list.
- Confirm `disable_zero_weight_rewards` is a method on the cfg (Flat calls it only for its exact class).
- Keep `time_out` as inherited.

### Registration (`$C/__init__.py`)

```python
gym.register(
    id="SourceCurriculum-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.source_curriculum_env_cfg:DeeproboticsLite3SourceCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3SourceCurriculumPPORunnerCfg",
    },
)
```

### Runner (`$C/agents/rsl_rl_ppo_cfg.py`)

```python
@configclass
class DeeproboticsLite3SourceCurriculumPPORunnerCfg(DeeproboticsLite3FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.save_interval = 50
        self.experiment_name = "deeprobotics_lite3_source_curriculum"
        self.algorithm.learning_rate = 3.844e-4
```

**Tests** (`tests/source_curriculum/test_curriculum_cfg_static.py`, Isaac-free, AST only): the new env cfg file contains the strings `seed=20_261_109`, `num_rows=NUM_ROWS`, `max_init_terrain_level=4`, `"threshold": 5.0`, `"consecutive_steps": 3`, `make_consistent`; `$C/__init__.py` registers `SourceCurriculum-Deeprobotics-Lite3-v0`; the runner sets `3.844e-4` and `deeprobotics_lite3_source_curriculum`. The real behaviour check is Task 4. Commit `feat(stage1): Source curriculum terrain, env cfg and runner`.

---

## Task 4: GPU check script (`scripts/tools/source_curriculum_check.py`)

Launch `AppLauncher(headless=True)` first (copy the launcher pattern from `scripts/tools/ea_lpl_run.py` child mode), then import `rl_training.tasks`, build `SourceCurriculum-Deeprobotics-Lite3-v0` with `parse_env_cfg(task, num_envs=4096)` and `gym.make`. Exit code 0 only if every check passes; print one `PASS/FAIL name detail` line per check and write a JSON report to `--output`.

Checks:
1. Active term sets equal exactly: terminations `{time_out, illegal_contact, bad_orientation_2, source_curriculum_safety, source_curriculum_tile_bounds, source_curriculum_completion, source_curriculum_omni_truncation}`; curriculum `{source_curriculum_levels}`.
2. Obs dims: policy 45, critic 48. `reward_mdp.gait_level == 1.0`. Exactly one friction palette (the importer did not raise).
3. After `env.reset()`: tile env levels in 0–4, omni env levels in 1–8, about 4096/20 envs per column (each column within ±5 of the mean), omni count about 819; every env origin equals `terrain_origins[level, type]`.
4. Robot root z minus 9-ray max hit z in [0.25, 0.45] for all envs after reset; base scan rays finite and within 0.08 m in xy of the root.
5. Tile commands: vx in [0.25, 0.5], vy 0, `is_heading_env` true, `is_standing_env` false; omni commands untouched by the tile logic (mix of slot types present).
6. Run 50 steps with the frozen model_1998 actor (`load_frozen_source_actor(OLD, device="cuda:0")`, deterministic, obs noise off for the check). Repeat check 4 on envs not reset; no NaN in obs, rewards, or contact forces.
7. Extreme-parameter masks: temporarily set robot root of one tile env and one omni env far outside both the tile and the flat region (write root state, step once) and confirm tile bounds fire only for the tile env and omni truncation only for the omni env; set progress ≥ 9.25 for one tile and one omni env and confirm completion fires only for the tile env.
8. Curriculum moves: force `source_curriculum_completion` on a tile env at level 2 and failure on another at level 2 (by state injection, then step); confirm levels 3 and 1 after their reset; a tile env with both ends at level 1.

Run: `cd W && PYTHONPATH=$PWD/source/rl_training CUDA_VISIBLE_DEVICES=0 /home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_curriculum_check.py --output /tmp/stage1_check.json`. If a check fails because of Task 2/3 code, fix it in this task and say so in the commit. Commit `test(stage1): GPU check for the Source curriculum env`.

---

## Task 5 (inline, controller): pilot and training

1. Seed checkpoint: `mkdir -p W/logs/rsl_rl/deeprobotics_lite3_source_curriculum/model1998_seed && cp <OLD> .../model_1998.pt` (`OLD` from `scripts/tools/source_adaptation_runtime.py:18`); `sha256sum` must equal `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`.
2. Pilot (GPU0, check ≥ 12288 MiB free first):
   ```
   cd W && PYTHONPATH=$PWD/source/rl_training CUDA_VISIBLE_DEVICES=0 /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
     scripts/reinforcement_learning/rsl_rl/train.py --task SourceCurriculum-Deeprobotics-Lite3-v0 --num_envs 4096 \
     --seed 42 --headless --resume --load_run model1998_seed --checkpoint model_1998.pt --max_iterations 30 \
     --run_name pilot_seed42
   ```
   Pass if: first logged iteration is 1998 and lr 3.844e-4; per-family `Curriculum/source_curriculum_levels/*` keys logged; no NaN loss; `model_2027.pt` saved and loads with `load_frozen_source_actor(path, expected_sha256=<its sha>)` (after Task 6). Record s/iteration and peak GPU memory. If out of memory, fall back to `--num_envs 2048` and record the deviation.
3. Full runs: seeds 42 (GPU0) and 43 (GPU1) in parallel, `--max_iterations 1500`, run names `full_seed42`, `full_seed43`, final checkpoint `model_3497.pt`. Run in background; check every 30 minutes (Monitor with an until-loop on the checkpoint file). One automatic retry on infrastructure failure (Isaac crash, CUDA init error) from the latest checkpoint; a second failure stops for the user.
4. Record wall time, final mean levels per family, and the s/iteration in the stage README.

Tasks 6–8 run as subagent tasks while training runs; they touch `source/` and `scripts/`, which is allowed because training does not use the clean-tree guard. Evaluations (Task 9) start only after Tasks 6–8 are committed and the tree is clean.

---

## Task 6: evaluator accepts other actors; stage-1 specs

**Files:** edit `$S/ea_lpl/specs.py`, `$S/source_scan/frozen_actor.py`, `scripts/tools/ea_lpl_run.py`, `$C/ea_lpl_env_cfg.py`, `tests/ea_lpl/test_specs.py`; create `scripts/tools/stage1_make_specs.py`, `tests/ea_lpl/test_stage1_specs.py`, `tests/source_scan/test_frozen_actor_expected_sha.py`.

specs.py changes:

```python
LEVELS = ("L1", "L2", "L3")
# validate_spec: level message becomes "level must be L1/L2/L3 and heading_hold boolean"
# optional "actor": {"path": absolute str, "sha256": 64 lowercase hex}; reject anything else with ValueError

def spec_actor(spec, *, default):
    """Return (path, sha256) for the actor of a spec; `default` is (OLD, ACTOR_SHA256)."""
    actor = spec.get("actor")
    if actor is None:
        return default
    return actor["path"], actor["sha256"]


def stage1_specs(policies, *, kind):
    """policies: {name: {"path": abs, "sha256": hex}}; kind in qualify / preview / extend."""
    if kind not in ("qualify", "preview", "extend"):
        raise ValueError(f"unknown stage-1 spec kind: {kind!r}")
    levels = {"qualify": ("L1", "L2"), "preview": ("L3",), "extend": ("L1", "L2")}[kind]
    prefix = {"qualify": "s1q", "preview": "s1p", "extend": "s1x"}[kind]
    offset = 480 if kind == "extend" else 0
    specs = []
    for name, actor in policies.items():
        for level in levels:
            for k in range(4):
                spec = dict(schema=SPEC_SCHEMA, label=f"{prefix}_{name}_{level.lower()}_{k}", mode="fixed",
                            level=level, heading_hold=True, columns=list(EQUAL8),
                            reset_table_seed=P0_TABLE_SEED, namespace=4,
                            slots=list(range(offset + 120 * k, offset + 120 * (k + 1))),
                            episodes_per_env=1, max_episode_steps=2000, codes=None, wall_limit_s=900,
                            actor=dict(actor))
                validate_spec(spec)
                specs.append(spec)
    return specs
```

Match the real `slots` and `columns` encodings used by `p0_specs()` (the plan assumes plain lists; if P0 stores them differently, follow P0). `namespace=4` keeps stage-1 keys disjoint from P0/P1 namespaces; check that 4 is unused in `p0_specs` and the P1 specs, otherwise pick the lowest unused value ≤ 8 and record it.

frozen_actor.py: add `expected_sha256=CHECKPOINT_SHA256` keyword. Default value keeps the old message `"checkpoint hash differs from the registered model_1998.pt"`; any other value raises `ValueError("checkpoint hash differs from the expected SHA256")` on mismatch. Validate the argument is 64 lowercase hex.

ea_lpl_run.py: at lines 234, 274 and 284 use `path, sha = spec_actor(spec, default=(OLD, ACTOR_SHA256))`; load with `load_frozen_source_actor(path, device="cuda:0", expected_sha256=sha)`; check `digest(path) == sha`; the manifest records `actor_sha256=sha` and `actor_path=str(path)`. P0/P1 specs (no `actor`) must behave byte-for-byte as before apart from the added `actor_path` key.

ea_lpl_env_cfg.py line 11: `source_allowed_levels: tuple = ("L1", "L2", "L3")`. Confirm L3 already exists in `_LEVEL_CANDIDATES` and `_source_sub_terrains("L3")` builds.

stage1_make_specs.py: CLI `--policy name=path` (repeatable; sha computed by the script), `--kind`, `--out-dir`; writes one JSON per spec and a `specs.txt` list.

Tests:
- test_specs.py line 25: `level="L3"` → `level="L4"` (L3 is now valid).
- test_stage1_specs.py: counts (3 policies × qualify = 24 specs; preview with 2 policies = 8; extend slots start at 480), label format, slots disjoint within a (policy, level), every spec validates, actor block preserved, bad kind and bad actor (relative path, uppercase hex, wrong length) rejected, `spec_actor` default path.
- test_frozen_actor_expected_sha.py: write a tiny fake checkpoint to tmp_path; wrong expected hash raises the new message; default path still raises the old message for a non-model_1998 file.
- Run `tests/ea_lpl tests/source_scan/test_frozen_actor_expected_sha.py` → all pass (39 baseline + new).

Commit `feat(stage1): evaluator actor selection and stage-1 specs`.

---

## Task 7: flat retention (`scripts/tools/stage1_flat_retention.py`)

GPU script, same launcher pattern. Args: `--checkpoint`, `--sha256`, `--output`, `--gpu`. Builds the Flat task cfg (`Flat-Deeprobotics-Lite3-v0` id from `$C/__init__.py`) with 1024 envs, `seed=20_261_109`, `cfg.observations.policy.enable_corruption = False`; wraps in `RslRlVecEnvWrapper(env, clip_actions=100)`; actor from `load_frozen_source_actor(path, expected_sha256=...)`, deterministic mean action; 3000 steps.

Metrics (computed on the command that was active before each step, envs not done this step):
- `rms_xy_error` = sqrt(mean((v_xy_body − cmd_xy)²)), `rms_yaw_error` = sqrt(mean((wz − cmd_wz)²)).
- The same two values per command category, categorised by value: standing (|cmd| all 0), only_x, only_y, only_ang, mixed.
- `falls_per_episode` = (episodes ended by illegal_contact or bad_orientation_2) / (episodes ended, any reason).
JSON output with the checkpoint sha, counts, and metrics. Unit-test the categorisation and metric helpers in `tests/ea_lpl/test_stage1_retention_metrics.py` by putting them in `$S/ea_lpl/retention.py` (torch only). Commit `feat(stage1): flat retention check`.

---

## Task 8: qualification verdicts and curves

**Files:** create `$S/ea_lpl/stage1.py`, `tests/ea_lpl/test_stage1_qualification.py`, `scripts/tools/stage1_qualification.py`, `scripts/tools/stage1_curves.py`.

stage1.py (pure Python):

```python
FAMILIES = ("flat", "multi_scale_rough", "slope", "multi_step_stairs", "slope_plus_rough", "friction_variation")


def family_of(terrain):
    """Map an evaluator condition label to its family; slope_rough_ must be tested before slope_."""
    if terrain.startswith("slope_rough_"):
        return "slope_plus_rough"
    if terrain.startswith("slope_"):
        return "slope"
    if terrain.startswith("stairs_"):
        return "multi_step_stairs"
    if terrain.startswith("friction_"):
        return "friction_variation"
    if terrain in ("flat", "multi_scale_rough"):
        return terrain
    raise ValueError(f"unknown condition label: {terrain!r}")
```

Use the real condition labels written by `S/ea_lpl/episodes.py` (`record["terrain"]`); adjust prefixes if they differ and keep the slope_rough-first rule.

- `family_table(records)` → per family: n, completed, rate, Clopper–Pearson 95% (reuse `summary.clopper_pearson`), safety endings, mean `projected_fraction` weighted by `steps`.
- `condition_counts(records)` → per condition (completed, n).
- `verdicts(records_by_level, retention, baseline_retention)` → dict Q1..Q4, each `{"pass": bool, "detail": ...}`:
  - Q1: every family pooled rate ≥ 0.90 at L1 and ≥ 0.85 at L2.
  - Q2: every one of the 30 conditions ≥ 24/32.
  - Q3: safety endings ≤ 24/480 at each level.
  - Q4: `rms_xy_error` and `rms_yaw_error` ≤ 1.10 × baseline; `falls_per_episode` ≤ baseline + 0.01.
- `choose_seed(results)`: among seeds, prefer those passing all of Q1–Q4; tie-break by higher L2 pooled completion, then fewer safety endings, then seed 42. Return the seed and the reason.

Tests: family_of on all 15 labels and an unknown label; each Q at its threshold edge (exactly 0.90 passes, 24/32 passes, 24/480 passes, 1.10× passes); choose_seed tie-break order.

stage1_qualification.py: reads run directories (pattern from `scripts/tools/ea_lpl_p0_summary.py`: `resolve_run` and `load`), retention JSONs, writes `qualification.json` and a Markdown table.

stage1_curves.py: reads the tensorboard event files of both full runs, plots a 2×3 grid (one panel per tile family) of `mean_level` and `completion_ema` vs iteration; seed 42 `#2a78d6`, seed 43 `#eb6834`; PNG + SVG. Commit `feat(stage1): qualification verdicts and curves`.

---

## Task 9 (inline, controller): evaluation, report, publish

Order (each eval run is an `ea_lpl_run.py --spec ... --gpu auto` call; tree must be clean):
1. Policies: `m1998` (OLD), `s42` and `s43` (`model_3497.pt` of each full run).
2. Qualification: `stage1_make_specs.py --kind qualify` with the three policies → 24 runs (3 × 2 levels × 4). Split across GPU0/GPU1.
3. Retention: `stage1_flat_retention.py` for the three policies.
4. Verdicts with `stage1_qualification.py`; `choose_seed`.
5. If neither seed passes Q1–Q4: extend both seeds once by 1000 iterations, as authorized by the master spec §5.4 (`--max_iterations 1000` resuming from model_3497; verify the runner's saved iteration convention before naming the final checkpoint). Generate `--kind extend` specs (namespace 4, slots 480–959) for m1998 and both extended seeds so comparisons use matched new starts; repeat retention for both extended seeds. If neither passes, stop at decision point S1-D and report. This corrects the earlier task text that extended only the better seed; it does not increase the master spec's budget.
6. L3 preview: `--kind preview` with θ1 and m1998 → 8 runs.
7. Curves.
8. `docs/lite3-two-stage-stage1-20261109/README.md` in Chinese: what was done, training facts (s/iteration, wall time, deviations), result tables (family × level for the three policies, 30-condition table, safety, retention), Q1–Q4 verdicts, seed choice, L3 preview, rulings and deviations, problems, and the detailed stage-2 plan. `SHA256SUMS` for checkpoints and JSON summaries; raw data stays in the evidence directory `/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/.git/ea-lpl-evidence/stage1/`.
9. Copy θ1 to the evidence directory with its SHA. Commit README, curves and summaries (explicit paths), push, verify `git ls-remote origin codex/lite3-ea-lpl-20261106` equals `git rev-parse HEAD`.
10. Stop and give the user the stage-1 report; wait for approval before stage 2.

## Self-review record

- Spec coverage: S1-1 (Task 3), S1-2 (Task 3 importer), S1-3/S1-4 (Task 1 rows and columns), S1-5 (Task 1 bounds, Task 2 levels), S1-6 (Task 2 command), S1-7 (Task 3 terminations), S1-8 (Task 1 moves, Task 2 levels), S1-9 (Task 3 base_height sensor), S1-10 (Task 3 events), S1-11 (runner keeps Flat clip), §5.4 (Task 5), §5.5 (Tasks 6, 7, 9), §5.6 (Task 8), §5.7 (Task 9), §5.8 8-row fallback (`ANCHOR_ROWS[8]`; switch by `NUM_ROWS` and `curriculum_num_rows`), §5.9 stops (Task 5 step 3, Task 9 step 5).
- Names used across tasks: `commanded_distance`, `tile_mask`, `source_curriculum_completion`, `time_out`, `base_velocity`, `OMNI_COLUMNS`, `curriculum_num_rows`, `spec_actor`, `expected_sha256` — consistent.
- Open points delegated to implementers with explicit instructions: real key names in `_LEVEL_CANDIDATES`, `SourceEpisodeLifecycle` API for the parity test, defining module of `source_terrain_tile_bounds`, P0 spec encodings.
