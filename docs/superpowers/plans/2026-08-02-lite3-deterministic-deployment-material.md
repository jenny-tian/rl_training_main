# Lite3 Deterministic Deployment Material Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give FlatOnline and HeldoutBoxes one explicit, reproducible Lite3 collision material so the frozen flat policy remains below the existing 30 rad/s safety limit after a direct standing reset.

**Architecture:** A CPU-only helper owns the canonical fixed values. Both deployment configs retain IsaacLab's startup material-assignment event but replace every range with one fixed value and use one bucket; the FlatOnline guard validates that contract. All other deployment dynamics randomization stays disabled, while the original Flat task remains unchanged.

**Tech Stack:** Python 3.11, PyTorch, pytest, IsaacLab/Isaac Sim, RSL-RL

---

### Task 1: Define the Canonical Material Contract

**Files:**
- Modify: `tests/online_adaptation/test_terrain_guards.py`
- Modify: `source/rl_training/rl_training/online_adaptation/terrain_guards.py`

- [ ] **Step 1: Write the failing helper test**

Add after `test_deterministic_joint_reset_preserves_default_standing_pose`:

```python
def test_deterministic_rigid_body_material_uses_one_fixed_bucket():
    helper = getattr(
        terrain_guards, "deterministic_rigid_body_material_params", None
    )

    assert helper is not None, "deterministic material helper is missing"
    assert helper() == {
        "static_friction_range": (0.65, 0.65),
        "dynamic_friction_range": (0.55, 0.55),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    }
```

- [ ] **Step 2: Run the test and verify RED**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/online_adaptation/test_terrain_guards.py::test_deterministic_rigid_body_material_uses_one_fixed_bucket -q
```

Expected: FAIL with `deterministic material helper is missing`.

- [ ] **Step 3: Add the minimal helper before `deterministic_root_reset_params`**

```python
def deterministic_rigid_body_material_params() -> dict[str, tuple[float, float] | int]:
    """Return the fixed collision material used by Lite3 deployment tasks."""
    return {
        "static_friction_range": (0.65, 0.65),
        "dynamic_friction_range": (0.55, 0.55),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    }
```

- [ ] **Step 4: Re-run Step 2 and verify `1 passed`**

### Task 2: Make the FlatOnline Guard Enforce the Material

**Files:**
- Modify: `tests/online_adaptation/test_terrain_guards.py`
- Modify: `source/rl_training/rl_training/online_adaptation/terrain_guards.py`

- [ ] **Step 1: Give `_flat_online_deployment_env_cfg` the required fake event**

Replace `randomize_rigid_body_material=None` with:

```python
randomize_rigid_body_material=SimpleNamespace(
    func=SimpleNamespace(__name__="randomize_rigid_body_material"),
    params={
        "asset_cfg": SimpleNamespace(name="robot", body_names=".*"),
        **terrain_guards.deterministic_rigid_body_material_params(),
    },
),
```

- [ ] **Step 2: Add rejection tests**

```python
@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("static_friction_range", (0.64, 0.65)),
        ("dynamic_friction_range", (0.50, 0.55)),
        ("restitution_range", (0.0, 0.1)),
        ("num_buckets", 2),
    ],
)
def test_flat_online_deployment_guard_rejects_changed_material(key, invalid):
    cfg = _flat_online_deployment_env_cfg()
    cfg.events.randomize_rigid_body_material.params[key] = invalid

    with pytest.raises(ValueError, match=key):
        terrain_guards.assert_flat_online_deployment_config(cfg)


def test_flat_online_deployment_guard_rejects_wrong_material_function():
    cfg = _flat_online_deployment_env_cfg()
    cfg.events.randomize_rigid_body_material.func.__name__ = "wrong_material"

    with pytest.raises(ValueError, match="randomize_rigid_body_material.func"):
        terrain_guards.assert_flat_online_deployment_config(cfg)
```

- [ ] **Step 3: Run the guard file and verify RED**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/online_adaptation/test_terrain_guards.py -q
```

Expected: the acceptance test fails because the current guard requires the material event to be `None`; the new rejection tests do not yet report the requested fields.

- [ ] **Step 4: Validate the event in `assert_flat_online_deployment_config`**

Remove `"randomize_rigid_body_material"` from the attributes that must be `None`, then add:

```python
material_event = getattr(events, "randomize_rigid_body_material", None)
if material_event is None:
    raise ValueError(
        "FlatOnline deployment requires events.randomize_rigid_body_material"
    )
material_func = getattr(material_event, "func", None)
_require_flat_online_value(
    "events.randomize_rigid_body_material.func",
    getattr(material_func, "__name__", None),
    "randomize_rigid_body_material",
)
material_params = getattr(material_event, "params", None)
if not isinstance(material_params, Mapping):
    raise ValueError(
        "FlatOnline deployment requires events.randomize_rigid_body_material.params mapping"
    )
for key, expected in deterministic_rigid_body_material_params().items():
    _require_flat_online_value(
        f"events.randomize_rigid_body_material.params[{key!r}]",
        material_params.get(key),
        expected,
    )
asset_cfg = material_params.get("asset_cfg")
_require_flat_online_attribute(
    asset_cfg,
    "name",
    "events.randomize_rigid_body_material.params['asset_cfg'].name",
    "robot",
)
_require_flat_online_attribute(
    asset_cfg,
    "body_names",
    "events.randomize_rigid_body_material.params['asset_cfg'].body_names",
    ".*",
)
```

- [ ] **Step 5: Re-run Step 3 and verify all guard tests pass**

### Task 3: Enable the Same Material in Both Deployment Tasks

**Files:**
- Modify: `tests/online_adaptation/test_flat_online_env_cfg.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_online_env_cfg.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/heldout_boxes_env_cfg.py`

- [ ] **Step 1: Import the material helper in the config tests**

Replace the existing terrain-guard import block with:

```python
from rl_training.online_adaptation.terrain_guards import (
    deterministic_joint_reset_params,
    deterministic_rigid_body_material_params,
    deterministic_root_reset_params,
)
```

- [ ] **Step 2: Replace the outdated material-disabled source test**

```python
def test_lite3_deployment_sources_keep_only_deterministic_material_initialization():
    for filename in ("flat_online_env_cfg.py", "heldout_boxes_env_cfg.py"):
        source = (LITE3_CONFIG_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        disabled_events = {
            ast.unparse(target)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
            for target in node.targets
            if ast.unparse(target).startswith("self.events.")
        }

        assert "self.events.randomize_rigid_body_material" not in disabled_events
        assert {
            "self.events.randomize_rigid_body_mass",
            "self.events.randomize_rigid_body_mass_base",
            "self.events.randomize_rigid_body_inertia",
            "self.events.randomize_com_positions",
            "self.events.randomize_actuator_gains",
        } <= disabled_events
```

- [ ] **Step 3: Add a resolved-config test for both tasks**

```python
@pytest.mark.isaaclab
@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("flat_online_env_cfg", "DeeproboticsLite3FlatOnlineEnvCfg"),
        ("heldout_boxes_env_cfg", "DeeproboticsLite3HeldoutBoxesEnvCfg"),
    ],
)
def test_lite3_deployment_configs_use_fixed_collision_material(
    module_name, class_name
):
    pytest.importorskip("pxr")
    module = importlib.import_module(
        "rl_training.tasks.manager_based.locomotion.velocity.config.quadruped."
        f"deeprobotics_lite3.{module_name}"
    )
    cfg = getattr(module, class_name)()
    event = cfg.events.randomize_rigid_body_material

    assert event is not None
    assert event.func.__name__ == "randomize_rigid_body_material"
    assert {
        key: event.params[key]
        for key in deterministic_rigid_body_material_params()
    } == deterministic_rigid_body_material_params()
    assert event.params["asset_cfg"].name == "robot"
    assert event.params["asset_cfg"].body_names == ".*"
```

Delete the obsolete assertion that FlatOnline's material event is `None`.

- [ ] **Step 4: Run config tests and verify RED**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/online_adaptation/test_flat_online_env_cfg.py -q
```

Expected: FAIL because FlatOnline removes the event and HeldoutBoxes still retains non-material dynamics randomization.

- [ ] **Step 5: Configure FlatOnline**

Import both helpers and replace the material-event deletion with:

```python
self.events.randomize_rigid_body_material.params.update(
    deterministic_rigid_body_material_params()
)
```

- [ ] **Step 6: Configure HeldoutBoxes**

Import both helpers and add:

```python
self.events.randomize_rigid_body_material.params.update(
    deterministic_rigid_body_material_params()
)
self.events.randomize_rigid_body_mass = None
self.events.randomize_rigid_body_mass_base = None
self.events.randomize_rigid_body_inertia = None
self.events.randomize_com_positions = None
self.events.randomize_actuator_gains = None
```

- [ ] **Step 7: Run both focused files and verify GREEN**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/online_adaptation/test_terrain_guards.py \
  tests/online_adaptation/test_flat_online_env_cfg.py -q
```

Expected: all focused tests pass.

### Task 4: Verify the Complete Software Change

**Files:**
- Verify: `source/rl_training/rl_training/online_adaptation/terrain_guards.py`
- Verify: both Lite3 deployment config files
- Verify: all tests

- [ ] **Step 1: Run the complete CPU suite**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests -q
```

Expected: zero failures.

- [ ] **Step 2: Run syntax and diff checks**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q source/rl_training/rl_training tests
git diff --check
```

Expected: both commands exit 0 without output.

- [ ] **Step 3: Inspect scope**

```bash
git status --short
git diff --stat
git diff --check
```

Expected: only approved standing-reset, command-handoff, material, test, and plan changes. `/home/tjx/Lite3_rl_deploy` remains untouched.

### Task 5: Validate the Frozen Policy in Isaac Sim

**Files:**
- Read: `logs/rsl_rl/deeprobotics_lite3_flat/2026-08-01_15-32-10_flat_stability_resume8800_seed42_20260801/model_9999.pt`
- Generate: evaluation artifacts under `/tmp`

- [ ] **Step 1: Run the 35-step FlatOnline diagnostic**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=source/rl_training:. \
LITE3_TASK=FlatOnline-Deeprobotics-Lite3-v0 \
LITE3_FORWARD_SPEED=0.3 LITE3_PROBE_STEPS=35 LITE3_PRINT_ROWS=0 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python /tmp/lite3_flat_height_probe.py
```

Expected: `max_joint_vel < 30`, no termination, and positive final x displacement.

- [ ] **Step 2: Run the actual FlatOnline path for 500 steps**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=source/rl_training:. \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_flat/2026-08-01_15-32-10_flat_stability_resume8800_seed42_20260801/model_9999.pt \
  --mode base --task FlatOnline-Deeprobotics-Lite3-v0 \
  --seed 7 --terrain-seed 7 --forward-speed 0.3 --max-steps 500 \
  --flat-retention-probe \
  --run-dir /tmp/lite3_eval_flat_fixed_material_500_20260802 \
  --device cuda:0 --headless
```

Expected: `summary.json` reports 500 steps, `fell: false`, and `safety_violation: false`.

- [ ] **Step 3: Requalify original Flat for 500 steps**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=source/rl_training:. \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_flat/2026-08-01_15-32-10_flat_stability_resume8800_seed42_20260801/model_9999.pt \
  --task Flat-Deeprobotics-Lite3-v0 --seeds 7 --steps-per-seed 500 \
  --max-mean-velocity-error 0.35 \
  --output /tmp/lite3_flat_original_seed7_500_fixed_material_20260802.json \
  --device cuda:0 --headless
```

Expected: `passed: true`, 500 steps, zero falls, and zero safety projections.

- [ ] **Step 4: Smoke-test HeldoutBoxes with the frozen policy**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=source/rl_training:. \
LITE3_TASK=HeldoutBoxes-Deeprobotics-Lite3-v0 \
LITE3_FORWARD_SPEED=0.3 LITE3_PROBE_STEPS=500 LITE3_PRINT_ROWS=0 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python /tmp/lite3_flat_height_probe.py
```

Expected: no joint speed at or above 30 rad/s before the held-out geometry and no early standing-handoff fall.

The real-robot material sweep remains a later hardware-deployment gate: run separate low, nominal, and high fixed-friction cases inside the PPO training distribution. It does not randomize the nominal online-learning run, and every case must retain the 30 rad/s and fall/tilt gates.

### Task 6: Commit the Verified Fix

**Files:**
- Commit only approved implementation, tests, and this plan after all checks pass.

- [ ] **Step 1: Stage exact files**

```bash
git add source/rl_training/rl_training/online_adaptation/terrain_guards.py
git add source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_online_env_cfg.py
git add source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/heldout_boxes_env_cfg.py
git add tests/online_adaptation/test_terrain_guards.py
git add tests/online_adaptation/test_flat_online_env_cfg.py
git add docs/superpowers/plans/2026-08-02-lite3-deterministic-deployment-material.md
```

- [ ] **Step 2: Inspect staged scope**

```bash
git diff --cached --check
git diff --cached --stat
```

Expected: no whitespace errors and no file outside the approved scope.

- [ ] **Step 3: Commit**

```bash
git commit -m "fix: set deterministic Lite3 deployment material"
```

Expected: success. If submission fails, retain the verified working tree and continue without blocking online-learning validation, per the user's instruction.
