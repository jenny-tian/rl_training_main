# Lite3 Ramped Startup Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the existing Lite3 PPO and final `0.5 m/s` FlatOnline target while exposing a reset-to-zero, acceleration-limited command that avoids the unsafe startup step.

**Architecture:** A dependency-light tensor helper performs bounded movement toward a target. The existing `UniformThresholdVelocityCommand` optionally stores sampled targets separately from exposed commands; ordinary tasks keep the current instantaneous behavior, while FlatOnline and held-out deployment configs enable an exact `(0.5, 0.5, 0.5)` acceleration tuple. The strict FlatOnline guard locks the ramp for deployment qualification; the separate strict-plane replay collector remains unchanged and keeps its own provenance boundary.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab/Isaac Sim 5.1, RSL-RL, Gymnasium, pytest, Git.

---

## File Structure

- Create `source/rl_training/rl_training/command_slew.py`: dependency-light validation and tensor slew helper.
- Create `tests/online_adaptation/test_command_slew.py`: CPU tests for positive/negative movement, no overshoot, shape validation, and disabled behavior.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py`: add opt-in target buffering, reset-to-zero behavior, and per-step command ramping.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_online_env_cfg.py`: enable the ramp for FlatOnline.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/heldout_boxes_env_cfg.py`: keep held-out evaluation command semantics aligned with FlatOnline.
- Modify `source/rl_training/rl_training/online_adaptation/terrain_guards.py`: require the exact ramp in strict FlatOnline deployment configuration.
- Modify `tests/online_adaptation/test_terrain_guards.py`: lock guard acceptance and rejection.
- Modify `tests/online_adaptation/test_flat_online_env_cfg.py`: lock resolved Isaac configuration and reset behavior.
- Modify `docs/lite3-online-learning-principles.md`: record diagnosis, ramp parameters, checkpoint identity, and measured qualification evidence.

### Task 1: Dependency-Light Command Slew Contract

**Files:**
- Create: `source/rl_training/rl_training/command_slew.py`
- Create: `tests/online_adaptation/test_command_slew.py`

- [ ] **Step 1: Write failing tensor tests**

Create `tests/online_adaptation/test_command_slew.py`:

```python
import pytest
import torch

from rl_training.command_slew import slew_toward


def test_slew_toward_moves_each_axis_in_both_directions_without_overshoot():
    current = torch.tensor([[0.0, 0.4, -0.1], [0.49, -0.49, 0.0]])
    target = torch.tensor([[0.5, 0.1, -0.5], [0.5, -0.5, 0.0]])

    result = slew_toward(current, target, max_delta=(0.1, 0.2, 0.3))

    assert torch.allclose(
        result,
        torch.tensor([[0.1, 0.2, -0.4], [0.5, -0.5, 0.0]]),
    )
    assert current.tolist() == [[0.0, 0.4, -0.1], [0.49, -0.49, 0.0]]


@pytest.mark.parametrize(
    "max_delta, message",
    [
        ((0.1, 0.2), "three components"),
        ((0.1, 0.0, 0.3), "positive"),
        ((0.1, float("inf"), 0.3), "finite"),
    ],
)
def test_slew_toward_rejects_invalid_limits(max_delta, message):
    with pytest.raises(ValueError, match=message):
        slew_toward(torch.zeros(1, 3), torch.ones(1, 3), max_delta=max_delta)


def test_slew_toward_rejects_mismatched_command_shapes():
    with pytest.raises(ValueError, match="same shape"):
        slew_toward(torch.zeros(1, 3), torch.ones(2, 3), max_delta=(0.1, 0.1, 0.1))
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_command_slew.py -q
```

Expected: collection fails with `ModuleNotFoundError: rl_training.command_slew`.

- [ ] **Step 3: Implement the minimal tensor helper**

Create `source/rl_training/rl_training/command_slew.py`:

```python
"""Dependency-light velocity-command slew limiting."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _validated_max_delta(
    max_delta: Sequence[float], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if len(max_delta) != 3:
        raise ValueError("max_delta must contain exactly three components")
    values = tuple(float(value) for value in max_delta)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("max_delta components must be finite")
    if not all(value > 0.0 for value in values):
        raise ValueError("max_delta components must be positive")
    return torch.tensor(values, device=device, dtype=dtype)


def slew_toward(
    current: torch.Tensor,
    target: torch.Tensor,
    *,
    max_delta: Sequence[float],
) -> torch.Tensor:
    if current.shape != target.shape:
        raise ValueError("current and target commands must have the same shape")
    if current.ndim == 0 or current.shape[-1] != 3:
        raise ValueError("velocity commands must have a final dimension of three")
    limit = _validated_max_delta(max_delta, device=current.device, dtype=current.dtype)
    delta = torch.clamp(target - current, min=-limit, max=limit)
    return current + delta
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: `5 passed`.

- [ ] **Step 5: Commit the helper**

```bash
git add source/rl_training/rl_training/command_slew.py \
  tests/online_adaptation/test_command_slew.py
git commit -m "feat: add velocity command slew contract"
```

If commit fails, preserve the working tree and continue as authorized.

### Task 2: Opt-In Slew-Limited Isaac Command

**Files:**
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py`
- Modify: `tests/online_adaptation/test_command_slew.py`

- [ ] **Step 1: Write failing source-contract tests**

Append to `tests/online_adaptation/test_command_slew.py`:

```python
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMMANDS = (
    ROOT
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py"
)


def test_uniform_threshold_command_declares_opt_in_acceleration_limits():
    tree = ast.parse(COMMANDS.read_text(encoding="utf-8"))
    cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "UniformThresholdVelocityCommandCfg"
    )
    assignment = next(
        node
        for node in cfg.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "command_acceleration_limits"
    )
    assert ast.unparse(assignment.value) == "None"


def test_slew_enabled_command_separates_target_from_exposed_command():
    source = COMMANDS.read_text(encoding="utf-8")
    assert "self.target_vel_command_b = self.vel_command_b.clone()" in source
    assert "self.vel_command_b[env_ids] = 0.0" in source
    assert "slew_toward(" in source
    assert "self._env.step_dt" in source
```

- [ ] **Step 2: Run and verify RED**

Run the Task 1 command. Expected: the two new assertions fail because the config field and target buffer do not exist.

- [ ] **Step 3: Add target buffering and reset behavior**

In `mdp/commands.py`, import `slew_toward`. Add to `UniformThresholdVelocityCommand.__init__` after the parent initialization:

```python
self.target_vel_command_b = self.vel_command_b.clone()
```

Override `reset` so only slew-enabled commands restart at zero:

```python
def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
    if self.cfg.command_acceleration_limits is not None:
        reset_ids = slice(None) if env_ids is None else env_ids
        self.vel_command_b[reset_ids] = 0.0
        self.target_vel_command_b[reset_ids] = 0.0
    return super().reset(env_ids)
```

Wrap the current `_resample_command` body by preserving the exposed command, executing all existing sampling/special-case logic unchanged, copying the result into the target buffer, and restoring the exposed command when slew is enabled:

```python
previous = self.vel_command_b[env_ids].clone()
super()._resample_command(env_ids)
# existing threshold and special-case overrides remain here
if self.cfg.command_acceleration_limits is not None:
    self.target_vel_command_b[env_ids] = self.vel_command_b[env_ids]
    self.vel_command_b[env_ids] = previous
```

- [ ] **Step 4: Add per-step ramping without changing default behavior**

Add this `_update_command` override:

```python
def _update_command(self):
    if self.cfg.command_acceleration_limits is None:
        super()._update_command()
        return

    if self.cfg.heading_command:
        raise ValueError("slew-limited heading commands are not supported")
    target = self.target_vel_command_b.clone()
    target[self.is_standing_env] = 0.0
    max_delta = tuple(
        float(limit) * float(self._env.step_dt)
        for limit in self.cfg.command_acceleration_limits
    )
    self.vel_command_b.copy_(
        slew_toward(self.vel_command_b, target, max_delta=max_delta)
    )
```

Add the config field to `UniformThresholdVelocityCommandCfg`:

```python
command_acceleration_limits: tuple[float, float, float] | None = None
"""Optional `(x, y, yaw)` command acceleration limits per second."""
```

- [ ] **Step 5: Run focused tests and compile the command module**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_command_slew.py -q
PYTHONPATH=source/rl_training:. \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training/command_slew.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py
```

Expected: all focused tests pass and compileall emits no output.

- [ ] **Step 6: Commit the Isaac command integration**

```bash
git add source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py \
  tests/online_adaptation/test_command_slew.py
git commit -m "feat: ramp opt-in velocity commands"
```

### Task 3: Enable and Guard the Lite3 Deployment Ramp

**Files:**
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_online_env_cfg.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/heldout_boxes_env_cfg.py`
- Modify: `source/rl_training/rl_training/online_adaptation/terrain_guards.py`
- Modify: `tests/online_adaptation/test_terrain_guards.py`
- Modify: `tests/online_adaptation/test_flat_online_env_cfg.py`

- [ ] **Step 1: Write failing guard tests**

In `_flat_online_deployment_env_cfg()` in `test_terrain_guards.py`, add:

```python
command_acceleration_limits=(0.5, 0.5, 0.5),
```

Add:

```python
@pytest.mark.parametrize(
    "invalid",
    [None, (0.6, 0.5, 0.5), (0.5, 0.5), [0.5, 0.5, 0.5]],
)
def test_flat_online_deployment_guard_rejects_changed_command_ramp(invalid):
    cfg = _flat_online_deployment_env_cfg()
    cfg.commands.base_velocity.command_acceleration_limits = invalid

    with pytest.raises(ValueError, match="command_acceleration_limits"):
        terrain_guards.assert_flat_online_deployment_config(cfg)
```

Add an AST test that both deployment config files assign the exact tuple:

```python
def test_lite3_deployment_configs_enable_the_same_command_ramp():
    for filename in ("flat_online_env_cfg.py", "heldout_boxes_env_cfg.py"):
        source = (LITE3_CONFIG_DIR / filename).read_text(encoding="utf-8")
        assert (
            "self.commands.base_velocity.command_acceleration_limits = "
            "(0.5, 0.5, 0.5)"
        ) in source
```

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_terrain_guards.py \
  tests/online_adaptation/test_flat_online_env_cfg.py -q
```

Expected: failures identify the missing config assignments and missing guard check.

- [ ] **Step 3: Enable the exact ramp in both deployment configs**

After `heading_command = False` in both Lite3 deployment configs, add:

```python
self.commands.base_velocity.command_acceleration_limits = (0.5, 0.5, 0.5)
```

Do not change `Flat-Deeprobotics-Lite3-v0` or `Rough-Deeprobotics-Lite3-v0`; their inherited value remains `None`.

- [ ] **Step 4: Lock the ramp in the strict FlatOnline guard**

In `assert_flat_online_deployment_config`, extend the command attribute checks with:

```python
("command_acceleration_limits", (0.5, 0.5, 0.5)),
```

The existing exact-value helper intentionally rejects lists as well as different tuple values, preserving canonical provenance.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass; Isaac-only tests remain explicitly skipped when `pxr` is unavailable.

- [ ] **Step 6: Run one launched Isaac config test**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
PYTHONPATH=source/rl_training:. python - <<'PY'
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device="cuda:0").app
try:
    import rl_training.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from rl_training.online_adaptation.terrain_guards import assert_flat_online_deployment_config
    cfg = parse_env_cfg("FlatOnline-Deeprobotics-Lite3-v0", device="cuda:0", num_envs=1)
    assert cfg.commands.base_velocity.command_acceleration_limits == (0.5, 0.5, 0.5)
    assert_flat_online_deployment_config(cfg)
finally:
    app.close()
PY
```

Expected: exit code 0.

- [ ] **Step 7: Commit configuration and guard changes**

```bash
git add source/rl_training/rl_training/online_adaptation/terrain_guards.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_online_env_cfg.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/heldout_boxes_env_cfg.py \
  tests/online_adaptation/test_terrain_guards.py \
  tests/online_adaptation/test_flat_online_env_cfg.py
git commit -m "fix: ramp lite3 deployment startup command"
```

### Task 4: Full CPU Verification

**Files:**
- Verify all intentional source and test changes.

- [ ] **Step 1: Run the complete CPU suite**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest -q
```

Expected: all tests pass; the prior baseline was `700 passed, 5 skipped`, so the total increases by the new tests with the same five explicit skips.

- [ ] **Step 2: Run static verification**

```bash
PYTHONPATH=source/rl_training:. \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training \
  scripts/reinforcement_learning/online_adaptation \
  tests/online_adaptation
git diff --check
git status --short
```

Expected: compileall and whitespace checks emit no diagnostics; status contains only intentional changes, including the independent foot-index fix if it has not yet been committed.

### Task 5: Deterministic GPU Startup Gate

**Files:**
- Temporary diagnostic: `/tmp/lite3_flat_height_probe.py`
- Existing checkpoint: `logs/rsl_rl/deeprobotics_lite3_flat/2026-08-01_15-32-10_flat_stability_resume8800_seed42_20260801/model_9999.pt`

- [ ] **Step 1: Extend only the temporary probe to log the exposed command**

Add this field to each row dictionary in `/tmp/lite3_flat_height_probe.py`:

```python
"command": wrapped_env.unwrapped.command_manager.get_command(
    "base_velocity"
).detach().cpu().tolist(),
```

Do not add diagnostic logging to production code.

- [ ] **Step 2: Confirm GPU availability and no conflicting FlatOnline process**

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
pgrep -af "python .*FlatOnline-Deeprobotics-Lite3-v0"
```

Expected: GPU 1 has enough free memory and no conflicting probe is running.

- [ ] **Step 3: Run the generated-flat eight-transition gate**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
PYTHONPATH=source/rl_training:. CUDA_VISIBLE_DEVICES=1 \
  python /tmp/lite3_flat_height_probe.py
```

Expected: command x begins near `0.01 m/s` and increases by about `0.01 m/s` per `0.02 s` environment step; every observed `joint_vel_max` is at most `30 rad/s`; no row is done. If the spike remains, stop and return to root-cause analysis before changing acceleration.

- [ ] **Step 4: Run the strict-plane comparison**

```bash
PYTHONPATH=source/rl_training:. CUDA_VISIBLE_DEVICES=1 LITE3_STRICT_PLANE=1 \
  python /tmp/lite3_flat_height_probe.py
```

Expected: the same ramp progression and no observed joint velocity above `30 rad/s`.

### Task 6: Qualification and Evidence

**Files:**
- Modify: `docs/lite3-online-learning-principles.md`
- Generated: deployment qualification JSON beside the checkpoint.

- [ ] **Step 1: Re-run five-seed source qualification**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=source/rl_training:. \
  python scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_flat/2026-08-01_15-32-10_flat_stability_resume8800_seed42_20260801/model_9999.pt \
  --task Flat-Deeprobotics-Lite3-v0 --seeds 7 11 19 23 31 \
  --steps-per-seed 2000 --max-mean-velocity-error 0.35 \
  --output /tmp/lite3_ramp_base_qualification_short.json \
  --device cuda:0 --headless
```

Expected: aggregate `passed: true`; zero falls, paused/projected, and nonfinite rows. This confirms the ordinary flat task was not changed.

- [ ] **Step 2: Run the 100-episode deployment qualification**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=source/rl_training:. \
  python scripts/reinforcement_learning/online_adaptation/qualify_flat_deployment.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_flat/2026-08-01_15-32-10_flat_stability_resume8800_seed42_20260801/model_9999.pt \
  --task FlatOnline-Deeprobotics-Lite3-v0 --seed 42 \
  --expected-episodes 100 --expected-steps 100000 \
  --output /tmp/lite3_ramp_base_qualification_deployment.json \
  --device cuda:0 --headless
```

Expected: exit code 0, `passed: true`, exactly 100 completed episodes and 100,000 attempted rows, with zero falls, premature terminations, paused/projected rows, and nonfinite rows.

- [ ] **Step 3: Preserve the formal replay provenance boundary**

Verify the collector's task boundary directly:

```bash
rg -n "assert_plane_only_terrain|STRICT_FLAT_BASELINE_TASK" \
  scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py \
  scripts/reinforcement_learning/online_adaptation/cli.py
test ! -e logs/rsl_rl/deeprobotics_lite3_flat/2026-08-01_15-32-10_flat_stability_resume8800_seed42_20260801/flat_baseline_manifest.json
test ! -e logs/rsl_rl/deeprobotics_lite3_flat/2026-08-01_15-32-10_flat_stability_resume8800_seed42_20260801/training_command.txt
```

Expected: the collector remains restricted to the strict-plane baseline task, and both commands exit zero because this candidate has neither a sealed manifest nor its original training-command file. Do not build a manifest with the current commit SHA for an older checkpoint and do not claim that FlatOnline ramp qualification is replay qualification. Formal replay remains a separate follow-up after authentic provenance inputs are recovered.

- [ ] **Step 4: Record exact evidence**

Append to `docs/lite3-online-learning-principles.md`:

- the pre-ramp generated/plane failures (`35.0339` and `44.0181 rad/s`);
- the chosen `(0.5, 0.5, 0.5)` acceleration tuple and one-second forward ramp;
- the unchanged checkpoint path and SHA-256;
- focused/full CPU test counts;
- eight-step generated/plane maxima;
- qualification path, hash, counts, and pass/fail outcome;
- the explicit replay provenance blocker and the fact that no replay claim was made.

- [ ] **Step 5: Final verification and evidence commit**

```bash
git diff --check
git status --short
git add docs/lite3-online-learning-principles.md
git commit -m "docs: record lite3 ramped startup evidence"
```

Expected: no whitespace errors. If commit fails, preserve the evidence file and continue as authorized.
