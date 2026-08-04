# Lite3 Height-Scan Residual Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed a fixed-size, non-privileged height-scan summary to the Lite3 residual SAC policy while preserving the frozen Flat PPO observation contract and failing closed on invalid scans.

**Architecture:** Keep `geometry` terrain features as the default legacy source. Add an explicit `height_scan` source that summarizes Isaac Lab ray hits into the existing ten terrain-feature slots. Carry a per-step validity mask outside the feature vector so the online agent can force zero residuals and exclude invalid steps from adaptation replay. Freeze the source and summary schema in runtime/checkpoint metadata.

**Tech Stack:** Python 3, PyTorch, Isaac Lab RayCaster, argparse, pytest, existing Lite3 online-adaptation checkpoint and replay APIs.

---

## File Map

- Create `source/rl_training/rl_training/online_adaptation/terrain_observation.py` for pure tensor summary and validation logic.
- Modify `source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py` for source selection, scanner access, validity, and metadata.
- Modify `source/rl_training/rl_training/online_adaptation/agent.py` for residual and adaptation-replay gating.
- Modify the online-adaptation CLI, worker, curriculum, and sealed-evaluation scripts for immutable configuration propagation.
- Add focused tests under `tests/online_adaptation/` for summary extraction, adapter behavior, agent gating, CLI/configuration propagation, and provenance.
- Update `docs/lite3-online-learning-principles.md` with the sensor contract after implementation.

### Task 1: Add Pure Height-Scan Summary

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/terrain_observation.py`
- Create: `tests/online_adaptation/test_terrain_observation.py`

- [ ] **Step 1: Write failing tests for deterministic summary and validation**

Use a synthetic `[batch, rows, cols]` relative-height grid. Assert the summary
order is exactly:

~~~text
[valid_fraction, global_min, global_max, global_mean, global_std,
 forward_min, forward_mean, forward_max, left_right_delta,
 forward_backward_delta]
~~~

Also test clipping and fixed scale, all-invalid input returning zeros plus
`valid=False`, and a batch/grid shape mismatch raising `ValueError`.

- [ ] **Step 2: Run the focused tests and verify the expected failure**

~~~bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_terrain_observation.py -q
~~~

Expected: collection fails because the new module does not exist.

- [ ] **Step 3: Implement the minimal pure helper**

Define a frozen configuration and stable return type:

~~~python
@dataclass(frozen=True)
class HeightScanConfig:
    rows: int
    cols: int
    height_scale_m: float = 0.10
    clip_m: float = 0.20
    min_valid_fraction: float = 0.90

def summarize_height_scan(
    relative_heights: torch.Tensor,
    config: HeightScanConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [batch, 10] normalized features and [batch, 1] validity."""
~~~

Mask non-finite hits before reductions, normalize by `height_scale_m`, clamp
to `[-clip_m / height_scale_m, clip_m / height_scale_m]`, and return detached
finite tensors. The forward half is the first half of configured grid rows;
left/right and forward/backward deltas use finite masked means.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the command from Step 2. Expected: all summary tests pass.

- [ ] **Step 5: Commit the pure summary**

~~~bash
git add source/rl_training/rl_training/online_adaptation/terrain_observation.py \
  tests/online_adaptation/test_terrain_observation.py
git commit -m "feat: summarize height scan terrain context"
~~~

### Task 2: Wire Height-Scan Source into the Isaac Lab Adapter

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py`
- Modify: `tests/online_adaptation/test_isaaclab_adapter.py`

- [ ] **Step 1: Add failing adapter tests**

Extend the fake scene with a fake `height_scanner` exposing
`data.ray_hits_w` and a fixed scanner origin. Assert that
`terrain_feature_source="height_scan"` returns the ten summary values and
that `adapter.terrain_valid` has shape `[1, 1]`. Add tests for missing
scanner, non-finite hits, and insufficient finite-hit fraction; each must
return finite zero features and `terrain_valid=False` without raising from
`frame()`.

Assert that the default `geometry` source preserves current frame shape,
values, and `step_height` behavior.

- [ ] **Step 2: Run focused adapter tests and verify the expected failure**

~~~bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_isaaclab_adapter.py -q
~~~

Expected: the new source-selection tests fail because the adapter has no
scanner source or validity property.

- [ ] **Step 3: Add explicit source/configuration validation**

Add `terrain_feature_source: Literal["geometry", "height_scan"] = "geometry"`,
`height_scanner_name="height_scanner"`, and a validated `HeightScanConfig`.
Keep `step_height` required only by the legacy geometry source. Resolve the
scanner lazily in height-scan mode so legacy/base runs do not depend on it.

- [ ] **Step 4: Compute scanner features through the pure helper**

Read `ray_hits_w[..., 2]` and the scanner/base world height, reshape to the
configured grid, call `summarize_height_scan`, cache the latest validity mask,
and append the returned ten values to the existing 49-dimensional frame.
Expose:

~~~python
@property
def terrain_valid(self) -> torch.Tensor: ...

def terrain_metadata(self) -> dict[str, object]: ...
~~~

`terrain_metadata()` includes source, schema version, grid shape,
normalization constants, and minimum valid fraction.

- [ ] **Step 5: Run focused adapter and summary tests**

~~~bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_terrain_observation.py \
  tests/online_adaptation/test_isaaclab_adapter.py -q
~~~

Expected: all tests pass, including legacy geometry compatibility.

- [ ] **Step 6: Commit adapter integration**

~~~bash
git add source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py \
  tests/online_adaptation/test_isaaclab_adapter.py
git commit -m "feat: expose optional height scan terrain features"
~~~

### Task 3: Gate Residual Actions and Adaptation Replay on Sensor Validity

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/agent.py`
- Modify: `tests/online_adaptation/test_publisher_agent.py`

- [ ] **Step 1: Write failing tests**

Call `begin_step(..., terrain_valid=torch.zeros(1, 1, dtype=torch.bool))` and
assert the decision residual is exactly zero, the published actor version is
`None`, and the completed transition enters global replay but not
`adaptation_replay`. Add a valid-mask test proving existing behavior is
unchanged and a shape-validation test for non-`[batch, 1]` masks.

- [ ] **Step 2: Run the focused tests and verify failure**

~~~bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_publisher_agent.py -q
~~~

Expected: new calls fail because `begin_step` has no `terrain_valid` argument.

- [ ] **Step 3: Add the optional validity gate**

Extend `begin_step` with `terrain_valid: torch.Tensor | None = None`.
Normalize `None` to all-true, validate its shape/device, and combine it with
the existing paused mask. Invalid rows use zero residual and a false pending
learning mask; the existing `finish_step` routing then excludes them from
adaptation replay without changing `TransitionBatch`.

- [ ] **Step 4: Run focused agent tests**

Run the command from Step 2. Expected: all publisher/agent tests pass.

- [ ] **Step 5: Commit the runtime gate**

~~~bash
git add source/rl_training/rl_training/online_adaptation/agent.py \
  tests/online_adaptation/test_publisher_agent.py
git commit -m "feat: fail closed on invalid terrain scans"
~~~

### Task 4: Add CLI, Runtime, and Metadata Propagation

**Files:**
- Modify: `scripts/reinforcement_learning/online_adaptation/cli.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/train_residual_sac.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/evaluate_flat_online_curriculum.py`
- Modify: `tests/online_adaptation/test_cli.py`
- Modify: `tests/online_adaptation/test_play_runtime.py`
- Modify: `tests/online_adaptation/test_run_flat_online_curriculum.py`
- Modify: `tests/online_adaptation/test_flat_online_acceptance.py`

- [ ] **Step 1: Add failing parser/configuration tests**

Add `--terrain-feature-source {geometry,height_scan}` and
`--terrain-min-valid-fraction` to play and curriculum parsers. Assert default
`geometry`, finite values in `(0, 1]`, rejection of invalid values, and worker
commands including the options only for `height_scan`.

- [ ] **Step 2: Run focused parser tests and verify failure**

~~~bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_cli.py \
  tests/online_adaptation/test_run_flat_online_curriculum.py \
  tests/online_adaptation/test_flat_online_acceptance.py -q
~~~

Expected: new option/configuration assertions fail.

- [ ] **Step 3: Add immutable configuration fields and forwarding**

Extend `TargetConfiguration` with the source and minimum valid fraction and
validate them. Forward both through development and sealed workers. Do not
include these algorithm fields in `FlatOnlineProtocol.sha256()`; record them
in the formal selected configuration and run manifest instead.

- [ ] **Step 4: Wire the adapter and agent in runtime loops**

Construct the adapter with the parsed source/configuration, read
`terrain_valid` before `begin_step`, pass it to the agent, and log validity and
invalid-scan counts in metrics and summaries. Include `terrain_metadata` and
source version in both experiment-checkpoint state copies so resumed runs reject
a mismatch before actor restoration.

- [ ] **Step 5: Run focused runtime tests**

Run the command from Step 2 and inspect generated worker commands. Expected:
geometry commands retain the legacy shape; height-scan commands carry the frozen
source and threshold.

- [ ] **Step 6: Commit runtime propagation**

~~~bash
git add scripts/reinforcement_learning/online_adaptation/cli.py \
  scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  scripts/reinforcement_learning/online_adaptation/train_residual_sac.py \
  scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  scripts/reinforcement_learning/online_adaptation/evaluate_flat_online_curriculum.py \
  tests/online_adaptation/test_cli.py \
  tests/online_adaptation/test_play_runtime.py \
  tests/online_adaptation/test_run_flat_online_curriculum.py \
  tests/online_adaptation/test_flat_online_acceptance.py
git commit -m "feat: propagate height scan residual configuration"
~~~

### Task 5: Enforce Checkpoint Provenance and Document Operations

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/checkpointing.py` only if a reusable metadata validator is required
- Modify: `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py`
- Modify: `tests/online_adaptation/test_checkpointing.py`
- Modify: `docs/lite3-online-learning-principles.md`

- [ ] **Step 1: Add checkpoint mismatch tests**

Save an experiment state with `terrain_feature_source="geometry"`, then load it
with an expected `height_scan` source and assert a deterministic provenance
error before actor/replay restoration. Add a round-trip test for metadata and
invalid-scan counters.

- [ ] **Step 2: Implement source validation**

Keep the generic checkpoint envelope schema unchanged. Add a small runtime
validator around the experiment state that checks source, schema version, grid
shape, normalization constants, and valid-hit threshold before
`OnlineAdaptationAgent.load_state_dict` is called.

- [ ] **Step 3: Document the operational contract**

Document that height scanning is opt-in for perceptive adaptation, the base
PPO remains proprioceptive, invalid scans force zero residual, and formal
flat-only baseline results continue to use `geometry` mode.

- [ ] **Step 4: Run focused checks**

~~~bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_checkpointing.py -q
PYTHONPATH=source/rl_training:. \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training/online_adaptation \
  scripts/reinforcement_learning/online_adaptation
git diff --check
~~~

Expected: all checkpoint tests pass, compileall exits zero, and diff check
prints no diagnostics.

- [ ] **Step 5: Commit provenance and docs**

~~~bash
git add source/rl_training/rl_training/online_adaptation/checkpointing.py \
  scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  tests/online_adaptation/test_checkpointing.py \
  docs/lite3-online-learning-principles.md
git commit -m "docs: specify height scan provenance and fallback"
~~~

### Task 6: Full Verification

**Files:** No source changes unless a test exposes a regression.

- [ ] **Step 1: Run the complete online-adaptation suite**

~~~bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation -q
~~~

Expected: zero failures; explicitly report Isaac-only skips.

- [ ] **Step 2: Compile modified modules**

~~~bash
PYTHONPATH=source/rl_training:. \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training/online_adaptation \
  scripts/reinforcement_learning/online_adaptation
~~~

Expected: exit code zero.

- [ ] **Step 3: Review source status without touching unrelated work**

~~~bash
git status --short
git diff --check
git log --oneline -8
~~~

Preserve all pre-existing dirty files and report that GPU simulation, real
sensor latency, and sim-to-real domain randomization were not validated in this
slice.

