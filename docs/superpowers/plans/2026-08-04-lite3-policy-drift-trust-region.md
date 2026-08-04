# Lite3 Policy-Drift Trust Region Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Gate residual Actor publication with an optional calibrated deterministic behavior-drift threshold and expose it through the formal worker configuration.

**Architecture:** Keep SAC optimization unchanged. The candidate publication transaction measures drift against the existing episode-local reference Actor, rejects an over-limit candidate before publication, and restores the full updater state. The CLI and curriculum supervisor carry the threshold as an immutable configuration value.

**Tech Stack:** Python 3, PyTorch, pytest, argparse, existing Lite3 online-adaptation package.

---

### Task 1: Repair the current data-contract test

**Files:**
- Modify: `tests/online_adaptation/test_package.py:91-104`

- [ ] **Step 1: Update the expected field list**

Add `"foot_edge_offset"` after `"foot_contact"` in the `RewardSignals`
field-order assertion. Keep the assertion explicit so future signal additions
still require an intentional contract update.

- [ ] **Step 2: Run the focused test and verify it passes**

Run:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_package.py::test_transition_batch_contract -q
```

Expected: `1 passed`.

- [ ] **Step 3: Commit the test-contract fix**

```bash
git add tests/online_adaptation/test_package.py
git commit -m "test: update reward signal contract"
```

### Task 2: Add the publication threshold API with failing tests first

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/actor_update.py:20-60`
- Modify: `tests/online_adaptation/test_actor_update.py`

- [ ] **Step 1: Add failing tests**

Add tests that call `update_and_publish_actor(..., max_residual_mse=...)` and
verify an over-limit `PolicyDrift` returns `accepted == False`, reason
`policy_residual_mse_limit`, no publisher call, version zero, and an exact
updater state round trip. Add a below-limit case that still publishes. Add
negative, NaN, and infinity validation coverage through the public helper;
`None` must remain the documented disabled value.

- [ ] **Step 2: Run the focused tests and verify the expected failure**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_actor_update.py -q
```

Expected: new tests fail because the helper does not yet accept or enforce the
threshold.

- [ ] **Step 3: Implement minimal validation and rejection**

Add a finite non-negative validator and an optional keyword argument to
`update_and_publish_actor`. After `drift_monitor.measure()` returns, return
`"policy_residual_mse_limit"` when the measured residual MSE is above the
configured threshold. Pass the rejection reason to the existing
`candidate_validator` path so `SACUpdater` restores its pre-update state.

- [ ] **Step 4: Run the focused tests and verify they pass**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_actor_update.py tests/online_adaptation/test_sac.py -q
```

Expected: all tests in both files pass.

- [ ] **Step 5: Commit the publication gate**

```bash
git add source/rl_training/rl_training/online_adaptation/actor_update.py \
  tests/online_adaptation/test_actor_update.py
git commit -m "feat: gate residual actor publication by drift"
```

### Task 3: Expose the threshold in the single-episode CLI

**Files:**
- Modify: `scripts/reinforcement_learning/online_adaptation/cli.py:270-280`
- Modify: `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py:1144-1156`
- Modify: `tests/online_adaptation/test_cli.py`
- Modify: `tests/online_adaptation/test_play_runtime.py`

- [ ] **Step 1: Add failing parser and call-site tests**

Assert the play parser defaults `max_policy_residual_mse` to `None`, accepts a
finite non-negative value, and rejects a negative value. Extend the existing
actor update call test to assert the parsed value is passed to
`update_and_publish_actor`.

- [ ] **Step 2: Run the focused tests and verify they fail**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_cli.py tests/online_adaptation/test_play_runtime.py -q
```

Expected: new assertions fail because the option and forwarding are absent.

- [ ] **Step 3: Add the optional CLI option and forward it**

Use the existing `_nonnegative_float` parser and pass
`args.max_policy_residual_mse` into `update_and_publish_actor`. Do not assign a
non-`None` default in this slice; existing runs must remain behaviorally
unchanged unless the threshold is explicitly selected.

- [ ] **Step 4: Run the focused tests and verify they pass**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_cli.py tests/online_adaptation/test_play_runtime.py -q
```

Expected: all tests in both files pass.

- [ ] **Step 5: Commit the CLI wiring**

```bash
git add scripts/reinforcement_learning/online_adaptation/cli.py \
  scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  tests/online_adaptation/test_cli.py tests/online_adaptation/test_play_runtime.py
git commit -m "feat: expose residual drift limit"
```

### Task 4: Carry the threshold through the formal curriculum configuration

**Files:**
- Modify: `scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py:40-120,160-215,230-270`
- Modify: `scripts/reinforcement_learning/online_adaptation/evaluate_flat_online_curriculum.py:75-145`
- Modify: `tests/online_adaptation/test_run_flat_online_curriculum.py`
- Modify: `tests/online_adaptation/test_flat_online_acceptance.py` only if configuration schema assertions require it

- [ ] **Step 1: Add failing configuration tests**

Assert `TargetConfiguration.max_policy_residual_mse` defaults to `None`, accepts
a finite non-negative value, rejects negative/NaN/infinite values, and that
`build_worker_command()` includes `--max-policy-residual-mse` when configured.
Assert the sealed evaluation worker forwards the frozen value from the formal
configuration.

- [ ] **Step 2: Run the focused tests and verify they fail**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_run_flat_online_curriculum.py \
  tests/online_adaptation/test_flat_online_acceptance.py -q
```

Expected: new configuration and forwarding assertions fail.

- [ ] **Step 3: Implement immutable configuration propagation**

Add the optional field and validation to `TargetConfiguration`; include it in
the worker command, the protocol/configuration hash input, and the sealed
evaluation command. Use the option only when the value is not `None`, so the
legacy command shape remains valid for uncalibrated development runs.

- [ ] **Step 4: Run focused tests and inspect generated commands**

Run the same pytest command from Step 2, then run:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest tests/online_adaptation -q
```

Expected: the focused tests pass; the full suite has no failures.

- [ ] **Step 5: Commit configuration propagation**

```bash
git add scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  scripts/reinforcement_learning/online_adaptation/evaluate_flat_online_curriculum.py \
  tests/online_adaptation/test_run_flat_online_curriculum.py \
  tests/online_adaptation/test_flat_online_acceptance.py
git commit -m "feat: freeze policy drift limit in curriculum config"
```

### Task 5: Document calibration and verify the completed slice

**Files:**
- Modify: `docs/lite3-online-learning-principles.md:620-638`
- Modify: `docs/superpowers/specs/2026-08-04-lite3-policy-drift-trust-region-design.md` if implementation details changed

- [ ] **Step 1: Document the operational rule**

State that thresholds are calibrated only on development episodes with
`calibrate_drift_threshold`, frozen in the selected configuration, and never
recomputed from sealed results. Clarify that `None` preserves the old behavior.

- [ ] **Step 2: Run final verification**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest tests/online_adaptation -q
PYTHONPATH=source/rl_training:. \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training/online_adaptation \
  scripts/reinforcement_learning/online_adaptation
git diff --check
```

Expected: pytest reports zero failures, compileall exits zero, and diff check
prints no diagnostics. If the pre-existing Isaac-only skips remain, report
their exact count rather than treating them as passes.

- [ ] **Step 3: Commit documentation**

```bash
git add docs/lite3-online-learning-principles.md \
  docs/superpowers/specs/2026-08-04-lite3-policy-drift-trust-region-design.md \
  docs/superpowers/plans/2026-08-04-lite3-policy-drift-trust-region.md
git commit -m "docs: define residual drift trust region"
```
