# Lite3 Safe Flat Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and seal a strict-flat Lite3 PPO that completes 100,000 deterministic replay transitions with zero falls and zero safety-invalid rows.

**Architecture:** A shared joint-conventions module owns PPO joint order and per-joint normalized action bounds. Training and every inference entry point apply the same bounds before the unchanged safety filter; qualification independently counts all safety interventions before provenance and replay publication are allowed.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab/Isaac Sim 5.1, RSL-RL, Gymnasium, pytest, JSON/JSONL, Git.

---

## File Structure

- Create `source/rl_training/rl_training/lite3_joint_conventions.py`: canonical PPO joint order, safe action bounds, tensor clipping, and physical-bound validation.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py`: configure the shared per-joint clips for all Lite3 execution tasks.
- Modify `source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py`: reorder raw Isaac joint tensors into PPO order.
- Modify `scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py`: clip frozen PPO outputs consistently and retain safety diagnostics.
- Modify `scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py`: execute clipped actions through the safety filter and record intervention evidence.
- Modify `source/rl_training/rl_training/online_adaptation/qualification.py`: make zero intervention counts part of the pass gate.
- Modify `scripts/reinforcement_learning/online_adaptation/train_residual_sac.py`: clip base PPO actions before residual control.
- Modify `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py`: clip current and next base actions before residual control and replay logging.
- Modify tests under `tests/online_adaptation/`: prove bounds, ordering, inference coverage, qualification gates, and collector diagnostics.
- Modify `docs/lite3-online-learning-principles.md`: record commands, hashes, timings, and both failed and successful evidence.

### Task 1: Canonical Joint Order and Safe Bounds

**Files:**
- Create: `source/rl_training/rl_training/lite3_joint_conventions.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py`
- Test: `tests/online_adaptation/test_lite3_joint_conventions.py`

- [ ] **Step 1: Write failing convention tests**

```python
import torch

from rl_training.lite3_joint_conventions import (
    LITE3_POLICY_JOINT_NAMES,
    LITE3_SAFE_ACTION_CLIP_BY_PATTERN,
    clip_lite3_policy_action,
    physical_targets_from_action,
)


def test_policy_joint_order_is_leg_major():
    assert LITE3_POLICY_JOINT_NAMES[:6] == (
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    )


def test_safe_clip_targets_stay_strictly_inside_safety_boundaries():
    lower = torch.tensor([[-100.0] * 12])
    upper = torch.tensor([[100.0] * 12])
    clipped_lower = clip_lite3_policy_action(lower)
    clipped_upper = clip_lite3_policy_action(upper)
    lower_target = physical_targets_from_action(clipped_lower)
    upper_target = physical_targets_from_action(clipped_upper)
    safety_lower = torch.tensor(([-0.51776993 + 0.02] + [-2.65507984 + 0.02] + [0.53533995 + 0.02]) * 4)
    safety_upper = torch.tensor(([0.51776993 - 0.02] + [0.29907990 - 0.02] + [2.78066015 - 0.02]) * 4)
    assert torch.all(lower_target > safety_lower)
    assert torch.all(upper_target < safety_upper)
    assert set(LITE3_SAFE_ACTION_CLIP_BY_PATTERN) == {
        ".*_HipX_joint", ".*_HipY_joint", ".*_Knee_joint"
    }
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_lite3_joint_conventions.py -q
```

Expected: collection fails because `rl_training.lite3_joint_conventions` does not exist.

- [ ] **Step 3: Implement the convention module**

Create immutable leg-major names and bounds. Use inward values derived from the current 0.99 soft limits and 2 cm safety margin:

```python
LITE3_POLICY_JOINT_NAMES = (
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
)
LITE3_SAFE_ACTION_LOWER = (-3.97, -7.93, -2.96) * 4
LITE3_SAFE_ACTION_UPPER = (3.97, 3.70, 5.83) * 4
LITE3_SAFE_ACTION_CLIP_BY_PATTERN = {
    ".*_HipX_joint": (-3.97, 3.97),
    ".*_HipY_joint": (-7.93, 3.70),
    ".*_Knee_joint": (-2.96, 5.83),
}


def clip_lite3_policy_action(action: torch.Tensor) -> torch.Tensor:
    lower = action.new_tensor(LITE3_SAFE_ACTION_LOWER)
    upper = action.new_tensor(LITE3_SAFE_ACTION_UPPER)
    if action.shape[-1] != len(LITE3_POLICY_JOINT_NAMES):
        raise ValueError(f"expected 12 Lite3 actions, got {action.shape[-1]}")
    return torch.maximum(torch.minimum(action, upper), lower)
```

Implement `physical_targets_from_action()` from defaults `(0.0, -0.65, 1.3) * 4` and scales `(0.125, 0.25, 0.25) * 4` for testable configuration validation.

- [ ] **Step 4: Apply identical clipping to every Lite3 environment**

In `DeeproboticsLite3RoughEnvCfg.__post_init__`, replace `{ ".*": (-100.0, 100.0) }` with a copied `LITE3_SAFE_ACTION_CLIP_BY_PATTERN`. Keeping this in the common Lite3 parent ensures strict-flat training and later online terrain execution use the same action manager contract.

- [ ] **Step 5: Run tests and commit**

Run the convention test and the Lite3 environment configuration tests. Expected: all pass.

```bash
git add source/rl_training/rl_training/lite3_joint_conventions.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py \
  tests/online_adaptation/test_lite3_joint_conventions.py
git commit -m "fix: constrain lite3 policy actions safely"
```

### Task 2: Ordered Adapter State

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py`
- Modify: `tests/online_adaptation/test_safety.py`

- [ ] **Step 1: Preserve the failing order test**

Use a raw tensor in joint-type order and IDs in leg-major order:

```python
def test_select_joint_order_maps_robot_storage_to_policy_action_order():
    robot_order = torch.tensor([[10, 20, 30, 40, 50, 60]])
    ordered = select_joint_order(robot_order, [0, 2, 4, 1, 3, 5])
    assert torch.equal(ordered, torch.tensor([[10, 30, 50, 20, 40, 60]]))
```

- [ ] **Step 2: Resolve and validate PPO-order IDs**

Import `LITE3_POLICY_JOINT_NAMES`, call:

```python
self.joint_ids, resolved = self.robot.find_joints(
    list(LITE3_POLICY_JOINT_NAMES), preserve_order=True
)
if tuple(resolved) != LITE3_POLICY_JOINT_NAMES:
    raise ValueError("Lite3 robot joints do not match the frozen PPO joint order")
```

Select joint position, default position, velocity, soft limits, and applied torque with these IDs in `frame()`, `safety_state()`, and `reward_signals()`.

- [ ] **Step 3: Run adapter and safety tests**

Run:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_safety.py \
  tests/online_adaptation/test_isaaclab_adapter.py -q
```

Expected: order test passes; Isaac-only tests are explicitly skipped in the normal CPU suite.

- [ ] **Step 4: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py \
  tests/online_adaptation/test_safety.py
git commit -m "fix: align lite3 adapter joint order"
```

### Task 3: Clip Every Frozen-Policy Inference Path

**Files:**
- Modify: `scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/train_residual_sac.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py`
- Modify: `tests/online_adaptation/test_collect_flat_replay.py`
- Create: `tests/online_adaptation/test_lite3_policy_clip_coverage.py`

- [ ] **Step 1: Write a failing deterministic-action test**

```python
def test_deterministic_base_action_clips_to_shared_lite3_bounds():
    action = collect_flat_replay.deterministic_base_action(
        lambda _observation: torch.tensor([[100.0, -100.0, 100.0] * 4]),
        torch.zeros(1, 3),
    )
    assert action.max().item() == 5.83
    assert action.min().item() == -7.93
```

Add an AST coverage test that requires `clip_lite3_policy_action` around every base-policy result in the collector, qualification worker, residual trainer, and both current/next action sites in the play worker.

- [ ] **Step 2: Run tests and verify RED**

Expected: deterministic action remains at `100`, and inference coverage reports unclipped call sites.

- [ ] **Step 3: Implement shared clipping**

Import `clip_lite3_policy_action` and apply it immediately after every frozen PPO inference, before any safety projection, environment step, residual decision, or replay field is built. Preserve `torch.inference_mode()` and detached outputs.

- [ ] **Step 4: Retain JSON-safe collector diagnostics**

Keep `safety_diagnostics(decision, tensors)` with paused/projected counts, reasons, and non-finite field names. On an invalid row only, record ordered joint names, raw clipped base action, final action, defaults, limits, and scales. Do not modify `safety_mask` or the immediate abort condition.

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_collect_flat_replay.py \
  tests/online_adaptation/test_lite3_policy_clip_coverage.py -q
git add scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py \
  scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  scripts/reinforcement_learning/online_adaptation/train_residual_sac.py \
  scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  tests/online_adaptation/test_collect_flat_replay.py \
  tests/online_adaptation/test_lite3_policy_clip_coverage.py
git commit -m "fix: clip lite3 base actions consistently"
```

### Task 4: Make Safety Evidence a Qualification Gate

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/qualification.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py`
- Modify: `tests/online_adaptation/test_qualification.py`

- [ ] **Step 1: Write failing aggregate tests**

Each test record includes `falls`, `paused_rows`, `projected_rows`, `nonfinite_rows`, `steps`, and `mean_velocity_error`. Add parameterized failures where exactly one safety count is one:

```python
@pytest.mark.parametrize("field", ["paused_rows", "projected_rows", "nonfinite_rows"])
def test_qualification_fails_on_any_safety_intervention(field):
    record = {
        "seed": 7, "falls": 0, "paused_rows": 0, "projected_rows": 0,
        "nonfinite_rows": 0, "steps": 2000, "mean_velocity_error": 0.1,
    }
    record[field] = 1
    report = build_qualification_report("source-task", "base.pt", [record], 0.35)
    assert report["aggregate"][field] == 1
    assert report["aggregate"]["passed"] is False
```

- [ ] **Step 2: Verify RED**

Expected: aggregate omits safety counts or incorrectly passes.

- [ ] **Step 3: Implement strict aggregation**

Validate every required count as an exact nonnegative integer. Sum all four failure counts, compute the mean error, and set `passed` only when all counts are zero and the error gate passes.

- [ ] **Step 4: Collect single-seed safety evidence**

Construct `SafetyFilter(SafetyLimits(...))` once per worker. For each step:

```python
base_action = clip_lite3_policy_action(policy(observation))
finite = bool(torch.isfinite(base_action).all().item())
decision = safety.project(
    base_action,
    torch.zeros_like(base_action),
    adapter.safety_state(torch.zeros(1, 1, dtype=torch.bool, device=base_action.device)),
    force_zero_residual=True,
)
paused_rows += int(decision.paused.sum().item())
projected_rows += int(decision.projected.sum().item())
nonfinite_rows += int(not finite)
observation, _, done, _ = wrapped_env.step(decision.action)
```

Write all counts into each worker record. The multi-seed process keeps rejecting malformed worker records.

- [ ] **Step 5: Run qualification tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_qualification.py -q
git add source/rl_training/rl_training/online_adaptation/qualification.py \
  scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  tests/online_adaptation/test_qualification.py
git commit -m "feat: qualify lite3 base policy safety"
```

### Task 5: Verify the Implementation Before GPU Work

**Files:**
- Modify only if diagnostics reveal a defect in Tasks 1–4.

- [ ] **Step 1: Run the complete CPU test suite**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest -q
```

Expected: zero failures; only explicitly marked Isaac-only skips.

- [ ] **Step 2: Compile and check whitespace**

```bash
PYTHONPATH=source/rl_training:. \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training \
  scripts/reinforcement_learning/online_adaptation
git diff --check
git status --short
```

Expected: compile and whitespace commands print no diagnostics; status contains only intentional changes.

- [ ] **Step 3: Commit any verification-only correction**

Stage only files changed to fix a reproduced verification failure and use a focused `fix:` commit. If no correction was needed, do not create an empty commit.

### Task 6: Train and Strictly Qualify the New PPO

**Files:**
- Generated: `/home/tjx/rl_training_main/logs/rsl_rl/deeprobotics_lite3_flat/*_flat_safe_seed42_20260731/`

- [ ] **Step 1: Record and launch the exact training command**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=source/rl_training:.
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-Deeprobotics-Lite3-v0 \
  --num_envs 4096 \
  --seed 42 \
  --max_iterations 10000 \
  --run_name flat_safe_seed42_20260731 \
  --device cuda:2 \
  --headless
```

Expected: training reaches iteration 9999 and writes `model_9999.pt`. Preserve the entire run directory even if training fails.

- [ ] **Step 2: Resolve exactly one run and checkpoint**

```bash
FLAT_ROOT=/home/tjx/rl_training_main/logs/rsl_rl/deeprobotics_lite3_flat
test "$(find "$FLAT_ROOT" -maxdepth 1 -type d -name '*_flat_safe_seed42_20260731' | wc -l)" -eq 1
FLAT_RUN="$(find "$FLAT_ROOT" -maxdepth 1 -type d -name '*_flat_safe_seed42_20260731')"
BASE_CHECKPOINT="$(find "$FLAT_RUN" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -1)"
test "$(basename "$BASE_CHECKPOINT")" = model_9999.pt
```

- [ ] **Step 3: Run enhanced five-seed qualification**

```bash
python scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --task Flat-Deeprobotics-Lite3-v0 \
  --seeds 7 11 19 23 31 \
  --steps-per-seed 2000 \
  --max-mean-velocity-error 0.35 \
  --output "$FLAT_RUN/base_qualification.json" \
  --device cuda:2 \
  --headless
```

Expected: aggregate has zero falls, paused rows, projected rows, and non-finite rows; `passed` is true. Stop if it fails.

- [ ] **Step 4: Build the provenance manifest**

Write the exact training command to `training_command.txt`. Pass `env.yaml`, `agent.yaml`, and `base_qualification.json` as repeated `--config` inputs so all three are hashed. Build `flat_baseline_manifest.json`, then independently verify checkpoint and report hashes against the manifest.

### Task 7: Probe Resets and Collect the Formal Replay

**Files:**
- Generated: `/home/tjx/rl_training_main/logs/online_adaptation/flat_replay_safe_seed42_reset_probe_20260731/`
- Generated: `/home/tjx/rl_training_main/logs/online_adaptation/flat_replay_safe_seed42_formal_20260731/`
- Modify: `docs/lite3-online-learning-principles.md`

- [ ] **Step 1: Run a reset-boundary probe**

Use the formal collector code path with a fresh output directory and inspect metrics through at least attempted row 2005. Because the formal CLI requires 100,000 transitions, run it normally and stop it only after the second reset evidence has been durably written. Verify rows around 1000 and 2000 have zero falls, paused rows, projected rows, and non-finite fields. Preserve the probe directory as diagnostic evidence; it is not the formal replay.

- [ ] **Step 2: Launch formal collection in a fresh directory**

```bash
python scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --base-manifest "$FLAT_RUN/flat_baseline_manifest.json" \
  --task Flat-Deeprobotics-Lite3-v0 \
  --seed 42 \
  --transitions 100000 \
  --output-dir /home/tjx/rl_training_main/logs/online_adaptation/flat_replay_safe_seed42_formal_20260731 \
  --device cuda:2 \
  --headless
```

Expected: collector exits zero and publishes `flat_replay.pt` plus `flat_replay_manifest.json`.

- [ ] **Step 3: Verify the sealed artifacts**

Check exactly 100,000 transitions, zero falls, zero safety-invalid rows, `frozen == true`, and recompute both checkpoint and replay SHA-256 hashes. Do not change code, thresholds, checkpoint, or seed after reading the formal result.

- [ ] **Step 4: Record evidence and verify once more**

Append commands, run directories, elapsed times, test counts, qualification aggregates, and SHA-256 hashes to `docs/lite3-online-learning-principles.md`. Run the complete CPU suite, compileall, and `git diff --check` again.

- [ ] **Step 5: Commit evidence locally**

```bash
git add docs/lite3-online-learning-principles.md
git commit -m "docs: record safe lite3 flat baseline evidence"
```

Do not push. If the local commit fails, retain the working-tree evidence and continue as previously authorized.
