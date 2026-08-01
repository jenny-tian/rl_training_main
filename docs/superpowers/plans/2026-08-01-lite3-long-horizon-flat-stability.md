# Lite3 Long-Horizon Flat Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and seal a Lite3 flat PPO that completes 100 deployment-matched qualification episodes and the formal 100,000-transition replay without a fall or safety-invalid row.

**Architecture:** A small Lite3 stability-contract module owns the unchanged runtime tilt limit and guard threshold. Lite3 training termination and reward terms consume that contract, while a separate deployment-matched qualification command evaluates the frozen PPO on the existing deterministic `FlatOnline` environment before the unchanged formal replay collector is allowed to run.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab/Isaac Sim 5.1, RSL-RL, Gymnasium, pytest, JSON/JSONL, Git.

---

## File Structure

- Create `source/rl_training/rl_training/lite3_stability_contract.py`: dependency-light constants and tensor functions for the Lite3 tilt boundary and guard-band penalty.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/events.py`: parameterize orientation termination with an XY-norm threshold.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`: expose the continuous guard-band penalty as an Isaac reward wrapper.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`: declare parameters for the generic orientation termination and reward term.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py`: bind Lite3 training to the shared `0.65` threshold and enable the guard-band reward.
- Create `tests/online_adaptation/test_lite3_stability_contract.py`: unit-test boundary geometry, penalty shape, and config wiring.
- Modify `source/rl_training/rl_training/online_adaptation/safety.py`: consume the shared Lite3 constants as defaults without changing runtime values.
- Modify `scripts/reinforcement_learning/online_adaptation/cli.py`: add a dedicated deployment qualification parser.
- Create `source/rl_training/rl_training/online_adaptation/deployment_qualification.py`: validate and aggregate exact 100-episode evidence independently of Isaac.
- Create `scripts/reinforcement_learning/online_adaptation/qualify_flat_deployment.py`: run the frozen PPO in `FlatOnline` and write strict qualification evidence.
- Create `tests/online_adaptation/test_deployment_qualification.py`: unit-test report validation and pass/fail gates.
- Modify `tests/online_adaptation/test_cli.py`: lock deployment qualification defaults.
- Modify `docs/lite3-online-learning-principles.md`: record the failed candidate, new commands, hashes, and final evidence.

### Task 1: Shared Lite3 Stability Contract

**Files:**
- Create: `source/rl_training/rl_training/lite3_stability_contract.py`
- Create: `tests/online_adaptation/test_lite3_stability_contract.py`

- [ ] **Step 1: Write failing tensor-contract tests**

Add tests that import the missing contract and establish exact behavior:

```python
import torch

from rl_training.lite3_stability_contract import (
    LITE3_MAX_PROJECTED_GRAVITY_XY,
    LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY,
    projected_gravity_xy_is_unsafe,
    tilt_guard_band_penalty,
)


def test_diagonal_tilt_uses_norm_not_individual_components():
    gravity = torch.tensor([[0.47, 0.47, -0.75]])
    assert gravity[:, :2].abs().amax().item() < 0.7
    assert projected_gravity_xy_is_unsafe(gravity).item()


def test_tilt_boundary_is_strictly_greater_than_unchanged_limit():
    gravity = torch.tensor([[0.65, 0.0, -1.0], [0.6501, 0.0, -1.0]])
    assert LITE3_MAX_PROJECTED_GRAVITY_XY == 0.65
    assert projected_gravity_xy_is_unsafe(gravity).tolist() == [False, True]


def test_guard_penalty_is_zero_then_monotonic_and_bounded():
    gravity = torch.tensor([[0.55, 0.0, -1.0], [0.60, 0.0, -1.0], [0.65, 0.0, -1.0]])
    penalty = tilt_guard_band_penalty(gravity)
    assert LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY == 0.55
    assert torch.allclose(penalty, torch.tensor([0.0, 0.25, 1.0]), atol=1.0e-5)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_lite3_stability_contract.py -q
```

Expected: collection fails with `ModuleNotFoundError: rl_training.lite3_stability_contract`.

- [ ] **Step 3: Implement the minimal contract**

Create:

```python
from __future__ import annotations

import torch

LITE3_MAX_PROJECTED_GRAVITY_XY = 0.65
LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY = 0.55


def projected_gravity_xy_is_unsafe(
    projected_gravity: torch.Tensor,
    *,
    max_xy_norm: float = LITE3_MAX_PROJECTED_GRAVITY_XY,
) -> torch.Tensor:
    return (projected_gravity[..., 2] > 0) | (
        projected_gravity[..., :2].norm(dim=-1) > max_xy_norm
    )


def tilt_guard_band_penalty(
    projected_gravity: torch.Tensor,
    *,
    guard_xy_norm: float = LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY,
    max_xy_norm: float = LITE3_MAX_PROJECTED_GRAVITY_XY,
) -> torch.Tensor:
    if not 0.0 <= guard_xy_norm < max_xy_norm:
        raise ValueError("tilt guard must satisfy 0 <= guard < maximum")
    normalized = (projected_gravity[..., :2].norm(dim=-1) - guard_xy_norm) / (
        max_xy_norm - guard_xy_norm
    )
    return normalized.clamp(0.0, 1.0).square()
```

- [ ] **Step 4: Run the contract tests and verify GREEN**

Run the command from Step 2. Expected: `3 passed`.

- [ ] **Step 5: Commit the contract**

```bash
git add source/rl_training/rl_training/lite3_stability_contract.py \
  tests/online_adaptation/test_lite3_stability_contract.py
git commit -m "feat: define lite3 stability contract"
```

If the commit fails, preserve the working-tree changes and continue as authorized.

### Task 2: Align Lite3 Training Termination and Reward

**Files:**
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/events.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py`
- Modify: `tests/online_adaptation/test_lite3_stability_contract.py`

- [ ] **Step 1: Write failing wiring tests**

Append AST/config tests that require the parameterized termination, reward wrapper, and Lite3 bindings:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_lite3_config_binds_training_to_shared_stability_contract():
    source = (ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py").read_text()
    assert 'self.terminations.bad_orientation_2.params["max_xy_norm"] = LITE3_MAX_PROJECTED_GRAVITY_XY' in source
    assert 'self.rewards.tilt_guard_band.weight = -20.0' in source
    assert 'self.rewards.tilt_guard_band.params["guard_xy_norm"] = LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY' in source
    assert 'self.rewards.tilt_guard_band.params["max_xy_norm"] = LITE3_MAX_PROJECTED_GRAVITY_XY' in source


def test_training_termination_delegates_to_xy_norm_contract_when_configured():
    source = (ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/events.py").read_text()
    assert "projected_gravity_xy_is_unsafe(asset.data.projected_gravity_b" in source
```

- [ ] **Step 2: Run focused tests and verify RED**

Run the Task 1 test command. Expected: the new wiring assertions fail.

- [ ] **Step 3: Parameterize the termination and add the reward wrapper**

Change `bad_orientation_2` to accept `max_xy_norm: float | None = None`. Preserve
the existing component-wise `0.7` behavior when the parameter is `None`; when
Lite3 supplies a threshold, return:

```python
return projected_gravity_xy_is_unsafe(
    asset.data.projected_gravity_b, max_xy_norm=max_xy_norm
)
```

Add this reward wrapper in `mdp/rewards.py`:

```python
def tilt_guard_band(
    env: ManagerBasedRLEnv,
    guard_xy_norm: float,
    max_xy_norm: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return tilt_guard_band_penalty(
        asset.data.projected_gravity_b,
        guard_xy_norm=guard_xy_norm,
        max_xy_norm=max_xy_norm,
    )
```

Declare a zero-weight `tilt_guard_band` `RewTerm` in `RewardsCfg`, and give
`bad_orientation_2` an explicit default `params={"max_xy_norm": None}` so
non-Lite3 environments preserve their component-wise behavior exactly.

- [ ] **Step 4: Bind Lite3 to the shared constants**

In `DeeproboticsLite3RoughEnvCfg.__post_init__`, set:

```python
self.rewards.tilt_guard_band.weight = -20.0
self.rewards.tilt_guard_band.params["guard_xy_norm"] = (
    LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY
)
self.rewards.tilt_guard_band.params["max_xy_norm"] = (
    LITE3_MAX_PROJECTED_GRAVITY_XY
)
self.terminations.bad_orientation_2.params["max_xy_norm"] = (
    LITE3_MAX_PROJECTED_GRAVITY_XY
)
```

- [ ] **Step 5: Run focused and configuration tests**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_lite3_stability_contract.py \
  tests/online_adaptation/test_terrain_guards.py -q
```

Expected: all selected tests pass; Isaac-dependent tests may be explicitly skipped.

- [ ] **Step 6: Commit the training alignment**

```bash
git add source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/events.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py \
  tests/online_adaptation/test_lite3_stability_contract.py
git commit -m "fix: align lite3 training tilt safety"
```

### Task 3: Make Runtime Safety Consume the Same Constants

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/safety.py`
- Modify: `tests/online_adaptation/test_safety.py`

- [ ] **Step 1: Write a failing default-contract test**

```python
from rl_training.lite3_stability_contract import (
    LITE3_MAX_PROJECTED_GRAVITY_XY,
    LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY,
)


def test_safety_defaults_share_lite3_training_contract():
    limits = SafetyLimits(residual_limits=torch.ones(12))
    assert limits.max_projected_gravity_xy == LITE3_MAX_PROJECTED_GRAVITY_XY
    assert limits.residual_guard_projected_gravity_xy == LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY
```

- [ ] **Step 2: Verify RED by requiring imports in the safety source**

Also assert the safety source names both shared constants; run `tests/online_adaptation/test_safety.py -q`. Expected: source/import assertion fails while numeric behavior remains unchanged.

- [ ] **Step 3: Replace literal defaults with shared constants**

Import both constants and use them as the two dataclass defaults. Do not change any comparisons or values.

- [ ] **Step 4: Run safety tests and commit**

Run `tests/online_adaptation/test_safety.py -q`; expected: all pass. Then:

```bash
git add source/rl_training/rl_training/online_adaptation/safety.py \
  tests/online_adaptation/test_safety.py
git commit -m "refactor: share lite3 tilt safety limits"
```

### Task 4: Pure Deployment Qualification Report

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/deployment_qualification.py`
- Create: `tests/online_adaptation/test_deployment_qualification.py`

- [ ] **Step 1: Write failing exact-evidence tests**

Define `_valid_record()` with seed 42, `attempted_rows=100_000`, `completed_episodes=100`, `timeouts=100`, all failure counts zero, `max_projected_gravity_xy=0.64`, `min_base_height=0.28`, and `mean_velocity_error=0.1`. Test:

```python
def test_deployment_report_accepts_exact_100_episode_evidence():
    report = build_deployment_qualification_report(
        task="FlatOnline-Deeprobotics-Lite3-v0",
        base_checkpoint="base.pt",
        record=_valid_record(),
        expected_seed=42,
        expected_steps=100_000,
        expected_episodes=100,
        max_mean_velocity_error=0.35,
        max_projected_gravity_xy=0.65,
    )
    assert report["passed"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("attempted_rows", 99_999), ("completed_episodes", 99), ("timeouts", 99),
     ("falls", 1), ("premature_terminations", 1), ("paused_rows", 1),
     ("projected_rows", 1), ("nonfinite_rows", 1)],
)
def test_deployment_report_rejects_incomplete_or_invalid_evidence(field, value):
    record = _valid_record()
    record[field] = value
    assert _report(record)["passed"] is False


def test_deployment_report_rejects_tilt_over_unchanged_threshold():
    record = _valid_record()
    record["max_projected_gravity_xy"] = 0.650001
    assert _report(record)["passed"] is False
```

- [ ] **Step 2: Run and verify RED**

Run the new test file. Expected: collection fails because the module is missing.

- [ ] **Step 3: Implement strict validation and aggregation**

Create `build_deployment_qualification_report(...)`. Validate exact nonnegative integer count fields (reject booleans), finite floats, exact task/seed/step/episode/timeout values, all failure counts equal zero, tilt `<= 0.65`, positive minimum height, and mean velocity error `<= 0.35`. Return the input record, thresholds, and one top-level `passed` boolean.

- [ ] **Step 4: Run tests and commit**

Run the new test file; expected: all pass. Then:

```bash
git add source/rl_training/rl_training/online_adaptation/deployment_qualification.py \
  tests/online_adaptation/test_deployment_qualification.py
git commit -m "feat: validate lite3 deployment qualification"
```

### Task 5: Deployment-Matched Qualification CLI and Worker

**Files:**
- Modify: `scripts/reinforcement_learning/online_adaptation/cli.py`
- Create: `scripts/reinforcement_learning/online_adaptation/qualify_flat_deployment.py`
- Modify: `tests/online_adaptation/test_cli.py`
- Modify: `tests/online_adaptation/test_deployment_qualification.py`

- [ ] **Step 1: Write failing CLI-default tests**

```python
def test_deployment_qualification_defaults_match_formal_distribution():
    args = build_deployment_qualify_parser().parse_args(["--base-checkpoint", "base.pt"])
    assert args.task == "FlatOnline-Deeprobotics-Lite3-v0"
    assert args.seed == 42
    assert args.expected_episodes == 100
    assert args.expected_steps == 100_000
    assert args.max_projected_gravity_xy == 0.65
    assert args.max_mean_velocity_error == 0.35
```

Add an AST test requiring the worker to reject every task except the exact
`FLAT_ONLINE_TASK`, run actions through `SafetyFilter`, call
`split_rsl_rl_endings`, abort when `safety_mask` is false, and call
`build_deployment_qualification_report`.

- [ ] **Step 2: Run and verify RED**

Run `test_cli.py` and `test_deployment_qualification.py`. Expected: missing parser/worker assertions fail.

- [ ] **Step 3: Add the dedicated parser**

Add `build_deployment_qualify_parser()` with required checkpoint and defaults from Step 1, plus `--output`, `--device`, and AppLauncher arguments. Keep the existing short qualification parser unchanged.

- [ ] **Step 4: Implement the single-process Isaac worker**

Use the existing frozen-policy setup from `qualify_base_policy.py`, but parse only `FlatOnline-Deeprobotics-Lite3-v0`, one environment, seed 42, and leave its 1,000-step time limit enabled. Before `gym.make`, assert the parsed task matches `default_flat_online_protocol()`. For every step:

```python
raw_action = policy(observation)
finite_rows = torch.isfinite(raw_action).all(dim=-1, keepdim=True)
action = clip_lite3_policy_action(raw_action)
decision = safety.project(
    action, torch.zeros_like(action), adapter.safety_state(timeout),
    force_zero_residual=True,
)
safety_mask = (~decision.paused.bool()) & (~decision.projected.bool()) & finite_rows
if not bool(safety_mask.all().item()):
    break
observation, _, done, extras = wrapped_env.step(decision.action)
terminated, truncated, _ = split_rsl_rl_endings(done, extras)
```

Account separately for timeout truncations and premature terminations. After an
episode boundary, replace the `SafetyFilter` with a new instance using the same
limits so its recurrent pause/contact state cannot leak across the wrapper's
automatic reset. Track extrema and velocity error, stop after 100,000 attempted
rows, write JSON through a temporary file followed by `Path.replace`, and return
exit code 0 only when `passed` is true. Never skip an invalid row.

- [ ] **Step 5: Run focused tests and commit**

Run both focused files; expected: all pass. Then:

```bash
git add scripts/reinforcement_learning/online_adaptation/cli.py \
  scripts/reinforcement_learning/online_adaptation/qualify_flat_deployment.py \
  tests/online_adaptation/test_cli.py \
  tests/online_adaptation/test_deployment_qualification.py
git commit -m "feat: qualify lite3 over 100 flat episodes"
```

### Task 6: CPU Regression Verification

**Files:**
- No production changes expected.

- [ ] **Step 1: Run the focused suite**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_lite3_stability_contract.py \
  tests/online_adaptation/test_deployment_qualification.py \
  tests/online_adaptation/test_qualification.py \
  tests/online_adaptation/test_cli.py \
  tests/online_adaptation/test_safety.py \
  tests/online_adaptation/test_collect_flat_replay.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run all CPU tests**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest tests -q
```

Expected: all tests pass with only explicit Isaac/GPU skips.

- [ ] **Step 3: Run static verification**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training scripts/reinforcement_learning/online_adaptation tests/online_adaptation
git diff --check
git status --short
```

Expected: compilation and whitespace checks succeed; status contains only intentional changes.

### Task 7: GPU Smoke Training

**Files:**
- Generated: `logs/rsl_rl/deeprobotics_lite3_flat/*_flat_stability_smoke_seed42_20260801/`

- [ ] **Step 1: Confirm GPU availability and absence of conflicting jobs**

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
pgrep -af "python .*train.py --task Flat-Deeprobotics-Lite3-v0"
```

Expected: one GPU has sufficient free memory and no conflicting flat training uses it.

- [ ] **Step 2: Run a one-iteration smoke test**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-Deeprobotics-Lite3-v0 --num_envs 64 --seed 42 \
  --max_iterations 1 --run_name flat_stability_smoke_seed42_20260801 \
  --device cuda:0 --headless
```

Expected: exit code 0, finite losses/rewards, and a saved resolved `env.yaml`.

- [ ] **Step 3: Verify the resolved safety contract**

```bash
rg -n -C 3 "tilt_guard_band|bad_orientation_2|max_xy_norm|weight: -20.0" \
  logs/rsl_rl/deeprobotics_lite3_flat/*_flat_stability_smoke_seed42_20260801/params/env.yaml
```

Expected: tilt reward is enabled with guard `0.55`, maximum `0.65`, and termination maximum `0.65`. Do not start full training if any value is absent or different.

### Task 8: Full PPO Training

**Files:**
- Generated: `logs/rsl_rl/deeprobotics_lite3_flat/*_flat_stability_seed42_20260801/`

- [ ] **Step 1: Start a unique 10,000-iteration run**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-Deeprobotics-Lite3-v0 --num_envs 4096 --seed 42 \
  --max_iterations 10000 --run_name flat_stability_seed42_20260801 \
  --device cuda:0 --headless
```

Expected: training completes normally with `model_9999.pt`. If run asynchronously, record PID/log path and monitor without starting another candidate.

- [ ] **Step 2: Record immutable training provenance inputs**

Create `training_command.txt` beside the checkpoint containing exactly the command from Step 1, then compute SHA-256 for `model_9999.pt`, `params/env.yaml`, and `params/agent.yaml`. Use a new run directory; never overwrite the prior failed run.

### Task 9: Two-Stage Qualification and Manifest

**Files:**
- Generated: short qualification JSON and deployment qualification JSON beside the new checkpoint.
- Generated: `flat_baseline_manifest.json` beside the new checkpoint.

- [ ] **Step 1: Run the existing five-seed screening**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  --base-checkpoint "$FLAT_RUN/model_9999.pt" \
  --task Flat-Deeprobotics-Lite3-v0 --seeds 7 11 19 23 31 \
  --steps-per-seed 2000 --max-mean-velocity-error 0.35 \
  --output "$FLAT_RUN/base_qualification_short.json" --device cuda:0 --headless
```

Expected: exit 0 and aggregate `passed: true` with all four failure counts zero.

- [ ] **Step 2: Run the deployment-matched 100-episode gate**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/online_adaptation/qualify_flat_deployment.py \
  --base-checkpoint "$FLAT_RUN/model_9999.pt" \
  --task FlatOnline-Deeprobotics-Lite3-v0 --seed 42 \
  --expected-episodes 100 --expected-steps 100000 \
  --output "$FLAT_RUN/base_qualification_deployment.json" \
  --device cuda:0 --headless
```

Expected: exit 0, `passed: true`, exactly 100,000 attempted rows, exactly 100 episodes/timeouts, zero failure counts, and maximum tilt at most `0.65`.

- [ ] **Step 3: Build the baseline manifest**

Pass `env.yaml`, `agent.yaml`, and both qualification JSON files through repeated
`--config` arguments to `build_flat_baseline_manifest.py`; pass
`training_command.txt` through its dedicated `--training-command-file` argument.
Expected: a new manifest whose hashes match all checkpoint/config/qualification
files and whose normalized training command matches the command file. The git SHA
must identify the training code commit, not a later evidence-only commit.

### Task 10: Reset Probe and Formal Replay

**Files:**
- Generated: `/home/tjx/rl_training_main/logs/online_adaptation/flat_replay_stability_seed42_reset_probe_20260801/`
- Generated: `/home/tjx/rl_training_main/logs/online_adaptation/flat_replay_stability_seed42_formal_20260801/`

- [ ] **Step 1: Run a fresh cross-reset probe**

Collect at least 2,200 attempted rows with the existing collector and new manifest. Expected: at least two timeout truncations, zero falls, zero safety-invalid rows, and no projection/pause/non-finite evidence. Preserve the probe directory.

- [ ] **Step 2: Run the unchanged formal collector**

Use the existing formal command with seed 42, exactly 100,000 accepted transitions, the new checkpoint/manifest, and the fresh formal output directory. Expected: exit 0; `flat_replay.pt`, final metrics, and final manifest are produced. Any invalid row aborts immediately and remains recorded.

- [ ] **Step 3: Validate final artifacts**

Verify the final manifest requires `transition_count == 100000`, `falls == 0`, `safety_invalid_rows == 0`, and that checkpoint/replay/config/report SHA-256 values match files on disk. Run the complete CPU suite, `compileall`, and `git diff --check` again before declaring completion.

- [ ] **Step 4: Record evidence and commit documentation**

Append the failed row-65,558 evidence, training run, checkpoint hash, both qualification reports, replay hash, durations, and exact commands to `docs/lite3-online-learning-principles.md`. Then:

```bash
git add docs/lite3-online-learning-principles.md
git commit -m "docs: record lite3 long-horizon stability evidence"
```

Do not push. If commit fails, report it and retain the documentation changes.
