# Lite3 Direct Reset Command Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the acceleration-limited forward target across a standing deployment reset so the frozen Lite3 policy receives the command lifecycle initialized by the environment.

**Architecture:** The IsaacLab command term remains the sole owner of command reset and slew state. The deployment wrapper validates standing state and reports the handoff generation, but does not mutate `vel_command_b` or `target_vel_command_b`.

**Tech Stack:** Python 3.11, PyTorch, pytest, IsaacLab/Isaac Sim, RSL-RL

---

### Task 1: Add the Command-Handoff Regression Test

**Files:**
- Modify: `tests/online_adaptation/test_deployment_startup.py`

- [ ] **Step 1: Add a fake acceleration-limited command term to `FakeEnv`**

Add these helpers above `FakeEnv`:

```python
class FakeVelocityCommand:
    def __init__(self):
        self.vel_command_b = torch.zeros(1, 3)
        self.target_vel_command_b = torch.tensor([[0.3, 0.0, 0.0]])


class FakeCommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "base_velocity"
        return self.command
```

Initialize them in `FakeEnv.__init__`:

```python
self.velocity_command = FakeVelocityCommand()
self.command_manager = FakeCommandManager(self.velocity_command)
```

Restore the underlying environment's intended command state in `FakeEnv.reset()`:

```python
self.velocity_command.vel_command_b.zero_()
self.velocity_command.target_vel_command_b[:] = torch.tensor([[0.3, 0.0, 0.0]])
```

- [ ] **Step 2: Write the failing wrapper test**

```python
def test_wrapper_preserves_acceleration_limited_target_command():
    env = FakeEnv()
    wrapped = Lite3DeploymentStartupWrapper(env, state_reader=reader)

    wrapped.reset()

    assert torch.equal(env.velocity_command.vel_command_b, torch.zeros(1, 3))
    assert torch.equal(
        env.velocity_command.target_vel_command_b,
        torch.tensor([[0.3, 0.0, 0.0]]),
    )
```

- [ ] **Step 3: Run the test and verify the expected failure**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/online_adaptation/test_deployment_startup.py::test_wrapper_preserves_acceleration_limited_target_command -q
```

Expected: FAIL because the actual target is `[0.0, 0.0, 0.0]` after wrapper reset.

### Task 2: Stop the Wrapper From Mutating Command State

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/deployment_startup.py`
- Test: `tests/online_adaptation/test_deployment_startup.py`

- [ ] **Step 1: Remove wrapper-owned command clearing**

Delete `_zero_commands()` and its invocation from `reset()`. The resulting reset sequence must be:

```python
def reset(self, **kwargs):
    observation, info = self.env.reset(**kwargs)
    self._set_startup_active(True)
    state = self.state_reader(self.unwrapped)
    self._validate_state(state, 0, 0.0)
    self._validate_handoff()
    self._set_startup_active(False)
    self.startup_generation += 1
    self._pending_generation = self.startup_generation
    info = dict(info)
    info["startup_generation"] = self.startup_generation
    return observation, info
```

- [ ] **Step 2: Run all startup-wrapper tests**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/online_adaptation/test_deployment_startup.py -q
```

Expected: all tests pass, including the new target-command regression test.

### Task 3: Verify the Complete Change

**Files:**
- Verify: `source/rl_training/rl_training/online_adaptation/deployment_startup.py`
- Verify: `tests/online_adaptation/test_deployment_startup.py`

- [ ] **Step 1: Run the complete CPU suite**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests -q
```

Expected: zero failures.

- [ ] **Step 2: Run syntax and diff validation**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q source/rl_training/rl_training tests
git diff --check
```

Expected: both commands exit 0 with no output.

- [ ] **Step 3: Run the frozen policy on the actual flat-retention track**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=source/rl_training:.
python scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_flat/2026-08-01_15-32-10_flat_stability_resume8800_seed42_20260801/model_9999.pt \
  --mode base --task FlatOnline-Deeprobotics-Lite3-v0 \
  --seed 7 --terrain-seed 7 --forward-speed 0.3 --max-steps 500 \
  --flat-retention-probe \
  --run-dir /tmp/lite3_eval_flat_command_handoff_20260802 \
  --device cuda:0 --headless
```

Expected: `summary.json` reports `steps: 500`, `fell: false`, and no joint-velocity safety violation. If it still terminates early, retain the regression fix and return to root-cause tracing using per-step metrics rather than relaxing safety limits.

- [ ] **Step 4: Inspect final scope**

```bash
git status --short
git diff --stat
git diff --check
```

Expected: only the approved standing-reset work, this command-handoff fix, their tests, and design/plan documents are present. `/home/tjx/Lite3_rl_deploy` remains untouched.
