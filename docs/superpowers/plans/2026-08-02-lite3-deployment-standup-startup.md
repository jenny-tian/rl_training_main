# Lite3 Deployment Stand-Up Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared, reset-scoped belly-down-to-standing startup procedure for Lite3 deployment tasks, then hand the unchanged PPO checkpoint control with the existing velocity-command ramp.

**Architecture:** A pure tensor trajectory module generates the two 1.5-second cubic stand-up phases. A deployment startup wrapper hides those physics steps from PPO/replay/online-learning counters and publishes a handoff generation. Startup-aware termination functions exempt only expected torso/shank contact and orientation while startup is active; finite state, joint limits, and the 30 rad/s speed limit remain enforced. All deployment-facing scripts construct the same wrapper through one shared factory; ordinary Flat/Rough training paths remain untouched.

**Tech Stack:** Python 3.11, PyTorch, Gymnasium, Isaac Lab 5.1, RSL-RL, pytest, CUDA.

---

## File Map

- Create `source/rl_training/rl_training/standup_trajectory.py`: dependency-light cubic interpolation, Lite3 folded/standing targets, and physical-target/action conversion helpers.
- Create `source/rl_training/rl_training/online_adaptation/deployment_startup.py`: startup state machine, Gym wrapper, generation metadata, safety checks, and shared deployment environment factory.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/events.py`: add startup-aware illegal-contact and orientation termination functions with strict default behavior outside startup.
- Modify `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`: expose startup-aware termination function configs without changing normal-task defaults.
- Modify Lite3 `flat_online_env_cfg.py` and `heldout_boxes_env_cfg.py`: set calibrated belly-down root height and startup-aware termination terms; retain the exact command ramp.
- Modify `online_adaptation/safety.py`: add a public temporal-state reset method used by the deployment session at handoff.
- Modify `qualify_flat_deployment.py`, `qualify_base_policy.py`, `collect_flat_replay.py`, `train_residual_sac.py`, and `play_online_adaptation.py`: replace direct deployment env construction with the shared factory and consume handoff generation/reset metadata.
- Create `tests/online_adaptation/test_standup_trajectory.py`: CPU trajectory and action-space contract tests.
- Create `tests/online_adaptation/test_deployment_startup.py`: fake-environment wrapper, lifecycle, safety, and task-scope tests.
- Modify `tests/online_adaptation/test_safety.py`, `test_terrain_guards.py`, `test_flat_online_env_cfg.py`, and `test_lite3_policy_clip_coverage.py`: lock reset API, startup config, calibrated height, and shared factory usage.
- Extend temporary `/tmp/lite3_flat_height_probe.py` only for calibration and startup telemetry; do not add diagnostic state to production code.
- Modify `docs/lite3-online-learning-principles.md`: record calibration, startup gate results, handoff gate results, and any remaining provenance boundary.

## Task 1: Pure Stand-Up Trajectory Contract

**Files:** create `standup_trajectory.py` and `test_standup_trajectory.py`.

- [ ] **Step 1: Write failing CPU tests** for `cubic_hermite_position`, `standup_targets`, and `physical_targets_to_policy_action`: exact phase endpoints, zero endpoint velocities, continuity at 1.5 s, no overshoot between targets, canonical 12-joint shape, finite-input validation, and the known targets `[0,-1.3545311211,2.5494772717] * 4` and `[0,-0.6535351663,1.2710916139] * 4`.
- [ ] **Step 2: Run the focused test file** with `PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest tests/online_adaptation/test_standup_trajectory.py -q`; verify collection or assertions fail because the module is absent.
- [ ] **Step 3: Implement the minimal pure module**. Use tensor broadcasting, validate final dimension 12, finite values, positive duration, and `0 <= elapsed <= duration`; do not mutate inputs. Convert physical targets with the inverse of `physical_targets_from_action`: `(target - default_joint_pos) / action_scale`, then clamp only with the existing `clip_lite3_policy_action` contract.
- [ ] **Step 4: Re-run focused tests and compile** the new module; expected all trajectory tests pass and compileall is silent.
- [ ] **Step 5: Commit** `feat: add lite3 standup trajectory contract`.

## Task 2: Startup-Aware Termination and Safety Reset

**Files:** `mdp/events.py`, `velocity_env_cfg.py`, `safety.py`, focused tests.

- [ ] **Step 1: Add failing tests** using a fake env with `startup_active=True/False`: startup-aware illegal contact returns false only while active; startup-aware orientation returns false only while active; default functions remain unchanged; `SafetyFilter.reset_temporal_state()` clears previous residual, pause, stable-step, and contact-step tensors.
- [ ] **Step 2: Run focused tests** and verify failures identify missing functions/method.
- [ ] **Step 3: Implement startup-aware functions** with the same signatures as existing `mdp.illegal_contact` and `mdp.bad_orientation_2`, reading `env.startup_active` and returning `torch.zeros(env.num_envs, dtype=torch.bool, device=...)` only for the startup exemption. The config default remains the existing function; deployment configs opt into the startup-aware functions explicitly.
- [ ] **Step 4: Implement `SafetyFilter.reset_temporal_state()`** by setting all four temporal fields to `None`, with no changes to projection behavior.
- [ ] **Step 5: Run `test_safety.py`, termination tests, compileall, and commit** `feat: add startup-aware safety lifecycle`.

## Task 3: Deployment Startup Wrapper

**Files:** create `deployment_startup.py` and `test_deployment_startup.py`.

- [ ] **Step 1: Write failing fake-env tests** for: task allowlist accepts only FlatOnline/HeldoutBoxes; reset runs two 1.5-second phases at the environment step size; internal steps are hidden from the outer caller; commands are zero during startup; final standing target is held for 10 consecutive 0.02 s checks; startup failure reports phase/time/joint/value; handoff increments generation and returns zero-command observation; automatic episode reset repeats startup.
- [ ] **Step 2: Run focused tests** and verify missing wrapper failures.
- [ ] **Step 3: Implement `Lite3DeploymentStartupWrapper`** around the already-created RSL/Gym environment. On reset, set `env.startup_active=True`, call the underlying reset, capture measured joint position/velocity, and advance internal actions generated by `standup_trajectory`. Use the existing action manager, not direct state writes. During phase 1 target the folded pose; during phase 2 target the standing pose; then hold the standing target until 10 consecutive valid foot-contact checks. Do not expose internal rewards, dones, or observations.
- [ ] **Step 4: Add mandatory per-step checks** for finite root/joint tensors, actual joint limits with a 0.02 rad margin, and `joint_vel.abs().amax() <= 30.0`. Raise `Lite3StartupError` with phase, elapsed time, joint index/name, measured value, and limit on violation. Validate final base height `>= 0.20 m`, projected-gravity XY norm `<= 0.65`, and all four foot force norms `> 1 N` for the required consecutive checks.
- [ ] **Step 5: At handoff** set `env.startup_active=False`, reset command current/target buffers, clear action-manager previous action, increment `startup_generation`, and return the post-handoff observation with `info["startup_generation"]`. Expose `consume_startup_generation()` so shared sessions can reset safety state exactly once.
- [ ] **Step 6: Run fake-env tests and commit** `feat: add lite3 deployment startup wrapper`.

## Task 4: Belly-Down Isaac Configuration and Shared Factory

**Files:** deployment configs, `deployment_startup.py`, shared-factory tests.

- [ ] **Step 1: Extend failing AST/config tests** to require a deterministic folded joint pose, a calibrated `startup_base_height`, startup-aware termination functions, and unchanged normal Flat/Rough task configuration.
- [ ] **Step 2: Implement the shared factory** `make_lite3_deployment_env(task, env_cfg, agent_cfg, gym_module, wrapper_cls=...)`: reject unsupported tasks, build the Gym/RSL wrapper once, then install `Lite3DeploymentStartupWrapper`; return a session object exposing `reset`, `step`, and `consume_startup_generation`. Do not alter ordinary training factory paths.
- [ ] **Step 3: Add the calibrated height as an exact deployment constant** only after the GPU probe in Task 5; keep the first implementation behind a test-visible constant so calibration cannot silently search at runtime.
- [ ] **Step 4: Run CPU config/factory tests and commit** `fix: configure lite3 belly-down deployment startup`.

## Task 5: Replace Direct Deployment Construction

**Files:** the five deployment-facing scripts listed in the file map and source-contract tests.

- [ ] **Step 1: Add failing source-contract tests** asserting each supported deployment entrypoint uses the shared factory and consumes `startup_generation` before its first policy/safety decision; assert no direct `gym.make` remains in those deployment paths except inside the factory.
- [ ] **Step 2: Replace each direct `gym.make` + `RslRlVecEnvWrapper` sequence** with the shared factory. Keep policy loading, replay writes, online updates, and qualification counters after the wrapper reset. On every generation, call `safety.reset_temporal_state()` and reset per-episode online state before calculating the next policy action.
- [ ] **Step 3: Preserve replay provenance boundaries**: the collector remains strict-plane-only and does not claim replay qualification for FlatOnline startup evidence.
- [ ] **Step 4: Run all affected CPU source tests and commit** `refactor: share lite3 deployment startup lifecycle`.

## Task 6: Isaac Belly-Down Calibration

**Files:** temporary `/tmp/lite3_flat_height_probe.py`; no production edits until the value is evidenced.

- [ ] **Step 1: Extend the temporary probe** with `LITE3_STARTUP_BASE_HEIGHT`, folded pose initialization, startup telemetry, contact-force logging, projected gravity, and max joint speed. Keep all changes under `/tmp`.
- [ ] **Step 2: Run a GPU height sweep** over explicit heights `0.05, 0.07, 0.09, 0.11, 0.13 m` using zero/PID startup control, recording torso/shank/foot contacts, penetration indicators, root height, and max speed. Select the lowest height that has torso and shank contact without material penetration or a speed spike above 30 rad/s; if none qualifies, stop and report the measured failure instead of guessing.
- [ ] **Step 3: Freeze the selected height** in both deployment configs and the strict guard; add a regression test for the exact value and a generated-flat/strict-plane configuration parse check.
- [ ] **Step 4: Run the complete three-second GPU stand-up gate** on generated flat and strict plane. Require all internal samples `<=30 rad/s`, finite state, valid joints, successful final height/tilt/contact checks, and no auto-reset.
- [ ] **Step 5: Commit** `test: calibrate lite3 belly-down startup height` with only the exact production constant/config tests; keep raw probe output in `/tmp` and summarize it in the evidence doc.

## Task 7: PPO Handoff Qualification and Evidence

**Files:** temporary probe, `docs/lite3-online-learning-principles.md`, qualification outputs.

- [ ] **Step 1: Run eight PPO transitions after successful handoff** on generated flat and strict plane with the unchanged checkpoint. Require zero termination/safety pause, command progression beginning at zero, and every joint speed `<=30 rad/s`.
- [ ] **Step 2: Run the existing five-seed ordinary-flat source qualification** to prove ordinary Flat behavior was not changed.
- [ ] **Step 3: Run the existing 100-episode/100,000-policy-step FlatOnline deployment qualification** through the shared wrapper; verify startup steps are absent from attempted-row counts and each episode begins with a fresh startup generation.
- [ ] **Step 4: Update `docs/lite3-online-learning-principles.md`** with exact checkpoint path/hash, calibrated base height, phase durations, startup and handoff maxima, qualification counts, and explicit replay-provenance blocker.
- [ ] **Step 5: Run full CPU suite, compileall, `git diff --check`, inspect status, and commit** `docs: record lite3 standup startup evidence`.

## Verification Commands

Use the repository's established environment:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest -q

PYTHONPATH=source/rl_training:. \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training scripts/reinforcement_learning/online_adaptation \
  tests/online_adaptation

git diff --check
```

GPU commands must activate `env_isaaclab`, set the worktree `PYTHONPATH`, and
use `CUDA_VISIBLE_DEVICES=1`. They require external execution approval when the
sandbox cannot access the NVIDIA driver. A failed GPU gate is a blocker for
qualification, not permission to relax safety thresholds.
