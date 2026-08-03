# Lite3 Track Completion Success Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` and complete the tasks in order. Steps use checkbox syntax.

**Goal:** Formal step and rough-terrain episodes pass only after Lite3 safely traverses the complete challenge, while reusing the frozen flat PPO and frozen flat replay.

**Architecture:** A dependency-light `TrackCompletion` model owns stage geometry, monotonic progress, strict completion, and finite cost. The Isaac worker uses it for formal challenges, emits complete evidence, and stops immediately after safe completion. Supervisors validate evidence instead of trusting `passed`; protocol fields make old residual state incompatible.

## Files

- Create `source/rl_training/rl_training/online_adaptation/track_completion.py` and `tests/online_adaptation/test_track_completion.py`.
- Modify `source/rl_training/rl_training/online_adaptation/terrain_protocol.py` and its tests.
- Modify `scripts/reinforcement_learning/online_adaptation/cli.py`, `play_online_adaptation.py`, and worker tests.
- Modify `run_flat_online_curriculum.py`, `evaluate_flat_online_curriculum.py`, and supervisor tests.
- Modify `docs/lite3-online-learning-principles.md`.

## Task 1: Track completion contract

- [ ] Add failing pure-Python tests for step and rough targets, leading-edge failure, monotonic `max_x`, immediate completion, simultaneous termination failure, finite failure costs, and invalid geometry.
- [ ] Run `tests/online_adaptation/test_track_completion.py -q` and confirm collection fails before implementation.
- [ ] Implement `TrackCompletion.start(...)`, `observe(base_x)`, `reached`, `remaining_m`, and `cost(fell=False, safety_violation=False)`. Step target is `x0 + flat_length / 2 + step_length + 0.5`; rough target is `x0 + track_length - flat_length / 2 - 0.5`. Reject non-finite values, unsupported stages, and targets outside the usable track.
- [ ] Run the focused tests and commit with `feat: define track completion contract`.

## Task 2: Version the protocol

- [ ] Add failing tests requiring `completion_mode="full_track_v1"`, `step_clearance_margin_m=0.5`, `track_end_buffer_m=0.5`, and `progress_cost_version="remaining_plus_safety_penalty_v1"` in `FlatOnlineProtocol` canonical JSON.
- [ ] Validate these fields as nonempty/finite/nonnegative and verify changing any field changes the protocol SHA-256.
- [ ] Run terrain protocol, checkpoint, and resume tests; commit with `fix: version full-track completion protocol`.

## Task 3: Enforce strict worker semantics

- [ ] Add failing CLI/runtime tests for `--require-track-completion`, reset-relative geometry, incomplete survival, leading-edge failure, safe crossing with immediate stop, simultaneous termination failure, and retention mode with completion disabled.
- [ ] Add the flag (default false). In formal mode, read post-reset base X, construct `TrackCompletion` from generated terrain dimensions, observe base X after each step, and evaluate environment termination/fall/safety before completion on that step.
- [ ] Define formal `passed` as `completion_reached and not fell and not safety_violation`; stop immediately after safe completion; emit finite `initial_x`, `max_x`, `completion_target_x`, `completion_remaining_m`, `completion_reached`, `completion_mode`, and `cost`.
- [ ] Run worker tests and `python -m compileall`; commit with `fix: require complete terrain traversal`. Stage only intended hunks because these files have existing user changes.

## Task 4: Fail-closed supervisor validation

- [ ] Add tests proving formal development/sealed commands include `--require-track-completion`, while flat-retention probe commands omit it.
- [ ] Add tests that reject `passed=true` without completion evidence, mode/stage/seed mismatch, non-finite or missing/nullable cost, inconsistent positions/remaining/reached/pass, non-positive target distance, and nonzero cost on a pass.
- [ ] Implement strict validation in `classify_worker_result()`; never convert missing cost to `0.0` or fall back to velocity error for formal episodes. Valid partial progress returns finite positive cost; valid completion returns zero.
- [ ] Run supervisor, acceptance, and protocol tests; commit with `fix: validate curriculum completion evidence`.

## Task 5: Verify and restart clean curriculum

- [ ] Document the exact geometry, immediate-stop rule, cost, protocol incompatibility, invalid run directories, and frozen artifact reuse in `docs/lite3-online-learning-principles.md`.
- [ ] Run the full CPU suite, compileall, and `git diff --check`. Expected baseline is at least `745 passed, 7 skipped`, plus new tests.
- [ ] Run one short incomplete and one reachable/synthetic completion Isaac smoke probe; inspect all completion and safety fields.
- [ ] Confirm an invalid curriculum directory fails protocol-hash compatibility before loading residual state.
- [ ] Start a fresh strict residual curriculum in a new output directory using only the frozen PPO and frozen flat replay, with no old residual checkpoint/state. Verify the command contains `--require-track-completion`.
- [ ] Commit documentation with `docs: record strict terrain completion protocol` and report the new run directory and first summary.

