# Lite3 Post-Update Rollback and Policy Drift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make performance rollback causally start after the current adaptation episode's first successful Actor update, and record finite policy-drift diagnostics before publishing each candidate Actor.

**Architecture:** `AdaptationPerformance` owns the pre-update/post-update window boundary, while `RuntimeCounters` carries successful Actor-update counts by episode from the updater thread to the control thread. A new `ActorDriftMonitor` freezes the episode reference Actor and fixed diagnostic inputs; a small actor-update coordinator invokes it through a transactional SAC candidate-validation hook before the double-buffered publisher swaps versions.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab runtime script, dataclasses, pytest, JSONL/TensorBoard-compatible scalar logs.

---

## File Map

| File | Responsibility |
|---|---|
| `source/rl_training/rl_training/online_adaptation/metrics.py` | Pre-update baseline windows and post-update rollback eligibility. |
| `source/rl_training/rl_training/online_adaptation/types.py` | Candidate rejection reason in immutable update statistics. |
| `source/rl_training/rl_training/online_adaptation/sac.py` | Transactional candidate-Actor validation before version acceptance. |
| `source/rl_training/rl_training/online_adaptation/policy_drift.py` | Frozen reference Actor, fixed diagnostic inputs, and scalar drift calculation. |
| `source/rl_training/rl_training/online_adaptation/actor_update.py` | Validate, diagnose, and publish one Actor update as one testable operation. |
| `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py` | Episode-aware counters, both runtime paths, background command handling, and logs. |
| `tests/online_adaptation/test_adaptation_metrics.py` | Performance boundary unit tests. |
| `tests/online_adaptation/test_play_runtime.py` | Pure runtime and episode-isolation tests. |
| `tests/online_adaptation/test_policy_drift.py` | Reference lifetime, fixed-batch, and drift-value tests. |
| `tests/online_adaptation/test_sac.py` | Candidate validation transaction tests. |
| `tests/online_adaptation/test_actor_update.py` | Publish/reject coordinator tests. |
| `docs/lite3-online-learning-principles.md` | Final implemented behavior and measured verification status. |

## Task 1: Add the Actor-update boundary to performance windows

**Files:**
- Modify: `tests/online_adaptation/test_adaptation_metrics.py`
- Modify: `source/rl_training/rl_training/online_adaptation/metrics.py`

- [ ] **Step 1: Replace the existing immediate-comparison assumptions with failing boundary tests**

Keep `add_window()` and configuration-validation tests, then add these tests:

```python
def test_pre_update_windows_replace_baseline_without_degradation():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()

    first = add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    second = add_window(monitor, velocity_error=0.40, tilt_rms=0.10)
    third = add_window(monitor, velocity_error=0.60, tilt_rms=0.10)

    assert first.cost == pytest.approx(0.30)
    assert second.cost == pytest.approx(0.50)
    assert third.cost == pytest.approx(0.70)
    assert monitor.best_cost == pytest.approx(0.70)
    assert monitor.degradation_count == 0
    assert all(not decision.degraded and not decision.should_rollback for decision in (first, second, third))
    assert not monitor.evaluation_enabled


def test_first_actor_update_preserves_baseline_and_clears_partial_window():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    assert monitor.add(velocity_error=9.0, tilt_rms=0.0, nonfoot_contact=0) is None

    monitor.mark_actor_updated()

    assert monitor.evaluation_enabled
    assert monitor.best_cost == pytest.approx(0.30)
    assert monitor.add(velocity_error=0.25, tilt_rms=0.10, nonfoot_contact=0) is None
    decision = monitor.add(velocity_error=0.25, tilt_rms=0.10, nonfoot_contact=0)
    assert decision.degraded
    assert not decision.should_rollback


def test_repeated_actor_update_notification_is_idempotent():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    monitor.mark_actor_updated()
    assert monitor.add(velocity_error=0.25, tilt_rms=0.10, nonfoot_contact=0) is None

    monitor.mark_actor_updated()

    decision = monitor.add(velocity_error=0.25, tilt_rms=0.10, nonfoot_contact=0)
    assert decision.degraded


def test_first_post_update_window_establishes_reference_without_pre_update_window():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    monitor.add(velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0)
    monitor.mark_actor_updated()

    decision = add_window(monitor, velocity_error=0.30, tilt_rms=0.10)

    assert decision.cost == pytest.approx(0.40)
    assert monitor.best_cost == pytest.approx(0.40)
    assert not decision.degraded


def test_two_complete_post_update_degraded_windows_request_rollback():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    monitor.mark_actor_updated()

    first = add_window(monitor, velocity_error=0.25, tilt_rms=0.10)
    second = add_window(monitor, velocity_error=0.25, tilt_rms=0.10)

    assert first.degraded and not first.should_rollback
    assert second.degraded and second.should_rollback
    assert second.rollback_reason == "performance_degradation"


def test_new_episode_disables_post_update_evaluation():
    monitor = AdaptationPerformance(window=2, improvement_ratio=0.05, degradation_ratio=0.10, patience=2)
    monitor.start_episode()
    add_window(monitor, velocity_error=0.20, tilt_rms=0.10)
    monitor.mark_actor_updated()

    monitor.start_episode()

    assert not monitor.evaluation_enabled
    assert monitor.best_cost is None
    assert monitor.degradation_count == 0
```

In the existing improvement and neutral-window tests, call `monitor.mark_actor_updated()` immediately after the pre-update baseline window. Keep the first-window baseline test unmarked so it continues to verify pre-update baseline collection.

- [ ] **Step 2: Run the boundary tests and verify RED**

Run:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_adaptation_metrics.py -q
```

Expected: FAIL because `evaluation_enabled` and `mark_actor_updated()` do not exist and pre-update degraded windows still increment patience.

- [ ] **Step 3: Implement the performance phase in `AdaptationPerformance`**

Add `_evaluation_enabled`, its property, and the idempotent boundary method; replace the decision section of `add()` with the following behavior:

```python
@property
def evaluation_enabled(self) -> bool:
    return self._evaluation_enabled

def start_episode(self) -> None:
    self._records.clear()
    self._best_cost = None
    self._degradation_count = 0
    self._evaluation_enabled = False

def mark_actor_updated(self) -> None:
    if self._evaluation_enabled:
        return
    self._evaluation_enabled = True
    self._records.clear()
    self._degradation_count = 0

def _decision_for_cost(self, cost: float) -> AdaptationPerformanceDecision:
    if not self._evaluation_enabled:
        self._best_cost = cost
        self._degradation_count = 0
        return AdaptationPerformanceDecision(cost, False, False, False, None)
    if self._best_cost is None:
        self._best_cost = cost
        return AdaptationPerformanceDecision(cost, False, False, False, None)

    improved = cost <= self._best_cost * (1.0 - self.improvement_ratio)
    degraded = cost > self._best_cost * (1.0 + self.degradation_ratio)
    if improved:
        self._best_cost = cost
        self._degradation_count = 0
    elif degraded:
        self._degradation_count += 1
    else:
        self._degradation_count = 0
    should_rollback = self._degradation_count >= self.patience
    return AdaptationPerformanceDecision(
        cost=cost,
        improved=improved,
        degraded=degraded,
        should_rollback=should_rollback,
        rollback_reason="performance_degradation" if should_rollback else None,
    )
```

Initialize `self._evaluation_enabled = False` in `__init__()`. In `add()`, retain finite validation and window aggregation, then finish with `return self._decision_for_cost(cost)`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: all `test_adaptation_metrics.py` tests pass.

- [ ] **Step 5: Commit and push Task 1**

```bash
git add source/rl_training/rl_training/online_adaptation/metrics.py \
  tests/online_adaptation/test_adaptation_metrics.py
git commit -m "fix: evaluate lite3 performance after actor update"
git push origin feat/lite3-online-adaptation
```

## Task 2: Carry successful Actor updates by adaptation episode

**Files:**
- Modify: `tests/online_adaptation/test_play_runtime.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py`

- [ ] **Step 1: Add failing counter and pure-runtime tests**

Add:

```python
def test_runtime_counters_isolate_actor_updates_by_episode():
    counters = RuntimeCounters()
    counters.record_actor_update(step=100, adaptation_episode=1)
    counters.record_actor_update(step=101, adaptation_episode=1)
    counters.record_actor_update(step=200, adaptation_episode=2)

    assert counters.actor_updates_for_episode(1) == 2
    assert counters.actor_updates_for_episode(2) == 1
    assert counters.actor_updates_for_episode(3) == 0


def test_runtime_does_not_roll_back_before_first_actor_update():
    runtime = make_orchestrator()
    enter_adapting(runtime)

    results = [
        runtime.observe_step(step=step, velocity_error=0.80, tilt_rms=0.20, nonfoot_contact=0, reward=0.1)
        for step in range(8, 16)
    ]

    assert runtime.active_episode == 1
    assert runtime.counters.performance_rollbacks == 0
    assert all(not result.commands for result in results)
    assert not runtime.adaptation_performance.evaluation_enabled


def test_runtime_rolls_back_after_two_complete_post_update_windows():
    runtime = make_orchestrator()
    enter_adapting(runtime)
    runtime.observe_step(step=8, velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(step=9, velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.record_actor_update(step=9)

    runtime.observe_step(step=10, velocity_error=0.50, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    first = runtime.observe_step(step=11, velocity_error=0.50, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(step=12, velocity_error=0.50, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    second = runtime.observe_step(step=13, velocity_error=0.50, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)

    assert not first.commands
    assert second.commands == (AdaptationCommand("rollback", 1, "performance_degradation"),)


def test_old_episode_update_does_not_enable_current_episode():
    runtime = make_orchestrator()
    enter_adapting(runtime)
    runtime.active_episode = 2
    runtime.adaptation_episode = 2
    runtime.adaptation_performance.start_episode()
    runtime.counters.record_actor_update(step=10, adaptation_episode=1)

    runtime.observe_step(step=11, velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)
    runtime.observe_step(step=12, velocity_error=0.20, tilt_rms=0.10, nonfoot_contact=0, reward=0.2)

    assert not runtime.adaptation_performance.evaluation_enabled


def test_safety_rollback_remains_active_before_actor_update():
    runtime = make_orchestrator()
    enter_adapting(runtime)

    result = runtime.observe_step(
        step=8,
        velocity_error=0.50,
        tilt_rms=0.20,
        nonfoot_contact=1,
        reward=0.1,
        paused=True,
        safety_reasons=("excessive_tilt",),
    )

    assert result.commands == (AdaptationCommand("rollback", 1, "excessive_tilt"),)


def test_control_timeout_rollback_remains_active_before_actor_update():
    runtime = make_orchestrator()
    enter_adapting(runtime)

    result = runtime.observe_step(
        step=8,
        velocity_error=0.50,
        tilt_rms=0.20,
        nonfoot_contact=0,
        reward=0.1,
        control_timed_out=True,
    )

    assert result.commands == (AdaptationCommand("rollback", 1, "control_timeout"),)
```

Add this helper near `make_orchestrator()` so all tests enter novelty consistently:

```python
def enter_adapting(runtime):
    for step in range(1, 6):
        runtime.observe_step(
            step=step,
            velocity_error=0.10,
            tilt_rms=0.02,
            nonfoot_contact=0,
            reward=1.0,
        )
    runtime.observe_step(step=6, velocity_error=0.50, tilt_rms=0.02, nonfoot_contact=0, reward=0.2)
    result = runtime.observe_step(
        step=7,
        velocity_error=0.50,
        tilt_rms=0.02,
        nonfoot_contact=0,
        reward=0.2,
    )
    assert result.commands == (AdaptationCommand("start_episode", 1),)
```

Replace the old `test_runtime_orchestrator_requests_performance_rollback` with the more explicit post-update test above so there is only one assertion of that behavior.

- [ ] **Step 2: Run the runtime tests and verify RED**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_play_runtime.py -q
```

Expected: FAIL because counters do not store per-episode counts and the orchestrator never calls `mark_actor_updated()`.

- [ ] **Step 3: Implement episode-aware counters and pure orchestration**

Extend `RuntimeCounters`:

```python
@dataclass
class RuntimeCounters:
    critic_updates: int = 0
    actor_updates: int = 0
    actor_updates_before_novelty: int = 0
    actor_updates_after_novelty: int = 0
    first_actor_update_step: int | None = None
    novelty_entries: int = 0
    rollbacks: int = 0
    performance_rollbacks: int = 0
    safety_rollbacks: int = 0
    state_durations: dict[str, int] = field(default_factory=dict)
    actor_updates_by_episode: dict[int, int] = field(default_factory=dict)

    def record_actor_update(self, *, step: int, adaptation_episode: int) -> None:
        self.actor_updates += 1
        if adaptation_episode > 0:
            self.actor_updates_after_novelty += 1
            self.actor_updates_by_episode[adaptation_episode] = (
                self.actor_updates_by_episode.get(adaptation_episode, 0) + 1
            )
        else:
            self.actor_updates_before_novelty += 1
        if self.first_actor_update_step is None:
            self.first_actor_update_step = step

    def actor_updates_for_episode(self, adaptation_episode: int) -> int:
        return self.actor_updates_by_episode.get(adaptation_episode, 0)
```

Keep the existing critic, novelty, state, and rollback methods unchanged. Immediately before the pure orchestrator calls `adaptation_performance.add(...)`, use only the active episode's count:

```python
if self.counters.actor_updates_for_episode(self.active_episode) > 0:
    self.adaptation_performance.mark_actor_updated()
```

Add `"performance_evaluation_enabled": self.adaptation_performance.evaluation_enabled` to pure `record_fields`.

When a performance rollback is requested or novelty leaves `ADAPTING`, call `self.adaptation_performance.start_episode()` after capturing the decision. This makes the returned state explicitly ineligible for further performance rollback while preserving the already-created rollback command.

Use this exact rollback/exit tail in the pure orchestrator:

```python
elif performance_decision.should_rollback:
    episode_to_rollback = self.active_episode
    rollback_reason = performance_decision.rollback_reason or "performance_degradation"
    novelty_decision = self.novelty_gate.force_cooldown()
    self.active_episode = 0
    self.counters.record_rollback(rollback_reason)
    self.last_rollback_reason = rollback_reason
    clear_adaptation_replay = True
    commands.append(AdaptationCommand("rollback", episode_to_rollback, rollback_reason))
    self.adaptation_performance.start_episode()
elif novelty_decision.left_adapting and self.active_episode > 0:
    self.active_episode = 0
    clear_adaptation_replay = True
    self.adaptation_performance.start_episode()
```

- [ ] **Step 4: Run focused runtime and metric tests**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_adaptation_metrics.py \
  tests/online_adaptation/test_play_runtime.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit and push Task 2**

```bash
git add scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  tests/online_adaptation/test_play_runtime.py
git commit -m "fix: isolate lite3 actor updates by episode"
git push origin feat/lite3-online-adaptation
```

## Task 3: Measure policy drift against a frozen episode reference

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/policy_drift.py`
- Create: `tests/online_adaptation/test_policy_drift.py`

- [ ] **Step 1: Write failing drift-monitor tests**

```python
import torch

from rl_training.online_adaptation.networks import ResidualActor
from rl_training.online_adaptation.policy_drift import ActorDriftMonitor
from rl_training.online_adaptation.types import TransitionBatch


def make_actor():
    return ResidualActor(2, 1, 4, torch.tensor([0.2]))


def make_batch(value=0.0):
    return TransitionBatch(
        history=torch.full((4, 3, 2), value),
        current_frame=torch.full((4, 2), value),
        base_action=torch.full((4, 1), value),
        residual_action=torch.zeros(4, 1),
        reward=torch.zeros(4, 1),
        next_history=torch.zeros(4, 3, 2),
        next_frame=torch.zeros(4, 2),
        next_base_action=torch.zeros(4, 1),
        done=torch.zeros(4, 1),
    )


def test_identical_actor_has_zero_policy_drift():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=3)

    drift = monitor.measure(actor, make_batch(), actor_version=7, update_step=100, adaptation_episode=3)

    assert drift.residual_mse == 0.0
    assert drift.joint_abs_change_mean == 0.0
    assert drift.joint_abs_change_max == 0.0
    assert drift.parameter_l2 == 0.0
    assert drift.adaptation_episode == 3
    assert drift.actor_version == 7
    assert drift.update_step == 100


def test_known_actor_change_has_positive_finite_drift():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    with torch.no_grad():
        actor.mean_layer.bias.add_(0.1)

    drift = monitor.measure(actor, make_batch(), actor_version=1, update_step=20, adaptation_episode=1)

    assert drift.residual_mse > 0.0
    assert drift.joint_abs_change_mean > 0.0
    assert drift.joint_abs_change_max > 0.0
    assert drift.parameter_l2 > 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in (
        drift.residual_mse,
        drift.joint_abs_change_mean,
        drift.joint_abs_change_max,
        drift.parameter_l2,
    ))


def test_monitor_reuses_first_diagnostic_inputs_within_episode():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    with torch.no_grad():
        actor.mean_layer.bias.add_(0.1)

    first = monitor.measure(actor, make_batch(0.0), actor_version=1, update_step=20, adaptation_episode=1)
    second = monitor.measure(actor, make_batch(100.0), actor_version=2, update_step=30, adaptation_episode=1)

    assert second.residual_mse == first.residual_mse
    assert second.joint_abs_change_mean == first.joint_abs_change_mean
    assert second.joint_abs_change_max == first.joint_abs_change_max


def test_new_episode_refreshes_reference_and_diagnostic_inputs():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    with torch.no_grad():
        actor.mean_layer.bias.add_(0.1)
    monitor.measure(actor, make_batch(), actor_version=1, update_step=20, adaptation_episode=1)

    monitor.start_episode(actor, adaptation_episode=2)
    drift = monitor.measure(actor, make_batch(2.0), actor_version=1, update_step=40, adaptation_episode=2)

    assert drift.residual_mse == 0.0
    assert drift.parameter_l2 == 0.0


def test_old_episode_clear_does_not_remove_new_reference():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=2)

    monitor.clear_episode(adaptation_episode=1)

    drift = monitor.measure(actor, make_batch(), actor_version=1, update_step=40, adaptation_episode=2)
    assert drift.residual_mse == 0.0
```

- [ ] **Step 2: Run the new test and verify RED**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_policy_drift.py -q
```

Expected: collection FAIL because `policy_drift.py` does not exist.

- [ ] **Step 3: Implement `ActorDriftMonitor`**

```python
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .networks import ResidualActor
from .types import TransitionBatch


@dataclass(frozen=True)
class PolicyDrift:
    residual_mse: float
    joint_abs_change_mean: float
    joint_abs_change_max: float
    parameter_l2: float
    adaptation_episode: int
    actor_version: int
    update_step: int


class ActorDriftMonitor:
    def __init__(self) -> None:
        self._reference_actor: ResidualActor | None = None
        self._diagnostic_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._adaptation_episode = 0

    @property
    def adaptation_episode(self) -> int:
        return self._adaptation_episode

    def start_episode(self, actor: ResidualActor, *, adaptation_episode: int) -> None:
        if adaptation_episode <= 0:
            raise ValueError("adaptation_episode must be positive")
        self._reference_actor = copy.deepcopy(actor)
        self._reference_actor.eval()
        self._reference_actor.requires_grad_(False)
        self._diagnostic_inputs = None
        self._adaptation_episode = adaptation_episode

    def clear_episode(self, *, adaptation_episode: int) -> None:
        if adaptation_episode != self._adaptation_episode:
            return
        self._reference_actor = None
        self._diagnostic_inputs = None
        self._adaptation_episode = 0

    def measure(
        self,
        actor: ResidualActor,
        batch: TransitionBatch,
        *,
        actor_version: int,
        update_step: int,
        adaptation_episode: int,
    ) -> PolicyDrift:
        if self._reference_actor is None or adaptation_episode != self._adaptation_episode:
            raise RuntimeError("no reference actor for adaptation episode")
        if self._diagnostic_inputs is None:
            self._diagnostic_inputs = (
                batch.history.detach().clone(),
                batch.current_frame.detach().clone(),
                batch.base_action.detach().clone(),
            )
        history, current_frame, base_action = self._diagnostic_inputs
        with torch.inference_mode():
            candidate_residual, _, _ = actor.sample(
                history, current_frame, base_action, deterministic=True
            )
            reference_residual, _, _ = self._reference_actor.sample(
                history, current_frame, base_action, deterministic=True
            )
            delta = candidate_residual - reference_residual
            joint_mean_abs = delta.abs().mean(dim=0)
            parameter_squared = torch.zeros((), device=delta.device)
            reference_parameters = dict(self._reference_actor.named_parameters())
            for name, parameter in actor.named_parameters():
                parameter_squared += (parameter - reference_parameters[name]).square().sum()
            values = (
                delta.square().mean(),
                joint_mean_abs.mean(),
                joint_mean_abs.max(),
                parameter_squared.sqrt(),
            )
        if not all(torch.isfinite(value).item() for value in values):
            raise FloatingPointError("non-finite policy drift")
        return PolicyDrift(
            residual_mse=float(values[0]),
            joint_abs_change_mean=float(values[1]),
            joint_abs_change_max=float(values[2]),
            parameter_l2=float(values[3]),
            adaptation_episode=adaptation_episode,
            actor_version=actor_version,
            update_step=update_step,
        )
```

- [ ] **Step 4: Run drift-monitor tests and verify GREEN**

Run the Step 2 command. Expected: 5 tests pass.

- [ ] **Step 5: Commit and push Task 3**

```bash
git add source/rl_training/rl_training/online_adaptation/policy_drift.py \
  tests/online_adaptation/test_policy_drift.py
git commit -m "feat: measure lite3 policy drift"
git push origin feat/lite3-online-adaptation
```

## Task 4: Make candidate validation part of the SAC transaction

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/types.py`
- Modify: `source/rl_training/rl_training/online_adaptation/sac.py`
- Modify: `tests/online_adaptation/test_sac.py`

- [ ] **Step 1: Add failing transaction tests**

```python
def test_candidate_validator_can_reject_and_restore_full_update():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    before = copy.deepcopy(updater.state_dict())
    calls = []

    def reject(candidate_actor, candidate_version):
        calls.append((candidate_actor, candidate_version))
        return "nonfinite_policy_drift"

    stats = updater.update(make_batch(), update_actor=True, candidate_validator=reject)

    assert calls == [(actor, 1)]
    assert not stats.accepted
    assert stats.rejection_reason == "nonfinite_policy_drift"
    assert updater.version == 0
    assert_state_dict_equal(updater.state_dict(), before)


def test_candidate_validator_is_not_called_for_critic_only_update():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())

    def reject(candidate_actor, candidate_version):
        raise AssertionError("critic-only update must not validate an Actor")

    stats = updater.update(make_batch(), update_actor=False, candidate_validator=reject)

    assert stats.accepted
    assert not stats.actor_updated


def test_unexpected_candidate_validator_error_restores_state_and_propagates():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    before = copy.deepcopy(updater.state_dict())

    def fail(candidate_actor, candidate_version):
        raise RuntimeError("diagnostic implementation failed")

    with pytest.raises(RuntimeError, match="diagnostic implementation failed"):
        updater.update(make_batch(), update_actor=True, candidate_validator=fail)

    assert_state_dict_equal(updater.state_dict(), before)
```

Add `import copy` and `import pytest`. Add a recursive tensor-aware `assert_state_dict_equal(actual, expected)` helper that compares dictionaries, lists, tuples, tensors, and scalar leaves so optimizer state is included.

```python
def assert_state_dict_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_dict_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_state_dict_equal(actual_item, expected_item)
        return
    assert actual == expected
```

- [ ] **Step 2: Run SAC tests and verify RED**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_sac.py -q
```

Expected: FAIL because `candidate_validator` and `rejection_reason` do not exist.

- [ ] **Step 3: Add rejection metadata and the transactional hook**

Extend `UpdateStats`:

```python
@dataclass(frozen=True)
class UpdateStats:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    mean_q: float
    grad_norm: float
    accepted: bool
    version: int
    critic_updated: bool = False
    actor_updated: bool = False
    rejection_reason: str | None = None
```

In `sac.py`, define:

```python
from collections.abc import Callable

CandidateActorValidator = Callable[[ResidualActor, int], str | None]


class CandidateActorRejected(RuntimeError):
    pass
```

Change `_rejected_stats(self, reason: str)` to populate `rejection_reason=reason`. Add `candidate_validator: CandidateActorValidator | None = None` to `update()`. Return `nonfinite_batch` for the initial batch check. After all Actor/Critic/target parameters pass existing finite checks, but before `self.version += 1`, run:

```python
if update_actor and candidate_validator is not None:
    rejection_reason = candidate_validator(self.actor, self.version + 1)
    if rejection_reason is not None:
        raise CandidateActorRejected(rejection_reason)
```

Use these exception branches around the existing transaction:

```python
except CandidateActorRejected as error:
    self.critic.requires_grad_(True)
    self.load_state_dict(before)
    return self._rejected_stats(str(error))
except FloatingPointError:
    self.critic.requires_grad_(True)
    self.load_state_dict(before)
    return self._rejected_stats("nonfinite_update")
except BaseException:
    self.critic.requires_grad_(True)
    self.load_state_dict(before)
    raise
```

Accepted stats explicitly use `rejection_reason=None`.

- [ ] **Step 4: Run SAC tests and verify GREEN**

Run the Step 2 command. Expected: all SAC tests pass, including full-state restoration.

- [ ] **Step 5: Commit and push Task 4**

```bash
git add source/rl_training/rl_training/online_adaptation/types.py \
  source/rl_training/rl_training/online_adaptation/sac.py \
  tests/online_adaptation/test_sac.py
git commit -m "feat: validate lite3 actor candidates transactionally"
git push origin feat/lite3-online-adaptation
```

## Task 5: Coordinate diagnostics and publication

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/actor_update.py`
- Create: `tests/online_adaptation/test_actor_update.py`

- [ ] **Step 1: Write failing coordinator tests**

Create tests using the small Actor/Critic and `make_batch()` pattern from Task 3:

```python
import torch

from rl_training.online_adaptation.actor_update import update_and_publish_actor
from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.policy_drift import ActorDriftMonitor
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher
from rl_training.online_adaptation.sac import SACConfig, SACUpdater
from rl_training.online_adaptation.types import TransitionBatch


def make_batch():
    return TransitionBatch(
        history=torch.randn(4, 3, 2),
        current_frame=torch.randn(4, 2),
        base_action=torch.randn(4, 1),
        residual_action=torch.zeros(4, 1),
        reward=torch.ones(4, 1),
        next_history=torch.randn(4, 3, 2),
        next_frame=torch.randn(4, 2),
        next_base_action=torch.randn(4, 1),
        done=torch.zeros(4, 1),
    )


def test_finite_candidate_is_diagnosed_then_published():
    actor = ResidualActor(2, 1, 4, torch.tensor([0.2]))
    updater = SACUpdater(actor, TwinCritic(2, 1, 4), SACConfig())
    publisher = DoubleBufferedActorPublisher(actor)
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    batch = make_batch()

    outcome = update_and_publish_actor(
        updater=updater,
        publisher=publisher,
        drift_monitor=monitor,
        critic_batch=batch,
        actor_batch=batch,
        adaptation_episode=1,
        update_step=50,
        episode_is_active=lambda: True,
    )

    assert outcome.stats.accepted
    assert outcome.policy_drift is not None
    assert outcome.policy_drift.actor_version == outcome.stats.version
    assert publisher.current()[1] == outcome.stats.version


def test_stale_candidate_is_restored_and_not_published():
    actor = ResidualActor(2, 1, 4, torch.tensor([0.2]))
    updater = SACUpdater(actor, TwinCritic(2, 1, 4), SACConfig())
    publisher = DoubleBufferedActorPublisher(actor)
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    batch = make_batch()

    outcome = update_and_publish_actor(
        updater=updater,
        publisher=publisher,
        drift_monitor=monitor,
        critic_batch=batch,
        actor_batch=batch,
        adaptation_episode=1,
        update_step=50,
        episode_is_active=lambda: False,
    )

    assert not outcome.stats.accepted
    assert outcome.stats.rejection_reason == "stale_adaptation_episode"
    assert outcome.policy_drift is None
    assert updater.version == 0
    assert publisher.current()[1] == 0


def test_nonfinite_drift_is_restored_and_not_published(monkeypatch):
    actor = ResidualActor(2, 1, 4, torch.tensor([0.2]))
    updater = SACUpdater(actor, TwinCritic(2, 1, 4), SACConfig())
    publisher = DoubleBufferedActorPublisher(actor)
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    monkeypatch.setattr(monitor, "measure", lambda *args, **kwargs: (_ for _ in ()).throw(FloatingPointError()))
    batch = make_batch()

    outcome = update_and_publish_actor(
        updater=updater,
        publisher=publisher,
        drift_monitor=monitor,
        critic_batch=batch,
        actor_batch=batch,
        adaptation_episode=1,
        update_step=50,
        episode_is_active=lambda: True,
    )

    assert not outcome.stats.accepted
    assert outcome.stats.rejection_reason == "nonfinite_policy_drift"
    assert outcome.policy_drift is None
    assert publisher.current()[1] == 0
```

- [ ] **Step 2: Run coordinator tests and verify RED**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_actor_update.py -q
```

Expected: collection FAIL because `actor_update.py` does not exist.

- [ ] **Step 3: Implement the coordinator**

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .policy_drift import ActorDriftMonitor, PolicyDrift
from .publisher import DoubleBufferedActorPublisher
from .sac import SACUpdater
from .types import TransitionBatch, UpdateStats


@dataclass(frozen=True)
class ActorUpdateOutcome:
    stats: UpdateStats
    policy_drift: PolicyDrift | None


def update_and_publish_actor(
    *,
    updater: SACUpdater,
    publisher: DoubleBufferedActorPublisher,
    drift_monitor: ActorDriftMonitor,
    critic_batch: TransitionBatch,
    actor_batch: TransitionBatch,
    adaptation_episode: int,
    update_step: int,
    episode_is_active: Callable[[], bool],
) -> ActorUpdateOutcome:
    policy_drift: PolicyDrift | None = None

    def validate_actor(candidate_actor, candidate_version):
        nonlocal policy_drift
        if not episode_is_active():
            return "stale_adaptation_episode"
        try:
            policy_drift = drift_monitor.measure(
                candidate_actor,
                actor_batch,
                actor_version=candidate_version,
                update_step=update_step,
                adaptation_episode=adaptation_episode,
            )
        except FloatingPointError:
            return "nonfinite_policy_drift"
        return None

    stats = updater.update(
        critic_batch=critic_batch,
        actor_batch=actor_batch,
        update_actor=True,
        candidate_validator=validate_actor,
    )
    if stats.accepted and stats.actor_updated:
        if policy_drift is None:
            raise RuntimeError("accepted Actor update has no policy drift diagnostics")
        publisher.publish(updater.actor, stats.version)
    else:
        policy_drift = None
    return ActorUpdateOutcome(stats=stats, policy_drift=policy_drift)
```

- [ ] **Step 4: Run coordinator, SAC, and drift tests**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_actor_update.py \
  tests/online_adaptation/test_policy_drift.py \
  tests/online_adaptation/test_sac.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit and push Task 5**

```bash
git add source/rl_training/rl_training/online_adaptation/actor_update.py \
  tests/online_adaptation/test_actor_update.py
git commit -m "feat: publish diagnosed lite3 actor updates"
git push origin feat/lite3-online-adaptation
```

## Task 6: Integrate the real updater loop and scalar logs

**Files:**
- Modify: `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py`
- Modify: `tests/online_adaptation/test_play_runtime.py`
- Modify: `docs/lite3-online-learning-principles.md`

- [ ] **Step 1: Add a failing scalar-log shape test**

Extract and test a pure conversion helper:

```python
from rl_training.online_adaptation.policy_drift import PolicyDrift


def test_policy_drift_record_is_flat_and_scalar_only():
    drift = PolicyDrift(
        residual_mse=0.01,
        joint_abs_change_mean=0.02,
        joint_abs_change_max=0.03,
        parameter_l2=0.04,
        adaptation_episode=2,
        actor_version=9,
        update_step=700,
    )

    assert policy_drift_record(drift) == {
        "policy_residual_mse": 0.01,
        "policy_joint_abs_change_mean": 0.02,
        "policy_joint_abs_change_max": 0.03,
        "policy_parameter_l2": 0.04,
        "policy_drift_episode": 2,
        "policy_drift_actor_version": 9,
        "policy_drift_update_step": 700,
    }
```

Import `PolicyDrift` and `policy_drift_record` in the existing test module.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_play_runtime.py::test_policy_drift_record_is_flat_and_scalar_only -q
```

Expected: FAIL because `policy_drift_record()` does not exist.

- [ ] **Step 3: Add the pure log conversion**

```python
def policy_drift_record(drift: PolicyDrift) -> dict[str, float | int]:
    return {
        "policy_residual_mse": drift.residual_mse,
        "policy_joint_abs_change_mean": drift.joint_abs_change_mean,
        "policy_joint_abs_change_max": drift.joint_abs_change_max,
        "policy_parameter_l2": drift.parameter_l2,
        "policy_drift_episode": drift.adaptation_episode,
        "policy_drift_actor_version": drift.actor_version,
        "policy_drift_update_step": drift.update_step,
    }
```

- [ ] **Step 4: Wire reference lifetime, candidate publication, and episode-safe logging**

Import `update_and_publish_actor`, `ActorDriftMonitor`, and `PolicyDrift`. Instantiate one `ActorDriftMonitor` beside `StableStateManager`.

Broaden the updater-loop snapshot annotation to `last_update: dict[str, Any] = {}` because accepted diagnostics and rejected update metadata include floats, integers, booleans, and `None`.

Extend `AdaptationCommand.kind` to `Literal["start_episode", "checkpoint", "rollback", "end_episode"]`. Both runtime paths emit `end_episode` with the episode being left when novelty exits `ADAPTING`; the pure path also calls `adaptation_performance.start_episode()` at that boundary.

Replace the Task 2 pure exit branch with:

```python
elif novelty_decision.left_adapting and self.active_episode > 0:
    episode_to_end = self.active_episode
    self.active_episode = 0
    clear_adaptation_replay = True
    self.adaptation_performance.start_episode()
    commands.append(AdaptationCommand("end_episode", episode_to_end))
```

Under `update_lock`, handle commands as follows:

```python
if command.kind == "start_episode":
    stable_manager.checkpoint(online_agent.updater)
    actor_drift_monitor.start_episode(
        online_agent.updater.actor,
        adaptation_episode=command.episode,
    )
    checkpointed_episode = command.episode
    with shared_lock:
        episode_is_active = active_episode == command.episode
    if episode_is_active:
        actor_gate_event.set()
elif command.kind == "checkpoint":
    if command.episode == checkpointed_episode:
        stable_manager.checkpoint(online_agent.updater)
elif command.kind == "rollback":
    if command.episode == checkpointed_episode:
        if stable_manager.rollback(online_agent.updater):
            online_agent.publisher.publish(online_agent.updater.actor, online_agent.updater.version)
        actor_drift_monitor.clear_episode(adaptation_episode=command.episode)
        checkpointed_episode = 0
        actor_gate_event.clear()
        with shared_lock:
            last_rollback_reason = command.reason
elif command.kind == "end_episode":
    if command.episode == checkpointed_episode:
        actor_drift_monitor.clear_episode(adaptation_episode=command.episode)
        checkpointed_episode = 0
        actor_gate_event.clear()
```

Replace direct full-Actor update/publication with:

```python
if actor_due:
    critic_batch = online_agent.sample_critic_batch()
    actor_batch = online_agent.sample_actor_batch()

    def episode_is_still_active() -> bool:
        with shared_lock:
            return active_episode == episode_snapshot

    with update_lock:
        outcome = update_and_publish_actor(
            updater=online_agent.updater,
            publisher=online_agent.publisher,
            drift_monitor=actor_drift_monitor,
            critic_batch=critic_batch,
            actor_batch=actor_batch,
            adaptation_episode=episode_snapshot,
            update_step=step_snapshot,
            episode_is_active=episode_is_still_active,
        )
    stats = outcome.stats
    policy_drift = outcome.policy_drift
```

When updating `last_update`, skip stale episode attempts entirely; otherwise include finite drift scalars:

```python
if stats is not None and stats.rejection_reason != "stale_adaptation_episode":
    with shared_lock:
        last_update.clear()
        last_update.update(asdict(stats))
        if policy_drift is not None:
            last_update.update(policy_drift_record(policy_drift))
        if stats.accepted and stats.critic_updated:
            counters.record_critic_update()
        if stats.accepted and stats.actor_updated:
            counters.record_actor_update(
                step=step_snapshot,
                adaptation_episode=episode_snapshot,
            )
```

Initialize `policy_drift = None` on every updater-loop iteration before choosing Actor or critic update.

In the Isaac control path, immediately before `adaptation_performance.add(...)`, read the active episode's successful count under `shared_lock` and mark the boundary:

```python
with shared_lock:
    performance_episode = active_episode
    successful_actor_updates = counters.actor_updates_for_episode(performance_episode)
if successful_actor_updates > 0:
    adaptation_performance.mark_actor_updated()
```

On performance rollback, call `adaptation_performance.start_episode()` after enqueuing the rollback. Use this exact main-loop exit branch so the updater thread clears only the matching reference Actor:

```python
elif novelty_decision.left_adapting and active_episode > 0:
    episode_to_end = active_episode
    actor_gate_event.clear()
    online_agent.clear_adaptation_replay()
    adaptation_performance.start_episode()
    with shared_lock:
        active_episode = 0
    commands.put(AdaptationCommand("end_episode", episode_to_end))
```

Add these fields to each JSONL record:

```python
"performance_evaluation_enabled": adaptation_performance.evaluation_enabled,
"actor_updates_in_episode": counters_snapshot["actor_updates_by_episode"].get(episode_snapshot, 0),
```

Use `counters_snapshot` because it is copied under `shared_lock`; do not read the mutable per-episode dictionary directly from the control thread.

Add these summary fields:

```python
"actor_updates_by_episode": counters.actor_updates_by_episode,
"last_update": dict(last_update),
```

- [ ] **Step 5: Update the principles document with implemented behavior**

Add this paragraph to section 9 and mirror its status in sections 13 and 14:

```markdown
性能退化判定只在控制线程观察到当前 adaptation episode 的首次成功 Actor 更新后启用。首次更新边界保留最近一个完整更新前基线，清除跨越边界的部分窗口；随后只有完整更新后窗口可以保存稳定快照或累计退化。每个成功发布的 Actor 版本还记录固定诊断 batch 上的确定性残差 MSE、每关节绝对变化的均值与最大值、相对 episode reference Actor 的参数 L2，以及 episode、版本和更新步数。这些指标仅用于观测，不进入 loss、阈值或安全门控；旧 episode 候选和非有限诊断在发布前被拒绝。
```

Do not claim GPU acceptance until Task 7 commands have produced the required summaries.

- [ ] **Step 6: Run focused tests and compile the runtime script**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_adaptation_metrics.py \
  tests/online_adaptation/test_play_runtime.py \
  tests/online_adaptation/test_policy_drift.py \
  tests/online_adaptation/test_sac.py \
  tests/online_adaptation/test_actor_update.py -q
PYTHONPATH=source/rl_training:. /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m py_compile \
  scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py
```

Expected: all focused tests pass; `py_compile` exits 0 with no output.

- [ ] **Step 7: Commit and push Task 6**

```bash
git add scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  tests/online_adaptation/test_play_runtime.py \
  docs/lite3-online-learning-principles.md
git commit -m "feat: log lite3 post-update policy drift"
git push origin feat/lite3-online-adaptation
```

## Task 7: Full verification and GPU evidence

**Files:**
- Modify after evidence: `docs/lite3-online-learning-principles.md`
- Generated outputs: `/home/tjx/rl_training_main/logs/online_adaptation/online_seed7_*_postupdate_drift_20260729_cuda2`

- [ ] **Step 1: Run all online-adaptation tests**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest tests/online_adaptation -q
```

Expected: zero failures; Isaac-only tests skip when `pxr` is unavailable outside an AppLauncher process.

- [ ] **Step 2: Run static verification**

```bash
PYTHONPATH=source/rl_training:. /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training/online_adaptation \
  scripts/reinforcement_learning/online_adaptation
git diff --check
```

Expected: both commands exit 0 without diagnostics.

- [ ] **Step 3: Run the 750-step GPU probe**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=source/rl_training:.
python scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  --base-checkpoint /home/tjx/rl_training_main/logs/online_adaptation/candidates/lite3_rl_deploy_policy_candidate.pt \
  --residual-checkpoint /home/tjx/rl_training_main/logs/online_adaptation/lite3_deploy_base_residual_formal_1024env_100000_resume742_20260727/residual_00045000.pt \
  --mode online --task HeldoutBoxes-Deeprobotics-Lite3-v0 \
  --seed 7 --forward-speed 0.3 --max-steps 750 \
  --device cuda:2 --control-timeout-ms 100 --headless \
  --run-dir /home/tjx/rl_training_main/logs/online_adaptation/online_seed7_750_postupdate_drift_20260729_cuda2
```

Expected: process exits 0; `summary.json` reports no fall/reset/safety violation, `actor_updates_before_novelty == 0`, and every accepted Actor update has finite policy-drift fields in `metrics.jsonl`.

- [ ] **Step 4: Run the 2000-step GPU evaluation**

Run the Step 3 command with:

```text
--max-steps 2000
--run-dir /home/tjx/rl_training_main/logs/online_adaptation/online_seed7_2000_postupdate_drift_20260729_cuda2
```

Expected safety gate: 2000 steps, zero reset, no fall, no safety violation. Any performance rollback must have the same episode's successful Actor update followed by two complete 50-step post-update windows.

- [ ] **Step 5: Audit causal ordering and drift values**

```bash
jq '{steps,reset_count,fell,safety_violation,actor_updates,actor_updates_before_novelty,actor_updates_by_episode,performance_rollbacks,last_update}' \
  /home/tjx/rl_training_main/logs/online_adaptation/online_seed7_2000_postupdate_drift_20260729_cuda2/summary.json
rg -n '"performance_evaluation_enabled": true|"policy_residual_mse"|"rollback_reason": "performance_degradation"' \
  /home/tjx/rl_training_main/logs/online_adaptation/online_seed7_2000_postupdate_drift_20260729_cuda2/metrics.jsonl
```

Expected: drift scalars are finite; no performance rollback precedes `performance_evaluation_enabled=true`; report actual performance outcome even if it fails the improvement threshold.

- [ ] **Step 6: Record evidence, run final checks, commit, and push**

Update the principles document with exact 750/2000-step results and test counts, then run:

```bash
git diff --check
git status --short
git add docs/lite3-online-learning-principles.md
git commit -m "docs: record lite3 policy drift evaluation"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/lite3-online-adaptation)"
git status --short --branch
```

Expected: local and remote SHA match; worktree has no uncommitted changes. If GitHub is unavailable, retain all local commits and report the exact ahead count without claiming synchronization.
