# Lite3 Novelty-Gated Online Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a true online-adaptation controller in which the critic learns continuously, the residual actor updates only after persistent proprioceptive novelty, and degraded updates roll back to the last stable full SAC state.

**Architecture:** Keep the frozen PPO policy and 50 Hz control path unchanged. Add a `NoveltyGate` driven only by velocity error, tilt, contacts, and reward; route safe transitions to a global replay and adapting transitions to a separate replay. Run critic updates continuously, actor/alpha updates at a lower rate only while adapting, and use short same-episode metric windows for rollback.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab, RSL-RL, pytest, TensorBoard JSONL metrics.

---

## File map

| File | Responsibility |
|---|---|
| `source/rl_training/rl_training/online_adaptation/novelty.py` | Observable-signal normalization and adaptation state machine. |
| `source/rl_training/rl_training/online_adaptation/types.py` | Critic/actor update ownership in update statistics. |
| `source/rl_training/rl_training/online_adaptation/replay_buffer.py` | Thread-safe reset needed by episode-local adaptation replay. |
| `source/rl_training/rl_training/online_adaptation/agent.py` | Global/adaptation replay ownership and transition routing. |
| `source/rl_training/rl_training/online_adaptation/sac.py` | Critic-only/full update split and full-state rollback. |
| `source/rl_training/rl_training/online_adaptation/metrics.py` | Episode-local performance windows. |
| `scripts/reinforcement_learning/online_adaptation/cli.py` | Runtime configuration. |
| `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py` | Gate, update schedules, rollback, and logging integration. |
| `scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py` | Multi-seed option pass-through. |
| `tests/online_adaptation/test_novelty.py` | Novelty state-machine unit tests. |
| `tests/online_adaptation/test_adaptation_metrics.py` | Performance-window tests. |
| `tests/online_adaptation/test_play_runtime.py` | Fake-runtime integration tests. |

## Task 1: Add the novelty gate as a pure state machine

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/novelty.py`
- Create: `tests/online_adaptation/test_novelty.py`

- [ ] **Step 1: Write failing calibration and persistent-entry tests.**

```python
from rl_training.online_adaptation.novelty import NoveltyConfig, NoveltyGate, NoveltyState


def make_gate():
    return NoveltyGate(
        NoveltyConfig(calibration_steps=3, enter_count=2, enter_window=3, recovery_steps=2, cooldown_steps=2)
    )


def test_calibration_never_enables_actor():
    gate = make_gate()
    for _ in range(3):
        decision = gate.observe(velocity_error=0.10, tilt_rms=0.02, nonfoot_contact=0, reward=1.0)
        assert decision.state == NoveltyState.CALIBRATING
        assert not decision.actor_update_enabled
    assert gate.observe(velocity_error=0.10, tilt_rms=0.02, nonfoot_contact=0, reward=1.0).state == NoveltyState.MONITORING


def test_persistent_novelty_enters_adapting():
    gate = make_gate()
    for _ in range(4):
        gate.observe(velocity_error=0.10, tilt_rms=0.02, nonfoot_contact=0, reward=1.0)
    first = gate.observe(velocity_error=0.50, tilt_rms=0.02, nonfoot_contact=0, reward=0.2)
    second = gate.observe(velocity_error=0.50, tilt_rms=0.02, nonfoot_contact=0, reward=0.2)
    assert first.state == NoveltyState.MONITORING
    assert second.entered_adapting
    assert second.actor_update_enabled
```

- [ ] **Step 2: Run the focused test and verify import failure.**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=source/rl_training:.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest tests/online_adaptation/test_novelty.py -q
```

Expected: collection fails because `rl_training.online_adaptation.novelty` does not exist.

- [ ] **Step 3: Implement the exact public interface.**

```python
class NoveltyState(str, Enum):
    CALIBRATING = "calibrating"
    MONITORING = "monitoring"
    ADAPTING = "adapting"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class NoveltyConfig:
    calibration_steps: int = 256
    enter_threshold: float = 2.5
    exit_threshold: float = 1.0
    enter_count: int = 10
    enter_window: int = 20
    recovery_steps: int = 100
    cooldown_steps: int = 100
    contact_count: int = 3
    contact_window: int = 10
    velocity_scale_floor: float = 0.05
    tilt_scale_floor: float = 0.02
    reward_scale_floor: float = 0.10
```

`NoveltyDecision` exposes `state`, `novelty_score`, `velocity_excess`, `tilt_excess`, `reward_drop`, `entered_adapting`, `left_adapting`, and `actor_update_enabled`. `observe` updates exponential mean/variance only in calibration/monitoring, requires `enter_count` threshold crossings in `enter_window`, and requires `recovery_steps` below the exit threshold before cooldown. `force_cooldown()` immediately closes the actor gate and resets episode counters.

- [ ] **Step 4: Add spike, recovery, contact, paused-step, and cooldown tests.**

Paused observations must return the current state without advancing any counter or baseline. A one-frame spike cannot trigger adaptation. Contact must occur at least `contact_count` times inside `contact_window` unless a continuous signal independently crosses the novelty threshold.

- [ ] **Step 5: Run the tests and commit.**

```bash
pytest tests/online_adaptation/test_novelty.py -q
git add source/rl_training/rl_training/online_adaptation/novelty.py tests/online_adaptation/test_novelty.py
git commit -m "feat: add proprioceptive novelty gate"
```

Expected: all novelty tests pass.

## Task 2: Split critic-only and actor SAC updates

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/types.py`
- Modify: `source/rl_training/rl_training/online_adaptation/sac.py`
- Modify: `tests/online_adaptation/test_sac.py`

- [ ] **Step 1: Write a failing critic-only test.**

```python
def test_critic_only_update_keeps_actor_and_version_unchanged():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    target_before = [parameter.detach().clone() for parameter in updater.target_critic.parameters()]
    stats = updater.update(critic_batch=make_batch(), actor_batch=None, update_actor=False)
    assert stats.accepted and stats.critic_updated and not stats.actor_updated
    assert stats.version == 0
    assert all(torch.equal(before, after) for before, after in zip(actor_before, actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(target_before, updater.target_critic.parameters()))
```

- [ ] **Step 2: Run the SAC test and verify the new call fails.**

```bash
pytest tests/online_adaptation/test_sac.py -q
```

Expected: `TypeError` for the new arguments or missing update-ownership fields.

- [ ] **Step 3: Implement the split update signature.**

Extend `UpdateStats` with defaulted `critic_updated: bool = False` and `actor_updated: bool = False`. Change the updater to:

```python
def update(
    self,
    critic_batch: TransitionBatch,
    actor_batch: TransitionBatch | None = None,
    *,
    update_actor: bool = True,
) -> UpdateStats:
```

Always train the critic and target critic from `critic_batch`. If `update_actor` is false, skip actor/alpha optimizers, return NaN actor/alpha losses with ownership flags set to false, and keep `version` unchanged. If true and `actor_batch` is omitted, reuse `critic_batch` to preserve `train_residual_sac.py` and existing callers; the online runtime passes a distinct adaptation batch. Calculate actor/alpha losses from the selected actor batch and increment the actor version exactly once. Keep the existing full `state_dict()` snapshot and finite-value rollback.

- [ ] **Step 4: Add a separate actor-batch test and preserve rejection behavior.**

The full update test must pass two batches with distinguishable tensor values and verify the actor path receives `actor_batch`. The NaN critic batch must reject the entire update without changing any state. A NaN actor batch must restore the preceding critic step from the atomic snapshot.

- [ ] **Step 5: Run and commit.**

```bash
pytest tests/online_adaptation/test_sac.py tests/online_adaptation/test_runtime.py -q
git add source/rl_training/rl_training/online_adaptation/types.py source/rl_training/rl_training/online_adaptation/sac.py tests/online_adaptation/test_sac.py
git commit -m "feat: separate critic and actor online updates"
```

Expected: all focused tests pass.

## Task 3: Add dual replay routing to the agent

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/replay_buffer.py`
- Modify: `source/rl_training/rl_training/online_adaptation/agent.py`
- Modify: `tests/online_adaptation/test_publisher_agent.py`

- [ ] **Step 1: Write failing routing and reset tests.**

Construct an agent, finish one transition with `adaptation_enabled=False` and another with `adaptation_enabled=True`, then assert:

```python
assert len(agent.replay) == 2
assert len(agent.adaptation_replay) == 1
agent.clear_adaptation_replay()
assert len(agent.adaptation_replay) == 0
```

Also test `should_update_actor` returns false when paused, actor gate is closed, or the adaptation replay has fewer than 128 items.

- [ ] **Step 2: Run the focused test and verify missing members.**

```bash
pytest tests/online_adaptation/test_publisher_agent.py -q
```

Expected: failures for `adaptation_replay`, `adaptation_enabled`, or `clear_adaptation_replay`.

- [ ] **Step 3: Implement the dual replay API.**

Add `ReplayBuffer.clear()` under the current lock. Instantiate `self.adaptation_replay` with the same tensor shapes as the global replay. Extend `finish_step(..., adaptation_enabled: bool = False)` so every safe transition enters `replay` and adapting transitions also enter `adaptation_replay`. Add:

```python
def sample_critic_batch(self, generator=None) -> TransitionBatch:
    return self.replay.sample(self.config.batch_size, self.device, generator=generator)

def sample_actor_batch(self, generator=None) -> TransitionBatch:
    return self.adaptation_replay.sample(self.config.batch_size, self.device, generator=generator)

def clear_adaptation_replay(self) -> None:
    self.adaptation_replay.clear()
```

Keep `sample_update_batch` as a compatibility alias for critic sampling until all callers migrate.

- [ ] **Step 4: Run and commit.**

```bash
pytest tests/online_adaptation/test_replay_buffer.py tests/online_adaptation/test_publisher_agent.py tests/online_adaptation/test_history.py -q
git add source/rl_training/rl_training/online_adaptation/replay_buffer.py source/rl_training/rl_training/online_adaptation/agent.py tests/online_adaptation/test_publisher_agent.py
git commit -m "feat: route novelty transitions to adaptation replay"
```

Expected: all focused tests pass.

## Task 4: Add episode-local performance monitoring

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/metrics.py`
- Modify: `source/rl_training/rl_training/online_adaptation/sac.py`
- Create: `tests/online_adaptation/test_adaptation_metrics.py`
- Modify: `tests/online_adaptation/test_sac.py`

- [ ] **Step 1: Write failing window and rollback tests.**

Create tests for a 2-sample window: the first complete window establishes the best cost, a 5% improvement creates a new stable checkpoint, and two consecutive windows more than 10% worse return a performance rollback decision. Use:

```python
cost = mean_velocity_error + mean_tilt_rms + 0.25 * contact_rate
```

- [ ] **Step 2: Run and verify import or behavior failure.**

```bash
pytest tests/online_adaptation/test_adaptation_metrics.py -q
```

Expected: missing `AdaptationPerformance` or missing decision fields.

- [ ] **Step 3: Implement `AdaptationPerformance`.**

The helper owns `window`, `improvement_ratio`, `degradation_ratio`, `patience`, `best_cost`, `degradation_count`, and the current records. `start_episode()` clears every episode-local value. `add(...)` returns no decision until a full window is present, then returns a dataclass with `cost`, `improved`, `degraded`, and `should_rollback`. Completed windows are non-overlapping so patience counts independent evidence.

- [ ] **Step 4: Extend stable-state observation without heldout inputs.**

Keep `StableStateManager.checkpoint()` and `rollback()`. Add an explicit method that receives an `AdaptationPerformanceDecision`; on improvement it checkpoints the full updater, on rollback it loads the last stable full updater state. Preserve `observe(score, updater)` for existing training callers.

- [ ] **Step 5: Run and commit.**

```bash
pytest tests/online_adaptation/test_adaptation_metrics.py tests/online_adaptation/test_sac.py tests/online_adaptation/test_reward_metrics.py -q
git add source/rl_training/rl_training/online_adaptation/metrics.py source/rl_training/rl_training/online_adaptation/sac.py tests/online_adaptation/test_adaptation_metrics.py tests/online_adaptation/test_sac.py
git commit -m "feat: add episode-local adaptation rollback metrics"
```

Expected: all focused tests pass.

## Task 5: Expose runtime configuration and comparison pass-through

**Files:**
- Modify: `scripts/reinforcement_learning/online_adaptation/cli.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py`
- Modify: `tests/online_adaptation/test_cli.py`
- Modify: `tests/online_adaptation/test_compare_online_adaptation.py`

- [ ] **Step 1: Write failing parser tests.**

```python
assert args.actor_updates_per_second == 1.0
assert args.novelty_calibration_steps == 256
assert args.novelty_enter_threshold == 2.5
assert args.adaptation_min_replay_size == 128
assert args.rollback_window == 50
assert args.rollback_patience == 2
assert args.rollback_degradation_ratio == 0.10
```

The comparison test monkeypatches `subprocess.run` and asserts the same options appear in the play command.

- [ ] **Step 2: Run and verify missing parser fields.**

```bash
pytest tests/online_adaptation/test_cli.py tests/online_adaptation/test_compare_online_adaptation.py -q
```

Expected: missing attributes or command tokens.

- [ ] **Step 3: Add exact CLI defaults.**

```python
parser.add_argument("--actor-updates-per-second", type=float, default=1.0)
parser.add_argument("--novelty-calibration-steps", type=int, default=256)
parser.add_argument("--novelty-enter-threshold", type=float, default=2.5)
parser.add_argument("--novelty-exit-threshold", type=float, default=1.0)
parser.add_argument("--novelty-enter-count", type=int, default=10)
parser.add_argument("--novelty-enter-window", type=int, default=20)
parser.add_argument("--novelty-recovery-steps", type=int, default=100)
parser.add_argument("--adaptation-min-replay-size", type=int, default=128)
parser.add_argument("--rollback-window", type=int, default=50)
parser.add_argument("--rollback-patience", type=int, default=2)
parser.add_argument("--rollback-degradation-ratio", type=float, default=0.10)
parser.add_argument("--stable-improvement-ratio", type=float, default=0.05)
parser.add_argument("--adaptation-cooldown-steps", type=int, default=100)
```

Keep `--updates-per-second` as the critic rate and pass every new option through `compare_online_adaptation.py`.

- [ ] **Step 4: Run and commit.**

```bash
pytest tests/online_adaptation/test_cli.py tests/online_adaptation/test_compare_online_adaptation.py -q
git add scripts/reinforcement_learning/online_adaptation/cli.py scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py tests/online_adaptation/test_cli.py tests/online_adaptation/test_compare_online_adaptation.py
git commit -m "feat: configure novelty-gated adaptation from CLI"
```

Expected: all parser and command tests pass.

## Task 6: Integrate gating, independent update rates, and rollback

**Files:**
- Modify: `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py`
- Create: `tests/online_adaptation/test_play_runtime.py`

- [ ] **Step 1: Write a fake-runtime contract test.**

Feed 256 normal records followed by persistent high-error records through a small orchestration helper extracted from the play script. Assert:

```python
assert actor_updates_before_novelty == 0
assert critic_updates_before_novelty > 0
assert actor_updates_after_novelty > 0
assert all("adaptation_state" in row for row in emitted_rows)
assert all("novelty_score" in row for row in emitted_rows)
```

Feed two degraded performance windows and assert the actor version returns to the stable version with `rollback_reason == "performance_degradation"`.

- [ ] **Step 2: Run and verify the integration test fails.**

```bash
pytest tests/online_adaptation/test_play_runtime.py -q
```

Expected: missing orchestration helper, fields, or split update paths.

- [ ] **Step 3: Construct runtime components from CLI values.**

Create `NoveltyConfig`, `NoveltyGate`, and `AdaptationPerformance` after creating the agent. Track `critic_updates`, `actor_updates`, `actor_updates_before_novelty`, `first_actor_update_step`, `adaptation_episode`, `performance_rollbacks`, `safety_rollbacks`, and state-duration counters.

Define the updater command payload locally in the orchestration layer so the control thread never mutates optimizer state:

```python
@dataclass(frozen=True)
class AdaptationCommand:
    kind: Literal["start_episode", "checkpoint", "rollback"]
    episode: int
    reason: str = ""


commands: queue.SimpleQueue[AdaptationCommand] = queue.SimpleQueue()
actor_gate_event = threading.Event()
```

The updater thread owns `current_episode_id`, `checkpointed_episode_id`, and all `StableStateManager` calls. The control thread owns only the gate and command queue.

- [ ] **Step 4: Route transitions without heldout leakage.**

After reward signals are available, compute novelty inputs and call `gate.observe(...)` before `finish_step`. If the decision enters adapting, first clear the adaptation replay and enqueue an episode-start command, then pass `decision.actor_update_enabled` to `finish_step(..., adaptation_enabled=...)`; the triggering transition is therefore the first item in the new adaptation replay. Read `adapter.heldout_mask()` only after this routing decision and only for metrics. Add an assertion or test seam showing no heldout value is accepted by `NoveltyGate`.

- [ ] **Step 5: Implement independent monotonic schedules in the updater thread.**

```python
next_critic = time.perf_counter()
next_actor = next_critic
while not stop_event.is_set():
    now = time.perf_counter()
    actor_due = now >= next_actor and agent.should_update_actor(
        len(agent.adaptation_replay), args.adaptation_min_replay_size, paused_event.is_set(), actor_gate_event.is_set()
    ) and checkpointed_episode_id == current_episode_id
    if actor_due:
        stats = updater.update(
            critic_batch=agent.sample_critic_batch(),
            actor_batch=agent.sample_actor_batch(),
            update_actor=True,
        )
        if stats.accepted and stats.actor_updated:
            publisher.publish(updater.actor, stats.version)
        next_actor += 1.0 / args.actor_updates_per_second
        next_critic = max(next_critic + 1.0 / args.updates_per_second, now)
    elif now >= next_critic and agent.should_update(len(agent.replay), args.min_replay_size, paused_event.is_set()):
        updater.update(critic_batch=agent.sample_critic_batch(), update_actor=False)
        next_critic += 1.0 / args.updates_per_second
    stop_event.wait(0.001)
```

The full actor update also performs that interval's critic update, so the configured critic rate is not silently increased. Use `update_lock` for both paths. If either configured frequency is non-positive, reject the CLI configuration before starting Isaac Sim.

- [ ] **Step 6: Wire snapshots and rollback.**

The control thread must never wait for a gradient update lock. Give it a `queue.SimpleQueue` of commands containing `start_episode`, `checkpoint`, or `rollback` plus episode id and reason. On `entered_adapting`, the control thread clears adaptation replay, starts the metric episode, increments the episode id, and enqueues `start_episode`; the updater thread processes that command under `update_lock`, checkpoints the full updater, and records `checkpointed_episode_id` before actor updates become eligible. During adapting, each valid step feeds `AdaptationPerformance`; improvement enqueues `checkpoint`, while degradation or safety enqueues `rollback`, closes the actor gate immediately, clears adaptation replay, and enters cooldown. The updater thread restores and publishes the stable actor before acknowledging rollback. This keeps snapshot and optimizer mutation serialized without blocking the 50 Hz control path.

- [ ] **Step 7: Add step and summary logging.**

Each JSONL record adds `adaptation_state`, `novelty_score`, three score components, `actor_update_enabled`, `critic_updates`, `actor_updates`, `adaptation_replay_size`, `adaptation_episode`, and `rollback_reason`. Summary adds pre/post-novelty actor updates, novelty entries, both rollback counts, first actor update step, and state durations. Keep `"heldout_mask_usage": "metrics_only"` in `config.json`.

- [ ] **Step 8: Run and commit.**

```bash
pytest tests/online_adaptation/test_play_runtime.py tests/online_adaptation/test_play_modes.py tests/online_adaptation/test_runtime.py -q
git add scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py tests/online_adaptation/test_play_runtime.py
git commit -m "feat: gate lite3 actor updates by novelty"
```

Expected: all runtime tests pass and the fake normal phase has zero actor updates.

## Task 7: Update acceptance checks and user documentation

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/acceptance.py`
- Modify: `tests/online_adaptation/test_acceptance_report.py`
- Modify: `docs/lite3-online-learning-principles.md`

- [ ] **Step 1: Write failing acceptance tests.**

Require online summaries to include `actor_updates_before_novelty`, `actor_updates_after_novelty`, `novelty_entries`, and `rollbacks`. Acceptance must fail when `actor_updates_before_novelty > 0`, even when the run does not fall.

- [ ] **Step 2: Implement acceptance checks.**

Preserve existing no-fall, safety, parameter-change, and first/second-half metric checks. Add `no_actor_updates_before_novelty` and `actor_updated_after_novelty`. Missing new fields produce a failed `complete_summary` check rather than an exception.

- [ ] **Step 3: Update the principles document.**

Explain the four-state gate, observable signals, critic/actor separation, adaptation replay, and same-episode rollback. State explicitly that the heldout mask remains evaluation-only and that safety checks are not weakened to make a run pass.

- [ ] **Step 4: Run and commit.**

```bash
pytest tests/online_adaptation/test_acceptance_report.py tests/online_adaptation/test_compare_online_adaptation.py -q
git add source/rl_training/rl_training/online_adaptation/acceptance.py tests/online_adaptation/test_acceptance_report.py docs/lite3-online-learning-principles.md
git commit -m "docs: define novelty-gated online acceptance"
```

Expected: all focused tests pass.

## Task 8: Full verification and GPU evaluation

**Files:**
- No planned source changes.
- Generated outputs: `/home/tjx/rl_training_main/logs/online_adaptation/novelty_gate_smoke_*` and comparison directories.

- [ ] **Step 1: Run the complete online-adaptation test suite.**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=source/rl_training:.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest tests/online_adaptation -q
```

Expected: zero failures.

- [ ] **Step 2: Compile all modified scripts.**

```bash
python -m py_compile scripts/reinforcement_learning/online_adaptation/cli.py scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Run a 600-step GPU smoke test.**

Use GPU2, the approved base checkpoint, `residual_00050000.pt`, `--control-timeout-ms 100`, critic 5 Hz, actor 1 Hz, and a new run directory. Verify the process exits normally, `metrics.jsonl` contains 600 records, critic updates are nonzero, and `actor_updates_before_novelty == 0`. Do not use this short run as performance acceptance.

- [ ] **Step 4: Run the full single-pass comparison.**

Run seeds 7, 11, 19, 23, and 31 at forward speed 0.3, 2000 steps, minimum 100 heldout samples, and 100 ms timeout. Inspect `comparison.json`, each `summary.json`, and actor-version timelines. Seeds without heldout coverage remain protocol-invalid and are not deleted.

- [ ] **Step 5: Verify and push repository state.**

```bash
git diff --check
git status --short --branch
git log --oneline -12
git push origin feat/lite3-online-adaptation
git status --short --branch
git rev-parse HEAD
git rev-parse origin/feat/lite3-online-adaptation
```

Expected: no whitespace errors or uncommitted source changes, and the final local and remote hashes match.

## Self-review checklist

- Novelty detection only accepts observable signals and never receives `heldout_mask`.
- Continuous critic learning is implemented by Task 6's independent critic schedule.
- Novelty-only actor learning is enforced by Task 3's replay routing and Task 6's actor gate.
- Actor versions increment only after successful actor updates; critic-only updates cannot publish a policy.
- Performance rollback uses two independent same-episode windows; safety rollback is immediate.
- The complete updater state, not only actor weights, is restored.
- Existing base, frozen-residual, and online modes remain available.
- Every spec requirement maps to a task and a verification command.
