# Lite3 Online Terrain Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Isaac Lab demonstration in which a frozen Lite3 PPO controller remains responsible for nominal locomotion while a bounded residual SAC policy updates its own weights during one continuous traversal of a held-out random-box track.

**Architecture:** Keep the existing PPO and its observation path unchanged. Add a pure-PyTorch online-adaptation package containing temporal encoding, replay, SAC, safety, reward, metrics, and double-buffered publication; connect it to Isaac Lab through a narrow adapter and two isolated environment configurations. Pretrain only on the existing rough/slope distribution, then evaluate base-only, frozen-residual, and online-residual modes on a separate held-out track.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab/Isaac Sim, Gymnasium, RSL-RL, pytest, TensorBoard/CSV/JSON logging, trimesh.

---

## Execution preconditions

- Execute in a dedicated worktree on branch `feat/lite3-online-adaptation`.
- `origin` must be the private repository `https://github.com/jenny-tian/rl_training_main.git`.
- Keep `https://github.com/DeepRoboticsLab/rl_training.git` as fetch-only remote `upstream`; never push to it.
- Before residual pretraining, qualify the base PPO on five source-terrain seeds for 1,000 control steps each. It must have zero falls and mean planar velocity error below 0.35 m/s. The existing `model_20.pt` is only a smoke-test checkpoint; if it fails this gate, resume the existing PPO training rather than weakening the gate.
- Use the Isaac Lab environment for simulator tests:

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/rl_training:$PYTHONPATH"
```

- Pure-PyTorch unit tests must run without launching Isaac Sim. Tests marked `isaaclab` may launch the simulator and run separately.

## File map

Create the following focused modules:

```text
source/rl_training/rl_training/online_adaptation/
├── __init__.py                  # public API only
├── types.py                     # transition, safety, reward, and metric records
├── history.py                   # fixed-length temporal observation buffer
├── replay_buffer.py             # CPU float16 replay with device-aware sampling
├── networks.py                  # GRU encoder, bounded actor, twin critic
├── sac.py                       # SAC losses, optimizers, target updates, checkpoints
├── safety.py                    # physical action projection and pause/recovery state
├── reward.py                    # onboard-compatible dense reward
├── metrics.py                   # rolling metrics and acceptance summaries
├── publisher.py                 # non-blocking double-buffered actor publication
├── agent.py                     # control-step orchestration independent of Isaac Lab
└── isaaclab_adapter.py          # reads Lite3 tensors and produces package records

source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/
├── terrains/
│   ├── __init__.py
│   └── heldout_boxes.py         # flat-to-box continuous track generator
└── config/quadruped/deeprobotics_lite3/
    ├── online_adapt_env_cfg.py  # source-distribution residual pretraining
    └── heldout_boxes_env_cfg.py # evaluation-only environment

scripts/reinforcement_learning/online_adaptation/
├── __init__.py
├── cli.py
├── qualify_base_policy.py
├── train_residual_sac.py
├── play_online_adaptation.py
└── compare_online_adaptation.py

tests/online_adaptation/
├── test_package.py
├── test_history.py
├── test_replay_buffer.py
├── test_networks.py
├── test_sac.py
├── test_safety.py
├── test_reward_metrics.py
├── test_isaaclab_adapter.py
├── test_publisher_agent.py
├── test_terrain_guards.py
├── test_heldout_boxes.py
├── test_cli.py
├── test_play_modes.py
└── test_acceptance_report.py
```

Modify:

```text
source/rl_training/setup.py
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/__init__.py
README.md
pytest.ini
```

## Task 1: Package discovery and shared data contracts

**Files:**
- Modify: `source/rl_training/setup.py:12-43`
- Create: `source/rl_training/rl_training/online_adaptation/__init__.py`
- Create: `source/rl_training/rl_training/online_adaptation/types.py`
- Create: `pytest.ini`
- Test: `tests/online_adaptation/test_package.py`

- [ ] **Step 1: Write the failing import and shape-contract test**

```python
from dataclasses import fields

import torch

from rl_training.online_adaptation.types import RewardSignals, SafetyState, TransitionBatch


def test_transition_batch_contract():
    batch = TransitionBatch(
        history=torch.zeros(4, 25, 49),
        current_frame=torch.zeros(4, 49),
        base_action=torch.zeros(4, 12),
        residual_action=torch.zeros(4, 12),
        reward=torch.zeros(4, 1),
        next_history=torch.zeros(4, 25, 49),
        next_frame=torch.zeros(4, 49),
        next_base_action=torch.zeros(4, 12),
        done=torch.zeros(4, 1),
    )
    assert batch.history.shape == (4, 25, 49)
    assert [field.name for field in fields(SafetyState)] == [
        "projected_gravity_xy",
        "base_height",
        "joint_pos",
        "joint_vel",
        "joint_lower",
        "joint_upper",
        "default_joint_pos",
        "action_scale",
        "nonfoot_contact",
        "control_timed_out",
    ]
    assert len(fields(RewardSignals)) == 10
```

- [ ] **Step 2: Run the test and verify the package does not exist**

Run:

```bash
pytest tests/online_adaptation/test_package.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'rl_training.online_adaptation'`.

- [ ] **Step 3: Add package discovery and exact shared records**

Change `source/rl_training/setup.py` to import `find_packages` and use it:

```python
from setuptools import find_packages, setup

# inside setup(...)
packages=find_packages(),
```

Create `types.py` with these records and no Isaac Lab imports:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransitionBatch:
    history: torch.Tensor
    current_frame: torch.Tensor
    base_action: torch.Tensor
    residual_action: torch.Tensor
    reward: torch.Tensor
    next_history: torch.Tensor
    next_frame: torch.Tensor
    next_base_action: torch.Tensor
    done: torch.Tensor


@dataclass(frozen=True)
class SafetyState:
    projected_gravity_xy: torch.Tensor
    base_height: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    default_joint_pos: torch.Tensor
    action_scale: torch.Tensor
    nonfoot_contact: torch.Tensor
    control_timed_out: torch.Tensor


@dataclass(frozen=True)
class RewardSignals:
    base_lin_vel_xy: torch.Tensor
    command_lin_vel_xy: torch.Tensor
    base_ang_vel_z: torch.Tensor
    command_ang_vel_z: torch.Tensor
    projected_gravity_xy: torch.Tensor
    base_lin_vel_z: torch.Tensor
    foot_slip: torch.Tensor
    foot_impact: torch.Tensor
    joint_power: torch.Tensor
    nonfoot_contact: torch.Tensor


@dataclass(frozen=True)
class SafetyDecision:
    action: torch.Tensor
    residual: torch.Tensor
    paused: torch.Tensor
    projected: torch.Tensor
    reasons: tuple[str, ...]


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
```

Export these names from `online_adaptation/__init__.py`.

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    isaaclab: requires Isaac Lab modules or a launched Isaac Sim application
```

- [ ] **Step 4: Run the test and package smoke check**

```bash
pytest tests/online_adaptation/test_package.py -v
python -c "import rl_training.online_adaptation as oa; print(oa.TransitionBatch.__name__)"
```

Expected: test passes and command prints `TransitionBatch`.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/setup.py source/rl_training/rl_training/online_adaptation pytest.ini tests/online_adaptation/test_package.py
git commit -m "feat: add online adaptation package contracts"
```

## Task 2: Temporal history buffer

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/history.py`
- Test: `tests/online_adaptation/test_history.py`

- [ ] **Step 1: Write tests for append, preview, and selective reset**

```python
import torch

from rl_training.online_adaptation.history import HistoryBuffer


def test_history_append_and_preview_do_not_double_append():
    history = HistoryBuffer(num_envs=2, history_length=3, frame_dim=2, device="cpu")
    history.append(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    preview = history.preview(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    assert torch.equal(history.snapshot()[:, -1], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(preview[:, -1], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))


def test_preview_resets_done_environment_before_appending():
    history = HistoryBuffer(num_envs=2, history_length=3, frame_dim=1, device="cpu")
    history.append(torch.tensor([[1.0], [2.0]]))
    preview = history.preview(torch.tensor([[3.0], [4.0]]), reset_mask=torch.tensor([True, False]))
    assert torch.equal(preview[0, :, 0], torch.tensor([0.0, 0.0, 3.0]))
    assert torch.equal(preview[1, :, 0], torch.tensor([0.0, 2.0, 4.0]))
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_history.py -v
```

Expected: import fails because `history.py` is absent.

- [ ] **Step 3: Implement the fixed-size buffer**

```python
from __future__ import annotations

import torch


class HistoryBuffer:
    def __init__(self, num_envs: int, history_length: int, frame_dim: int, device: str | torch.device):
        self._data = torch.zeros(num_envs, history_length, frame_dim, device=device)

    def append(self, frame: torch.Tensor) -> None:
        if frame.shape != self._data[:, -1].shape:
            raise ValueError(f"expected frame shape {self._data[:, -1].shape}, got {frame.shape}")
        self._data = torch.roll(self._data, shifts=-1, dims=1)
        self._data[:, -1].copy_(frame)

    def preview(self, frame: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        candidate = self._data.clone()
        if reset_mask is not None:
            candidate[reset_mask] = 0.0
        candidate = torch.roll(candidate, shifts=-1, dims=1)
        candidate[:, -1].copy_(frame)
        return candidate

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._data.zero_()
        else:
            self._data[env_ids] = 0.0

    def snapshot(self) -> torch.Tensor:
        return self._data.clone()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/online_adaptation/test_history.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/history.py tests/online_adaptation/test_history.py
git commit -m "feat: add temporal proprioception history"
```

## Task 3: Memory-bounded replay buffer

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/replay_buffer.py`
- Test: `tests/online_adaptation/test_replay_buffer.py`

- [ ] **Step 1: Write overwrite and sample tests**

```python
import torch

from rl_training.online_adaptation.replay_buffer import ReplayBuffer


def make_transition(value: float, batch: int = 2):
    return dict(
        history=torch.full((batch, 3, 2), value),
        current_frame=torch.full((batch, 2), value),
        base_action=torch.full((batch, 1), value),
        residual_action=torch.full((batch, 1), value),
        reward=torch.full((batch, 1), value),
        next_history=torch.full((batch, 3, 2), value + 1),
        next_frame=torch.full((batch, 2), value + 1),
        next_base_action=torch.full((batch, 1), value + 1),
        done=torch.zeros(batch, 1),
    )


def test_replay_overwrites_oldest_and_samples_float32():
    replay = ReplayBuffer(capacity=3, history_length=3, frame_dim=2, action_dim=1, storage_dtype=torch.float16)
    replay.add_batch(**make_transition(1.0))
    replay.add_batch(**make_transition(2.0))
    assert len(replay) == 3
    batch = replay.sample(batch_size=3, device="cpu", generator=torch.Generator().manual_seed(0))
    assert batch.history.dtype == torch.float32
    assert batch.history.shape == (3, 3, 2)
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_replay_buffer.py -v
```

Expected: import fails.

- [ ] **Step 3: Implement preallocated CPU storage**

Implement `ReplayBuffer` with these exact behaviors:

```python
class ReplayBuffer:
    def __init__(self, capacity, history_length, frame_dim, action_dim, storage_dtype=torch.float16):
        self.capacity = capacity
        self._size = 0
        self._cursor = 0
        self._history = torch.empty(capacity, history_length, frame_dim, dtype=storage_dtype, device="cpu")
        self._current_frame = torch.empty(capacity, frame_dim, dtype=storage_dtype, device="cpu")
        self._base_action = torch.empty(capacity, action_dim, dtype=storage_dtype, device="cpu")
        self._residual_action = torch.empty(capacity, action_dim, dtype=storage_dtype, device="cpu")
        self._reward = torch.empty(capacity, 1, dtype=torch.float32, device="cpu")
        self._next_history = torch.empty_like(self._history)
        self._next_frame = torch.empty_like(self._current_frame)
        self._next_base_action = torch.empty_like(self._base_action)
        self._done = torch.empty(capacity, 1, dtype=torch.float32, device="cpu")

    def __len__(self):
        return self._size
```

`add_batch` must iterate over batch rows, write at `_cursor`, wrap modulo capacity, and update `_size`. `sample` must use `torch.randint(..., generator=generator)`, convert floating observation/action tensors to float32 on the requested device, and return `TransitionBatch`.

- [ ] **Step 4: Run tests and memory check**

```bash
pytest tests/online_adaptation/test_replay_buffer.py -v
python - <<'PY'
from rl_training.online_adaptation.replay_buffer import ReplayBuffer
r = ReplayBuffer(10_000, 25, 49, 12)
print(sum(t.numel() * t.element_size() for t in vars(r).values() if hasattr(t, "numel")) / 1024**2)
PY
```

Expected: tests pass; allocated tensor memory remains below 70 MiB.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/replay_buffer.py tests/online_adaptation/test_replay_buffer.py
git commit -m "feat: add bounded online replay buffer"
```

## Task 4: Temporal residual actor and twin critic

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/networks.py`
- Test: `tests/online_adaptation/test_networks.py`

- [ ] **Step 1: Write deterministic shape, bound, and initialization tests**

```python
import torch

from rl_training.online_adaptation.networks import ResidualActor, TwinCritic


LIMITS = torch.tensor([0.24, 0.20, 0.20] * 4)


def test_actor_is_near_zero_and_bounded():
    actor = ResidualActor(frame_dim=49, action_dim=12, history_dim=64, action_limits=LIMITS)
    history = torch.zeros(8, 25, 49)
    frame = torch.zeros(8, 49)
    base = torch.zeros(8, 12)
    action, log_prob, mean = actor.sample(history, frame, base, deterministic=False)
    assert action.shape == (8, 12)
    assert log_prob.shape == (8, 1)
    assert torch.all(action.abs() <= LIMITS + 1e-6)
    assert mean.abs().max().item() < 1e-3


def test_twin_critic_returns_two_columns():
    critic = TwinCritic(frame_dim=49, action_dim=12, history_dim=64)
    q1, q2 = critic(torch.zeros(8, 25, 49), torch.zeros(8, 49), torch.zeros(8, 12), torch.zeros(8, 12))
    assert q1.shape == q2.shape == (8, 1)
    assert q1.data_ptr() != q2.data_ptr()
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_networks.py -v
```

Expected: import fails.

- [ ] **Step 3: Implement the network APIs**

Use a GRU temporal encoder and ELU MLPs:

```python
class HistoryEncoder(nn.Module):
    def __init__(self, frame_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(frame_dim)
        self.gru = nn.GRU(frame_dim, hidden_dim, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(self.norm(history))
        return hidden[-1]
```

`ResidualActor` must concatenate `[history_latent, current_frame, base_action]`, use hidden sizes `[128, 128]`, output mean and log standard deviation, clamp log standard deviation to `[-5, 1]`, tanh-squash samples, apply the standard tanh log-probability correction, and multiply by a registered `action_limits` buffer. Initialize the final mean layer weights and bias uniformly in `[-1e-4, 1e-4]`.

`TwinCritic` must use one temporal encoder and two independent MLP heads. Each head receives `[history_latent, current_frame, base_action, residual_action]` and returns one scalar Q value.

- [ ] **Step 4: Run tests**

```bash
pytest tests/online_adaptation/test_networks.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/networks.py tests/online_adaptation/test_networks.py
git commit -m "feat: add temporal residual SAC networks"
```

## Task 5: SAC updater, finite-update rejection, and stable checkpoints

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/sac.py`
- Test: `tests/online_adaptation/test_sac.py`

- [ ] **Step 1: Write tests for one valid update and one rejected update**

```python
import torch

from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.sac import SACConfig, SACUpdater, StableStateManager
from rl_training.online_adaptation.types import TransitionBatch


def make_batch(reward_value=1.0):
    return TransitionBatch(
        history=torch.randn(32, 25, 49),
        current_frame=torch.randn(32, 49),
        base_action=torch.randn(32, 12),
        residual_action=torch.zeros(32, 12),
        reward=torch.full((32, 1), reward_value),
        next_history=torch.randn(32, 25, 49),
        next_frame=torch.randn(32, 49),
        next_base_action=torch.randn(32, 12),
        done=torch.zeros(32, 1),
    )


def test_sac_update_changes_version_and_target_parameters():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    critic = TwinCritic(49, 12, 64)
    updater = SACUpdater(actor, critic, SACConfig())
    before = [p.detach().clone() for p in updater.target_critic.parameters()]
    stats = updater.update(make_batch())
    assert stats.accepted
    assert stats.version == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, updater.target_critic.parameters()))


def test_nonfinite_batch_is_rejected_without_version_change():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    stats = updater.update(make_batch(float("nan")))
    assert not stats.accepted
    assert stats.version == 0


def test_stable_state_rolls_back_after_sustained_degradation():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    manager = StableStateManager(patience=3, min_delta=0.01)
    manager.observe(score=1.0, updater=updater)
    updater.version = 7
    assert not manager.observe(score=0.9, updater=updater)
    assert not manager.observe(score=0.8, updater=updater)
    assert manager.observe(score=0.7, updater=updater)
    assert updater.version == 0
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_sac.py -v
```

Expected: import fails.

- [ ] **Step 3: Implement exact SAC behavior**

Define:

```python
@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float = -6.0
    max_grad_norm: float = 1.0
```

`SACUpdater.update` must:

1. Reject any batch containing non-finite values.
2. Compute target `reward + (1-done)*gamma*(min(target_q1,target_q2)-alpha*next_log_prob)` under `torch.no_grad()`.
3. Update both critics using mean squared error.
4. Freeze critic parameters while computing Actor loss `(alpha*log_prob-min(q1,q2)).mean()`.
5. Update `log_alpha` toward `target_entropy`.
6. Clip Actor and Critic gradient norms to `max_grad_norm`.
7. Reject the update and restore the pre-update state if any loss, gradient, or resulting parameter is non-finite.
8. Soft-update the target critic only after an accepted update.
9. Increment `version` only after an accepted update.

Add `state_dict`, `load_state_dict`, `save(path)`, and `load(path, map_location)` methods that include Actor, Critic, target Critic, all optimizers, `log_alpha`, and `version`.

Add `StableStateManager(patience=5, min_delta=0.01)`. On a score improvement it deep-copies the complete updater state as the stable state and resets degradation count. On a non-improvement it increments the count; once patience is reached it restores the stable state, resets the count, and returns `True`. If no stable state has been recorded, it must never report a rollback.

- [ ] **Step 4: Run tests**

```bash
pytest tests/online_adaptation/test_sac.py -v
```

Expected: 3 tests pass with finite losses and deterministic rollback.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/sac.py tests/online_adaptation/test_sac.py
git commit -m "feat: add safe residual SAC updater"
```

## Task 6: Physical safety projection and pause/recovery state machine

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/safety.py`
- Test: `tests/online_adaptation/test_safety.py`

- [ ] **Step 1: Write projection and hysteresis tests**

```python
import torch

from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState


def state(tilt=0.0, height=0.35):
    return SafetyState(
        projected_gravity_xy=torch.tensor([[tilt, 0.0]]),
        base_height=torch.tensor([[height]]),
        joint_pos=torch.zeros(1, 12),
        joint_vel=torch.zeros(1, 12),
        joint_lower=torch.full((1, 12), -1.0),
        joint_upper=torch.full((1, 12), 1.0),
        default_joint_pos=torch.zeros(1, 12),
        action_scale=torch.full((1, 12), 0.25),
        nonfoot_contact=torch.zeros(1, 1, dtype=torch.bool),
        control_timed_out=torch.zeros(1, 1, dtype=torch.bool),
    )


def test_rate_and_joint_target_projection():
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2), residual_rate_limit=0.05))
    decision = filt.project(torch.zeros(1, 12), torch.ones(1, 12), state())
    assert torch.allclose(decision.residual, torch.full((1, 12), 0.05))
    assert decision.projected.item()


def test_unsafe_tilt_pauses_and_requires_stable_window_to_resume():
    filt = SafetyFilter(SafetyLimits(residual_limits=torch.full((12,), 0.2), stable_steps_to_resume=3))
    assert filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state(tilt=0.8)).paused.item()
    for _ in range(2):
        assert filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state()).paused.item()
    assert not filt.project(torch.zeros(1, 12), torch.zeros(1, 12), state()).paused.item()
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_safety.py -v
```

- [ ] **Step 3: Implement limits and projection order**

Define `SafetyLimits` with defaults:

```python
@dataclass(frozen=True)
class SafetyLimits:
    residual_limits: torch.Tensor
    residual_rate_limit: float = 0.02
    max_projected_gravity_xy: float = 0.65
    min_base_height: float = 0.20
    max_joint_velocity: float = 30.0
    stable_steps_to_resume: int = 25
```

`SafetyFilter.project` must apply this order:

1. Determine unsafe environments from gravity, height, joint velocity, non-foot contact, and timeout.
2. Update a per-environment paused mask with stable-window hysteresis.
3. Clamp raw residual to configured normalized-action limits.
4. Clamp change relative to previous accepted residual.
5. Set paused-environment residuals to zero.
6. Convert `base_action + residual` to physical target using `default_joint_pos + action_scale * action`.
7. Clamp physical targets to joint limits with a 0.02 rad margin.
8. Convert the projected physical targets back into normalized actions.
9. Return `SafetyDecision` and persist the accepted residual.

- [ ] **Step 4: Run tests**

```bash
pytest tests/online_adaptation/test_safety.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/safety.py tests/online_adaptation/test_safety.py
git commit -m "feat: add residual safety projection"
```

## Task 7: Onboard-compatible reward and rolling metrics

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/reward.py`
- Create: `source/rl_training/rl_training/online_adaptation/metrics.py`
- Test: `tests/online_adaptation/test_reward_metrics.py`

- [ ] **Step 1: Write reward monotonicity and half-track summary tests**

```python
import torch

from rl_training.online_adaptation.metrics import RollingMetrics
from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
from rl_training.online_adaptation.types import RewardSignals


def signals(error=0.0, tilt=0.0):
    return RewardSignals(
        base_lin_vel_xy=torch.tensor([[1.0 - error, 0.0]]),
        command_lin_vel_xy=torch.tensor([[1.0, 0.0]]),
        base_ang_vel_z=torch.zeros(1, 1),
        command_ang_vel_z=torch.zeros(1, 1),
        projected_gravity_xy=torch.tensor([[tilt, 0.0]]),
        base_lin_vel_z=torch.zeros(1, 1),
        foot_slip=torch.zeros(1, 1),
        foot_impact=torch.zeros(1, 1),
        joint_power=torch.zeros(1, 1),
        nonfoot_contact=torch.zeros(1, 1),
    )


def test_reward_decreases_with_tracking_error_and_tilt():
    residual = torch.zeros(1, 12)
    good = compute_online_reward(signals(), residual, residual, RewardWeights())
    bad = compute_online_reward(signals(error=0.5, tilt=0.5), residual, residual, RewardWeights())
    assert good.item() > bad.item()


def test_metrics_reports_first_and_second_half():
    metrics = RollingMetrics(window=3)
    for index in range(6):
        metrics.update(step=index, in_heldout=True, velocity_error=1.0 - 0.1 * index, tilt_rms=0.5 - 0.05 * index, nonfoot_contact=0)
    summary = metrics.heldout_summary()
    assert summary["second_half_velocity_error"] < summary["first_half_velocity_error"]
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_reward_metrics.py -v
```

- [ ] **Step 3: Implement bounded reward**

Use this formula with configurable weights:

```python
velocity_term = torch.exp(-4.0 * torch.sum((signals.base_lin_vel_xy - signals.command_lin_vel_xy) ** 2, dim=-1, keepdim=True))
yaw_term = torch.exp(-2.0 * (signals.base_ang_vel_z - signals.command_ang_vel_z) ** 2)
tilt = torch.sum(signals.projected_gravity_xy**2, dim=-1, keepdim=True)
residual_cost = torch.mean(residual**2, dim=-1, keepdim=True)
residual_rate_cost = torch.mean((residual - previous_residual) ** 2, dim=-1, keepdim=True)
reward = (
    2.0 * velocity_term
    + 0.5 * yaw_term
    - 2.0 * tilt
    - 0.5 * signals.base_lin_vel_z**2
    - 0.05 * signals.foot_slip
    - 0.02 * signals.foot_impact
    - 0.001 * signals.joint_power
    - 0.10 * residual_cost
    - 0.05 * residual_rate_cost
    - 2.0 * signals.nonfoot_contact
)
return torch.clamp(reward, -10.0, 5.0)
```

`RewardWeights` must expose every numeric coefficient. `RollingMetrics` must keep raw held-out samples and a fixed-size rolling deque, write JSON-serializable dictionaries, and compute first-half/second-half means without dropping failures.

- [ ] **Step 4: Run tests**

```bash
pytest tests/online_adaptation/test_reward_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/reward.py source/rl_training/rl_training/online_adaptation/metrics.py tests/online_adaptation/test_reward_metrics.py
git commit -m "feat: add online reward and adaptation metrics"
```

## Task 8: Isaac Lab proprioception and safety adapter

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py`
- Test: `tests/online_adaptation/test_isaaclab_adapter.py`

- [ ] **Step 1: Write a pure frame-composition test**

```python
import pytest
import torch

from rl_training.online_adaptation.isaaclab_adapter import compose_proprio_frame

pytestmark = pytest.mark.isaaclab


def test_compose_proprio_frame_has_fixed_order_and_49_dimensions():
    terms = [
        torch.full((2, 3), 1.0),
        torch.full((2, 3), 2.0),
        torch.full((2, 12), 3.0),
        torch.full((2, 12), 4.0),
        torch.full((2, 4), 5.0),
        torch.full((2, 12), 6.0),
        torch.full((2, 3), 7.0),
    ]
    frame = compose_proprio_frame(*terms)
    assert frame.shape == (2, 49)
    assert torch.equal(frame[:, :3], terms[0])
    assert torch.equal(frame[:, -3:], terms[-1])
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_isaaclab_adapter.py -v
```

- [ ] **Step 3: Implement a narrow runtime adapter**

`compose_proprio_frame` must concatenate in this exact order:

```text
base angular velocity (3), projected gravity (3), relative joint position (12),
joint velocity (12), binary foot contact (4), previous final action (12), command (3)
```

Implement `Lite3IsaacLabAdapter` that resolves and stores foot and non-foot body IDs once. Its methods must be:

```python
def frame(self, previous_action: torch.Tensor) -> torch.Tensor:
    robot = self.robot
    forces = self.contact_sensor.data.net_forces_w[:, self.foot_ids]
    contacts = (torch.linalg.norm(forces, dim=-1) > self.contact_threshold).float()
    command = self.env.command_manager.get_command("base_velocity")[:, :3]
    return compose_proprio_frame(
        robot.data.root_ang_vel_b,
        robot.data.projected_gravity_b,
        robot.data.joint_pos - robot.data.default_joint_pos,
        robot.data.joint_vel,
        contacts,
        previous_action,
        command,
    )

def safety_state(self, control_timed_out: torch.Tensor) -> SafetyState:
    limits = self.robot.data.soft_joint_pos_limits
    nonfoot_forces = self.contact_sensor.data.net_forces_w[:, self.nonfoot_ids]
    return SafetyState(
        projected_gravity_xy=self.robot.data.projected_gravity_b[:, :2],
        base_height=self.robot.data.root_pos_w[:, 2:3],
        joint_pos=self.robot.data.joint_pos,
        joint_vel=self.robot.data.joint_vel,
        joint_lower=limits[..., 0],
        joint_upper=limits[..., 1],
        default_joint_pos=self.robot.data.default_joint_pos,
        action_scale=self.action_scale.expand_as(self.robot.data.joint_pos),
        nonfoot_contact=(torch.linalg.norm(nonfoot_forces, dim=-1) > self.contact_threshold).any(dim=1, keepdim=True),
        control_timed_out=control_timed_out,
    )

def reward_signals(self) -> RewardSignals:
    foot_forces = torch.linalg.norm(self.contact_sensor.data.net_forces_w[:, self.foot_ids], dim=-1)
    contacts = foot_forces > self.contact_threshold
    foot_speed = torch.linalg.norm(self.robot.data.body_lin_vel_w[:, self.foot_ids, :2], dim=-1)
    command = self.env.command_manager.get_command("base_velocity")
    nonfoot_forces = self.contact_sensor.data.net_forces_w[:, self.nonfoot_ids]
    return RewardSignals(
        base_lin_vel_xy=self.robot.data.root_lin_vel_b[:, :2],
        command_lin_vel_xy=command[:, :2],
        base_ang_vel_z=self.robot.data.root_ang_vel_b[:, 2:3],
        command_ang_vel_z=command[:, 2:3],
        projected_gravity_xy=self.robot.data.projected_gravity_b[:, :2],
        base_lin_vel_z=self.robot.data.root_lin_vel_b[:, 2:3],
        foot_slip=(foot_speed * contacts.float()).mean(dim=1, keepdim=True),
        foot_impact=foot_forces.max(dim=1, keepdim=True).values,
        joint_power=torch.abs(self.robot.data.joint_vel * self.robot.data.applied_torque).mean(dim=1, keepdim=True),
        nonfoot_contact=(torch.linalg.norm(nonfoot_forces, dim=-1) > self.contact_threshold).any(dim=1, keepdim=True).float(),
    )

def heldout_mask(self) -> torch.Tensor:
    return self.robot.data.root_pos_w[:, 0:1] >= self.track_boundary_x
```

Use `robot.data.root_ang_vel_b`, `robot.data.projected_gravity_b`, relative joint positions, joint velocities, `contact_sensor.data.net_forces_w`, command manager output, root linear velocity, foot body velocities, applied torque, and joint limits. `heldout_mask` may compare root world X against the configured track boundary but must only be consumed by metrics/logging; do not expose it through `frame`, `safety_state`, reward, replay, or Actor inputs.

Initialize the adapter's per-joint physical action scale in Lite3 joint order as `torch.tensor([0.125, 0.25, 0.25] * 4)`. Read `track_boundary_x` from the held-out environment config; for the 20 m track with a 3 m flat prefix centered by TerrainGenerator, the boundary is `-7.0` m.

- [ ] **Step 4: Run pure test and Isaac import smoke check**

```bash
pytest tests/online_adaptation/test_isaaclab_adapter.py -v
python -c "from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter; print(Lite3IsaacLabAdapter.__name__)"
```

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py tests/online_adaptation/test_isaaclab_adapter.py
git commit -m "feat: add Lite3 online adaptation adapter"
```

## Task 9: Source-terrain leakage guard and training environment

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/terrain_guards.py`
- Create: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/online_adapt_env_cfg.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/__init__.py`
- Test: `tests/online_adaptation/test_terrain_guards.py`

- [ ] **Step 1: Write leakage guard tests using fake terrain objects**

```python
from types import SimpleNamespace

import pytest

from rl_training.online_adaptation.terrain_guards import assert_heldout_terrains_disabled


def terrain(boxes=0.0, stairs=0.0, stairs_inv=0.0):
    sub = {
        "boxes": SimpleNamespace(proportion=boxes),
        "pyramid_stairs": SimpleNamespace(proportion=stairs),
        "pyramid_stairs_inv": SimpleNamespace(proportion=stairs_inv),
    }
    return SimpleNamespace(sub_terrains=sub)


def test_guard_accepts_zero_heldout_proportions():
    assert_heldout_terrains_disabled(terrain())


def test_guard_rejects_any_heldout_probability():
    with pytest.raises(ValueError, match="held-out terrain leakage"):
        assert_heldout_terrains_disabled(terrain(boxes=0.01))
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_terrain_guards.py -v
```

- [ ] **Step 3: Implement guard and source config**

`assert_heldout_terrains_disabled` must inspect `boxes`, `pyramid_stairs`, and `pyramid_stairs_inv`, collect any proportion greater than zero, and raise one `ValueError` listing all leaks.

`DeeproboticsLite3OnlineAdaptEnvCfg` must subclass `DeeproboticsLite3RoughEnvCfg`, call the guard after `super().__post_init__()`, set `scene.num_envs = 1024`, keep height scan disabled for the policy, keep command and domain randomization behavior from the parent, and set a distinct experiment/task name without enabling held-out terrain.

Register:

```python
gym.register(
    id="OnlineAdapt-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.online_adapt_env_cfg:DeeproboticsLite3OnlineAdaptEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughPPORunnerCfg",
    },
)
```

- [ ] **Step 4: Run guard tests and list environment**

```bash
pytest tests/online_adaptation/test_terrain_guards.py -v
python scripts/tools/list_envs.py | rg 'OnlineAdapt-Deeprobotics-Lite3-v0'
```

Expected: tests pass and the environment ID is printed.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/terrain_guards.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3 tests/online_adaptation/test_terrain_guards.py
git commit -m "feat: add leakage-safe residual training environment"
```

## Task 10: Continuous held-out flat-to-box track

**Files:**
- Create: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/__init__.py`
- Create: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/heldout_boxes.py`
- Create: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/heldout_boxes_env_cfg.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/__init__.py`
- Test: `tests/online_adaptation/test_heldout_boxes.py`

- [ ] **Step 1: Write deterministic mesh tests**

```python
import numpy as np
import pytest

from rl_training.tasks.manager_based.locomotion.velocity.terrains.heldout_boxes import (
    HeldoutBoxesTrackCfg,
    heldout_boxes_track,
)

pytestmark = pytest.mark.isaaclab


def test_track_is_deterministic_and_has_flat_then_variable_region():
    cfg = HeldoutBoxesTrackCfg(
        proportion=1.0,
        size=(20.0, 4.0),
        flat_length=3.0,
        lane_width=2.0,
        grid_width=0.4,
        grid_height_range=(0.04, 0.12),
    )
    cfg.seed = 7
    meshes, origin = heldout_boxes_track(0.5, cfg)
    vertices = np.concatenate([mesh.vertices for mesh in meshes])
    flat_z = vertices[vertices[:, 0] < cfg.flat_length - 0.1, 2]
    heldout_z = vertices[vertices[:, 0] > cfg.flat_length + 0.5, 2]
    assert origin[0] < cfg.flat_length
    assert flat_z.max() <= 1e-6
    assert heldout_z.max() > 0.04
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_heldout_boxes.py -v
```

- [ ] **Step 3: Implement custom terrain and evaluation config**

Create `HeldoutBoxesTrackCfg(SubTerrainBaseCfg)` with:

```python
function = heldout_boxes_track
flat_length: float = 3.0
lane_width: float = 2.0
grid_width: float = 0.4
grid_height_range: tuple[float, float] = (0.04, 0.12)
```

`heldout_boxes_track` must create one continuous base slab, keep the first `flat_length` meters level, tile the remaining X range with columns of width `grid_width`, sample each top height uniformly in `[-height, height]` where `height` interpolates by difficulty, and return an origin centered laterally at X=1.0 m on the flat segment. Construct a local generator with `np.random.default_rng(getattr(cfg, "seed", 0))`; do not read or mutate global NumPy state.

`DeeproboticsLite3HeldoutBoxesEnvCfg` must:

- subclass the Lite3 rough config;
- replace the terrain generator with one row, one column, size `(20.0, 4.0)`, only `HeldoutBoxesTrackCfg(proportion=1.0)`, fixed seed, no curriculum;
- spawn at the custom origin with yaw 0;
- set a constant forward command of 0.5 m/s and zero lateral/yaw commands;
- expose `heldout_start_x = -7.0` for metrics-only segmentation;
- use one environment by default;
- disable time-out termination for the continuous demonstration while retaining bad-orientation protection;
- disable policy corruption and all reset/push randomization during evaluation.

Register `HeldoutBoxes-Deeprobotics-Lite3-v0`.

- [ ] **Step 4: Run mesh test and a headless environment smoke test**

```bash
pytest tests/online_adaptation/test_heldout_boxes.py -v
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=HeldoutBoxes-Deeprobotics-Lite3-v0 \
  --checkpoint logs/rsl_rl/deeprobotics_lite3_rough/2026-07-24_14-42-38/model_20.pt \
  --num_envs=1 --headless --video_length=50
```

Expected: terrain loads, Lite3 spawns on the flat segment, and 50 simulation steps complete without configuration errors.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3 tests/online_adaptation/test_heldout_boxes.py
git commit -m "feat: add held-out flat-to-box evaluation track"
```

## Task 11: Double-buffered publication and algorithm-independent agent

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/publisher.py`
- Create: `source/rl_training/rl_training/online_adaptation/agent.py`
- Test: `tests/online_adaptation/test_publisher_agent.py`

- [ ] **Step 1: Write publication and warm-up tests**

```python
import torch

from rl_training.online_adaptation.agent import OnlineAdaptationAgent
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher


def test_publisher_swaps_complete_versions():
    actor = torch.nn.Linear(2, 1, bias=False)
    publisher = DoubleBufferedActorPublisher(actor)
    updated = torch.nn.Linear(2, 1, bias=False)
    updated.weight.data.fill_(3.0)
    publisher.publish(updated, version=4)
    published, version = publisher.current()
    assert version == 4
    assert torch.equal(published.weight, updated.weight)


def test_agent_does_not_request_update_before_minimum_replay():
    assert not OnlineAdaptationAgent.should_update(replay_size=255, min_replay_size=256, paused=False)
    assert OnlineAdaptationAgent.should_update(replay_size=256, min_replay_size=256, paused=False)
    assert not OnlineAdaptationAgent.should_update(replay_size=256, min_replay_size=256, paused=True)
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_publisher_agent.py -v
```

- [ ] **Step 3: Implement publication and step orchestration**

`DoubleBufferedActorPublisher` must keep two inference-only deep copies on the same device. `publish(training_actor, version)` loads weights into the inactive copy before acquiring a short lock to swap the active index and version. `current()` returns the active Actor and version; it must never observe a partially loaded state.

Define:

```python
@dataclass(frozen=True)
class AgentConfig:
    history_length: int = 25
    frame_dim: int = 49
    action_dim: int = 12
    replay_capacity: int = 10_000
    min_replay_size: int = 256
    batch_size: int = 128
    updates_per_second: float = 5.0
```

`OnlineAdaptationAgent` must own `HistoryBuffer`, `ReplayBuffer`, `SafetyFilter`, `SACUpdater`, and publisher. Expose:

```python
def begin_step(self, frame, base_action, safety_state, deterministic):
    if self._pending is not None:
        raise RuntimeError("finish_step must be called before the next begin_step")
    self.history.append(frame)
    history = self.history.snapshot()
    actor, version = self.publisher.current()
    with torch.no_grad():
        residual, _, _ = actor.sample(history, frame, base_action, deterministic=deterministic)
    decision = self.safety.project(base_action, residual, safety_state)
    self._pending = PendingStep(history, frame, base_action, decision.residual, version)
    return decision

def finish_step(self, next_frame, next_base_action, reward, done):
    if self._pending is None:
        raise RuntimeError("begin_step must be called before finish_step")
    next_history = self.history.preview(next_frame, reset_mask=done.squeeze(-1).bool())
    self.replay.add_batch(
        history=self._pending.history,
        current_frame=self._pending.frame,
        base_action=self._pending.base_action,
        residual_action=self._pending.residual_action,
        reward=reward,
        next_history=next_history,
        next_frame=next_frame,
        next_base_action=next_base_action,
        done=done.float(),
    )
    done_ids = torch.nonzero(done.squeeze(-1), as_tuple=False).squeeze(-1)
    if done_ids.numel() > 0:
        self.history.reset(done_ids)
    self._pending = None

def sample_update_batch(self):
    return self.replay.sample(self.config.batch_size, self.device)

@staticmethod
def should_update(replay_size: int, min_replay_size: int, paused: bool) -> bool:
    return replay_size >= min_replay_size and not paused
```

Define a private frozen `PendingStep` dataclass with `history`, `frame`, `base_action`, `residual_action`, and `actor_version` fields matching the values above.

`begin_step` appends the current frame once, snapshots history, samples the published Actor, applies safety, and stores a pending transition. `finish_step` creates next history using `HistoryBuffer.preview(next_frame, reset_mask=done)`, stores the batch, resets completed environments, and clears pending state. It must raise if called out of order.

- [ ] **Step 4: Run tests**

```bash
pytest tests/online_adaptation/test_publisher_agent.py -v
```

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/publisher.py source/rl_training/rl_training/online_adaptation/agent.py tests/online_adaptation/test_publisher_agent.py
git commit -m "feat: add online adaptation agent runtime"
```

## Task 12: Base PPO qualification and residual SAC source pretraining

**Files:**
- Create: `scripts/reinforcement_learning/online_adaptation/__init__.py`
- Create: `scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py`
- Create: `scripts/reinforcement_learning/online_adaptation/train_residual_sac.py`
- Create: `scripts/reinforcement_learning/online_adaptation/cli.py`
- Test: `tests/online_adaptation/test_cli.py`

- [ ] **Step 1: Write CLI default tests**

```python
from scripts.reinforcement_learning.online_adaptation.cli import build_qualify_parser, build_train_parser


def test_train_defaults_keep_heldout_task_out_of_pretraining():
    args = build_train_parser().parse_args(["--base-checkpoint", "base.pt"])
    assert args.task == "OnlineAdapt-Deeprobotics-Lite3-v0"
    assert "HeldoutBoxes" not in args.task
    assert args.min_replay_size == 256
    assert args.replay_capacity == 10_000


def test_qualification_defaults_are_source_only_and_strict():
    args = build_qualify_parser().parse_args(["--base-checkpoint", "base.pt"])
    assert args.task == "OnlineAdapt-Deeprobotics-Lite3-v0"
    assert args.seeds == [7, 11, 19, 23, 31]
    assert args.steps_per_seed == 1_000
    assert args.max_mean_velocity_error == 0.35
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_cli.py -v
```

- [ ] **Step 3: Implement parsers and the base qualification gate**

The train parser must require `--base-checkpoint` and support `--task`, `--num-envs`, `--seed`, `--max-steps`, `--device`, `--log-dir`, `--save-interval`, `--min-replay-size`, `--replay-capacity`, and AppLauncher arguments.

The qualification parser must support `--seeds 7 11 19 23 31`, `--steps-per-seed 1000`, `--max-mean-velocity-error 0.35`, `--output`, and AppLauncher arguments.

`qualify_base_policy.py` must run the frozen base policy with residual identically zero on the source-only task, disable timeout termination, count any other done as a fall, compute planar velocity error from the adapter, and write one JSON record per seed plus an aggregate record. Exit non-zero unless all seeds have zero falls and aggregate mean velocity error is at most 0.35 m/s.

- [ ] **Step 4: Qualify the current base checkpoint or resume PPO training**

```bash
python scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_rough/2026-07-24_14-42-38/model_20.pt \
  --headless --output logs/online_adaptation/base_qualification.json
```

Expected for the current short checkpoint: it may fail the quality gate. If it fails, resume the existing PPO task:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Rough-Deeprobotics-Lite3-v0 --headless --max_iterations=10000
```

Select the resulting converged checkpoint and rerun `qualify_base_policy.py`. Do not begin residual pretraining until the qualification command exits 0.

- [ ] **Step 5: Implement the residual pretraining loop**

The training script must:

1. Launch Isaac Sim before importing simulator-dependent modules.
2. Reject any task name containing `HeldoutBoxes`.
3. Instantiate `OnlineAdapt-Deeprobotics-Lite3-v0` and the existing RSL-RL runner.
4. Load the base checkpoint, retrieve its inference policy, set all underlying policy parameters to `requires_grad_(False)`, and assert no parameter is trainable.
5. Build the adapter, residual networks, safety filter, replay, and updater.
6. For each vectorized step, compute base action under `torch.inference_mode()`, compute bounded residual, step the environment, compute the onboard-compatible reward, store transitions, and update SAC after 256 samples.
7. Use a strong residual magnitude penalty from Task 7 and deterministic Actor evaluation every save interval.
8. Every 250 steps feed the rolling source-terrain score to `StableStateManager`; log and save whenever it performs a rollback.
9. Save zero-padded checkpoints such as `residual_00000100.pt`, plus `latest.pt`, YAML/JSON configuration, TensorBoard scalars, and a terrain-leakage assertion result.
10. Never import the held-out environment module directly.

- [ ] **Step 6: Run CLI test and 300-step smoke training**

```bash
pytest tests/online_adaptation/test_cli.py -v
python scripts/reinforcement_learning/online_adaptation/train_residual_sac.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_rough/2026-07-24_14-42-38/model_20.pt \
  --num-envs=32 --max-steps=300 --headless \
  --log-dir logs/online_adaptation/smoke
```

Expected: no held-out terrain is loaded, replay grows, losses remain finite once updates begin, and `latest.pt` is created.

- [ ] **Step 7: Commit**

```bash
git add scripts/reinforcement_learning/online_adaptation tests/online_adaptation/test_cli.py
git commit -m "feat: add residual SAC pretraining workflow"
```

## Task 13: Single-pass asynchronous online-learning demonstration

**Files:**
- Create: `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/cli.py`
- Test: `tests/online_adaptation/test_play_modes.py`

- [ ] **Step 1: Write mode and information-leakage tests**

```python
from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser


def test_play_modes_are_explicit_and_default_to_online():
    args = build_play_parser().parse_args([
        "--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt"
    ])
    assert args.mode == "online"
    assert args.task == "HeldoutBoxes-Deeprobotics-Lite3-v0"


def test_base_mode_does_not_require_residual_checkpoint():
    args = build_play_parser().parse_args(["--base-checkpoint", "base.pt", "--mode", "base"])
    assert args.residual_checkpoint is None
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_play_modes.py -v
```

- [ ] **Step 3: Implement the three-mode demonstration**

Support modes:

```text
base              frozen PPO, residual identically zero, no updater thread
residual_frozen   pretrained residual Actor, no gradient updates
online            pretrained residual plus asynchronous SAC updates
```

The script must:

- force one environment and no automatic reset;
- use identical seed and constant forward command for all modes;
- collect replay from the first control frame;
- start updates only when replay size is at least 256 and safety is not paused;
- run a background updater paced by `updates_per_second`, publish only accepted versions through the double buffer, and stop cleanly through `threading.Event`;
- measure adapter + base inference + residual inference + safety projection time separately from `env.step` and rendering; if this control-compute duration exceeds 20 ms, set `control_timed_out` and continue with the frozen base action;
- fail the run if the environment reports done; do not reset and continue;
- use `heldout_mask` only when calling metrics/logging;
- write `metrics.jsonl`, `summary.json`, `config.json`, `residual_before.pt`, `residual_after.pt`, and TensorBoard events;
- record update count, parameter version, safety reasons, base/residual action norms, velocity error, tilt RMS, contacts, and loop time;
- every 250 held-out steps evaluate the rolling score with `StableStateManager` and publish the restored stable Actor immediately after a rollback;
- close updater thread, environment, video writer, and simulation app in `finally` blocks.

- [ ] **Step 4: Run tests and a 400-step online smoke demo**

```bash
pytest tests/online_adaptation/test_play_modes.py -v
python scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_rough/2026-07-24_14-42-38/model_20.pt \
  --residual-checkpoint logs/online_adaptation/smoke/latest.pt \
  --mode online --seed 7 --max-steps 400 --headless \
  --run-dir logs/online_adaptation/demo_smoke
```

Expected: run does not reset, parameters before/after differ after accepted updates, and summary includes `updates`, `safety_fallbacks`, and held-out half metrics.

- [ ] **Step 5: Commit**

```bash
git add scripts/reinforcement_learning/online_adaptation tests/online_adaptation/test_play_modes.py
git commit -m "feat: add single-pass online adaptation demo"
```

## Task 14: Three-way comparison and acceptance reporting

**Files:**
- Create: `scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py`
- Create: `source/rl_training/rl_training/online_adaptation/acceptance.py`
- Test: `tests/online_adaptation/test_acceptance_report.py`

- [ ] **Step 1: Write acceptance calculation tests**

```python
import pytest

from rl_training.online_adaptation.acceptance import evaluate_run


def test_acceptance_requires_velocity_tilt_and_no_reset():
    result = evaluate_run({
        "reset_count": 0,
        "fell": False,
        "first_half_velocity_error": 1.0,
        "second_half_velocity_error": 0.65,
        "first_half_tilt_rms": 0.5,
        "second_half_tilt_rms": 0.39,
        "first_half_nonfoot_contacts": 2,
        "second_half_nonfoot_contacts": 2,
        "parameter_changed": True,
        "safety_violation": False,
    })
    assert result["passed"]
    assert result["velocity_improvement"] == pytest.approx(0.35)
    assert result["tilt_improvement"] == pytest.approx(0.22)
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/online_adaptation/test_acceptance_report.py -v
```

- [ ] **Step 3: Implement evaluator and comparison runner**

`evaluate_run` must require:

- no reset and no fall;
- velocity error improvement `>= 0.30`;
- tilt RMS improvement `>= 0.20`;
- second-half non-foot contacts no greater than first-half contacts;
- residual parameters changed;
- no safety-bound violation.

`compare_online_adaptation.py` must run `base`, `residual_frozen`, and `online` for every requested seed using `subprocess.run(..., check=True)`, preserve failed run directories, aggregate all seeds, and produce:

```text
comparison.csv
comparison.json
comparison.md
velocity_error.png
tilt_rms.png
success_rate.png
```

The report must separately show within-run first/second-half improvement and between-mode improvement. It must never discard failed seeds.

The aggregate comparison passes only when the online mode succeeds on at least 7 of 10 seeds, its mean second-half velocity error is at least 10% lower than `residual_frozen`, and its mean second-half tilt RMS is no worse than `residual_frozen`.

- [ ] **Step 4: Run tests and one-seed comparison smoke test**

```bash
pytest tests/online_adaptation/test_acceptance_report.py -v
python scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_rough/2026-07-24_14-42-38/model_20.pt \
  --residual-checkpoint logs/online_adaptation/smoke/latest.pt \
  --seeds 7 --max-steps 400 --headless \
  --output-dir logs/online_adaptation/comparison_smoke
```

Expected: all three mode directories and aggregate report files exist.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/acceptance.py scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py tests/online_adaptation/test_acceptance_report.py
git commit -m "feat: add online adaptation acceptance reporting"
```

## Task 15: Full verification, documentation, and private push

**Files:**
- Modify: `README.md`
- Modify: `source/rl_training/rl_training/online_adaptation/__init__.py`

- [ ] **Step 1: Document exact workflows and safety limits**

Add README commands for:

1. Confirming the base PPO checkpoint.
2. Running source-distribution residual pretraining.
3. Running each of the three held-out modes.
4. Running the 10-seed comparison.
5. Locating logs, reports, before/after checkpoints, and videos.

State prominently that this release is simulation-only, does not authorize unprotected real-robot online exploration, and does not use height scans in the online policy.

- [ ] **Step 2: Run pure unit tests**

```bash
pytest tests/online_adaptation -v -m "not isaaclab"
```

Expected: all pure tests pass.

- [ ] **Step 3: Run formatting and static checks**

```bash
python -m compileall source/rl_training/rl_training/online_adaptation scripts/reinforcement_learning/online_adaptation
python -m isort --check-only source/rl_training/rl_training/online_adaptation scripts/reinforcement_learning/online_adaptation tests/online_adaptation
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Run simulator integration smoke tests**

```bash
python scripts/tools/list_envs.py | rg 'OnlineAdapt|HeldoutBoxes'
python scripts/reinforcement_learning/online_adaptation/train_residual_sac.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_rough/2026-07-24_14-42-38/model_20.pt \
  --num-envs=32 --max-steps=300 --headless \
  --log-dir logs/online_adaptation/final_smoke
python scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_rough/2026-07-24_14-42-38/model_20.pt \
  --residual-checkpoint logs/online_adaptation/final_smoke/latest.pt \
  --seeds 7 --max-steps 400 --headless \
  --output-dir logs/online_adaptation/final_comparison_smoke
```

Expected: both tasks register; training and all three comparison modes finish without reset or non-finite updates.

- [ ] **Step 5: Run the full held-out evaluation**

After residual pretraining is long enough for source-terrain validation to stabilize, resolve the qualified base checkpoint and run:

```bash
TASK_BASE_CHECKPOINT=$(find logs/rsl_rl/deeprobotics_lite3_rough -name 'model_*.pt' | sort -V | tail -1)
test -n "$TASK_BASE_CHECKPOINT"
python scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py \
  --base-checkpoint "$TASK_BASE_CHECKPOINT" \
  --residual-checkpoint logs/online_adaptation/lite3_source/latest.pt \
  --seeds 7 11 19 23 31 43 59 71 83 97 \
  --max-steps 1750 --video \
  --output-dir logs/online_adaptation/heldout_10_seed
```

Expected: reports include every seed. If the success thresholds are not met, keep the reports and iterate on source-only hyperparameters or safety bounds without introducing held-out boxes into pretraining.

- [ ] **Step 6: Final review and commit**

```bash
git status --short
git diff --check
git add README.md source/rl_training/rl_training/online_adaptation
git commit -m "docs: document Lite3 online terrain adaptation"
```

- [ ] **Step 7: Push only to the private origin and verify synchronization**

```bash
git remote -v
git push origin feat/lite3-online-adaptation
git fetch origin
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/lite3-online-adaptation)"
```

Expected: `origin` is `jenny-tian/rl_training_main`, upstream push URL is disabled, and local/remote hashes match.
