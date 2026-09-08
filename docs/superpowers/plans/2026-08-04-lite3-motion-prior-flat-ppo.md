# Lite3 Motion Prior Flat PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Improve `Flat-Deeprobotics-Lite3-v0` walking quality by using externally collected and Lite3-retargeted trajectories as a provenance-bearing motion prior while keeping the PPO actor proprioceptive and the critic privileged.

**Architecture:** A standalone trajectory package defines a strict `.npz` schema, validates metadata, and retargets external quadruped data into Lite3's canonical 12-DoF order. Isaac Lab reward functions lazily load one validated reference and compare joint, foot, contact, clearance, symmetry, and smoothness signals at the current episode phase. Flat PPO enables these terms only when an explicit reference path and `prior_on`/`prior_decay` mode are configured; `prior_off` remains the causal baseline.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Isaac Lab manager-based rewards, pytest.

---

### Task 1: Define the canonical trajectory schema and provenance contract

**Files:**
- Create: `source/rl_training/rl_training/motion_prior/__init__.py`
- Create: `source/rl_training/rl_training/motion_prior/schema.py`
- Test: `tests/online_adaptation/test_motion_prior_schema.py`

- [ ] **Step 1: Write failing schema tests**

```python
def test_round_trip_preserves_reference_arrays_and_metadata(tmp_path):
    trajectory = make_trajectory()
    path = tmp_path / "walk.npz"
    trajectory.save(path)
    loaded = MotionTrajectory.load(path)
    assert np.array_equal(loaded.joint_pos, trajectory.joint_pos)
    assert loaded.metadata["source_id"] == "synthetic-go1"

def test_schema_rejects_bad_shape_or_missing_provenance(tmp_path):
    with pytest.raises(ValueError, match="joint_pos"):
        MotionTrajectory(time=np.zeros(3), joint_pos=np.zeros((3, 11)),
                         foot_pos_b=np.zeros((3, 4, 3)), contact=np.zeros((3, 4)),
                         metadata={}).validate()
```

- [ ] **Step 2: Run `PYTHONPATH=source/rl_training:. pytest tests/online_adaptation/test_motion_prior_schema.py -q` and verify the new imports/functions fail.**

- [ ] **Step 3: Implement `MotionTrajectory`**

```python
@dataclass(frozen=True)
class MotionTrajectory:
    time: np.ndarray                 # [T]
    joint_pos: np.ndarray            # [T, 12], radians, Lite3 order
    foot_pos_b: np.ndarray           # [T, 4, 3], metres, body frame
    contact: np.ndarray              # [T, 4], bool or {0,1}
    command: np.ndarray | None = None  # [T, 3], vx, vy, wz
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "MotionTrajectory": ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> "MotionTrajectory": ...
```

Validation requires finite, strictly increasing time, equal frame counts, final dimensions `[12]`, `[4,3]`, `[4]`, optional command `[3]`, and metadata keys `source_id`, `source_format`, `retargeter`, `created_utc`, and `flat_only_split`. Store metadata as a JSON string in the NPZ and reject object arrays and missing required keys.

- [ ] **Step 4: Run the focused schema tests and commit the self-contained schema.**

### Task 2: Add external-trajectory normalization and Lite3 retargeting

**Files:**
- Create: `source/rl_training/rl_training/motion_prior/retarget.py`
- Create: `scripts/reinforcement_learning/motion_prior/retarget_motion.py`
- Test: `tests/online_adaptation/test_motion_prior_retarget.py`

- [ ] **Step 1: Write failing retarget tests**

```python
def test_source_joint_mapping_applies_sign_scale_and_offset():
    source = np.zeros((2, 12)); source[:, 1] = 0.5
    result = retarget_joint_trajectory(source, SOURCE_GO1_ORDER, Lite3RetargetConfig())
    assert np.allclose(result[:, 1], -0.65 - 0.5)

def test_foot_space_retarget_solves_lite3_leg_ik_and_preserves_contact():
    feet = standing_lite3_feet(5)
    result = retarget_motion(time=np.arange(5) / 50.0, foot_pos_b=feet,
                             contact=np.ones((5, 4)), metadata=source_metadata())
    assert result.joint_pos.shape == (5, 12)
    assert np.allclose(result.foot_pos_b, feet, atol=2e-4)
```

- [ ] **Step 2: Run the focused tests to confirm missing mapping/IK behavior fails.**

- [ ] **Step 3: Implement the minimal retargeter**

```python
SOURCE_GO1_ORDER = (...)
LITE3_ORDER = (...)

@dataclass(frozen=True)
class Lite3RetargetConfig:
    joint_names: tuple[str, ...] = LITE3_POLICY_JOINT_NAMES
    default_joint_pos: tuple[float, ...] = (0.0, -0.65, 1.3) * 4
    joint_sign: tuple[float, ...] = (1.0, 1.0, 1.0) * 4
    joint_offset: tuple[float, ...] = (0.0,) * 12
    joint_scale: tuple[float, ...] = (1.0,) * 12
    thigh_length: float = 0.20
    shank_length: float = 0.21

def retarget_joint_trajectory(...): ...
def foot_positions_from_lite3_joints(...): ...
def lite3_joints_from_foot_positions(...): ...
def retarget_motion(...): ...
```

Use explicit source-name mapping, sign/scale/offset normalization, finite/range checks, and the existing Lite3 leg dimensions. The analytic IK operates in each leg's sagittal plane after subtracting the hip anchor; preserve lateral `HipX` from the input or set it from the foot lateral coordinate. Reject unreachable feet instead of silently clipping. `retarget_motion` accepts either source joints or body-frame feet, derives contact from a supplied boolean array or foot speed/height thresholds, and emits `MotionTrajectory` metadata recording source IDs and mapping parameters.

- [ ] **Step 4: Implement the CLI**

```text
python scripts/reinforcement_learning/motion_prior/retarget_motion.py \
  --input source.npz --output lite3_reference.npz \
  --source-id go1-run-001 --source-format go1_telemetry \
  --flat-only-split train
```

The CLI must refuse absent provenance flags, print frame count/duration/contact ratio, and never write into the repository's formal baseline data directory by default.

- [ ] **Step 5: Run retarget tests and a CLI `--help` smoke test, then commit.**

### Task 3: Implement the reference manager and schedule

**Files:**
- Create: `source/rl_training/rl_training/motion_prior/manager.py`
- Test: `tests/online_adaptation/test_motion_prior_manager.py`

- [ ] **Step 1: Write failing manager tests**

```python
def test_manager_interpolates_and_loops_reference(tmp_path):
    ref = make_trajectory(frames=3, dt=0.1); ref.save(tmp_path / "ref.npz")
    manager = MotionPriorManager(tmp_path / "ref.npz", device="cpu")
    sample = manager.sample(torch.tensor([0.05, 0.35]))
    assert sample.joint_pos.shape == (2, 12)
    assert torch.allclose(sample.joint_pos[0], torch.full((12,), 0.5))

@pytest.mark.parametrize("mode,expected", [("on", 1.0), ("off", 0.0)])
def test_prior_schedule_modes(mode, expected):
    assert prior_weight(mode, 10, 100, 2.0) == pytest.approx(expected)
```

- [ ] **Step 2: Implement validated lazy loading, linear interpolation, looping by duration, and `prior_weight(mode, step, decay_steps, weight)` for `on`, `decay`, and `off`. Cache manager instances on the environment by absolute path and expose provenance metadata for logging.**

- [ ] **Step 3: Run focused manager tests and commit.**

### Task 4: Add Motion-Priors-style Isaac Lab reward terms

**Files:**
- Create: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/motion_prior_rewards.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/__init__.py`
- Test: `tests/online_adaptation/test_motion_prior_rewards.py`

- [ ] **Step 1: Write failing reward tests with a small fake env/articulation**

```python
def test_joint_and_foot_reference_rewards_are_maximal_at_reference():
    env = fake_env_at_reference()
    assert torch.allclose(motion_prior_joint_pos_exp(env, ...), torch.ones(1))
    assert torch.allclose(motion_prior_foot_pos_exp(env, ...), torch.ones(1))

def test_contact_and_clearance_rewards_distinguish_wrong_phase():
    env = fake_env_with_swing_clearance()
    assert motion_prior_contact_phase(env, ...)[0] > 0
    assert motion_prior_swing_clearance(env, ...)[0] > 0
```

- [ ] **Step 2: Implement rewards using the manager's body-frame samples:** exponential joint/foot errors; binary contact agreement; swing-foot clearance only for reference swing frames; joint mirror symmetry; and finite-difference joint smoothness. Every term multiplies the schedule scale and returns `[num_envs]`. Missing/invalid references produce a zero tensor with a one-time warning, preserving `prior_off` behavior.

- [ ] **Step 3: Run reward tests without launching Isaac Sim, then commit.**

### Task 5: Wire optional prior terms into flat Lite3 PPO

**Files:**
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_env_cfg.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py` (only if shared reward term parameters need Lite3 body names)
- Test: `tests/online_adaptation/test_flat_motion_prior_cfg.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_flat_prior_is_off_by_default_and_actor_observation_is_unchanged():
    cfg = DeeproboticsLite3FlatEnvCfg(); cfg.__post_init__()
    assert cfg.motion_prior_mode == "off"
    assert cfg.observations.policy.height_scan is None

def test_flat_prior_on_enables_only_reference_rewards(tmp_path):
    cfg = DeeproboticsLite3FlatEnvCfg(motion_prior_path=str(tmp_path / "ref.npz"), motion_prior_mode="on")
    assert cfg.rewards.motion_prior_joint_pos.weight > 0
```

- [ ] **Step 2: Add optional class fields `motion_prior_path: str | None`, `motion_prior_mode: str`, `motion_prior_weight`, and `motion_prior_decay_steps`; add zero-weight reward definitions and set body/joint names in Lite3 flat `__post_init__`. Keep path unset and all terms removed for the formal flat-only baseline.**

- [ ] **Step 3: Run config tests and import the task config to catch manager/configclass errors.**

- [ ] **Step 4: Document the three comparison runs (`prior_off`, `prior_on`, `prior_decay`) and provenance/flat-only split rules in `docs/lite3-motion-prior.md`, then commit.**

### Task 6: End-to-end verification and training handoff

**Files:**
- Test: `tests/online_adaptation/test_motion_prior_end_to_end.py`
- Modify: `docs/lite3-online-learning-principles.md` if needed to link the new workflow.

- [ ] **Step 1: Add an end-to-end synthetic test covering NPZ -> manager -> reward and a no-reference `prior_off` path.**

- [ ] **Step 2: Run focused tests:**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/pytest tests/online_adaptation/test_motion_prior_*.py -q
```

- [ ] **Step 3: Run the existing regression suite:**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/pytest tests/online_adaptation -q
```

Expected: existing tests remain green; no GPU PPO claim is made from unit tests. The actual training handoff is `train.py --task=Flat-Deeprobotics-Lite3-v0` with separately logged `prior_off`, `prior_on`, and `prior_decay` runs.

---

## Self-review

- External data is never bundled or silently used by the baseline; provenance is required at serialization and the prior is opt-in.
- The actor observation group remains proprioceptive; privileged critic height scan remains unchanged.
- Retargeting rejects unreachable/invalid motion rather than hiding errors through clipping.
- Reward terms cover foot position, contact/phase, swing clearance, symmetry, and smoothness, with explicit comparison modes.
- Unit tests cover schema, mapping/IK, interpolation/scheduling, reward behavior, configuration, and the end-to-end path.
