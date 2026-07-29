# Lite3 Flat-Only Online Step Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a reproducible Isaac Lab pipeline in which a strictly flat-trained Lite3 PPO acquires 4-6 cm step and +/-4 cm uneven-terrain capability through bounded, persistent online residual SAC learning.

**Architecture:** Keep the flat PPO frozen and enforce zero residual authority during initialization and reset warmup. Add explicit terrain/provenance protocols, termination-aware and n-step replay, transactional cross-process learner checkpoints, stratified flat/online Critic data, a flat Actor anchor, residual authority scheduling, and a parent curriculum process that launches one protected Isaac episode per worker process. Freeze the selected development configuration before a one-shot 20-seed combined-course evaluation.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab/Isaac Sim 5.1, RSL-RL, Gymnasium, trimesh, pytest, JSON/JSONL, TensorBoard.

**Design:** [`docs/superpowers/specs/2026-07-29-lite3-flat-only-online-step-learning-design.md`](../specs/2026-07-29-lite3-flat-only-online-step-learning-design.md)

---

## File Responsibility Map

New focused modules:

- `source/rl_training/rl_training/online_adaptation/provenance.py`: SHA-256, immutable JSON writes, baseline/replay/protocol manifests.
- `source/rl_training/rl_training/online_adaptation/terrain_protocol.py`: curriculum stages, development/sealed seeds, stage-to-geometry mapping, protocol hashing.
- `source/rl_training/rl_training/online_adaptation/nstep.py`: batched 1/3/5-step transition accumulation.
- `source/rl_training/rl_training/online_adaptation/checkpointing.py`: atomic full experiment checkpoint and RNG capture/restore.
- `source/rl_training/rl_training/online_adaptation/curriculum.py`: episode budgets, rolling advancement, residual gain schedule, stop reasons.
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/flat_online_track.py`: deterministic flat-prefix, low-step and uneven-track geometry.
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_online_env_cfg.py`: one-environment protected online curriculum task.
- `scripts/reinforcement_learning/online_adaptation/build_flat_baseline_manifest.py`: inspect and hash a qualified flat PPO run.
- `scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py`: collect immutable plane-only transitions from the frozen PPO.
- `scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py`: CPU parent process; launch one Isaac worker per protected episode and persist curriculum state.
- `scripts/reinforcement_learning/online_adaptation/evaluate_flat_online_curriculum.py`: run four-mode development/sealed comparisons and write strict acceptance reports.
- `scripts/reinforcement_learning/online_adaptation/select_flat_online_configuration.py`: select target/replay/anchor settings from development-only evidence.
- `scripts/reinforcement_learning/online_adaptation/freeze_flat_online_configuration.py`: hash and lock the formal configuration before sealed evaluation.

Existing modules remain responsible for their current boundaries:

- `types.py`, `replay_buffer.py`, `sac.py`: transition schema, storage and learning semantics.
- `networks.py`, `agent.py`, `actor_update.py`: residual policy, executed authority and transactional publication.
- `play_online_adaptation.py`: single Isaac worker episode; no curriculum policy decisions.
- `terrain_guards.py`: positive environment assertions.
- `acceptance.py`: pure report evaluation, with no environment or training access.

## Common Verification Environment

Use this prefix for every CPU test command:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest
```

Isaac-only tests remain marked `@pytest.mark.isaaclab`. Normal CPU runs must pass with only those tests skipped.

Every task-local `git rev-list --left-right --count` command below must print `0 0`; stop the next task if commit, push, fetch or alignment verification fails.

---

### Task 1: Enforce Positive Plane-Only Initialization

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/terrain_guards.py`
- Modify: `source/rl_training/rl_training/online_adaptation/__init__.py`
- Modify: `scripts/reinforcement_learning/rsl_rl/train.py:114-130`
- Modify: `scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py:94-101`
- Test: `tests/online_adaptation/test_terrain_guards.py`

- [ ] **Step 1: Write failing positive-assertion tests**

```python
def test_plane_guard_accepts_only_plane_without_generator():
    assert_plane_only_terrain(SimpleNamespace(terrain_type="plane", terrain_generator=None))


@pytest.mark.parametrize(
    ("terrain_type", "generator"),
    [("generator", None), ("plane", object()), ("generator", object())],
)
def test_plane_guard_rejects_every_non_plane_configuration(terrain_type, generator):
    cfg = SimpleNamespace(terrain_type=terrain_type, terrain_generator=generator)
    with pytest.raises(ValueError, match="strict plane-only terrain required"):
        assert_plane_only_terrain(cfg)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_terrain_guards.py -q
```

Expected: collection fails because `assert_plane_only_terrain` is not defined.

- [ ] **Step 3: Implement the positive guard and wire both flat entry points**

Add to `terrain_guards.py`:

```python
def assert_plane_only_terrain(terrain_cfg: Any) -> None:
    terrain_type = getattr(terrain_cfg, "terrain_type", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_type != "plane" or terrain_generator is not None:
        raise ValueError(
            "strict plane-only terrain required: "
            f"terrain_type={terrain_type!r}, terrain_generator={terrain_generator!r}"
        )
```

In `rsl_rl/train.py`, immediately after `env_cfg.seed` is set:

```python
if args_cli.task == "Flat-Deeprobotics-Lite3-v0":
    from rl_training.online_adaptation.terrain_guards import assert_plane_only_terrain

    assert_plane_only_terrain(env_cfg.scene.terrain)
```

In the single-seed qualification path, call the same guard before `gym.make` and reject every task except `Flat-Deeprobotics-Lite3-v0`.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_terrain_guards.py \
  tests/online_adaptation/test_qualification.py -q
```

Expected: all selected tests pass; Isaac-only configuration tests may skip.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/terrain_guards.py \
  source/rl_training/rl_training/online_adaptation/__init__.py \
  scripts/reinforcement_learning/rsl_rl/train.py \
  scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  tests/online_adaptation/test_terrain_guards.py \
  tests/online_adaptation/test_qualification.py
git commit -m "feat: enforce lite3 plane-only initialization"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 2: Create Reproducible Baseline Manifests

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/provenance.py`
- Create: `scripts/reinforcement_learning/online_adaptation/build_flat_baseline_manifest.py`
- Create: `tests/online_adaptation/test_provenance.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/cli.py`

- [ ] **Step 1: Write failing manifest tests**

```python
def test_manifest_hashes_checkpoint_config_and_command(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "params.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")

    manifest = build_flat_baseline_manifest(
        checkpoint=checkpoint,
        config_paths=[config],
        training_command=["train.py", "--task", "Flat-Deeprobotics-Lite3-v0"],
        git_sha="abc123",
        task="Flat-Deeprobotics-Lite3-v0",
        seed=42,
        num_envs=4096,
        max_iterations=10_000,
        versions={"python": "3.11", "isaaclab": "5.1"},
    )

    assert manifest["schema_version"] == 1
    assert manifest["terrain_assertion"] == "plane-only"
    assert manifest["files"][str(checkpoint)]["sha256"] == sha256_file(checkpoint)
    assert manifest["training"]["seed"] == 42


def test_atomic_json_write_never_leaves_temporary_file(tmp_path):
    output = tmp_path / "manifest.json"
    write_json_atomic(output, {"ok": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    assert list(tmp_path.glob("*.tmp")) == []
```

- [ ] **Step 2: Verify RED**

Run the new test file. Expected: import fails because `provenance.py` does not exist.

- [ ] **Step 3: Implement hashing, immutable manifest construction and atomic JSON writing**

The public API must be:

```python
def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def build_flat_baseline_manifest(
    *,
    checkpoint: str | Path,
    config_paths: list[str | Path],
    training_command: list[str],
    git_sha: str,
    task: str,
    seed: int,
    num_envs: int,
    max_iterations: int,
    versions: dict[str, str],
) -> dict[str, object]:
    if task != "Flat-Deeprobotics-Lite3-v0":
        raise ValueError(f"flat baseline manifest rejects task {task!r}")
    paths = [Path(checkpoint), *(Path(path) for path in config_paths)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest inputs missing: {missing}")
    files = {
        str(path.resolve()): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(paths, key=lambda item: str(item.resolve()))
    }
    return {
        "schema_version": 1,
        "terrain_assertion": "plane-only",
        "task": task,
        "git_sha": git_sha,
        "training": {
            "seed": int(seed),
            "num_envs": int(num_envs),
            "max_iterations": int(max_iterations),
            "command": list(training_command),
        },
        "versions": dict(sorted(versions.items())),
        "files": files,
    }

def write_json_atomic(path: str | Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
```

`build_flat_baseline_manifest` must reject any task other than `Flat-Deeprobotics-Lite3-v0`, reject missing files, sort file keys, and use `allow_nan=False` when writing.

- [ ] **Step 4: Implement the CLI**

The command must accept:

```text
--checkpoint PATH --config PATH (repeat --config for every config file)
--seed INT --num-envs INT --max-iterations INT
--training-command-file PATH --output PATH
```

It obtains the Git SHA with `git rev-parse HEAD`, reads package versions through `importlib.metadata.version`, builds the manifest, and refuses to overwrite an existing output whose content differs.

- [ ] **Step 5: Run tests and CLI help**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_provenance.py -q
PYTHONPATH=source/rl_training:. \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/online_adaptation/build_flat_baseline_manifest.py --help
```

Expected: tests pass and help exits 0.

- [ ] **Step 6: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/provenance.py \
  scripts/reinforcement_learning/online_adaptation/build_flat_baseline_manifest.py \
  scripts/reinforcement_learning/online_adaptation/cli.py \
  tests/online_adaptation/test_provenance.py
git commit -m "feat: record lite3 flat baseline provenance"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 3: Guarantee Exact Zero Residual Authority

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/networks.py`
- Modify: `source/rl_training/rl_training/online_adaptation/agent.py`
- Modify: `tests/online_adaptation/test_networks.py`
- Modify: `tests/online_adaptation/test_publisher_agent.py`

- [ ] **Step 1: Write failing zero-authority tests**

```python
def test_actor_deterministic_mean_is_exactly_zero_at_initialization():
    actor = ResidualActor(49, 12, 64, LIMITS)
    history = torch.randn(4, 25, 49)
    frame = torch.randn(4, 49)
    base = torch.randn(4, 12)
    _, _, mean = actor.sample(history, frame, base, deterministic=True)
    assert torch.equal(mean, torch.zeros_like(mean))


def test_zero_authority_suppresses_stochastic_samples_exactly():
    agent = make_agent(residual_gain=0.0)
    decision = agent.begin_step(
        torch.randn(1, 2),
        torch.zeros(1, 1),
        safety_state(),
        deterministic=False,
    )
    assert torch.equal(decision.residual, torch.zeros_like(decision.residual))
```

- [ ] **Step 2: Verify RED**

Expected: deterministic mean is only near zero and `AgentConfig` has no residual gain.

- [ ] **Step 3: Implement exact initialization and executed gain**

In `ResidualActor.__init__`:

```python
nn.init.zeros_(self.mean_layer.weight)
nn.init.zeros_(self.mean_layer.bias)
```

Add to `AgentConfig`:

```python
residual_gain: float = 1.0
```

Validate `0.0 <= residual_gain <= 1.0`, store a lock-protected `_residual_gain`, and expose:

```python
@property
def residual_gain(self) -> float:
    with self._residual_gain_lock:
        return self._residual_gain

def set_residual_gain(self, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("residual gain must be in [0, 1]")
    with self._residual_gain_lock:
        self._residual_gain = float(value)
```

Multiply the sampled residual by the snapshotted gain before `SafetyFilter.project`.

- [ ] **Step 4: Verify exact-zero behavior and existing safety behavior**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_networks.py \
  tests/online_adaptation/test_publisher_agent.py \
  tests/online_adaptation/test_safety.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/networks.py \
  source/rl_training/rl_training/online_adaptation/agent.py \
  tests/online_adaptation/test_networks.py \
  tests/online_adaptation/test_publisher_agent.py
git commit -m "feat: enforce zero lite3 residual authority"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 4: Define the Curriculum and Sealed Seed Protocol

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/terrain_protocol.py`
- Create: `tests/online_adaptation/test_terrain_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

```python
def test_protocol_has_five_ordered_stages_and_disjoint_seeds():
    protocol = default_flat_online_protocol()
    assert [stage.name for stage in protocol.stages] == [
        "step_2cm", "step_4cm", "step_6cm", "rough_2cm", "rough_4cm"
    ]
    assert protocol.development_seeds == (101, 103, 107, 109, 113)
    assert len(protocol.sealed_seeds) == 20
    assert set(protocol.development_seeds).isdisjoint(protocol.sealed_seeds)


def test_sealed_geometry_combines_step_and_rough_segments():
    geometry = sealed_geometry(seed=1009)
    assert geometry.step_height in (0.04, 0.05, 0.06)
    assert geometry.rough_amplitude == 0.04
    assert geometry.seed == 1009
```

- [ ] **Step 2: Verify RED**

Expected: module import fails.

- [ ] **Step 3: Implement immutable protocol dataclasses**

Use:

```python
@dataclass(frozen=True)
class TerrainGeometry:
    step_height: float
    rough_amplitude: float
    seed: int


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    step_height: float
    rough_amplitude: float
    max_episodes: int = 20


@dataclass(frozen=True)
class FlatOnlineProtocol:
    stages: tuple[CurriculumStage, ...]
    development_seeds: tuple[int, ...]
    sealed_seeds: tuple[int, ...]
    max_total_episodes: int = 100
    max_total_steps: int = 200_000
    max_steps_per_episode: int = 2_000
```

Use sealed seeds `(1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123)`. Derive sealed step height with `(0.04, 0.05, 0.06)[seed % 3]`.

- [ ] **Step 4: Add canonical JSON and SHA-256 methods**

`FlatOnlineProtocol.canonical_json()` uses sorted keys, compact separators and `allow_nan=False`; `sha256()` hashes its UTF-8 bytes. Tests assert two default instances have the same hash.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_terrain_protocol.py -q
git add source/rl_training/rl_training/online_adaptation/terrain_protocol.py \
  tests/online_adaptation/test_terrain_protocol.py
git commit -m "feat: define lite3 terrain evaluation protocol"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 5: Build Deterministic Low-Step and Uneven Tracks

**Files:**
- Create: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/flat_online_track.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/__init__.py`
- Create: `tests/online_adaptation/test_flat_online_track.py`

- [ ] **Step 1: Write geometry tests first**

```python
def test_combined_track_contains_flat_prefix_exact_step_and_seeded_roughness():
    cfg = FlatOnlineTrackCfg(
        proportion=1.0,
        size=(20.0, 4.0),
        flat_length=3.0,
        step_height=0.06,
        step_length=2.0,
        rough_amplitude=0.04,
        rough_cell_width=0.4,
        lane_width=2.0,
        seed=1009,
    )
    meshes, origin = flat_online_track(1.0, cfg)
    vertices = np.concatenate([mesh.vertices for mesh in meshes])
    assert origin[0] < cfg.flat_length
    assert vertices[vertices[:, 0] < 2.5, 2].max() <= 1e-6
    step = vertices[(vertices[:, 0] > 3.2) & (vertices[:, 0] < 4.8), 2]
    assert step.max() == pytest.approx(0.06)
    rough = vertices[vertices[:, 0] > 6.0, 2]
    rough_top = rough[rough > -0.5]
    assert rough_top.max() <= 0.04 + 1e-6
    assert rough_top.min() >= -0.04 - 1e-6
```

- [ ] **Step 2: Verify RED**

Expected: terrain module does not exist.

- [ ] **Step 3: Implement geometry with explicit segments**

Validate all dimensions and amplitudes. Build a zero-height base, one full-width lane step box from `flat_length` for `step_length`, and seeded rough cells after the step. Do not use Isaac global RNG; use `np.random.default_rng(cfg.seed)`.

`FlatOnlineTrackCfg` must expose `flat_length`, `lane_width`, `step_height`, `step_length`, `rough_amplitude`, `rough_cell_width`, and `seed`.

- [ ] **Step 4: Test determinism and seed variation**

Add assertions that identical seeds produce equal vertex arrays and different seeds change only the rough segment.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_flat_online_track.py -q
git add source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains \
  tests/online_adaptation/test_flat_online_track.py
git commit -m "feat: add lite3 low-step curriculum track"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 6: Register a Runtime-Configurable Protected Curriculum Environment

**Files:**
- Create: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_online_env_cfg.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/__init__.py`
- Modify: `source/rl_training/rl_training/online_adaptation/terrain_protocol.py`
- Modify: `tests/online_adaptation/test_terrain_guards.py`
- Create: `tests/online_adaptation/test_flat_online_env_cfg.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_apply_geometry_updates_both_generator_and_subterrain_seed():
    subterrain = SimpleNamespace(step_height=0.0, rough_amplitude=0.0, seed=0)
    generator = SimpleNamespace(seed=0, sub_terrains={"flat_online": subterrain})
    terrain = SimpleNamespace(terrain_generator=generator)
    env_cfg = SimpleNamespace(scene=SimpleNamespace(terrain=terrain))

    apply_geometry_to_env_cfg(env_cfg, TerrainGeometry(0.06, 0.04, 1009))

    assert generator.seed == 1009
    assert subterrain.seed == 1009
    assert subterrain.step_height == 0.06
    assert subterrain.rough_amplitude == 0.04
```

- [ ] **Step 2: Verify RED**

Expected: `apply_geometry_to_env_cfg` is undefined.

- [ ] **Step 3: Implement the config mutator and Isaac environment config**

Register `FlatOnline-Deeprobotics-Lite3-v0`. The environment derives from `DeeproboticsLite3RoughEnvCfg` but replaces the terrain generator with one `FlatOnlineTrackCfg`, one row/column, no curriculum, no external pushes, deterministic reset, one environment and fixed forward command. Do not hardcode geometry seed in `__post_init__`; the worker must call `apply_geometry_to_env_cfg` before `gym.make`.

- [ ] **Step 4: Verify the Isaac-only config**

The test must assert one environment, no terrain curriculum, deterministic reset parameters, and the registered RSL-RL entry point is the flat PPO config rather than the rough PPO config.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_flat_online_env_cfg.py \
  tests/online_adaptation/test_terrain_guards.py -q
git add source/rl_training/rl_training/online_adaptation/terrain_protocol.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3 \
  tests/online_adaptation/test_flat_online_env_cfg.py \
  tests/online_adaptation/test_terrain_guards.py
git commit -m "feat: register lite3 flat-online terrain task"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 7: Replace Ambiguous `done` with Explicit Transition Semantics

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/types.py`
- Modify: `source/rl_training/rl_training/online_adaptation/replay_buffer.py`
- Modify: `source/rl_training/rl_training/online_adaptation/agent.py`
- Modify: `tests/online_adaptation/test_replay_buffer.py`
- Modify: `tests/online_adaptation/test_package.py`
- Modify: `tests/online_adaptation/test_actor_update.py`
- Modify: `tests/online_adaptation/test_policy_drift.py`
- Modify: `tests/online_adaptation/test_sac.py`
- Modify: `tests/online_adaptation/test_publisher_agent.py`

- [ ] **Step 1: Change test factories first**

Replace each `done=` fixture field with:

```python
terminated=torch.zeros(batch, 1),
truncated=torch.zeros(batch, 1),
bootstrap_mask=torch.ones(batch, 1),
discount=torch.full((batch, 1), 0.99),
safety_mask=torch.ones(batch, 1),
curriculum_stage_id=torch.zeros(batch, 1, dtype=torch.int64),
episode_id=torch.zeros(batch, 1, dtype=torch.int64),
```

Add a replay test asserting every sampled field survives storage exactly.

- [ ] **Step 2: Verify RED**

Expected: `TransitionBatch` rejects the new fields.

- [ ] **Step 3: Update the transition schema and ReplayBuffer storages**

`TransitionBatch` must use the four explicit ending fields and remove `done`. It also stores audit-only `safety_mask`, `curriculum_stage_id`, and `episode_id` tensors; none of these audit fields may enter Actor or Critic observations. `ReplayBuffer.add_batch` and `sample` must store and return every field. Add input validation requiring binary masks, non-negative integer IDs, and `0 <= discount <= 1`.

- [ ] **Step 4: Update `OnlineAdaptationAgent.finish_step`**

Use this signature:

```python
def finish_step(
    self,
    next_frame: torch.Tensor,
    next_base_action: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    discount: torch.Tensor,
    safety_mask: torch.Tensor,
    curriculum_stage_id: torch.Tensor,
    episode_id: torch.Tensor,
    adaptation_enabled: bool = False,
) -> None:
```

Reset history on `terminated | truncated`. Safety-invalid rows continue to be removed through `valid_mask`.

- [ ] **Step 5: Run all affected tests**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_replay_buffer.py \
  tests/online_adaptation/test_package.py \
  tests/online_adaptation/test_actor_update.py \
  tests/online_adaptation/test_policy_drift.py \
  tests/online_adaptation/test_sac.py \
  tests/online_adaptation/test_publisher_agent.py -q
test -z "$(rg -l '\.done\b|done=' source/rl_training/rl_training/online_adaptation)"
```

Expected: all selected tests pass and the source scan is empty.

- [ ] **Step 6: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation \
  tests/online_adaptation
git commit -m "refactor: make lite3 transition endings explicit"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 8: Serialize Replay Buffers and Collect Immutable Flat Data

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/replay_buffer.py`
- Create: `scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/cli.py`
- Modify: `tests/online_adaptation/test_replay_buffer.py`
- Create: `tests/online_adaptation/test_collect_flat_replay.py`

- [ ] **Step 1: Write failing replay round-trip and freeze tests**

```python
def test_replay_state_round_trip_preserves_cursor_size_and_rows():
    replay = make_replay(capacity=3)
    replay.add_batch(**make_transition(1.0))
    replay.add_batch(**make_transition(2.0))
    restored = make_replay(capacity=3)
    restored.load_state_dict(replay.state_dict())
    assert restored.state_dict().keys() == replay.state_dict().keys()
    assert_restored_storage_equal(restored, replay)


def test_frozen_replay_rejects_add_and_clear():
    replay = make_replay(capacity=3)
    replay.freeze()
    with pytest.raises(RuntimeError, match="replay buffer is frozen"):
        replay.add_batch(**make_transition(1.0))
    with pytest.raises(RuntimeError, match="replay buffer is frozen"):
        replay.clear()
```

- [ ] **Step 2: Verify RED**

Expected: methods are missing.

- [ ] **Step 3: Implement locked `state_dict`, `load_state_dict`, `freeze` and `frozen`**

The state must include schema version, dimensions, dtype name, size, cursor, frozen flag, and clones of every storage tensor. Loading validates every dimension and capacity before mutating the buffer.

- [ ] **Step 4: Implement the flat replay collector**

The script only accepts `Flat-Deeprobotics-Lite3-v0`, calls `assert_plane_only_terrain`, runs the frozen PPO with deterministic actions, writes 1-step explicit transitions, freezes the replay, and saves:

```text
flat_replay.pt
flat_replay_manifest.json
metrics.jsonl
```

The manifest uses the exact keys `baseline_manifest_sha256`, `protocol_sha256`, `task`, `seed`, `transition_count`, `replay_sha256`, `falls`, `safety_invalid_rows`, and `frozen`. A formal manifest requires `transition_count >= 100000`, `falls == 0`, `safety_invalid_rows == 0`, and `frozen is True`.

Its parser requires `--base-checkpoint`, `--base-manifest`, `--output-dir`, and accepts `--task Flat-Deeprobotics-Lite3-v0`, `--seed`, `--transitions`, `--device`, and `--headless`. It rejects `--transitions` below 100,000 for a formal collection.

- [ ] **Step 5: Test argument validation without launching Isaac**

Use a parser test to prove rough/held-out tasks are rejected before `AppLauncher` construction.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_replay_buffer.py \
  tests/online_adaptation/test_collect_flat_replay.py -q
git add source/rl_training/rl_training/online_adaptation/replay_buffer.py \
  scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py \
  scripts/reinforcement_learning/online_adaptation/cli.py \
  tests/online_adaptation/test_replay_buffer.py \
  tests/online_adaptation/test_collect_flat_replay.py
git commit -m "feat: persist immutable lite3 flat replay"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 9: Add Batched N-Step Accumulation

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/nstep.py`
- Create: `tests/online_adaptation/test_nstep.py`
- Modify: `source/rl_training/rl_training/online_adaptation/agent.py`

- [ ] **Step 1: Write exact 1/3/5-step tests**

```python
@pytest.mark.parametrize("n_step", [1, 3, 5])
def test_nstep_accumulates_reward_discount_and_next_state(n_step):
    accumulator = NStepAccumulator(num_envs=1, n_step=n_step, gamma=0.9)
    emitted = []
    for index in range(n_step):
        emitted.extend(accumulator.append(make_raw_transition(reward=1.0, marker=index)))
    assert len(emitted) == 1
    transition = emitted[0]
    assert transition.reward.item() == pytest.approx(sum(0.9**i for i in range(n_step)))
    assert transition.discount.item() == pytest.approx(0.9**n_step)
    assert transition.next_frame.item() == n_step - 1


def test_true_termination_disables_bootstrap_but_truncation_keeps_it():
    terminated = run_terminal_case(terminated=True, truncated=False)
    truncated = run_terminal_case(terminated=False, truncated=True)
    assert terminated.bootstrap_mask.item() == 0.0
    assert truncated.bootstrap_mask.item() == 1.0
```

- [ ] **Step 2: Verify RED**

Expected: module import fails.

- [ ] **Step 3: Implement `RawTransition` and `NStepAccumulator`**

`RawTransition` has the same tensor fields as `TransitionBatch` except `discount`; it carries one-step reward plus `terminated`, `truncated`, `bootstrap_mask`, `safety_mask`, `curriculum_stage_id`, and `episode_id`. Use one deque per environment. `append` emits the oldest complete transition when the queue reaches `n_step`, and flushes every prefix on termination/truncation. Aggregation computes `sum(gamma**i * reward_i)`, stores `gamma**actual_length`, uses the last next-state fields, and propagates the minimum bootstrap/safety mask. `state_dict` and `load_state_dict` must preserve pending queues for cross-process checkpoints.

- [ ] **Step 4: Integrate the accumulator before ReplayBuffer insertion**

`AgentConfig` gains `n_step: int = 1` and `gamma: float = 0.99`. `finish_step` feeds raw transitions to the accumulator and inserts each emitted aggregate into global and adaptation replay according to its original valid mask.

- [ ] **Step 5: Run tests for n-step, replay, agent and history reset**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_nstep.py \
  tests/online_adaptation/test_replay_buffer.py \
  tests/online_adaptation/test_publisher_agent.py \
  tests/online_adaptation/test_history.py -q
```

Expected: all pass for 1, 3 and 5 steps.

- [ ] **Step 6: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/nstep.py \
  source/rl_training/rl_training/online_adaptation/agent.py \
  tests/online_adaptation/test_nstep.py \
  tests/online_adaptation/test_publisher_agent.py
git commit -m "feat: add lite3 n-step transition accumulation"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 10: Make SAC Targets Timeout-Aware and Observable

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/sac.py`
- Modify: `source/rl_training/rl_training/online_adaptation/types.py`
- Modify: `tests/online_adaptation/test_sac.py`
- Modify: `tests/online_adaptation/test_play_runtime.py`

- [ ] **Step 1: Write a deterministic target test**

```python
def test_target_uses_saved_discount_and_bootstrap_mask():
    batch = make_batch()
    batch = replace(
        batch,
        reward=torch.tensor([[2.0], [2.0]]),
        discount=torch.tensor([[0.9], [0.9**3]]),
        bootstrap_mask=torch.tensor([[0.0], [1.0]]),
    )
    target = compute_sac_target(
        reward=batch.reward,
        discount=batch.discount,
        bootstrap_mask=batch.bootstrap_mask,
        next_q=torch.tensor([[10.0], [10.0]]),
        alpha_log_prob=torch.zeros(2, 1),
    )
    torch.testing.assert_close(target, torch.tensor([[2.0], [2.0 + 0.9**3 * 10.0]]))
```

- [ ] **Step 2: Verify RED**

Expected: `compute_sac_target` is undefined.

- [ ] **Step 3: Implement and use the pure target function**

```python
def compute_sac_target(
    *,
    reward: torch.Tensor,
    discount: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    next_q: torch.Tensor,
    alpha_log_prob: torch.Tensor,
) -> torch.Tensor:
    return reward + bootstrap_mask * discount * (next_q - alpha_log_prob)
```

Replace `(1 - done) * gamma` in `SACUpdater.update` with saved discount and bootstrap mask.

- [ ] **Step 4: Extend `UpdateStats`**

Add these exact dataclass fields after the existing defaults and update every constructor, test factory and event record:

```python
mean_target_q: float | None = None
mean_td_error: float | None = None
mean_flat_q: float | None = None
mean_online_q: float | None = None
flat_anchor_loss: float | None = None
flat_batch_fraction: float = 0.0
n_step: int = 1
```

Rejected stats use finite defaults where required by JSON and `None` for unavailable optional metrics.

- [ ] **Step 5: Run SAC, actor update and runtime logging tests**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_sac.py \
  tests/online_adaptation/test_actor_update.py \
  tests/online_adaptation/test_play_runtime.py -q
```

Expected: targets differ correctly for termination and truncation; JSON event rows remain finite.

- [ ] **Step 6: Commit**

```bash
git add source/rl_training/rl_training/online_adaptation/sac.py \
  source/rl_training/rl_training/online_adaptation/types.py \
  tests/online_adaptation/test_sac.py \
  tests/online_adaptation/test_play_runtime.py
git commit -m "feat: make lite3 sac targets timeout-aware"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 11: Add Transactional Full Experiment Checkpoints

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/checkpointing.py`
- Modify: `source/rl_training/rl_training/online_adaptation/sac.py`
- Modify: `source/rl_training/rl_training/online_adaptation/agent.py`
- Modify: `source/rl_training/rl_training/online_adaptation/novelty.py`
- Create: `tests/online_adaptation/test_checkpointing.py`
- Modify: `tests/online_adaptation/test_sac.py`
- Modify: `tests/online_adaptation/test_novelty.py`

- [ ] **Step 1: Write failing atomic checkpoint and RNG tests**

```python
def test_experiment_checkpoint_round_trip_restores_learning_and_rng(tmp_path):
    state = make_experiment_state()
    path = tmp_path / "experiment.pt"
    save_experiment_checkpoint(path, state)
    perturb_all_components(state)
    restored = load_experiment_checkpoint(path, expected_protocol_sha256="protocol-hash")
    assert_experiment_state_equal(restored, make_experiment_state())
    assert (tmp_path / "experiment.pt.sha256").exists()


def test_corrupt_checkpoint_preserves_previous_good_snapshot(tmp_path):
    path = tmp_path / "experiment.pt"
    save_experiment_checkpoint(path, make_experiment_state(version=1))
    simulate_failed_atomic_save(path, make_experiment_state(version=2))
    restored = load_experiment_checkpoint(path, expected_protocol_sha256="protocol-hash")
    assert restored["updater"]["version"] == 1
```

- [ ] **Step 2: Verify RED**

Expected: checkpoint module missing.

- [ ] **Step 3: Add state APIs to mutable components**

Implement `state_dict/load_state_dict` for `StableStateManager`, `NoveltyGate`, `NStepAccumulator`, ReplayBuffer, and OnlineAdaptationAgent. Agent load must re-publish the loaded updater Actor/version and reset control-only history/safety state for `BASE_WARMUP`.

- [ ] **Step 4: Implement RNG capture and restore**

Capture Python `random`, NumPy, CPU Torch and all available CUDA RNG states. Restore CUDA states only when device count matches; otherwise raise a descriptive error instead of silently changing reproducibility.

- [ ] **Step 5: Implement atomic save with checksum**

Write to a sibling temporary file, `flush` and `os.fsync`, calculate SHA-256, atomically `os.replace` the checkpoint, then atomically replace the checksum sidecar. Load verifies checksum, schema version and protocol hash before constructing runtime objects.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_checkpointing.py \
  tests/online_adaptation/test_sac.py \
  tests/online_adaptation/test_novelty.py -q
git add source/rl_training/rl_training/online_adaptation/checkpointing.py \
  source/rl_training/rl_training/online_adaptation/agent.py \
  source/rl_training/rl_training/online_adaptation/sac.py \
  source/rl_training/rl_training/online_adaptation/novelty.py \
  tests/online_adaptation/test_checkpointing.py \
  tests/online_adaptation/test_sac.py \
  tests/online_adaptation/test_novelty.py
git commit -m "feat: persist complete lite3 online learner state"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 12: Mix Flat and Online Critic Replay Deterministically

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/replay_buffer.py`
- Modify: `source/rl_training/rl_training/online_adaptation/agent.py`
- Modify: `tests/online_adaptation/test_replay_buffer.py`
- Modify: `tests/online_adaptation/test_publisher_agent.py`

- [ ] **Step 1: Write failing stratified sample tests**

```python
def test_stratified_batch_uses_requested_flat_fraction():
    flat = filled_replay(value=1.0, count=20, frozen=True)
    online = filled_replay(value=2.0, count=20)
    sampled = sample_stratified_batch(
        flat_replay=flat,
        online_replay=online,
        batch_size=8,
        flat_fraction=0.75,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )
    assert sampled.source_is_flat.sum().item() == 6
    assert sampled.flat_fraction == pytest.approx(0.75)
    assert torch.equal(sampled.transitions.reward[:6], torch.ones(6, 1))
    assert torch.equal(sampled.transitions.reward[6:], torch.full((2, 1), 2.0))
```

- [ ] **Step 2: Verify RED**

Expected: no stratified result type or sampler.

- [ ] **Step 3: Add batch concatenation and a separate audit result**

Do not add replay ownership to `TransitionBatch`; the same schema must remain serializable by either replay. Define the sampler result in `replay_buffer.py`:

```python
@dataclass(frozen=True)
class StratifiedBatch:
    transitions: TransitionBatch
    source_is_flat: torch.Tensor
    flat_fraction: float
```

Implement `concatenate_transition_batches` with strict field/shape checks. `sample_stratified_batch` concatenates flat rows first, online rows second, and creates the matching boolean `source_is_flat` tensor solely for logging and tests.

- [ ] **Step 4: Implement the 75/25 to 50/50 schedule**

`AgentConfig` gains `flat_fraction_initial=0.75`, `flat_fraction_ready=0.50`, and `online_ready_size=1024`. `sample_critic_batch` uses 75/25 below the threshold and 50/50 at or above it. The actual sampled fraction is returned in `UpdateStats`.

- [ ] **Step 5: Prewarm only the Critics from strict flat replay**

Add `flat_critic_warmup_updates: int = 1000` and `prewarm_critics_from_flat()`. On a new experiment, before the first target-terrain action, run exactly that many Critic/target-Critic updates from frozen flat replay while Actor, Actor optimizer, entropy parameter and residual output remain bitwise unchanged. Persist `flat_critic_warmup_complete` and the update count in the experiment checkpoint so reset/resume cannot repeat prewarming. A test snapshots every Actor tensor before prewarm, proves at least one Critic tensor changes, and proves the Actor snapshot remains exact.

- [ ] **Step 6: Preserve legacy behavior when no flat replay is loaded**

If `flat_replay is None`, sample 100% online and record fraction 0.0. Existing infrastructure regressions must continue to run.

- [ ] **Step 7: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_replay_buffer.py \
  tests/online_adaptation/test_publisher_agent.py -q
git add source/rl_training/rl_training/online_adaptation/replay_buffer.py \
  source/rl_training/rl_training/online_adaptation/agent.py \
  tests/online_adaptation/test_replay_buffer.py \
  tests/online_adaptation/test_publisher_agent.py
git commit -m "feat: mix lite3 flat and online replay"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 13: Anchor the Actor on Flat Data

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/sac.py`
- Modify: `source/rl_training/rl_training/online_adaptation/actor_update.py`
- Modify: `tests/online_adaptation/test_sac.py`
- Modify: `tests/online_adaptation/test_actor_update.py`

- [ ] **Step 1: Write failing anchor tests**

```python
def test_flat_anchor_penalizes_deterministic_residual_mean():
    updater = make_updater(SACConfig(flat_anchor_weight=2.0))
    force_actor_mean(updater.actor, value=0.1)
    stats = updater.update(
        critic_batch=make_batch(),
        actor_batch=make_batch(),
        flat_anchor_batch=make_batch(),
    )
    assert stats.accepted
    assert stats.flat_anchor_loss == pytest.approx(0.02, rel=0.2)


def test_zero_anchor_weight_matches_legacy_actor_loss():
    first = make_seeded_updater(SACConfig(flat_anchor_weight=0.0))
    second = clone_updater(first)
    legacy = first.update(make_batch(), make_batch())
    anchored = second.update(make_batch(), make_batch(), flat_anchor_batch=make_batch())
    assert anchored.actor_loss == pytest.approx(legacy.actor_loss)
```

- [ ] **Step 2: Verify RED**

Expected: config and update argument are missing.

- [ ] **Step 3: Implement the anchor**

Add `flat_anchor_weight: float = 0.0` to `SACConfig`. For the flat batch, call deterministic Actor sampling and add:

```python
flat_anchor_loss = flat_mean.square().mean()
actor_loss = actor_loss + self.config.flat_anchor_weight * flat_anchor_loss
```

Reject non-finite flat batches transactionally. Critic targets remain based on the stratified Critic batch.

- [ ] **Step 4: Pass an independent flat batch through `update_and_publish_actor`**

The signature gains `flat_anchor_batch: TransitionBatch | None`; tests prove publication failure still restores optimizer and Actor state including the anchor step.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_sac.py \
  tests/online_adaptation/test_actor_update.py -q
git add source/rl_training/rl_training/online_adaptation/sac.py \
  source/rl_training/rl_training/online_adaptation/actor_update.py \
  tests/online_adaptation/test_sac.py \
  tests/online_adaptation/test_actor_update.py
git commit -m "feat: anchor lite3 residual actor on flat data"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 14: Implement Residual Gain and Curriculum State Machines

**Files:**
- Create: `source/rl_training/rl_training/online_adaptation/curriculum.py`
- Create: `tests/online_adaptation/test_curriculum.py`
- Modify: `source/rl_training/rl_training/online_adaptation/policy_drift.py`
- Modify: `tests/online_adaptation/test_policy_drift.py`

- [ ] **Step 1: Write failing gain schedule tests**

```python
def test_gain_starts_zero_opens_to_quarter_and_advances_after_five_safe_episodes():
    scheduler = ResidualGainScheduler()
    assert scheduler.gain == 0.0
    scheduler.open_for_adaptation()
    assert scheduler.gain == 0.25
    for _ in range(5):
        scheduler.record_episode(safe=True, rolled_back=False, performance_degraded=False)
    assert scheduler.gain == 0.5


def test_any_rollback_reduces_one_gain_tier():
    scheduler = ResidualGainScheduler(initial_gain=0.75)
    scheduler.record_episode(safe=False, rolled_back=True, performance_degraded=True)
    assert scheduler.gain == 0.5
```

- [ ] **Step 2: Write failing curriculum budget tests**

```python
def test_coordinator_advances_after_ten_episode_eighty_percent_window():
    coordinator = CurriculumCoordinator(default_flat_online_protocol())
    for passed in (True, True, True, True, True, True, True, True, False, False):
        coordinator.record_episode(EpisodeOutcome(passed=passed, fell=False, steps=1000, cost=0.2))
    assert coordinator.stage.name == "step_4cm"


def test_coordinator_stops_at_either_budget():
    by_episode = CurriculumCoordinator(default_flat_online_protocol())
    by_episode.load_counts(total_episodes=100, total_steps=10_000)
    assert by_episode.stop_reason == "episode_budget_exhausted"
    by_steps = CurriculumCoordinator(default_flat_online_protocol())
    by_steps.load_counts(total_episodes=10, total_steps=200_000)
    assert by_steps.stop_reason == "step_budget_exhausted"
```

- [ ] **Step 3: Verify RED**

Expected: curriculum module missing.

- [ ] **Step 4: Implement exact state machines**

`ResidualGainScheduler` uses tiers `(0.0, 0.25, 0.5, 0.75, 1.0)`. `CurriculumCoordinator` keeps per-stage and total counters, a 10-episode outcome deque, best finite cost, no-improvement count, round-robin development seed index, and stop reason. Any fall blocks advancement. Ten outcomes with at least eight passes, zero falls and `flat_retention_passed=True` advance the stage. Ten outcomes without a new best cost stop with `no_improvement`.

- [ ] **Step 5: Add complete state round trips**

Tests must restore stage, seed index, rolling windows, budgets and gain exactly from `state_dict`.

- [ ] **Step 6: Calibrate the candidate drift threshold only on development data**

Add `calibrate_drift_threshold(samples, quantile=0.99, multiplier=1.25)`. It rejects empty/non-finite/negative samples and returns `torch.quantile(samples, 0.99) * 1.25`. The supervisor collects accepted-candidate drift only during zero-gain and first-tier development episodes, records the full sample count, quantile and multiplier, and freezes the resulting threshold in the selected development configuration. Sealed evaluation may load but never recompute it.

- [ ] **Step 7: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_curriculum.py \
  tests/online_adaptation/test_policy_drift.py -q
git add source/rl_training/rl_training/online_adaptation/curriculum.py \
  source/rl_training/rl_training/online_adaptation/policy_drift.py \
  tests/online_adaptation/test_curriculum.py \
  tests/online_adaptation/test_policy_drift.py
git commit -m "feat: coordinate lite3 protected terrain curriculum"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 15: Extend the Single-Episode Worker

**Files:**
- Modify: `scripts/reinforcement_learning/online_adaptation/cli.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py`
- Modify: `tests/online_adaptation/test_cli.py`
- Modify: `tests/online_adaptation/test_play_runtime.py`
- Modify: `tests/online_adaptation/test_play_modes.py`

- [ ] **Step 1: Add parser tests for worker-only arguments**

The parser must accept:

```text
--terrain-stage step_2cm
--terrain-seed 101
--protocol-sha256 HASH
--experiment-checkpoint-in PATH
--experiment-checkpoint-out PATH
--flat-replay PATH
--flat-anchor-weight FLOAT
--n-step {1,3,5}
--residual-gain FLOAT
--evaluation-only
```

Tests reject invalid stage names, sealed seeds in training mode, and online mode without checkpoint output.

- [ ] **Step 2: Verify RED**

Expected: parser rejects the new arguments.

- [ ] **Step 3: Configure geometry before `gym.make`**

After `parse_env_cfg`, resolve development or sealed geometry from the protocol and call `apply_geometry_to_env_cfg`. Record stage, seed, protocol hash and generated geometry in `config.json`.

- [ ] **Step 4: Load and save complete experiment state**

Load the immutable flat replay and optional experiment checkpoint before the updater thread starts. Force gain 0 during control warmup, then apply the coordinator-provided gain. On normal completion, environment termination, safety rollback or exception, stop/join the updater thread and atomically write the complete output checkpoint. `--evaluation-only` must disable commands, optimizer steps and checkpoint mutation.

- [ ] **Step 5: Map Gymnasium endings explicitly**

For a five-item Gymnasium step, use `terminated` and `truncated` directly. For `RslRlVecEnvWrapper`'s four-item step, require `extras["time_outs"]` with the same batch shape as `done`, then compute:

```python
truncated = done.bool() & extras["time_outs"].bool()
terminated = done.bool() & ~extras["time_outs"].bool()
bootstrap_mask = (~terminated).to(dtype=torch.float32)
```

Raise `RuntimeError("cannot distinguish termination from truncation")` before replay insertion if `time_outs` is absent or malformed. Safety intervention and control timeout override `bootstrap_mask` to zero and set `safety_mask` to zero; those rows remain Actor-invalid.

- [ ] **Step 6: Test deployable-information isolation**

Add a runtime test that injects two different geometry objects, stage IDs and seeds while keeping robot history and command identical. Assert Actor inputs and reward terms are bitwise equal. Add a source scan assertion that `terrain_stage`, `terrain_seed`, step height, rough amplitude and road-end coordinates are only referenced by environment configuration, audit logging and offline completion logic, never by `ResidualActor`, `compute_online_reward`, novelty gating or publication acceptance.

- [ ] **Step 7: Write an immutable `episode_summary.json`**

Include pass/fall/safety outcome, completion boundary, actual interaction steps, stage/seed, gain, checkpoint hashes, updates, rollbacks, velocity error, tilt RMS, contacts and flat/online sample fractions. Use `allow_nan=False`.

- [ ] **Step 8: Run runtime tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_cli.py \
  tests/online_adaptation/test_play_runtime.py \
  tests/online_adaptation/test_play_modes.py -q
git add scripts/reinforcement_learning/online_adaptation/cli.py \
  scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  tests/online_adaptation/test_cli.py \
  tests/online_adaptation/test_play_runtime.py \
  tests/online_adaptation/test_play_modes.py
git commit -m "feat: run persistent lite3 curriculum episodes"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 16: Add the Parent Curriculum Supervisor

**Files:**
- Create: `scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py`
- Create: `tests/online_adaptation/test_run_flat_online_curriculum.py`
- Modify: `scripts/reinforcement_learning/online_adaptation/cli.py`

- [ ] **Step 1: Write failing command construction tests**

```python
def test_worker_command_contains_exact_stage_seed_budget_and_checkpoint_paths(tmp_path):
    request = EpisodeRequest(stage="step_2cm", seed=101, episode=1, gain=0.25)
    command = build_worker_command(make_args(tmp_path), request)
    assert command_value(command, "--terrain-stage") == "step_2cm"
    assert command_value(command, "--terrain-seed") == "101"
    assert command_value(command, "--max-steps") == "2000"
    assert command_value(command, "--residual-gain") == "0.25"
    assert command_value(command, "--n-step") == "3"
    assert command_value(command, "--critic-replay-mode") == "dual"
    assert command_value(command, "--flat-anchor-weight") == "1.0"
    assert "--experiment-checkpoint-out" in command
```

- [ ] **Step 2: Write failing resume and failed-worker tests**

The supervisor must resume from `curriculum_state.json`, refuse a different protocol hash, count a failed worker as a failed episode when a valid failure summary exists, and stop without advancing when no valid summary/checkpoint exists.

- [ ] **Step 3: Implement a CPU-only parent process and exact experiment CLI**

The parent never imports Isaac. It loads the protocol/coordinator, selects the next round-robin development seed, creates an immutable episode directory, launches `play_online_adaptation.py` with `subprocess.run(check=False)`, validates output hashes, records `EpisodeOutcome`, updates gain, and atomically saves coordinator state.

Require `--base-checkpoint`, `--base-manifest`, `--output-dir`, and `--flat-replay`. Add `--configuration` for a selector-produced JSON file, plus direct ablation flags `--target-semantics {legacy-done,timeout-aware}`, `--n-step {1,3,5}`, `--critic-replay-mode {online-only,dual}`, `--flat-fraction-initial`, `--flat-fraction-ready`, `--flat-anchor-weight`, `--flat-critic-warmup-updates`, `--max-episodes`, `--max-total-steps`, `--max-stage-episodes`, `--device`, and `--headless`. Direct flags may fill fields omitted by a partial target configuration, but a conflicting duplicate value is rejected. Every value is copied into `run_manifest.json` and forwarded to each worker. The defaults are timeout-aware, 3-step, dual replay, 0.75/0.50 mixing, anchor weight 1.0, 1,000 flat-only Critic warmup updates, 100 episodes, 200,000 steps and 20 episodes per stage. On every clean stop, write the latest complete learner state to `<output-dir>/experiment.pt` and its checksum sidecar.

- [ ] **Step 4: Enforce budgets before launching**

Refuse any worker request that would exceed 100 episodes or 200,000 steps. Every worker receives `--max-steps min(2000, remaining_steps)`.

- [ ] **Step 5: Gate every stage advancement on a flat-retention probe**

After each complete 10-episode target-terrain window, launch deterministic `base` and `learned_frozen` plane workers on seeds `(7, 11, 19, 23, 31)` with updates disabled. Store these probe steps in a separate `evaluation_steps` counter, never in the online interaction budget. Pass `flat_retention_passed=True` only when completion is within 5 percentage points, velocity error and tilt RMS are each at most 1.05 times base, and no fall/contact count increases. A failed probe blocks stage advancement and is included in no-improvement stopping.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_run_flat_online_curriculum.py -q
git add scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  scripts/reinforcement_learning/online_adaptation/cli.py \
  tests/online_adaptation/test_run_flat_online_curriculum.py
git commit -m "feat: supervise lite3 cross-reset online curriculum"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 17: Implement Strict Four-Mode and Sealed Acceptance

**Files:**
- Modify: `source/rl_training/rl_training/online_adaptation/acceptance.py`
- Create: `scripts/reinforcement_learning/online_adaptation/evaluate_flat_online_curriculum.py`
- Create: `scripts/reinforcement_learning/online_adaptation/select_flat_online_configuration.py`
- Create: `scripts/reinforcement_learning/online_adaptation/freeze_flat_online_configuration.py`
- Create: `tests/online_adaptation/test_flat_online_acceptance.py`
- Create: `tests/online_adaptation/test_flat_online_configuration.py`
- Modify: `tests/online_adaptation/test_acceptance_report.py`

- [ ] **Step 1: Write failing pure acceptance tests**

```python
def test_strict_acceptance_requires_all_confirmed_thresholds():
    report = evaluate_flat_online_experiment(
        mode_records=passing_mode_records(),
        flat_base=flat_record(completion_rate=1.0, velocity_error=0.10, tilt_rms=0.05),
        flat_learned=flat_record(completion_rate=0.95, velocity_error=0.105, tilt_rms=0.0525),
    )
    assert report["passed"]
    assert report["checks"] == {
        "twenty_sealed_seeds": True,
        "learned_success_at_least_80_percent": True,
        "improves_base_by_30_points": True,
        "improves_initial_zero_by_30_points": True,
        "zero_evaluation_falls": True,
        "flat_completion_retained": True,
        "flat_velocity_retained": True,
        "flat_tilt_retained": True,
        "flat_contacts_not_increased": True,
        "within_training_budget": True,
    }
```

Add one parametrized test that flips each check to false and proves aggregate failure.

- [ ] **Step 2: Verify RED**

Expected: evaluator is missing.

- [ ] **Step 3: Implement the pure evaluator**

Require exactly the 20 protocol sealed seeds, no duplicates, deterministic `learned_frozen`, no updates during evaluation, success at least 16/20, both improvements at least 0.30, zero falls, completion no more than 0.05 below flat base, velocity/tilt at most 1.05 times flat base, no added contacts, and training budgets within bounds.

- [ ] **Step 4: Implement the evaluation driver**

Run `base`, `initial_zero_residual`, and `learned_frozen` on each sealed combined course in separate Isaac processes. `online_learning` comes from immutable training summaries, not sealed updates. Run flat retention for base and learned-frozen with the same flat seeds. Freeze and verify configuration/protocol/checkpoint hashes before the first sealed worker.

The parser requires `--formal-configuration` and accepts `--output-dir`, `--device`, and `--headless`. It refuses a writable formal configuration, any hash mismatch, any sealed result already present in the output directory, or an existing `<output-dir>/sealed_evaluation_started.json` for the same formal hash. Before launching the first worker it atomically creates that sibling marker; infrastructure-invalid reruns therefore require a new output directory.

- [ ] **Step 5: Implement deterministic development selection and configuration freezing**

`select_flat_online_configuration.py` has two CPU-only subcommands. `select-target` requires the timeout-aware 1-, 3-, and 5-step run directories and writes `selected_target_configuration.json`. `select-final` requires that target configuration plus the selected-target online-only, dual-without-anchor, and dual-with-anchor run directories and writes `selected_development_configuration.json`. Both verify identical development seeds, baseline hash and budgets. Selection maximizes pass rate, then minimizes falls, safety aborts, flat-retention cost and interaction steps in that order. The final JSON contains every numeric algorithm value and every source hash; neither subcommand imports or reads the sealed seed results. It also writes `q_calibration.json` from the recorded flat/online Q, target-Q and TD-error distributions. If source and online Q scales fail the documented finite/range checks, selection stops with `q_calibration_required`; Cal-QL is not silently enabled and requires a separately reviewed one-factor task.

`freeze_flat_online_configuration.py` requires `--selected-configuration`, `--base-manifest`, `--flat-replay-manifest`, `--experiment-checkpoint`, and `--output`. It embeds the canonical protocol, hashes every input plus the current Git commit, writes with `allow_nan=False`, verifies a read-back, and changes mode to `0444`. Tests prove changed inputs, writable output, duplicate freeze, and development/sealed seed overlap are rejected.

- [ ] **Step 6: Write JSON, CSV and Markdown without NaN**

Outputs are `sealed_runs.jsonl`, `sealed_comparison.csv`, `acceptance.json`, and `acceptance.md`. Existing historical `evaluate_run` remains unchanged for old artifacts.

- [ ] **Step 7: Run tests and commit**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest \
  tests/online_adaptation/test_flat_online_acceptance.py \
  tests/online_adaptation/test_flat_online_configuration.py \
  tests/online_adaptation/test_acceptance_report.py -q
git add source/rl_training/rl_training/online_adaptation/acceptance.py \
  scripts/reinforcement_learning/online_adaptation/evaluate_flat_online_curriculum.py \
  scripts/reinforcement_learning/online_adaptation/select_flat_online_configuration.py \
  scripts/reinforcement_learning/online_adaptation/freeze_flat_online_configuration.py \
  tests/online_adaptation/test_flat_online_acceptance.py \
  tests/online_adaptation/test_flat_online_configuration.py \
  tests/online_adaptation/test_acceptance_report.py
git commit -m "feat: evaluate lite3 flat-to-terrain learning strictly"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 18: Complete Automated and GPU Causal Verification

**Files:**
- Modify after evidence: `docs/lite3-online-learning-principles.md`
- Generated outputs: `/home/tjx/rl_training_main/logs/online_adaptation/flat_online_*`

- [ ] **Step 1: Run the full automated suite**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest tests/online_adaptation -q
PYTHONPATH=source/rl_training:. \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training/online_adaptation \
  scripts/reinforcement_learning/online_adaptation
git diff --check
```

Expected: zero failures, only explicit Isaac-only skips, no compile or whitespace diagnostics.

- [ ] **Step 2: Run a flat zero-authority GPU probe**

Run 500 control steps with `FlatOnline-Deeprobotics-Lite3-v0`, a qualified flat PPO, `--mode residual_frozen`, `--residual-gain 0`, and no residual checkpoint. Assert no reset/fall/safety violation, every logged residual norm is exactly zero, and base/final actions differ by at most `1e-6`.

- [ ] **Step 3: Run a 2 cm cross-process reset probe**

Run three 2 cm development episodes with a tiny test budget. Force one safe abort in episode 2. Verify episode 3 restores updater version, both replay sizes, n-step queue state, RNG state, stable snapshot, gain tier and budget; immediately after reset it executes zero residual during `BASE_WARMUP`.

- [ ] **Step 4: Run 1/3/5-step and replay-mix GPU smoke runs**

Use the same base checkpoint, stage and seed. Each run is 2,000 steps. Confirm finite Q/target/TD metrics and exact configured source fractions. These are causal probes, not performance selection.

- [ ] **Step 5: Update the principles document and commit evidence**

Record commands, run directories, hashes, test count and failures as well as successes. State explicitly that infrastructure probes do not prove learned terrain capability.

```bash
git add docs/lite3-online-learning-principles.md
git commit -m "docs: record lite3 flat-online causal probes"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

---

### Task 19: Train, Select on Development Seeds, Freeze, and Evaluate Once

**Files:**
- Modify after results: `docs/lite3-online-learning-principles.md`
- Generated artifacts: `logs/rsl_rl/deeprobotics_lite3_flat/*`
- Generated artifacts: `logs/online_adaptation/flat_online_formal_20260729/*`

- [ ] **Step 1: Train the formal flat PPO**

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=source/rl_training:.
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-Deeprobotics-Lite3-v0 \
  --num_envs 4096 \
  --seed 42 \
  --max_iterations 10000 \
  --run_name flat_only_seed42_20260729 \
  --headless
```

Do not substitute a rough-policy checkpoint. Preserve the complete run directory.

- [ ] **Step 2: Qualify and manifest the flat PPO**

Resolve exactly one run directory and its highest-versioned checkpoint, then run the fixed five flat qualification seeds and build the manifest:

```bash
FLAT_ROOT=/home/tjx/rl_training_main/logs/rsl_rl/deeprobotics_lite3_flat
test "$(find "$FLAT_ROOT" -maxdepth 1 -type d -name '*_flat_only_seed42_20260729' | wc -l)" -eq 1
FLAT_RUN="$(find "$FLAT_ROOT" -maxdepth 1 -type d -name '*_flat_only_seed42_20260729')"
BASE_CHECKPOINT="$(find "$FLAT_RUN" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -1)"
test -n "$BASE_CHECKPOINT"
test -f "$BASE_CHECKPOINT"
printf '%s\n' 'python scripts/reinforcement_learning/rsl_rl/train.py --task Flat-Deeprobotics-Lite3-v0 --num_envs 4096 --seed 42 --max_iterations 10000 --run_name flat_only_seed42_20260729 --headless' > "$FLAT_RUN/training_command.txt"

python scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --task Flat-Deeprobotics-Lite3-v0 \
  --seeds 7 11 19 23 31 \
  --steps-per-seed 2000 \
  --max-mean-velocity-error 0.35 \
  --output "$FLAT_RUN/base_qualification.json" \
  --device cuda:2 \
  --headless
python -c 'import json,sys; report=json.load(open(sys.argv[1])); assert report["aggregate"]["passed"]' \
  "$FLAT_RUN/base_qualification.json"

mapfile -t FLAT_CONFIGS < <(find "$FLAT_RUN" -type f -path '*/params/*.yaml' | sort)
test "${#FLAT_CONFIGS[@]}" -ge 2
CONFIG_ARGS=()
for path in "${FLAT_CONFIGS[@]}"; do CONFIG_ARGS+=(--config "$path"); done
python scripts/reinforcement_learning/online_adaptation/build_flat_baseline_manifest.py \
  --checkpoint "$BASE_CHECKPOINT" \
  "${CONFIG_ARGS[@]}" \
  --seed 42 \
  --num-envs 4096 \
  --max-iterations 10000 \
  --training-command-file "$FLAT_RUN/training_command.txt" \
  --output "$FLAT_RUN/flat_baseline_manifest.json"
test -f "$FLAT_RUN/flat_baseline_manifest.json"
```

Stop if the plane assertion, qualification, unique-run check, checkpoint resolution or hashing fails.

- [ ] **Step 3: Collect and freeze flat replay**

Collect and verify exactly 100,000 deterministic plane transitions:

```bash
FLAT_REPLAY_DIR=/home/tjx/rl_training_main/logs/online_adaptation/flat_replay_seed42_20260729
python scripts/reinforcement_learning/online_adaptation/collect_flat_replay.py \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --base-manifest "$FLAT_RUN/flat_baseline_manifest.json" \
  --task Flat-Deeprobotics-Lite3-v0 \
  --seed 42 \
  --transitions 100000 \
  --output-dir "$FLAT_REPLAY_DIR" \
  --device cuda:2 \
  --headless
python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["transition_count"] == 100000; assert m["falls"] == 0; assert m["safety_invalid_rows"] == 0; assert m["frozen"] is True' \
  "$FLAT_REPLAY_DIR/flat_replay_manifest.json"
test -f "$FLAT_REPLAY_DIR/flat_replay.pt"
```

The collector also verifies that the supplied baseline manifest hashes the exact checkpoint before launching Isaac.

- [ ] **Step 4: Execute development ablations in the fixed order**

Run the target ablations from the same zero-residual initialization. All runs use protocol development seeds `(101, 103, 107, 109, 113)` and identical 100-episode/200,000-step budgets:

```bash
DEV_ROOT=/home/tjx/rl_training_main/logs/online_adaptation/flat_online_development_20260729
COMMON_ARGS=(
  --base-checkpoint "$BASE_CHECKPOINT"
  --base-manifest "$FLAT_RUN/flat_baseline_manifest.json"
  --flat-replay "$FLAT_REPLAY_DIR/flat_replay.pt"
  --max-episodes 100 --max-total-steps 200000 --max-stage-episodes 20
  --critic-replay-mode online-only --flat-anchor-weight 0
  --device cuda:2 --headless
)
python scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  "${COMMON_ARGS[@]}" --output-dir "$DEV_ROOT/legacy_1step" \
  --target-semantics legacy-done --n-step 1
python scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  "${COMMON_ARGS[@]}" --output-dir "$DEV_ROOT/timeout_1step" \
  --target-semantics timeout-aware --n-step 1
python scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  "${COMMON_ARGS[@]}" --output-dir "$DEV_ROOT/timeout_3step" \
  --target-semantics timeout-aware --n-step 3
python scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  "${COMMON_ARGS[@]}" --output-dir "$DEV_ROOT/timeout_5step" \
  --target-semantics timeout-aware --n-step 5

python scripts/reinforcement_learning/online_adaptation/select_flat_online_configuration.py \
  select-target \
  --run "$DEV_ROOT/timeout_1step" \
  --run "$DEV_ROOT/timeout_3step" \
  --run "$DEV_ROOT/timeout_5step" \
  --output "$DEV_ROOT/selected_target_configuration.json"
```

Then hold the selected target fixed and change only Critic replay, followed by only Actor anchoring:

```bash
TARGET_COMMON=(
  --base-checkpoint "$BASE_CHECKPOINT"
  --base-manifest "$FLAT_RUN/flat_baseline_manifest.json"
  --flat-replay "$FLAT_REPLAY_DIR/flat_replay.pt"
  --configuration "$DEV_ROOT/selected_target_configuration.json"
  --max-episodes 100 --max-total-steps 200000 --max-stage-episodes 20
  --device cuda:2 --headless
)
python scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  "${TARGET_COMMON[@]}" --output-dir "$DEV_ROOT/online_only_no_anchor" \
  --critic-replay-mode online-only --flat-anchor-weight 0
python scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  "${TARGET_COMMON[@]}" --output-dir "$DEV_ROOT/dual_no_anchor" \
  --critic-replay-mode dual --flat-fraction-initial 0.75 \
  --flat-fraction-ready 0.50 --flat-anchor-weight 0
python scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  "${TARGET_COMMON[@]}" --output-dir "$DEV_ROOT/dual_with_anchor" \
  --critic-replay-mode dual --flat-fraction-initial 0.75 \
  --flat-fraction-ready 0.50 --flat-anchor-weight 1.0

python scripts/reinforcement_learning/online_adaptation/select_flat_online_configuration.py \
  select-final \
  --target-configuration "$DEV_ROOT/selected_target_configuration.json" \
  --run "$DEV_ROOT/online_only_no_anchor" \
  --run "$DEV_ROOT/dual_no_anchor" \
  --run "$DEV_ROOT/dual_with_anchor" \
  --output "$DEV_ROOT/selected_development_configuration.json"
```

Save every failed run. The selector validates the fixed seeds and one-factor comparisons before producing either configuration.

- [ ] **Step 5: Run the full five-stage protected curriculum**

```bash
python scripts/reinforcement_learning/online_adaptation/run_flat_online_curriculum.py \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --base-manifest "$FLAT_RUN/flat_baseline_manifest.json" \
  --flat-replay "$FLAT_REPLAY_DIR/flat_replay.pt" \
  --configuration "$DEV_ROOT/selected_development_configuration.json" \
  --output-dir /home/tjx/rl_training_main/logs/online_adaptation/flat_online_formal_20260729 \
  --device cuda:2 \
  --headless
```

Immediately before this command, re-resolve and validate every selected input:

```bash
test "$(find "$FLAT_ROOT" -maxdepth 1 -type d -name '*_flat_only_seed42_20260729' | wc -l)" -eq 1
test -f "$BASE_CHECKPOINT"
test -f "$FLAT_RUN/flat_baseline_manifest.json"
test -f "$FLAT_REPLAY_DIR/flat_replay.pt"
test -f "$DEV_ROOT/selected_development_configuration.json"
```

The supervisor records the resolved paths and hashes in the run manifest and enforces at most 100 episodes and 200,000 steps.

- [ ] **Step 6: Freeze the formal configuration**

Create and verify the read-only formal configuration:

```bash
FORMAL_ROOT=/home/tjx/rl_training_main/logs/online_adaptation/flat_online_formal_20260729
python scripts/reinforcement_learning/online_adaptation/freeze_flat_online_configuration.py \
  --selected-configuration "$DEV_ROOT/selected_development_configuration.json" \
  --base-manifest "$FLAT_RUN/flat_baseline_manifest.json" \
  --flat-replay-manifest "$FLAT_REPLAY_DIR/flat_replay_manifest.json" \
  --experiment-checkpoint "$FORMAL_ROOT/experiment.pt" \
  --output "$FORMAL_ROOT/formal_configuration.json"
test ! -w "$FORMAL_ROOT/formal_configuration.json"
```

The frozen file contains protocol, baseline, flat replay, experiment checkpoint, Git and CLI hashes. From this point, do not alter code, thresholds, checkpoint choice or seed lists based on sealed outcomes.

- [ ] **Step 7: Execute the one-shot 20-seed sealed evaluation**

```bash
python scripts/reinforcement_learning/online_adaptation/evaluate_flat_online_curriculum.py \
  --formal-configuration /home/tjx/rl_training_main/logs/online_adaptation/flat_online_formal_20260729/formal_configuration.json \
  --output-dir /home/tjx/rl_training_main/logs/online_adaptation/flat_online_sealed_20260729 \
  --device cuda:2 \
  --headless
```

Expected output is a valid acceptance report, not necessarily a pass. Do not rerun a statistically valid failure. Infrastructure-invalid runs retain logs and receive a new experiment identifier after a tested fix.

- [ ] **Step 8: Document the honest result**

Update `docs/lite3-online-learning-principles.md` with all four modes, per-seed outcomes, confidence intervals, training budget, falls, rollbacks, flat retention, configuration hashes and whether every strict criterion passed. Preserve the simulation-only boundary: no real Lite3 command is authorized, and future hardware use still requires operator reset, residual/velocity zeroing, damping or emergency stop after a fall, an independent watchdog, harness and physical emergency stop. Never describe partial improvement as “learned stairs” unless the complete acceptance report passes.

- [ ] **Step 9: Run final review, commit, push and verify remote alignment**

```bash
PYTHONPATH=source/rl_training:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/pytest tests/online_adaptation -q
git diff --check
git add docs/lite3-online-learning-principles.md
git commit -m "docs: report lite3 flat-to-terrain evaluation"
git push origin feat/lite3-online-adaptation
git fetch origin feat/lite3-online-adaptation
git rev-list --left-right --count HEAD...origin/feat/lite3-online-adaptation
```

Expected: tests pass, push succeeds, and remote divergence is `0 0`.

---

## Plan Completion Gate

The implementation is complete only when Tasks 1-18 pass their automated and causal checks and Task 19 produces either:

1. a strict passing `acceptance.json`, or
2. an immutable failing report that identifies which confirmed threshold was missed without weakening the threshold or hiding failed episodes.

Every effective task ends with a local commit and push to the private `jenny-tian/rl_training_main` repository. If GitHub is unreachable, retain the local commit and record the exact local/remote SHA until the push can be retried.
