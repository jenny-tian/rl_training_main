# EA-LPL P0：航向闭环基础策略与目标条件 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 θ₀（model_1998 加航向闭环）在带初始扰动的 Source L1、L2 条件上的基线，并按预登记规则选出 P1 的目标条件。

**Architecture:** 新增 `rl_training.ea_lpl` 纯张量模块（初始状态表、配置检查、回合切分、统计）和一个 EA-LPL 环境配置，复用 `SourceTrainingVecEnv`、安全过滤器和现有监督模式。一个通用运行脚本 `scripts/tools/ea_lpl_run.py` 负责 GPU 运行，一个汇总脚本负责统计和目标选择。

**Tech Stack:** Isaac Lab / Isaac Sim（env_isaaclab），RSL-RL 5.0.1，PyTorch，numpy，pytest

**Spec:** `docs/lite3-ea-lpl-research-plan-v1-20261105.md`（D1、D4、D5、D10，第 5 节 P0 规则）；执行规则见 `docs/lite3-ea-lpl-codex-brief-v1-20261105.md`

## Global Constraints

- Python：`/home/tjx/anaconda3/envs/env_isaaclab/bin/python`；测试命令前缀 `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。
- 基础策略 `model_1998.pt`，SHA256 `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`，只读。
- 120 个环境，共享地形库 1 行，控制周期 0.02 s，每回合最多 2000 控制步，完成条件为前进 ≥ 9 m 后连续稳定 50 步；这些不改。
- 安全过滤器上限不改：残差上限 (0.24, 0.20, 0.20)×4，速率 0.02/步。
- 航向闭环：转向指令 = clip(0.5 × wrap(0 − 航向), −0.8, 0.8) rad/s；前进 0.3 m/s，横向 0。
- 初始扰动：x ∈ [−0.2, 0.2] m，y ∈ [−0.1, 0.1] m，yaw ∈ [−0.1, 0.1] rad；表种子 20261106；P0 命名空间 1；命名空间 9 保留给封存实验，禁止使用。
- GPU 准入：所选 GPU 空闲显存 ≥ 12288 MiB，GPU0、GPU1 都可用，可共享。
- 原始数据写到 `$(git rev-parse --git-common-dir)/ea-lpl-evidence/p0/`，不进 Git。
- 不访问 boxes、pyramid_stairs、pyramid_stairs_inv。

## Review Focus

1. 航向闭环被悄悄关闭：只设 `heading_command=True` 时转向指令会被 `ranges.ang_vel_z=(0,0)` 限幅为 0，或被纯前进槽位关闭。Task 1 的 `check_heading_hold` 测试和 Task 4 的启动检查必须拦住。
2. 初始状态表没生效：复位事件没被替换时，所有环境从同一位置出发。Task 4 启动时逐环境核对位置和航向偏移。
3. 多回合运行中地形列漂移：原分配器在复位时换列。Task 1 的固定列分配器测试，加 Task 3 对多回合需要固定列的规格检查。
4. 终止步记录成复位后的状态：状态必须在奖励计算钩子里（复位前）采集。Task 3 的回合切分测试检查事件与 done 一致。
5. L2 被训练配置拒绝：`source_allowed_levels` 改动由 Task 1 覆盖，Task 5 的 L2 运行验证。

## File Structure

| 文件 | 操作 | 职责 |
|---|---|---|
| `source/rl_training/rl_training/source_scan/parallel_terrain_allocation.py` | 修改 | 新增 `FixedColumnTerrainAllocator`，`assign_source_terrain_for_parallel_reset` 接受 `fixed_columns` |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py` | 修改 | 课程项透传 `fixed_columns` |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_training_env_cfg.py` | 修改 | `source_allowed_levels` |
| `source/rl_training/rl_training/source_scan/training_env.py` | 修改 | 等级检查读取 `source_allowed_levels` |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/ea_lpl_env_cfg.py` | 新建 | EA-LPL 环境配置 |
| `source/rl_training/rl_training/ea_lpl/__init__.py` | 新建 | 包标记 |
| `source/rl_training/rl_training/ea_lpl/reset_table.py` | 新建 | 确定性初始状态表与复位事件 |
| `source/rl_training/rl_training/ea_lpl/cfg_checks.py` | 新建 | 航向闭环配置检查与指令公式 |
| `source/rl_training/rl_training/ea_lpl/specs.py` | 新建 | 运行规格校验与 P0 规格 |
| `source/rl_training/rl_training/ea_lpl/episodes.py` | 新建 | 多回合切分 |
| `source/rl_training/rl_training/ea_lpl/summary.py` | 新建 | 置信区间、条件表、配对检验、目标选择 |
| `scripts/tools/ea_lpl_run.py` | 新建 | 监督进程与仿真子进程 |
| `scripts/tools/ea_lpl_make_specs.py` | 新建 | 写出规格文件 |
| `scripts/tools/ea_lpl_p0_summary.py` | 新建 | P0 汇总与目标选择 |
| `tests/ea_lpl/test_*.py` | 新建 | CPU 测试 |

---

### Task 1: 固定列分配、允许 L2、EA-LPL 配置与航向闭环检查（预计 2 小时）

**Files:**
- Modify: `source/rl_training/rl_training/source_scan/parallel_terrain_allocation.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:67-76`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_training_env_cfg.py:12-20`
- Modify: `source/rl_training/rl_training/source_scan/training_env.py:119-122`
- Create: `source/rl_training/rl_training/ea_lpl/__init__.py`, `source/rl_training/rl_training/ea_lpl/cfg_checks.py`
- Create: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/ea_lpl_env_cfg.py`
- Test: `tests/ea_lpl/test_fixed_columns.py`, `tests/ea_lpl/test_cfg_checks.py`

**Interfaces:**
- Produces: `FixedColumnTerrainAllocator(num_envs, block_counts, sampler_seed, fixed_columns, *, bank_rows=None)`；`assign_source_terrain_for_parallel_reset(env, env_ids, sampler_seed, fixed_columns=None)`；`check_heading_hold(command_cfg, *, enabled, stiffness=0.5, max_yaw_rate=0.8)`；`heading_yaw_rate_command(heading, *, stiffness=0.5, max_rate=0.8) -> torch.Tensor`；配置类 `DeeproboticsLite3EaLplEnvCfg`，字段 `ea_heading_hold`、`ea_columns`、`ea_reset_table_seed`、`ea_reset_namespace`、`ea_reset_slots`。

- [ ] **Step 1: 写失败的测试**

`tests/ea_lpl/test_fixed_columns.py`：

```python
from types import SimpleNamespace

import pytest
import torch

from rl_training.source_scan.parallel_terrain_allocation import (
    SOURCE_RESET_BLOCK_COUNTS,
    FixedColumnTerrainAllocator,
    assign_source_terrain_for_parallel_reset,
)


def fake_env(num_envs):
    labels = tuple(SOURCE_RESET_BLOCK_COUNTS)
    sub = {name: SimpleNamespace(proportion=1.0 / len(labels)) for name in labels}
    generator = SimpleNamespace(num_rows=num_envs, num_cols=len(labels), sub_terrains=sub)
    cfg = SimpleNamespace(terrain_type="generator", use_terrain_origins=True,
                          terrain_generator=generator, shared_bank_rows=None)
    origins = torch.arange(num_envs * len(labels) * 3, dtype=torch.float32).reshape(num_envs, len(labels), 3)
    terrain = SimpleNamespace(cfg=cfg, terrain_origins=origins, env_origins=torch.zeros(num_envs, 3),
                              terrain_types=torch.zeros(num_envs, dtype=torch.long),
                              terrain_levels=torch.zeros(num_envs, dtype=torch.long))
    return SimpleNamespace(num_envs=num_envs, scene=SimpleNamespace(terrain=terrain))


def test_fixed_allocator_keeps_columns_and_counts_episodes():
    columns = tuple(i % 15 for i in range(30))
    allocator = FixedColumnTerrainAllocator(30, SOURCE_RESET_BLOCK_COUNTS, 7, columns, bank_rows=1)
    first = allocator.reset(list(range(30)))
    assert [a.column for a in first] == list(columns)
    assert all(a.row == 0 for a in first)
    again = allocator.reset([3, 1])
    assert [(a.env_id, a.column, a.episode_index) for a in again] == [(1, 1, 1), (3, 3, 1)]


def test_fixed_allocator_rejects_bad_columns():
    with pytest.raises(ValueError):
        FixedColumnTerrainAllocator(3, SOURCE_RESET_BLOCK_COUNTS, 0, (0, 1), bank_rows=1)
    with pytest.raises(ValueError):
        FixedColumnTerrainAllocator(2, SOURCE_RESET_BLOCK_COUNTS, 0, (0, 15), bank_rows=1)
    allocator = FixedColumnTerrainAllocator(2, SOURCE_RESET_BLOCK_COUNTS, 0, (0, 1))
    with pytest.raises(ValueError):
        allocator.reset([0])


def test_parallel_reset_with_fixed_columns_never_changes_columns():
    env = fake_env(30)
    columns = tuple((3 * i) % 15 for i in range(30))
    assign_source_terrain_for_parallel_reset(env, list(range(30)), 5, fixed_columns=columns)
    assert env.scene.terrain.terrain_types.tolist() == list(columns)
    assign_source_terrain_for_parallel_reset(env, [4, 9], 5, fixed_columns=columns)
    assert env.scene.terrain.terrain_types.tolist() == list(columns)
    assert env._source_parallel_terrain_allocator.assignments[4].episode_index == 1


def test_parallel_reset_rejects_changed_fixed_columns():
    env = fake_env(30)
    columns = tuple(i % 15 for i in range(30))
    assign_source_terrain_for_parallel_reset(env, list(range(30)), 5, fixed_columns=columns)
    with pytest.raises(RuntimeError):
        assign_source_terrain_for_parallel_reset(env, [0], 5, fixed_columns=tuple(reversed(columns)))
```

`tests/ea_lpl/test_cfg_checks.py`：

```python
import math
from types import SimpleNamespace

import pytest
import torch

from rl_training.ea_lpl.cfg_checks import check_heading_hold, heading_yaw_rate_command


def command_cfg(**overrides):
    ranges = SimpleNamespace(lin_vel_x=(0.3, 0.3), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.8, 0.8), heading=(0.0, 0.0))
    values = dict(heading_command=True, heading_control_stiffness=0.5, rel_heading_envs=1.0, rel_standing_envs=0.0,
                  rel_zero_vel_envs=0.0, rel_only_lin_y_envs=0.0, rel_only_lin_x_envs=0.0, rel_only_ang_z_envs=0.0,
                  resampling_time_range=(1.0e9, 1.0e9), ranges=ranges)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_heading_hold_accepts_registered_settings():
    check_heading_hold(command_cfg(), enabled=True)


def test_heading_hold_rejects_zero_yaw_range_and_pure_forward_slot():
    cfg = command_cfg()
    cfg.ranges.ang_vel_z = (0.0, 0.0)
    with pytest.raises(ValueError, match="ang_vel_z"):
        check_heading_hold(cfg, enabled=True)
    with pytest.raises(ValueError, match="rel_only_lin_x_envs"):
        check_heading_hold(command_cfg(rel_only_lin_x_envs=1.0), enabled=True)


def test_disabled_heading_matches_source_baseline():
    cfg = command_cfg(heading_command=False, rel_heading_envs=0.0, rel_only_lin_x_envs=1.0)
    cfg.ranges.ang_vel_z = (0.0, 0.0)
    check_heading_hold(cfg, enabled=False)


def test_yaw_rate_command_formula():
    heading = torch.tensor([0.0, 0.2, -0.2, 3.0, -3.0])
    command = heading_yaw_rate_command(heading)
    assert torch.allclose(command[:3], torch.tensor([0.0, -0.1, 0.1]))
    assert math.isclose(float(command[3]), -0.8)
    assert math.isclose(float(command[4]), 0.8)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_fixed_columns.py tests/ea_lpl/test_cfg_checks.py -q`
Expected: FAIL，`ImportError`（`FixedColumnTerrainAllocator`、`rl_training.ea_lpl` 不存在）。

- [ ] **Step 3: 实现固定列分配器**

在 `parallel_terrain_allocation.py` 的 `ReservedRowTerrainAllocator` 类之后加入：

```python
class FixedColumnTerrainAllocator:
    """Keep every environment on one registered column for all of its episodes.

    ``fixed_columns[i]`` is the column of environment i. Rows follow the
    reserved-row layout, or ``env_id % bank_rows`` for an explicit shared bank.
    """

    def __init__(self, num_envs, block_counts, sampler_seed, fixed_columns, *, bank_rows=None):
        ReservedRowTerrainAllocator._positive_int(num_envs, "num_envs")
        if type(sampler_seed) is not int or sampler_seed < 0:
            raise ValueError("sampler_seed must be an explicit nonnegative integer")
        labels = tuple(block_counts)
        columns = tuple(fixed_columns)
        if len(columns) != num_envs:
            raise ValueError("fixed_columns must list one column per environment")
        if any(type(c) is not int or not 0 <= c < len(labels) for c in columns):
            raise ValueError("fixed_columns must contain registered column indices")
        if bank_rows is not None and (type(bank_rows) is not int or not 1 <= bank_rows <= num_envs):
            raise ValueError("bank_rows must be between one and the environment count")
        self._num_envs = num_envs
        self._labels = labels
        self._columns = columns
        self._bank_rows = bank_rows
        self.sampler_seed = sampler_seed
        self._assignment_count = 0
        self._active: list[TerrainAssignment | None] = [None] * num_envs
        self._episodes = [0] * num_envs

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self._bank_rows or self._num_envs), len(self._labels)

    @property
    def column_labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def block_size(self) -> int:
        return self._num_envs

    @property
    def assignments(self) -> tuple[TerrainAssignment | None, ...]:
        return tuple(self._active)

    def reset(self, env_ids: Sequence[int]) -> tuple[TerrainAssignment, ...]:
        if not isinstance(env_ids, Sequence) or isinstance(env_ids, (str, bytes)):
            raise ValueError("env_ids must be a sequence of integer IDs")
        ids = tuple(env_ids)
        if any(type(i) is not int or not 0 <= i < self._num_envs for i in ids):
            raise ValueError("env_ids must contain in-range integer IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("env_ids must be unique")
        ids = tuple(sorted(ids))
        if not ids:
            return ()
        if self._assignment_count == 0 and ids != tuple(range(self._num_envs)):
            raise ValueError("the first reset must initialize every environment")
        result = []
        for env_id in ids:
            column = self._columns[env_id]
            assignment = TerrainAssignment(
                env_id=env_id,
                row=env_id if self._bank_rows is None else env_id % self._bank_rows,
                column=column,
                subterrain=self._labels[column],
                assignment_index=self._assignment_count,
                episode_index=self._episodes[env_id],
                block_index=self._assignment_count // self._num_envs,
                index_in_block=self._assignment_count % self._num_envs,
            )
            self._active[env_id] = assignment
            self._episodes[env_id] += 1
            self._assignment_count += 1
            result.append(assignment)
        return tuple(result)
```

修改 `assign_source_terrain_for_parallel_reset`：签名加 `fixed_columns=None`；`signature` 一行改为

```python
    fixed = None if fixed_columns is None else tuple(int(c) for c in fixed_columns)
    signature = (num_envs, labels, proportions, tuple(terrain_origins.shape), bank_rows, fixed)
```

`if is_new_allocator:` 分支的最前面加：

```python
        if fixed is not None:
            candidate_allocator = FixedColumnTerrainAllocator(
                num_envs, SOURCE_RESET_BLOCK_COUNTS, sampler_seed, fixed, bank_rows=bank_rows
            )
        elif bank_rows is None:
```

原来的 `if bank_rows is None:` 改成上面的 `elif`，其余不动。`__all__` 加入 `"FixedColumnTerrainAllocator"`。

- [ ] **Step 4: 课程项透传、允许 L2**

`curriculums.py`：

```python
def source_terrain_parallel_reset_sampler(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], sampler_seed: int,
    fixed_columns: Sequence[int] | None = None,
) -> None:
    """Assign one reserved-row Source tile before the selected assets reset."""

    from rl_training.source_scan.parallel_terrain_allocation import (
        assign_source_terrain_for_parallel_reset,
    )

    assign_source_terrain_for_parallel_reset(env, env_ids, sampler_seed, fixed_columns=fixed_columns)
```

`source_training_env_cfg.py`：类里加字段 `source_allowed_levels: tuple = ("L1",)`，检查改为 `self.source_terrain_level not in self.source_allowed_levels`，报错文字保持原样。

`training_env.py` 第 119 行附近：

```python
        allowed = getattr(cfg, "source_allowed_levels", ("L1",))
        if (cfg.source_terrain_level not in allowed or cfg.seed != seeds.ppo
```

- [ ] **Step 5: 配置检查模块**

`source/rl_training/rl_training/ea_lpl/__init__.py`：

```python
"""EA-LPL online adaptation on the frozen Lite3 base policy."""
```

`source/rl_training/rl_training/ea_lpl/cfg_checks.py`：

```python
"""Configuration checks that need no simulator import."""
from __future__ import annotations

import torch


def check_heading_hold(command_cfg, *, enabled, stiffness=0.5, max_yaw_rate=0.8):
    """Reject settings under which Isaac Lab would clip or disable the heading loop."""
    common = dict(rel_standing_envs=0.0, rel_zero_vel_envs=0.0, rel_only_lin_y_envs=0.0, rel_only_ang_z_envs=0.0)
    if enabled:
        expected = dict(common, heading_command=True, heading_control_stiffness=stiffness,
                        rel_heading_envs=1.0, rel_only_lin_x_envs=0.0)
        yaw_range = (-max_yaw_rate, max_yaw_rate)
    else:
        expected = dict(common, heading_command=False, rel_heading_envs=0.0, rel_only_lin_x_envs=1.0)
        yaw_range = (0.0, 0.0)
    for name, value in expected.items():
        actual = getattr(command_cfg, name)
        if actual != value:
            raise ValueError(f"heading setting {name}={actual!r}, expected {value!r}")
    ranges = command_cfg.ranges
    for name, value in dict(lin_vel_x=(0.3, 0.3), lin_vel_y=(0.0, 0.0), ang_vel_z=yaw_range,
                            heading=(0.0, 0.0)).items():
        actual = tuple(getattr(ranges, name))
        if actual != tuple(value):
            raise ValueError(f"command range {name}={actual!r}, expected {tuple(value)!r}")
    if tuple(command_cfg.resampling_time_range) != (1.0e9, 1.0e9):
        raise ValueError("commands must not resample inside an episode")


def heading_yaw_rate_command(heading, *, stiffness=0.5, max_rate=0.8):
    """Isaac Lab heading loop with target heading 0: clip(k * wrap(0 - heading))."""
    error = torch.atan2(torch.sin(-heading), torch.cos(-heading))
    return (stiffness * error).clamp(-max_rate, max_rate)
```

- [ ] **Step 6: EA-LPL 环境配置**

`ea_lpl_env_cfg.py`（这一步先不接复位表，Task 2 再加）：

```python
"""EA-LPL scene: frozen base policy, optional heading hold, explicit columns and reset table."""
from isaaclab.utils import configclass

from .source_adaptation_env_cfg import DeeproboticsLite3SourceAdaptationEnvCfg


@configclass
class DeeproboticsLite3EaLplEnvCfg(DeeproboticsLite3SourceAdaptationEnvCfg):
    source_allowed_levels: tuple = ("L1", "L2")
    ea_heading_hold: bool = True
    ea_heading_stiffness: float = 0.5
    ea_max_yaw_rate: float = 0.8
    ea_columns: tuple | None = None
    ea_reset_table_seed: int | None = None
    ea_reset_namespace: int = 0
    ea_reset_slots: tuple | None = None

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.rel_standing_envs = 0.0
        command.rel_zero_vel_envs = 0.0
        command.rel_only_lin_y_envs = 0.0
        command.rel_only_ang_z_envs = 0.0
        command.ranges.lin_vel_x = (0.3, 0.3)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.heading = (0.0, 0.0)
        if self.ea_heading_hold:
            command.heading_command = True
            command.heading_control_stiffness = self.ea_heading_stiffness
            command.rel_heading_envs = 1.0
            command.rel_only_lin_x_envs = 0.0
            command.ranges.ang_vel_z = (-self.ea_max_yaw_rate, self.ea_max_yaw_rate)
        else:
            command.heading_command = False
            command.rel_heading_envs = 0.0
            command.rel_only_lin_x_envs = 1.0
            command.ranges.ang_vel_z = (0.0, 0.0)
        if self.ea_columns is not None:
            columns = tuple(int(c) for c in self.ea_columns)
            if len(columns) != self.source_parallel_sampler_num_envs or any(not 0 <= c < 15 for c in columns):
                raise ValueError("ea_columns must list one registered column per environment")
            self.curriculum.source_terrain_parallel_reset_sampler.params["fixed_columns"] = columns
```

不需要注册新的 gym 任务：运行脚本把这个配置实例传给 `gym.make("SourceTerrain-Deeprobotics-Lite3-v0", cfg=cfg)`。

- [ ] **Step 7: 运行测试，确认通过**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl tests/source_scan -q -k "fixed or cfg_checks or allocation or training_env"`
Expected: 新增 8 项 PASS；已有分配器和训练环境测试仍通过。

- [ ] **Step 8: Commit**

```bash
git add source/rl_training/rl_training/source_scan/parallel_terrain_allocation.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_training_env_cfg.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/ea_lpl_env_cfg.py \
  source/rl_training/rl_training/source_scan/training_env.py source/rl_training/rl_training/ea_lpl tests/ea_lpl
git commit -m "feat(ea-lpl): fixed terrain columns, L2 opt-in and heading-hold configuration"
git push && git ls-remote --heads origin codex/lite3-ea-lpl-20261106
```

---

### Task 2: 确定性初始状态表（预计 1.5 小时）

**Files:**
- Create: `source/rl_training/rl_training/ea_lpl/reset_table.py`
- Modify: `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/ea_lpl_env_cfg.py`
- Test: `tests/ea_lpl/test_reset_table.py`

**Interfaces:**
- Consumes: Task 1 的 `DeeproboticsLite3EaLplEnvCfg`。
- Produces: `DEFAULT_RESET_RANGES`、`NAMESPACES`、`initial_state_offsets(table_seed, namespace, slots, episodes, ranges=DEFAULT_RESET_RANGES) -> np.ndarray[n, 3]`（列为 x、y、yaw）、`reset_root_state_from_table(env, env_ids, table_seed, namespace=0, slots=None, ranges=None, asset_cfg=None)`；每次复位后 `env._ea_last_reset_keys = {env_id: (namespace, slot, episode)}`，`env._ea_reset_episode_counter` 为每个环境的回合计数。

- [ ] **Step 1: 写失败的测试**

`tests/ea_lpl/test_reset_table.py`：

```python
import math
from types import SimpleNamespace

import numpy as np
import torch

from rl_training.ea_lpl.reset_table import DEFAULT_RESET_RANGES, initial_state_offsets, reset_root_state_from_table


class FakeAsset:
    def __init__(self, n):
        state = torch.zeros(n, 13)
        state[:, 2] = 0.375
        state[:, 3] = 1.0
        self.data = SimpleNamespace(default_root_state=state)
        self.pose = self.velocity = self.ids = None

    def write_root_pose_to_sim(self, pose, env_ids):
        self.pose, self.ids = pose.clone(), env_ids.clone()

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.velocity = velocity.clone()


class FakeScene:
    def __init__(self, asset, origins):
        self.asset, self.env_origins = asset, origins

    def __getitem__(self, name):
        assert name == "robot"
        return self.asset


def fake_env(n):
    origins = torch.stack((torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.zeros(n)), dim=1)
    return SimpleNamespace(num_envs=n, scene=FakeScene(FakeAsset(n), origins))


def test_offsets_are_deterministic_bounded_and_keyed():
    a = initial_state_offsets(20261106, 1, list(range(100)), [0] * 100)
    b = initial_state_offsets(20261106, 1, list(range(100)), [0] * 100)
    c = initial_state_offsets(20261106, 2, list(range(100)), [0] * 100)
    assert a.shape == (100, 3) and np.array_equal(a, b) and not np.array_equal(a, c)
    for column, name in enumerate(("x", "y", "yaw")):
        low, high = DEFAULT_RESET_RANGES[name]
        assert low <= a[:, column].min() and a[:, column].max() <= high
    assert len({tuple(row) for row in a}) == 100


def test_zero_ranges_reproduce_nominal_reset():
    env = fake_env(4)
    zero = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    reset_root_state_from_table(env, torch.arange(4), 1, ranges=zero)
    asset = env.scene.asset
    expected = asset.data.default_root_state[:, :3] + env.scene.env_origins
    assert torch.equal(asset.pose[:, :3], expected)
    assert torch.equal(asset.pose[:, 3:7], asset.data.default_root_state[:, 3:7])
    assert torch.equal(asset.velocity, torch.zeros(4, 6))


def test_counters_slots_and_yaw_quaternion():
    env = fake_env(4)
    ranges = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.1, 0.1)}
    reset_root_state_from_table(env, torch.arange(4), 7, namespace=3, slots=(10, 11, 12, 13), ranges=ranges)
    reset_root_state_from_table(env, torch.tensor([1, 3]), 7, namespace=3, slots=(10, 11, 12, 13), ranges=ranges)
    assert env._ea_last_reset_keys == {1: (3, 11, 1), 3: (3, 13, 1)}
    assert env._ea_reset_episode_counter.tolist() == [1, 2, 1, 2]
    quat = env.scene.asset.pose[0, 3:7]
    assert torch.allclose(quat, torch.tensor([math.cos(0.05), 0.0, 0.0, math.sin(0.05)]), atol=1e-7)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_reset_table.py -q`
Expected: FAIL，`ModuleNotFoundError: rl_training.ea_lpl.reset_table`。

- [ ] **Step 3: 实现**

`source/rl_training/rl_training/ea_lpl/reset_table.py`：

```python
"""Deterministic perturbed initial states keyed by (seed, namespace, slot, episode).

The offsets do not consume the simulator's global random stream, so two arms
with the same keys start from identical states.
"""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_RESET_RANGES = {"x": (-0.2, 0.2), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}
NAMESPACES = {"nominal": 0, "p0_eval": 1, "p1_grid": 2, "p1_confirm": 3,
              "p2_train": 4, "p2_eval": 5, "p2_flat": 6, "sealed": 9}


def initial_state_offsets(table_seed, namespace, slots, episodes, ranges=DEFAULT_RESET_RANGES):
    if len(slots) != len(episodes):
        raise ValueError("slots and episodes must have the same length")
    bounds = (ranges["x"], ranges["y"], ranges["yaw"])
    rows = []
    for slot, episode in zip(slots, episodes):
        key = [int(table_seed), int(namespace), int(slot), int(episode)]
        if min(key) < 0:
            raise ValueError("reset keys must be nonnegative")
        draw = np.random.default_rng(key).random(3)
        rows.append([low + (high - low) * value for (low, high), value in zip(bounds, draw)])
    return np.asarray(rows, dtype=np.float64).reshape(len(slots), 3)


def _yaw_quat_wxyz(yaw):
    half = 0.5 * yaw
    zero = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)


def _quat_mul_wxyz(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), dim=-1)


def reset_root_state_from_table(env, env_ids, table_seed, namespace=0, slots=None, ranges=None, asset_cfg=None):
    """Isaac Lab reset event: default root state + env origin + tabulated (x, y, yaw) offset, zero velocity."""
    asset = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    default = asset.data.default_root_state
    ids = torch.as_tensor(env_ids, device=default.device, dtype=torch.long).reshape(-1)
    id_list = ids.cpu().tolist()
    counters = getattr(env, "_ea_reset_episode_counter", None)
    if counters is None:
        counters = torch.zeros(env.num_envs, dtype=torch.long)
        env._ea_reset_episode_counter = counters
    slot_list = [i if slots is None else int(slots[i]) for i in id_list]
    episodes = counters[id_list].tolist()
    counters[id_list] += 1
    offsets = torch.as_tensor(
        initial_state_offsets(table_seed, namespace, slot_list, episodes, ranges or DEFAULT_RESET_RANGES),
        dtype=default.dtype, device=default.device)
    root = default[ids].clone()
    positions = root[:, 0:3] + env.scene.env_origins[ids]
    positions[:, 0] += offsets[:, 0]
    positions[:, 1] += offsets[:, 1]
    orientations = _quat_mul_wxyz(root[:, 3:7], _yaw_quat_wxyz(offsets[:, 2]))
    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=ids)
    asset.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:13]), env_ids=ids)
    env._ea_last_reset_keys = {i: (int(namespace), s, e) for i, s, e in zip(id_list, slot_list, episodes)}
```

在 `ea_lpl_env_cfg.py` 顶部加导入，`__post_init__` 末尾加复位表：

```python
from isaaclab.managers import SceneEntityCfg

from rl_training.ea_lpl.reset_table import DEFAULT_RESET_RANGES, reset_root_state_from_table
```

```python
        if self.ea_reset_table_seed is not None:
            slots = None if self.ea_reset_slots is None else tuple(int(s) for s in self.ea_reset_slots)
            if slots is not None and len(slots) != self.source_parallel_sampler_num_envs:
                raise ValueError("ea_reset_slots must list one slot per environment")
            if not 0 <= int(self.ea_reset_namespace) < 9:
                raise ValueError("namespace 9 is reserved for sealed runs")
            self.events.randomize_reset_base.func = reset_root_state_from_table
            self.events.randomize_reset_base.params = {
                "asset_cfg": SceneEntityCfg("robot"),
                "table_seed": int(self.ea_reset_table_seed),
                "namespace": int(self.ea_reset_namespace),
                "slots": slots,
                "ranges": dict(DEFAULT_RESET_RANGES),
            }
```

Isaac Lab 会按函数签名核对事件参数：必填参数 `table_seed` 必须给出，其余参数都有默认值，参数名不能多也不能写错。

- [ ] **Step 4: 运行测试，确认通过**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_reset_table.py -q`
Expected: 3 项 PASS。

- [ ] **Step 5: Commit**

```bash
git add source/rl_training/rl_training/ea_lpl/reset_table.py tests/ea_lpl/test_reset_table.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/ea_lpl_env_cfg.py
git commit -m "feat(ea-lpl): deterministic perturbed initial-state table"
git push
```

---

### Task 3: 运行规格、多回合切分和统计（预计 2 小时）

**Files:**
- Create: `source/rl_training/rl_training/ea_lpl/specs.py`, `source/rl_training/rl_training/ea_lpl/episodes.py`, `source/rl_training/rl_training/ea_lpl/summary.py`
- Test: `tests/ea_lpl/test_specs.py`, `tests/ea_lpl/test_episodes.py`, `tests/ea_lpl/test_summary.py`

**Interfaces:**
- Produces: `SPEC_SCHEMA = "ea_lpl_run_spec_v1"`、`NUM_ENVS = 120`、`EQUAL8`、`P0_TABLE_SEED = 20261106`、`validate_spec(spec) -> dict`、`p0_specs() -> list[dict]`；`episode_records(arrays, events, assignments, *, episodes_per_env=1, control_dt=0.02) -> list[dict]`；`clopper_pearson(k, n, alpha=0.05) -> (low, high)`、`condition_table(records, *, level) -> list[dict]`、`paired_mcnemar(records_a, records_b) -> dict`、`select_target_conditions(rows, *, max_success=0.6, max_safety=0.10, min_progress_when_zero=4.0) -> (eligible, chosen)`。
- 记录字段（后续阶段沿用）：`env_id, episode_index, column, terrain, steps, seconds, completed, hold, termination_reasons, safety_ending, bounds_ending, timeout, forward_progress_m, terminal_lateral_m, lateral_peak_m, terminal_yaw_deg, velocity_rms_error_mps, vx_mps, reward_per_control, projected_fraction`。
- 状态数组 15 列：`vx, vy, vz, 速度误差平方, 离地高度, 前进距离(5), 关节偏差, 关节速度均方根, 倾斜, 横向位置(9), 横滚代理, 俯仰代理, 航向(12), 偏航角速度, 转向指令(14)`。

- [ ] **Step 1: 写失败的测试**

`tests/ea_lpl/test_specs.py`：

```python
import pytest

from rl_training.ea_lpl.specs import EQUAL8, p0_specs, validate_spec


def test_p0_specs_are_valid_and_paired():
    specs = {spec["label"]: validate_spec(spec) for spec in p0_specs()}
    assert set(specs) == {"p0_e0n_l1", "p0_e0p_l1", "p0_e1p_l1", "p0_e1p_l2", "p0_e0p_l2"}
    assert specs["p0_e0n_l1"]["columns"] is None and specs["p0_e0n_l1"]["reset_table_seed"] is None
    for name in ("p0_e0p_l1", "p0_e1p_l1", "p0_e1p_l2", "p0_e0p_l2"):
        assert specs[name]["columns"] == list(EQUAL8)
        assert specs[name]["slots"] == list(range(120))
    assert specs["p0_e1p_l1"]["heading_hold"] and not specs["p0_e0p_l1"]["heading_hold"]


def test_spec_rejections():
    spec = dict(p0_specs()[1])
    with pytest.raises(ValueError):
        validate_spec(dict(spec, namespace=9))
    with pytest.raises(ValueError):
        validate_spec(dict(spec, columns=None, episodes_per_env=2))
    with pytest.raises(ValueError):
        validate_spec(dict(spec, max_episode_steps=1000))
    with pytest.raises(ValueError):
        validate_spec(dict(spec, level="L3"))
```

`tests/ea_lpl/test_episodes.py`：

```python
import numpy as np
import pytest

from rl_training.ea_lpl.episodes import episode_records


def synthetic(done_steps):
    steps, n = 10, len(done_steps)
    done = np.zeros((steps, n), dtype=bool)
    events, assignments = [], []
    for env, ends in enumerate(done_steps):
        start = 0
        for end in ends:
            done[end - 1, env] = True
            events.append(dict(env_id=env, control_step=end, episode_step=end - start, hold=0,
                               completed=False, terms=["source_terrain_tile_bounds"]))
            start = end
        assignments.append(dict(env_id=env, column=0, subterrain="flat"))
    state = np.zeros((steps, n, 15))
    state[:, :, 5] = np.arange(steps)[:, None]
    arrays = dict(done=done, state=state, rewards=np.zeros((steps, n, 3)), actions=np.zeros((steps, n, 5)),
                  columns=np.zeros((steps, n), dtype=np.int64))
    return arrays, events, assignments


def test_records_split_by_done_flags():
    arrays, events, assignments = synthetic([[4, 9], [6, 10]])
    records = episode_records(arrays, events, assignments, episodes_per_env=2)
    assert [(r["env_id"], r["episode_index"], r["steps"]) for r in records] == [(0, 0, 4), (0, 1, 5), (1, 0, 6), (1, 1, 4)]
    assert records[1]["forward_progress_m"] == 8.0
    assert records[0]["bounds_ending"] and not records[0]["completed"]


def test_missing_episode_or_shifted_event_is_rejected():
    arrays, events, assignments = synthetic([[4, 9], [6]])
    with pytest.raises(ValueError):
        episode_records(arrays, events, assignments, episodes_per_env=2)
    events[0] = dict(events[0], control_step=3)
    with pytest.raises(ValueError):
        episode_records(arrays, events, assignments, episodes_per_env=1)
```

`tests/ea_lpl/test_summary.py`：

```python
import math

from rl_training.ea_lpl.summary import clopper_pearson, condition_table, paired_mcnemar, select_target_conditions


def record(env_id, column, terrain, completed, progress=9.3, safety=False, lateral=0.1):
    return dict(env_id=env_id, episode_index=0, column=column, terrain=terrain, steps=1000 + env_id,
                completed=completed, safety_ending=safety, bounds_ending=not completed and not safety, timeout=False,
                forward_progress_m=progress, terminal_lateral_m=lateral, lateral_peak_m=abs(lateral),
                terminal_yaw_deg=1.0)


def test_clopper_pearson_known_values():
    assert math.isclose(clopper_pearson(0, 8)[1], 1 - 0.025 ** (1 / 8), rel_tol=1e-6)
    assert math.isclose(clopper_pearson(8, 8)[0], 0.025 ** (1 / 8), rel_tol=1e-6)
    low, high = clopper_pearson(4, 8)
    assert math.isclose(low, 0.15701, abs_tol=1e-4) and math.isclose(high, 0.84299, abs_tol=1e-4)


def test_condition_table_and_mcnemar():
    first = [record(i, 2, "slope_uphill", False, progress=4.0) for i in range(8)]
    second = [record(i, 2, "slope_uphill", i < 6, progress=9.3 if i < 6 else 4.0) for i in range(8)]
    row = condition_table(second, level="L1")[0]
    assert (row["n"], row["completed"], row["distinct_trajectories"]) == (8, 6, 8)
    test = paired_mcnemar(first, second)
    assert (test["only_first"], test["only_second"]) == (0, 6)
    assert math.isclose(test["p_value"], 2 * 0.5 ** 6)


def test_target_selection_rule():
    rows = [
        dict(level="L1", column=0, terrain="flat", n=8, completed=2, rate=0.25, safety=0, progress_mean_m=8.0),
        dict(level="L1", column=4, terrain="slope_cross_slope_left", n=8, completed=2, rate=0.25, safety=0, progress_mean_m=6.0),
        dict(level="L1", column=5, terrain="slope_cross_slope_right", n=8, completed=0, rate=0.0, safety=0, progress_mean_m=5.4),
        dict(level="L1", column=2, terrain="slope_uphill", n=8, completed=0, rate=0.0, safety=0, progress_mean_m=3.9),
        dict(level="L2", column=6, terrain="stairs_ascending", n=8, completed=3, rate=0.375, safety=0, progress_mean_m=7.0),
        dict(level="L2", column=7, terrain="stairs_descending", n=8, completed=2, rate=0.25, safety=2, progress_mean_m=7.0),
        dict(level="L2", column=3, terrain="slope_downhill", n=8, completed=7, rate=0.875, safety=0, progress_mean_m=9.3),
    ]
    eligible, chosen = select_target_conditions(rows)
    assert {(c["level"], c["column"]) for c in eligible} == {("L1", 4), ("L1", 5), ("L2", 6)}
    assert [(c["level"], c["column"]) for c in chosen] == [("L1", 4), ("L1", 5), ("L2", 6)]
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_specs.py tests/ea_lpl/test_episodes.py tests/ea_lpl/test_summary.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现 `specs.py`**

```python
"""Run specifications for the EA-LPL runner (P0 subset; later phases extend validate_spec)."""
from __future__ import annotations

SPEC_SCHEMA = "ea_lpl_run_spec_v1"
NUM_ENVS = 120
LEVELS = ("L1", "L2")
EQUAL8 = tuple(column for column in range(15) for _ in range(8))
P0_TABLE_SEED = 20261106
REQUIRED = {"schema", "label", "mode", "level", "heading_hold", "columns", "reset_table_seed", "namespace",
            "slots", "episodes_per_env", "max_episode_steps", "codes", "wall_limit_s"}


def _int_list(values, *, length, low, high=None):
    return (isinstance(values, list) and len(values) == length
            and all(type(v) is int and v >= low and (high is None or v < high) for v in values))


def validate_spec(spec):
    missing = REQUIRED - set(spec)
    if missing:
        raise ValueError(f"spec is missing {sorted(missing)}")
    if spec["schema"] != SPEC_SCHEMA or spec["mode"] != "fixed":
        raise ValueError("unknown schema or mode")
    if spec["level"] not in LEVELS or type(spec["heading_hold"]) is not bool:
        raise ValueError("level must be L1/L2 and heading_hold boolean")
    if spec["columns"] is not None and not _int_list(spec["columns"], length=NUM_ENVS, low=0, high=15):
        raise ValueError("columns must list 120 registered column indices")
    if type(spec["episodes_per_env"]) is not int or spec["episodes_per_env"] < 1:
        raise ValueError("episodes_per_env must be a positive integer")
    if spec["episodes_per_env"] > 1 and spec["columns"] is None:
        raise ValueError("multiple episodes per environment require fixed columns")
    if spec["max_episode_steps"] != 2000:
        raise ValueError("the episode limit is fixed at 2000 control steps")
    seed = spec["reset_table_seed"]
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("reset_table_seed must be a nonnegative integer or null")
    if type(spec["namespace"]) is not int or not 0 <= spec["namespace"] < 9:
        raise ValueError("namespace must be in 0..8; 9 is reserved for sealed runs")
    if spec["slots"] is not None and (seed is None or not _int_list(spec["slots"], length=NUM_ENVS, low=0)):
        raise ValueError("slots need a reset table seed and 120 nonnegative integers")
    if spec["codes"] is not None:
        raise ValueError("residual codes are added in P1")
    if type(spec["wall_limit_s"]) not in (int, float) or spec["wall_limit_s"] <= 0:
        raise ValueError("wall_limit_s must be positive")
    return spec


def p0_specs():
    common = dict(schema=SPEC_SCHEMA, mode="fixed", episodes_per_env=1, max_episode_steps=2000,
                  codes=None, wall_limit_s=900)
    perturbed = dict(columns=list(EQUAL8), reset_table_seed=P0_TABLE_SEED, namespace=1, slots=list(range(NUM_ENVS)))
    return [
        dict(common, label="p0_e0n_l1", level="L1", heading_hold=False, columns=None,
             reset_table_seed=None, namespace=0, slots=None),
        dict(common, label="p0_e0p_l1", level="L1", heading_hold=False, **perturbed),
        dict(common, label="p0_e1p_l1", level="L1", heading_hold=True, **perturbed),
        dict(common, label="p0_e1p_l2", level="L2", heading_hold=True, **perturbed),
        dict(common, label="p0_e0p_l2", level="L2", heading_hold=False, **perturbed),
    ]
```

- [ ] **Step 4: 实现 `episodes.py`**

```python
"""Split per-step recordings into episodes using terminal events (terminal sample included)."""
from __future__ import annotations

import numpy as np

SAFETY_TERMS = {"source_training_safety", "illegal_contact", "bad_orientation_2"}
BOUNDS_TERM = "source_terrain_tile_bounds"
TIMEOUT_TERM = "source_training_timeout"


def episode_records(arrays, events, assignments, *, episodes_per_env=1, control_dt=0.02):
    done = np.asarray(arrays["done"], dtype=bool)
    state = np.asarray(arrays["state"], dtype=np.float64)
    rewards = np.asarray(arrays["rewards"], dtype=np.float64)
    actions = np.asarray(arrays["actions"], dtype=np.float64)
    columns = np.asarray(arrays["columns"])
    steps, n = done.shape
    if state.shape[:2] != (steps, n) or len(assignments) != n:
        raise ValueError("arrays and assignments have inconsistent dimensions")
    by_env = {}
    for event in events:
        by_env.setdefault(event["env_id"], []).append(event)
    records = []
    for env_id in range(n):
        ends = sorted(by_env.get(env_id, []), key=lambda e: e["control_step"])
        if [e["control_step"] for e in ends] != (np.flatnonzero(done[:, env_id]) + 1).tolist():
            raise ValueError(f"terminal events and done flags disagree for env {env_id}")
        if len(ends) < episodes_per_env:
            raise ValueError(f"env {env_id} finished {len(ends)} of {episodes_per_env} episodes")
        column = int(assignments[env_id]["column"])
        start = 0
        for index, event in enumerate(ends[:episodes_per_env]):
            end = event["control_step"]
            if event["episode_step"] != end - start:
                raise ValueError("episode length differs from the terminal snapshot")
            window = slice(start, end)
            if not (columns[window, env_id] == column).all():
                raise ValueError("terrain column changed inside an episode")
            values = state[window, env_id]
            reasons = list(event["terms"])
            completed = bool(event["completed"])
            if completed and (event["hold"] < 50 or values[-1, 5] < 9.0):
                raise ValueError("completion lacks progress and stable-hold evidence")
            records.append(dict(
                env_id=env_id, episode_index=index, column=column, terrain=assignments[env_id]["subterrain"],
                steps=end - start, seconds=(end - start) * control_dt, completed=completed, hold=event["hold"],
                termination_reasons=reasons, safety_ending=bool(SAFETY_TERMS.intersection(reasons)),
                bounds_ending=BOUNDS_TERM in reasons, timeout=TIMEOUT_TERM in reasons,
                forward_progress_m=float(values[-1, 5]), terminal_lateral_m=float(values[-1, 9]),
                lateral_peak_m=float(np.abs(values[:, 9]).max()), terminal_yaw_deg=float(np.degrees(values[-1, 12])),
                velocity_rms_error_mps=float(np.sqrt(values[:, 3].mean())), vx_mps=float(values[:, 0].mean()),
                reward_per_control=float(rewards[window, env_id].sum(-1).mean() * control_dt),
                projected_fraction=float(actions[window, env_id, 2].mean())))
            start = end
    return records
```

- [ ] **Step 5: 实现 `summary.py`**

```python
"""Exact binomial intervals, paired tests and the registered P0 target rule (no scipy needed)."""
from __future__ import annotations

import math

import numpy as np


def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def _bisect(condition):
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if condition(middle):
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def clopper_pearson(k, n, alpha=0.05):
    if type(n) is not int or n <= 0 or type(k) is not int or not 0 <= k <= n:
        raise ValueError("need integers 0 <= k <= n, n > 0")
    lower = 0.0 if k == 0 else _bisect(lambda p: 1 - _binom_cdf(k - 1, n, p) >= alpha / 2)
    upper = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) <= alpha / 2)
    return lower, upper


def condition_table(records, *, level):
    rows = []
    for column in sorted({r["column"] for r in records}):
        group = [r for r in records if r["column"] == column]
        n, k = len(group), sum(bool(r["completed"]) for r in group)
        low, high = clopper_pearson(k, n)
        distinct = {(r["steps"], round(r["terminal_lateral_m"], 6), round(r["forward_progress_m"], 6)) for r in group}
        rows.append(dict(
            level=level, column=column, terrain=group[0]["terrain"], n=n, completed=k, rate=k / n,
            ci_low=low, ci_high=high, bounds=sum(r["bounds_ending"] for r in group),
            safety=sum(r["safety_ending"] for r in group), timeouts=sum(r["timeout"] for r in group),
            progress_mean_m=float(np.mean([r["forward_progress_m"] for r in group])),
            lateral_peak_mean_m=float(np.mean([r["lateral_peak_m"] for r in group])),
            terminal_abs_yaw_mean_deg=float(np.mean([abs(r["terminal_yaw_deg"]) for r in group])),
            distinct_trajectories=len(distinct)))
    return rows


def paired_mcnemar(records_a, records_b):
    a = {r["env_id"]: bool(r["completed"]) for r in records_a if r["episode_index"] == 0}
    b = {r["env_id"]: bool(r["completed"]) for r in records_b if r["episode_index"] == 0}
    if set(a) != set(b):
        raise ValueError("paired runs cover different environments")
    only_a = sum(a[i] and not b[i] for i in a)
    only_b = sum(b[i] and not a[i] for i in a)
    m = only_a + only_b
    p_value = 1.0 if m == 0 else min(1.0, 2 * _binom_cdf(min(only_a, only_b), m, 0.5))
    return dict(pairs=len(a), both=sum(a[i] and b[i] for i in a), only_first=only_a, only_second=only_b,
                p_value=p_value)


def _mirror(terrain):
    if terrain.endswith("cross_slope_left"):
        return terrain[: -len("left")] + "right"
    if terrain.endswith("cross_slope_right"):
        return terrain[: -len("right")] + "left"
    return None


def select_target_conditions(rows, *, max_success=0.6, max_safety=0.10, min_progress_when_zero=4.0):
    """Research plan section 5, P0 rule. Rows come from condition_table of heading-hold runs."""
    eligible = []
    for row in rows:
        if row["terrain"] == "flat" or row["rate"] > max_success or row["safety"] / row["n"] > max_safety:
            continue
        if row["completed"] == 0 and row["progress_mean_m"] < min_progress_when_zero:
            continue
        eligible.append(dict(level=row["level"], column=row["column"], terrain=row["terrain"], rate=row["rate"],
                             progress_mean_m=row["progress_mean_m"],
                             mechanism="lateral" if "cross_slope" in row["terrain"] else "sagittal"))
    eligible.sort(key=lambda c: (0 if 0.1 <= c["rate"] <= max_success else 1, abs(c["rate"] - 0.3),
                                 -c["progress_mean_m"], c["level"], c["column"]))
    chosen = []
    lateral = next((c for c in eligible if c["mechanism"] == "lateral"), None)
    if lateral is not None:
        chosen.append(lateral)
        twin = next((c for c in eligible if c["level"] == lateral["level"]
                     and c["terrain"] == _mirror(lateral["terrain"])), None)
        if twin is not None:
            chosen.append(twin)
    sagittal = next((c for c in eligible if c["mechanism"] == "sagittal"), None)
    if sagittal is not None:
        chosen.append(sagittal)
    return eligible, chosen
```

- [ ] **Step 6: 运行测试，确认通过**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl -q`
Expected: 全部 PASS（本任务新增 7 项）。

- [ ] **Step 7: Commit**

```bash
git add source/rl_training/rl_training/ea_lpl/specs.py source/rl_training/rl_training/ea_lpl/episodes.py \
  source/rl_training/rl_training/ea_lpl/summary.py tests/ea_lpl
git commit -m "feat(ea-lpl): run specs, episode splitting and exact statistics"
git push
```

---

### Task 4: 运行脚本与规格文件（预计 3 小时，含一次冒烟运行）

**Files:**
- Create: `scripts/tools/ea_lpl_run.py`, `scripts/tools/ea_lpl_make_specs.py`

**Interfaces:**
- Consumes: Task 1 到 Task 3 的全部接口；`source_adaptation_runtime.OLD/digest/write_json/environment_identity`；`source_delivery_light.record_phase/close_comparison_environment/lifecycle_deadline/check_scans`；`source_live_ppo_supervisor.group_exists/terminate_group`；`rl_training.source_scan.frozen_actor.load_frozen_source_actor`。
- Produces: 命令 `ea_lpl_run.py --spec <json> --output <dir> [--gpu auto|0|1]`。运行目录产物：`spec.json`、`manifest.json`、`runtime.log`、`phases.jsonl`、`progress.json`、`<label>.npz`（键 `rewards, state, columns, actions, done`）、`<label>_events.json`、`result.json`（`status`、`transitions`、`case.records`、`case.initial`）、`execution.json`。P1 会在同一文件里加残差码分支。

- [ ] **Step 1: 写 `ea_lpl_make_specs.py`**

```python
"""Write EA-LPL run specifications as JSON files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training")]
from rl_training.ea_lpl.specs import p0_specs, validate_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("p0",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in p0_specs():
        validate_spec(spec)
        (args.out / f"{spec['label']}.json").write_text(json.dumps(spec, indent=2) + "\n")
        print(args.out / f"{spec['label']}.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 `ea_lpl_run.py`**

```python
"""EA-LPL runner: frozen base policy episodes on the Source scene.

The supervisor chooses a GPU, records a manifest, enforces deadlines and
cleans the process group. The child owns Isaac Sim and records the first
`episodes_per_env` episodes of every environment.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
from source_adaptation_runtime import OLD, digest, environment_identity, write_json  # noqa: E402
from source_delivery_light import check_scans, close_comparison_environment, lifecycle_deadline, record_phase  # noqa: E402
from source_live_ppo_supervisor import group_exists, terminate_group  # noqa: E402

ACTOR_SHA256 = "87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c"
MIN_FREE_MIB = 12288
STATE_NAMES = ["vx_mps", "vy_mps", "vz_mps", "velocity_error_squared", "height_above_scan_m", "progress_m",
               "joint_deviation_l2", "joint_velocity_rms", "tilt_xy_norm", "lateral_m", "roll_proxy_rad",
               "pitch_proxy_rad", "yaw_rad", "yaw_rate_radps", "yaw_rate_command_radps"]
ACTION_NAMES = ["mean_action_rms", "clipped_joint_fraction", "projected", "paused", "executed_delta_max"]


def make_ea_environment(spec):
    import gymnasium as gym
    import rl_training.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rl_training.source_scan.training_env import SourceTrainingVecEnv
    from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.ea_lpl_env_cfg import (
        DeeproboticsLite3EaLplEnvCfg,
    )
    cfg = DeeproboticsLite3EaLplEnvCfg(
        source_terrain_level=spec["level"], ea_heading_hold=spec["heading_hold"],
        ea_columns=None if spec["columns"] is None else tuple(spec["columns"]),
        ea_reset_table_seed=spec["reset_table_seed"], ea_reset_namespace=spec["namespace"],
        ea_reset_slots=None if spec["slots"] is None else tuple(spec["slots"]))
    cfg.sim.device = "cuda:0"
    raw = gym.make("SourceTerrain-Deeprobotics-Lite3-v0", cfg=cfg)
    return SourceTrainingVecEnv(RslRlVecEnvWrapper(raw, clip_actions=None))


def check_initial_state(base, spec):
    import numpy as np
    from rl_training.ea_lpl.cfg_checks import check_heading_hold, heading_yaw_rate_command
    from rl_training.ea_lpl.reset_table import DEFAULT_RESET_RANGES, initial_state_offsets
    check_heading_hold(base.cfg.commands.base_velocity, enabled=spec["heading_hold"])
    data = base.scene["robot"].data
    n = base.num_envs
    if spec["reset_table_seed"] is None:
        expected = np.zeros((n, 3))
    else:
        slots = spec["slots"] or list(range(n))
        expected = initial_state_offsets(spec["reset_table_seed"], spec["namespace"], slots, [0] * n,
                                         DEFAULT_RESET_RANGES)
        if getattr(base, "_ea_last_reset_keys", None) != {i: (spec["namespace"], slots[i], 0) for i in range(n)}:
            raise ValueError("reset table keys differ from the spec")
    xy = (data.root_link_pos_w[:, :2] - base.scene.env_origins[:, :2] - data.default_root_state[:, :2])
    if np.abs(xy.cpu().double().numpy() - expected[:, :2]).max() > 1e-4:
        raise ValueError("initial x/y offsets differ from the reset table")
    yaw_error = data.heading_w.cpu().double().numpy() - expected[:, 2]
    if np.abs(np.arctan2(np.sin(yaw_error), np.cos(yaw_error))).max() > 1e-4:
        raise ValueError("initial heading differs from the reset table")
    if spec["heading_hold"]:
        # Isaac Lab samples a random yaw rate at resample time; apply the heading loop before the first action.
        base.command_manager.compute(dt=0.0)
        command = base.command_manager.get_command("base_velocity")[:, 2]
        if float((command - heading_yaw_rate_command(data.heading_w)).abs().max()) > 1e-5:
            raise ValueError("heading-hold command does not follow the registered formula")


def collect_fixed(wrapped, actor, spec, folder, app, report):
    import numpy as np
    import torch
    from rl_training.ea_lpl.episodes import episode_records
    base = wrapped.unwrapped
    n = base.num_envs
    if n != 120 or abs(base.step_dt - 0.02) > 1e-9:
        raise ValueError("EA-LPL runs require 120 environments at 50 Hz")
    identity = environment_identity(base)
    if spec["columns"] is not None and [a["column"] for a in identity["assignments"]] != spec["columns"]:
        raise ValueError("initial terrain columns differ from the spec")
    check_initial_state(base, spec)
    observations = wrapped.get_observations()
    manager = base.reward_manager
    original_compute = manager.compute
    captured = {}
    chunks = {key: [] for key in ("rewards", "state", "columns", "actions", "done")}
    events = []
    finished = np.zeros(n, dtype=np.int64)
    target = spec["episodes_per_env"]

    def compute(dt):
        value = original_compute(dt)
        data = base.scene["robot"].data
        velocity = data.root_lin_vel_b
        gravity = data.projected_gravity_b
        command = base.command_manager.get_command("base_velocity")
        ground = base.scene.sensors["height_scanner_base"].data.ray_hits_w[:, :, 2].mean(-1)
        relative = data.root_link_pos_w - base.scene.terrain.env_origins
        w, x, y, z = data.root_link_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        state = torch.stack((
            velocity[:, 0], velocity[:, 1], velocity[:, 2], (velocity[:, :2] - command[:, :2]).square().sum(-1),
            data.root_link_pos_w[:, 2] - ground, relative[:, 0], (data.joint_pos - data.default_joint_pos).norm(dim=-1),
            data.joint_vel.square().mean(-1).sqrt(), gravity[:, :2].norm(dim=-1), relative[:, 1],
            torch.asin(gravity[:, 1].clamp(-1, 1)), torch.asin(gravity[:, 0].clamp(-1, 1)), yaw,
            data.root_ang_vel_b[:, 2], command[:, 2]), dim=-1)
        captured.update(rewards=manager._step_reward.clone(), state=state,
                        columns=base.scene.terrain.terrain_types.clone())
        return value

    manager.compute = compute
    try:
        with torch.inference_mode():
            for step in range(1, spec["max_episode_steps"] * target + 1):
                if not app.is_running():
                    raise RuntimeError("simulator stopped")
                action = actor(observations)
                observations, reward, done, extras = wrapped.step(action)
                report["transitions"] += n
                trace = wrapped.last_trace
                metrics = torch.stack((
                    action.square().mean(-1).sqrt(), (trace.raw_policy_action != trace.base_action).float().mean(-1),
                    trace.decision.projected[:, 0].float(), trace.decision.paused[:, 0].float(),
                    (trace.executed_action - action).abs().amax(-1)), dim=-1)
                for key in ("rewards", "state", "columns"):
                    chunks[key].append(captured[key].cpu().numpy().copy())
                chunks["actions"].append(metrics.cpu().numpy().copy())
                chunks["done"].append(done.cpu().numpy().copy())
                terminal = extras["source_terminal"]
                if terminal is not None:
                    ids = terminal["ids"].cpu().tolist()
                    ending = {k: terminal[k].cpu().tolist() for k in ("steps", "hold", "completed")}
                    terms = {k: v.cpu().tolist() for k, v in terminal["terms"].items()}
                    for j, i in enumerate(ids):
                        events.append(dict(env_id=i, control_step=step, episode_index=int(finished[i]),
                                           episode_step=ending["steps"][j], hold=ending["hold"][j],
                                           completed=ending["completed"][j],
                                           terms=[k for k, v in terms.items() if v[j]]))
                        finished[i] += 1
                check_scans(base)
                if step % 200 == 0:
                    done_envs = int((finished >= target).sum())
                    write_json(folder / "progress.json", dict(step=step, transitions=report["transitions"],
                                                              finished_envs=done_envs))
                    print(f"[ea-lpl] {spec['label']} step={step} finished={done_envs}/{n}", flush=True)
                if (finished >= target).all():
                    break
    finally:
        manager.compute = original_compute
        arrays = {key: np.asarray(value) for key, value in chunks.items()}
        np.savez_compressed(folder / f"{spec['label']}.npz", **arrays)
        write_json(folder / f"{spec['label']}_events.json", events)
    if not all(np.isfinite(arrays[key]).all() for key in ("rewards", "state", "actions")):
        raise FloatingPointError("recorded data contain nonfinite values")
    if (finished < target).any():
        raise RuntimeError("some environments did not finish their registered episodes")
    records = episode_records(arrays, events, identity["assignments"], episodes_per_env=target,
                              control_dt=base.step_dt)
    return dict(label=spec["label"], initial=identity, records=records, controls=len(arrays["done"]),
                state_names=STATE_NAMES, action_names=ACTION_NAMES)


def child(spec_path, folder):
    from rl_training.ea_lpl.specs import validate_spec
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    report = dict(status="starting", transitions=0, label=spec["label"])
    app = wrapped = None
    stack_file = (folder / "python_stacks.log").open("w")
    faulthandler.enable(file=stack_file)

    def phase(name, limit):
        faulthandler.cancel_dump_traceback_later()
        record_phase(folder, name, spec["label"], limit)
        if limit:
            faulthandler.dump_traceback_later(max(1, limit - 5), file=stack_file)

    try:
        phase("application_start", 180)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device="cuda:0").app
        import torch
        from rl_training.source_scan.frozen_actor import load_frozen_source_actor
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        phase("environment_start", 300)
        wrapped = make_ea_environment(spec)
        actor = load_frozen_source_actor(OLD, device="cuda:0")
        phase("controls", 600 * spec["episodes_per_env"])
        report["case"] = collect_fixed(wrapped, actor, spec, folder, app, report)
        report["status"] = "passed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(folder / "result.json", report)
        if wrapped is not None:
            phase("environment_close", 30)
            close_comparison_environment(wrapped)
        if app is not None:
            phase("application_close", 30)
            app.close()
        phase("finished", None)
        faulthandler.disable()
        stack_file.close()
    return 0 if report["status"] == "passed" else 1


def gpu_free_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                  text=True, timeout=10)
    return {int(a): int(b) for a, b in (line.split(",") for line in out.strip().splitlines())}, out


def choose_gpu(request):
    free, _ = gpu_free_mib()
    candidates = [0, 1] if request == "auto" else [int(request)]
    usable = [(free[i], i) for i in candidates if free.get(i, 0) >= MIN_FREE_MIB]
    if not usable:
        raise RuntimeError(f"no GPU among {candidates} has {MIN_FREE_MIB} MiB free: {free}")
    return max(usable)[1]


def supervise(spec_path, folder, gpu_request):
    from rl_training.ea_lpl.specs import validate_spec
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    if digest(OLD) != ACTOR_SHA256:
        raise ValueError("base actor checkpoint changed")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "source", "scripts"], cwd=ROOT, text=True).strip():
        raise RuntimeError("commit source and script changes before a GPU run")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    gpu = choose_gpu(gpu_request)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "spec.json", spec)
    write_json(folder / "manifest.json", dict(
        schema="ea_lpl_run_manifest_v1", label=spec["label"], revision=revision,
        spec_sha256=digest(folder / "spec.json"), actor_sha256=ACTOR_SHA256, gpu=gpu,
        gpu_free_mib_before=gpu_free_mib()[0], python=sys.executable, argv=sys.argv, started_unix=time.time()))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    started, reason = time.monotonic(), None
    with (folder / "runtime.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", __file__, "--child", "--spec", str(folder / "spec.json"),
                                    "--output", str(folder)], cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                now = time.monotonic()
                reason = (f"wall limit {spec['wall_limit_s']}s exceeded" if now - started > spec["wall_limit_s"]
                          else lifecycle_deadline(folder, now))
                if reason:
                    break
                time.sleep(1)
        finally:
            if group_exists(process.pid):
                terminate_group(process, 15)
    result = folder / "result.json"
    status = json.loads(result.read_text())["status"] if result.exists() else "missing"
    receipt = dict(exit_code=process.poll(), deadline_failure=reason, elapsed_s=time.monotonic() - started,
                   process_group_empty=not group_exists(process.pid), status=status, finished_unix=time.time())
    write_json(folder / "execution.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if status == "passed" and reason is None and receipt["exit_code"] == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="auto", choices=("auto", "0", "1"))
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    raise SystemExit(child(args.spec, args.output) if args.child else supervise(args.spec, args.output, args.gpu))
```

- [ ] **Step 3: 生成规格并提交**

```bash
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
$PY scripts/tools/ea_lpl_make_specs.py p0 --out docs/lite3-ea-lpl-p0-20261106/specs
git add scripts/tools/ea_lpl_run.py scripts/tools/ea_lpl_make_specs.py docs/lite3-ea-lpl-p0-20261106/specs
git commit -m "feat(ea-lpl): supervised runner and P0 specifications"
git push
```

- [ ] **Step 4: 冒烟运行（E1p-L1 正式运行同时充当冒烟）**

```bash
EVID=$(cd "$(git rev-parse --git-common-dir)" && pwd)/ea-lpl-evidence/p0
$PY scripts/tools/ea_lpl_run.py --spec docs/lite3-ea-lpl-p0-20261106/specs/p0_e1p_l1.json --output $EVID/p0_e1p_l1 --gpu auto
```

Expected: `execution.json` 中 `status` 为 `passed`、`exit_code` 为 0；`result.json` 里 120 条记录；npz 的 `state[..., 14]`（转向指令）不全为 0。失败时按任务书第 3 节：基础设施问题重试 1 次（输出目录加 `_retry1`），代码问题修复、提交后重跑。

---

### Task 5: P0 正式运行（预计 1 小时）

**Files:** 无新代码。

- [ ] **Step 1: 两张卡并行跑剩余规格**

```bash
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
EVID=$(cd "$(git rev-parse --git-common-dir)" && pwd)/ea-lpl-evidence/p0
S=docs/lite3-ea-lpl-p0-20261106/specs
# E0n 用 GPU0，与 11-05 归档使用同一张卡，便于逐环境复现
$PY scripts/tools/ea_lpl_run.py --spec $S/p0_e0n_l1.json --output $EVID/p0_e0n_l1 --gpu 0 &
$PY scripts/tools/ea_lpl_run.py --spec $S/p0_e0p_l1.json --output $EVID/p0_e0p_l1 --gpu 1 &
wait
$PY scripts/tools/ea_lpl_run.py --spec $S/p0_e1p_l2.json --output $EVID/p0_e1p_l2 --gpu auto &
$PY scripts/tools/ea_lpl_run.py --spec $S/p0_e0p_l2.json --output $EVID/p0_e0p_l2 --gpu auto &
wait
```

两张卡里只有一张满足准入时，改为依次运行。`p0_e0p_l2` 是可选项，GPU 紧张时可以不跑。

Expected: 每个目录 `execution.json` 的 `status` 为 `passed`。L2 建图失败时记录原因，P0 只用 L1 结果继续，不算停止条件。

---

### Task 6: 汇总、目标选择与报告（预计 2 小时）

**Files:**
- Create: `scripts/tools/ea_lpl_p0_summary.py`
- Create: `docs/lite3-ea-lpl-p0-20261106/README.md`, `summary.json`, `conditions.csv`, `p1_inputs.json`, `SHA256SUMS`
- Create: `docs/lite3-ea-lpl-progress-log.md`

**Interfaces:**
- Consumes: Task 3 的统计函数；各运行目录的 `result.json`；归档 `docs/lite3-source-full-episode-comparison-20261105/raw/runtime/result.json`（`cases[0]` 为 original）。
- Produces: `p1_inputs.json` = `{"schema": "ea_lpl_p1_inputs_v1", "targets": [{"level", "column", "terrain", "rate", "progress_mean_m", "mechanism"}]}`，P1 读取。

- [ ] **Step 1: 写汇总脚本**

```python
"""Summarize P0 runs, check the archived reproduction and select P1 targets by the registered rule."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training")]
from rl_training.ea_lpl.summary import condition_table, paired_mcnemar, select_target_conditions  # noqa: E402

ARCHIVE = ROOT / "docs/lite3-source-full-episode-comparison-20261105/raw/runtime/result.json"
RUNS = ("p0_e0n_l1", "p0_e0p_l1", "p0_e1p_l1", "p0_e1p_l2", "p0_e0p_l2")


def load(folder):
    report = json.loads((folder / "result.json").read_text())
    if report["status"] != "passed":
        raise ValueError(f"{folder} did not pass")
    return report["case"]


def reproduction(case):
    archived = json.loads(ARCHIVE.read_text())["cases"][0]
    if archived["label"] != "original":
        raise ValueError("archived original case not found")
    old = {r["env_id"]: (bool(r["completed"]), r["steps"]) for r in archived["first_episodes"]}
    new = {r["env_id"]: (bool(r["completed"]), r["steps"]) for r in case["records"]}
    mismatched = sorted(i for i in old if old[i] != new.get(i))
    return dict(archived_completed=sum(v[0] for v in old.values()), reproduced_completed=sum(v[0] for v in new.values()),
                identical=not mismatched, mismatched_env_ids=mismatched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = {name: load(args.evidence / name) for name in RUNS if (args.evidence / name / "result.json").exists()}
    for required in ("p0_e0n_l1", "p0_e0p_l1", "p0_e1p_l1"):
        if required not in cases:
            raise ValueError(f"missing required run {required}")
    tables = {name: condition_table(case["records"], level=name[-2:].upper()) for name, case in cases.items()}
    heading_effect = {}
    for level in ("l1", "l2"):
        off, on = cases.get(f"p0_e0p_{level}"), cases.get(f"p0_e1p_{level}")
        if off and on:
            heading_effect[level] = paired_mcnemar(off["records"], on["records"])
    hold_rows = tables["p0_e1p_l1"] + tables.get("p0_e1p_l2", [])
    eligible, chosen = select_target_conditions(hold_rows)
    summary = dict(reproduction=reproduction(cases["p0_e0n_l1"]), tables=tables, heading_effect=heading_effect,
                   eligible=eligible, targets=chosen)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "conditions.csv").open("w", newline="") as stream:
        fields = ["run"] + list(tables["p0_e1p_l1"][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, rows in tables.items():
            for row in rows:
                writer.writerow(dict(row, run=name))
    (args.out / "p1_inputs.json").write_text(json.dumps(dict(schema="ea_lpl_p1_inputs_v1", targets=chosen),
                                                        indent=2) + "\n")
    print(json.dumps(dict(reproduction=summary["reproduction"], heading_effect=heading_effect, targets=chosen),
                     indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行汇总**

```bash
$PY scripts/tools/ea_lpl_p0_summary.py --evidence $EVID --out docs/lite3-ea-lpl-p0-20261106
(cd $EVID && find . -type f -print0 | sort -z | xargs -0 sha256sum) > docs/lite3-ea-lpl-p0-20261106/SHA256SUMS
```

Expected: `reproduction.identical` 为 true（逐环境复现归档 87/120）。不一致时最多排查 2 小时：确认 E0n 用的是 GPU0、分配器和命令配置与归档一致，把差异和原因写进报告，然后继续。这一项不是停止条件。

- [ ] **Step 3: 写 README（模板见任务书第 8 节）**

主要结果表固定列：`条件 | 等级 | E0p 完成/8 | E1p 完成/8 | 95% 区间 | 越界 | 安全终止 | 平均前进 m | 平均横向峰值 m | 末端 |偏航| 均值°`。另列：E0n 复现结果、航向闭环效果（配对 McNemar：只有关闭时完成、只有开启时完成、p 值）、合格条件和选中条件。

- [ ] **Step 4: 进度日志与提交**

新建 `docs/lite3-ea-lpl-progress-log.md`，第一条写 P0 结论和下一步。

```bash
git add scripts/tools/ea_lpl_p0_summary.py docs/lite3-ea-lpl-p0-20261106 docs/lite3-ea-lpl-progress-log.md
git commit -m "results(ea-lpl): P0 heading-hold baseline and target conditions"
git push && git ls-remote --heads origin codex/lite3-ea-lpl-20261106
```

- [ ] **Step 5: 衔接**

`p1_inputs.json` 的 `targets` 非空：直接开始 P1。为空：进入决策点 D-0，停下等待用户。
