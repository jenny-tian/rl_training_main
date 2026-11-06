# EA-LPL P1：动作基、改进空间与执行响应诊断 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 P0 选出的目标条件上，测量三个二维动作基在安全过滤器上限内的改进空间，按预登记规则选出 P2 使用的动作基、目标条件和预言码，并给出 C1-a、C1-b 两种设定下执行感知探索（EA）是否值得做的判定。

**Architecture:** 新增残差执行路径（`ResidualSafetyBridge` 和 `SourceTrainingVecEnv.step_residual`），运行脚本支持每个环境一个固定码并记录过滤器回放数据；影子过滤器和开环回放在 CPU 上离线计算执行响应；改进空间的筛选、确认和选择都由纯函数完成。

**Tech Stack:** 同 P0。

**Spec:** `docs/lite3-ea-lpl-research-plan-v1-20261105.md`（D6、D8，第 5 节 P1 规则）；输入为 P0 的 `docs/lite3-ea-lpl-p0-20261106/p1_inputs.json`。

## Global Constraints

- P0 的全部约束继续有效。
- 残差请求 r = V·tanh(u)，从回合第 1 步开始，整回合保持；过滤器按 0.02/步的速率把残差逐步加上去。
- 筛选：每个目标条件 73 个码（3 个动作基 × 24 个非零码 + 1 个零码）× 8 个起点，命名空间 2，起点槽位 0 到 7。
- 确认：命名空间 3，起点槽位 0 到 15。
- 表种子 20261106。σ0 = 0.6，宏动作 Δ = 10 步。
- 判定阈值（不得修改）：筛选入围差值 ≥ 2/8；确认通过差值 ≥ 4/16 且安全终止 ≤ 零码 + 1；EA：C1-a 为 p_proj ≥ 5% 或 p_guard ≥ 5%，C1-b 为宏动作偏差 ≥ 0.3 的决策点比例 ≥ 20% 且预测误差中位数 ≤ 0.3。

## Review Focus

1. 残差路径与原路径不一致：零码时 `step_residual` 必须与 `step` 逐环境结果相同。Task 4 的零码复核运行检查。
2. 影子计算改动了真实过滤器的记忆：Task 3 测试逐字节比较。
3. 码超出过滤器上限，造成未登记的截断：Task 1 测试检查每个动作基在 z ∈ {±1}² 的角点都在上限内。
4. 回放与实际执行不一致，导致 EA 诊断不可信：Task 6 用实际固定码回放，最大误差必须 ≤ 1e-6。
5. 确认阶段复用了筛选起点，造成赢家诅咒：Task 7 的确认规格必须使用命名空间 3。

## File Structure

| 文件 | 操作 | 职责 |
|---|---|---|
| `source/rl_training/rl_training/ea_lpl/basis.py` | 新建 | 三个动作基与批量残差请求 |
| `source/rl_training/rl_training/ea_lpl/residual_bridge.py` | 新建 | 基础动作加残差请求经过安全过滤器 |
| `source/rl_training/rl_training/source_scan/training_env.py` | 修改 | `step` 拆出 `_advance`，新增 `step_residual` |
| `source/rl_training/rl_training/ea_lpl/shadow.py` | 新建 | 不改真实过滤器的影子计算、有限差分 Jacobian |
| `source/rl_training/rl_training/ea_lpl/specs.py` | 修改 | 规格支持残差码 |
| `source/rl_training/rl_training/ea_lpl/headroom.py` | 新建 | 网格码、规格打包、筛选、确认、动作基选择 |
| `source/rl_training/rl_training/ea_lpl/replay.py` | 新建 | 回放保真检查、C1-a 统计、C1-b 开环切换诊断 |
| `scripts/tools/ea_lpl_run.py` | 修改 | 残差码分支与回放记录 |
| `scripts/tools/ea_lpl_make_specs.py` | 修改 | 生成 P1 筛选和确认规格 |
| `scripts/tools/ea_lpl_p1_analysis.py` | 新建 | P1 汇总、EA 判定、写出 `p2_inputs.json` |

---

### Task 1: 动作基（预计 1 小时）

**Files:**
- Create: `source/rl_training/rl_training/ea_lpl/basis.py`
- Test: `tests/ea_lpl/test_basis.py`

**Interfaces:**
- Produces: `LEGS`、`RESIDUAL_LIMITS = (0.24, 0.20, 0.20)`、`BASIS_NAMES = ("sag", "roll", "hipx")`、`basis(name, *, device=None, dtype=torch.float32) -> Tensor[12, 2]`、`stacked_bases(*, device=None, dtype=torch.float32) -> Tensor[3, 12, 2]`、`residual_request(basis_index, u, bases) -> Tensor[n, 12]`。

- [ ] **Step 1: 写失败的测试**

```python
import itertools
import math

import torch

from rl_training.ea_lpl.basis import BASIS_NAMES, RESIDUAL_LIMITS, basis, residual_request, stacked_bases


def test_bases_are_orthogonal_and_inside_the_filter_envelope():
    limits = torch.tensor(RESIDUAL_LIMITS * 4, dtype=torch.float64)
    for name in BASIS_NAMES:
        v = basis(name, dtype=torch.float64)
        assert v.shape == (12, 2)
        gram = v.T @ v
        assert abs(float(gram[0, 1])) < 1e-12 and float(gram[0, 0]) > 0
        assert torch.isclose(gram[0, 0], gram[1, 1])
        for z in itertools.product((-1.0, 1.0), repeat=2):
            assert torch.all((v @ torch.tensor(z, dtype=torch.float64)).abs() <= limits + 1e-12)
    assert torch.count_nonzero(basis("hipx")[[1, 2, 4, 5, 7, 8, 10, 11]]) == 0
    assert torch.count_nonzero(basis("sag")[[0, 3, 6, 9]]) == 0
    assert torch.count_nonzero(basis("roll")[[0, 3, 6, 9]]) == 0


def test_residual_request_uses_each_environment_basis():
    bases = stacked_bases(dtype=torch.float64)
    index = torch.tensor([0, 1, 2])
    u = torch.tensor([[2.0, 0.0], [0.0, -1.0], [1.0, 1.0]], dtype=torch.float64)
    r = residual_request(index, u, bases)
    assert torch.allclose(r[0], bases[0][:, 0] * math.tanh(2.0))
    assert torch.allclose(r[1], -bases[1][:, 1] * math.tanh(1.0))
    assert torch.allclose(r[2], bases[2] @ torch.tanh(u[2]))
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_basis.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现**

```python
"""Two-dimensional residual bases in canonical leg-major Lite3 order (HipX, HipY, Knee per leg).

Columns are scaled to the Source SafetyFilter residual limits, so V @ z with
z in [-1, 1]^2 never exceeds them. The HipY:Knee ratio -1:2 follows C0. The
hipx pair spans both lateral shift and stance width regardless of the left
and right HipX axis conventions.
"""
from __future__ import annotations

import torch

LEGS = ("FL", "FR", "HL", "HR")
RESIDUAL_LIMITS = (0.24, 0.20, 0.20)
BASIS_NAMES = ("sag", "roll", "hipx")


def _synergy(legs):
    column = torch.zeros(12, dtype=torch.float64)
    for leg in legs:
        i = LEGS.index(leg)
        column[3 * i + 1] = -0.10
        column[3 * i + 2] = 0.20
    return column


def _hipx(signs):
    column = torch.zeros(12, dtype=torch.float64)
    for i, sign in enumerate(signs):
        column[3 * i] = 0.12 * sign
    return column


_BASES = {
    "sag": torch.stack((_synergy(("FL", "FR")), _synergy(("HL", "HR"))), dim=1),
    "roll": torch.stack((_synergy(("FL", "HL")), _synergy(("FR", "HR"))), dim=1),
    "hipx": torch.stack((_hipx((1, 1, 1, 1)), _hipx((1, -1, 1, -1))), dim=1),
}


def basis(name, *, device=None, dtype=torch.float32):
    if name not in _BASES:
        raise ValueError(f"unknown basis {name!r}")
    return _BASES[name].to(device=device, dtype=dtype)


def stacked_bases(*, device=None, dtype=torch.float32):
    return torch.stack([basis(name, device=device, dtype=dtype) for name in BASIS_NAMES])


def residual_request(basis_index, u, bases):
    """r_i = V[basis_index[i]] @ tanh(u_i)."""
    return torch.einsum("njk,nk->nj", bases[basis_index], torch.tanh(u))
```

- [ ] **Step 4: 运行测试，确认通过；Commit**

```bash
git add source/rl_training/rl_training/ea_lpl/basis.py tests/ea_lpl/test_basis.py
git commit -m "feat(ea-lpl): residual bases scaled to the safety envelope"
git push
```

---

### Task 2: 残差执行路径（预计 1.5 小时）

**Files:**
- Create: `source/rl_training/rl_training/ea_lpl/residual_bridge.py`
- Modify: `source/rl_training/rl_training/source_scan/training_env.py:152-181`
- Test: `tests/ea_lpl/test_residual_bridge.py`

**Interfaces:**
- Produces: `ResidualSafetyTrace`（字段 `raw_policy_action, base_action, residual_request, requested_action, executed_action, decision, filter_state_before, safety_state`）；`ResidualSafetyBridge(adapter, safety_filter, *, action_clip=clip_lite3_policy_action).step(raw_base_action, residual_request, control_timed_out)`；`SourceTrainingVecEnv.step_residual(base_actions, residual_request)`，返回值与 `step` 相同，`last_trace` 为 `ResidualSafetyTrace`。

- [ ] **Step 1: 写失败的测试**

```python
import pytest
import torch

from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState

LIMITS = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4))


def nominal_state(n=2, contact=False):
    zeros = torch.zeros(n, 12)
    return SafetyState(projected_gravity_xy=torch.zeros(n, 2), base_height=torch.full((n, 1), 0.35),
                       joint_pos=zeros.clone(), joint_vel=zeros.clone(), joint_lower=torch.full((n, 12), -10.0),
                       joint_upper=torch.full((n, 12), 10.0), default_joint_pos=zeros.clone(),
                       action_scale=torch.full((n, 12), 0.25), nonfoot_contact=torch.full((n, 1), contact),
                       control_timed_out=torch.zeros(n, 1, dtype=torch.bool))


class FixedAdapter:
    def __init__(self, state):
        self.state = state

    def safety_state(self, control_timed_out):
        return self.state


def test_residual_ramps_at_the_rate_limit_then_holds():
    bridge = ResidualSafetyBridge(FixedAdapter(nominal_state()), SafetyFilter(LIMITS))
    request = torch.full((2, 12), 0.1)
    timed_out = torch.zeros(2, 1, dtype=torch.bool)
    executed = [bridge.step(torch.zeros(2, 12), request, timed_out).executed_action[0, 0].item() for _ in range(7)]
    assert executed == pytest.approx([0.02, 0.04, 0.06, 0.08, 0.10, 0.10, 0.10])


def test_contact_guard_zeroes_the_residual_and_keeps_the_request():
    bridge = ResidualSafetyBridge(FixedAdapter(nominal_state(contact=True)), SafetyFilter(LIMITS))
    trace = bridge.step(torch.zeros(2, 12), torch.full((2, 12), 0.1), torch.zeros(2, 1, dtype=torch.bool))
    assert torch.equal(trace.executed_action, torch.zeros(2, 12))
    assert torch.equal(trace.residual_request, torch.full((2, 12), 0.1))
    assert bool(trace.decision.projected.all())


def test_nonfinite_request_is_rejected():
    bridge = ResidualSafetyBridge(FixedAdapter(nominal_state()), SafetyFilter(LIMITS))
    request = torch.zeros(2, 12)
    request[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        bridge.step(torch.zeros(2, 12), request, torch.zeros(2, 1, dtype=torch.bool))
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_residual_bridge.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现残差桥**

```python
"""Frozen base action plus a requested residual through the deployment SafetyFilter."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.types import SafetyDecision, SafetyState
from rl_training.source_scan.bridge import FrozenPolicySafetyBridge


@dataclass(frozen=True)
class ResidualSafetyTrace:
    raw_policy_action: torch.Tensor
    base_action: torch.Tensor
    residual_request: torch.Tensor
    requested_action: torch.Tensor
    executed_action: torch.Tensor
    decision: SafetyDecision
    filter_state_before: dict
    safety_state: SafetyState


class ResidualSafetyBridge:
    def __init__(self, adapter, safety_filter, *, action_clip=clip_lite3_policy_action):
        self.adapter = adapter
        self.safety_filter = safety_filter
        self.action_clip = action_clip

    def step(self, raw_base_action, residual_request, control_timed_out):
        FrozenPolicySafetyBridge._require_finite("raw policy action", raw_base_action)
        FrozenPolicySafetyBridge._require_finite("residual request", residual_request)
        if raw_base_action.shape != residual_request.shape or raw_base_action.shape[1] != 12:
            raise ValueError("base actions and residual requests must both be [num_envs, 12]")
        base = self.action_clip(raw_base_action)
        state = self.adapter.safety_state(control_timed_out)
        FrozenPolicySafetyBridge.validate_state(state, base)
        before = self.safety_filter.temporal_snapshot(base)
        decision = self.safety_filter.project(base, residual_request, state)
        FrozenPolicySafetyBridge._require_finite("executed safety action", decision.action)
        return ResidualSafetyTrace(
            raw_policy_action=raw_base_action.detach().clone(), base_action=base.detach().clone(),
            residual_request=residual_request.detach().clone(),
            requested_action=(base + residual_request).detach().clone(),
            executed_action=decision.action.detach().clone(), decision=decision,
            filter_state_before={k: v.detach().clone() for k, v in before.items()}, safety_state=state)
```

- [ ] **Step 4: 拆分 `SourceTrainingVecEnv.step`**

把 `step` 中从 `if bool(trace.decision.paused.any()):` 到 `return observations, rewards, dones, extras` 的部分原样移到新方法 `_advance(self, trace)`，`step` 变成：

```python
    def step(self, actions):
        if actions.shape != (self.num_envs, 12):
            raise ValueError("Source PPO requires [num_envs, 12] sampled actions")
        return self._advance(self.runtime.bridge.step(actions, self.runtime.zeros))

    def step_residual(self, base_actions, residual_request):
        if base_actions.shape != (self.num_envs, 12) or residual_request.shape != (self.num_envs, 12):
            raise ValueError("residual steps require [num_envs, 12] base actions and residual requests")
        if self._residual_bridge is None:
            from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
            self._residual_bridge = ResidualSafetyBridge(self.runtime, self.runtime.safety)
        return self._advance(self._residual_bridge.step(base_actions, residual_request, self.runtime.zeros))
```

`__init__` 末尾加 `self._residual_bridge = None`。`runtime.bridge` 的零残差路径不变，所以已有的 Source 测试和 P0 结果不受影响。

- [ ] **Step 5: 运行测试，确认通过**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl tests/source_scan -q -k "residual or training_env or bridge"`
Expected: 新增 3 项 PASS，已有 `training_env`、`bridge` 测试仍通过。

- [ ] **Step 6: Commit**

```bash
git add source/rl_training/rl_training/ea_lpl/residual_bridge.py source/rl_training/rl_training/source_scan/training_env.py tests/ea_lpl/test_residual_bridge.py
git commit -m "feat(ea-lpl): residual execution path through the deployment safety filter"
git push
```

---

### Task 3: 影子过滤器与执行响应（预计 1.5 小时）

**Files:**
- Create: `source/rl_training/rl_training/ea_lpl/shadow.py`
- Test: `tests/ea_lpl/test_shadow.py`

**Interfaces:**
- Consumes: Task 1 的 `basis`。
- Produces: `filter_from_snapshot(limits, snapshot) -> SafetyFilter`、`request_from_codes(V, u) -> Tensor[n, 12]`（V 为 `[n, 12, 2]`）、`rollout_mean_action(limits, snapshot, base_actions, states, request) -> Tensor[n, 12]`、`finite_difference_jacobian(limits, snapshot, base_actions, states, V, u_ref, delta) -> Tensor[n, 12, 2]`、`parameterization_jacobian(V, u_ref, delta)`、`sensitivity(J) -> Tensor[n, 2, 2]`、`relative_deviation(M, M0) -> Tensor[n]`。

- [ ] **Step 1: 写失败的测试**

```python
import torch

from rl_training.ea_lpl.basis import basis
from rl_training.ea_lpl.shadow import (finite_difference_jacobian, parameterization_jacobian, relative_deviation,
                                       request_from_codes, sensitivity)
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState

LIMITS = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4))


def nominal_state(n, contact=False):
    zeros = torch.zeros(n, 12)
    return SafetyState(projected_gravity_xy=torch.zeros(n, 2), base_height=torch.full((n, 1), 0.35),
                       joint_pos=zeros.clone(), joint_vel=zeros.clone(), joint_lower=torch.full((n, 12), -10.0),
                       joint_upper=torch.full((n, 12), 10.0), default_joint_pos=zeros.clone(),
                       action_scale=torch.full((n, 12), 0.25), nonfoot_contact=torch.full((n, 1), contact),
                       control_timed_out=torch.zeros(n, 1, dtype=torch.bool))


def snapshot_with(n, previous):
    live = SafetyFilter(LIMITS)
    live.temporal_snapshot(torch.zeros(n, 12))
    live._previous_residual = previous.clone()
    return live.temporal_snapshot(torch.zeros(n, 12))


def test_shadow_calls_leave_the_live_filter_unchanged():
    live = SafetyFilter(LIMITS)
    live.project(torch.zeros(2, 12), torch.full((2, 12), 0.05), nominal_state(2))
    before = live.temporal_snapshot(torch.zeros(2, 12))
    v = basis("sag").expand(2, 12, 2)
    finite_difference_jacobian(LIMITS, before, [torch.zeros(2, 12)] * 3, [nominal_state(2)] * 3, v,
                               torch.zeros(2, 2), 0.6)
    after = live.temporal_snapshot(torch.zeros(2, 12))
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_small_delta_at_the_hold_point_equals_the_parameterization():
    v = basis("sag").expand(2, 12, 2)
    u_ref = torch.tensor([[1.0, -0.5], [0.0, 0.0]])
    snapshot = snapshot_with(2, request_from_codes(v, u_ref))
    j = finite_difference_jacobian(LIMITS, snapshot, [torch.zeros(2, 12)], [nominal_state(2)], v, u_ref, 0.05)
    assert torch.allclose(j, parameterization_jacobian(v, u_ref, 0.05), atol=1e-5)


def test_rate_limit_shrinks_the_exploration_scale_response():
    v = basis("sag").expand(1, 12, 2)
    snapshot = snapshot_with(1, torch.zeros(1, 12))
    m = sensitivity(finite_difference_jacobian(LIMITS, snapshot, [torch.zeros(1, 12)], [nominal_state(1)], v,
                                               torch.zeros(1, 2), 0.6))
    m0 = sensitivity(parameterization_jacobian(v, torch.zeros(1, 2), 0.6))
    assert float(relative_deviation(m, m0)[0]) > 0.3


def test_contact_guard_gives_zero_sensitivity():
    v = basis("roll").expand(1, 12, 2)
    snapshot = snapshot_with(1, torch.zeros(1, 12))
    j = finite_difference_jacobian(LIMITS, snapshot, [torch.zeros(1, 12)], [nominal_state(1, contact=True)], v,
                                   torch.zeros(1, 2), 0.05)
    assert torch.count_nonzero(j) == 0
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_shadow.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现**

```python
"""Side-effect-free SafetyFilter evaluations for execution-sensitivity estimates.

Every evaluation starts from a copied pre-decision snapshot; the live filter
is never called. A response over several steps holds one residual request
while replaying recorded base actions and safety states (open loop).
"""
from __future__ import annotations

import torch

from rl_training.online_adaptation.safety import SafetyFilter


def filter_from_snapshot(limits, snapshot):
    shadow = SafetyFilter(limits)
    shadow._previous_residual = snapshot["filter_previous_residual_action"].clone()
    shadow._paused = snapshot["filter_paused"].clone()
    shadow._stable_steps = snapshot["filter_stable_steps"].clone()
    shadow._contact_steps = snapshot["filter_contact_steps"].clone()
    return shadow


def request_from_codes(v, u):
    return torch.einsum("njk,nk->nj", v, torch.tanh(u))


def rollout_mean_action(limits, snapshot, base_actions, states, request):
    if not states or len(base_actions) != len(states):
        raise ValueError("need one base action per recorded state")
    shadow = filter_from_snapshot(limits, snapshot)
    total = torch.zeros_like(base_actions[0])
    for base, state in zip(base_actions, states):
        total = total + shadow.project(base, request, state).action
    return total / len(states)


def finite_difference_jacobian(limits, snapshot, base_actions, states, v, u_ref, delta):
    columns = []
    for i in range(u_ref.shape[1]):
        step = torch.zeros_like(u_ref)
        step[:, i] = delta
        plus = rollout_mean_action(limits, snapshot, base_actions, states, request_from_codes(v, u_ref + step))
        minus = rollout_mean_action(limits, snapshot, base_actions, states, request_from_codes(v, u_ref - step))
        columns.append((plus - minus) / (2 * delta))
    return torch.stack(columns, dim=-1)


def parameterization_jacobian(v, u_ref, delta):
    """The same finite difference without the filter: only the tanh parameterization."""
    scale = (torch.tanh(u_ref + delta) - torch.tanh(u_ref - delta)) / (2 * delta)
    return v * scale[:, None, :]


def sensitivity(j):
    return j.transpose(-1, -2) @ j


def relative_deviation(m, m0):
    return torch.linalg.matrix_norm(m - m0) / torch.linalg.matrix_norm(m0).clamp_min(1e-12)
```

- [ ] **Step 4: 运行测试，确认通过；Commit**

```bash
git add source/rl_training/rl_training/ea_lpl/shadow.py tests/ea_lpl/test_shadow.py
git commit -m "feat(ea-lpl): shadow safety filter and finite-difference execution response"
git push
```

---

### Task 4: 残差码规格、运行脚本分支与零码复核（预计 3 小时，含一次运行）

**Files:**
- Modify: `source/rl_training/rl_training/ea_lpl/specs.py`
- Create: `source/rl_training/rl_training/ea_lpl/headroom.py`（本任务只写网格与打包部分）
- Modify: `scripts/tools/ea_lpl_run.py`, `scripts/tools/ea_lpl_make_specs.py`
- Test: `tests/ea_lpl/test_specs.py`, `tests/ea_lpl/test_headroom.py`

**Interfaces:**
- Produces: 规格字段 `codes = {"basis": [120 个名称], "u": [[a, b] × 120], "code_name": [120 个字符串], "record_replay": bool}`；`GRID_VALUES`、`P1_TABLE_SEED = 20261106`、`GRID_NAMESPACE = 2`、`CONFIRM_NAMESPACE = 3`、`grid_codes() -> list[dict]`（73 项，第一项 `code_name == "zero"`）、`grid_rows(targets, *, starts=8)`、`pack_specs(rows, *, label_prefix, namespace, wall_limit_s=900)`；回放文件 `<label>_replay.npz`，键 `base_action, executed_action, gravity_xy, base_height, joint_vel, nonfoot_contact, control_timed_out`（形状 `[T, 120, ·]`）与 `joint_lower, joint_upper, default_joint_pos, action_scale, request, basis_index, u, episode_end`。

- [ ] **Step 1: 写失败的测试**

`tests/ea_lpl/test_specs.py` 追加：

```python
from rl_training.ea_lpl.headroom import grid_codes, grid_rows, pack_specs


def test_residual_code_specs_validate():
    targets = [dict(level="L1", column=4, terrain="slope_cross_slope_left")]
    specs = pack_specs(grid_rows(targets), label_prefix="p1_grid", namespace=2)
    assert len(specs) == 5
    for spec in specs:
        validate_spec(spec)
        assert len(spec["codes"]["basis"]) == 120
    names = [n for spec in specs for n in spec["codes"]["code_name"] if n != "pad"]
    assert len(names) == 73 * 8 and names.count("zero") == 8


def test_code_specs_need_fixed_columns_and_bounded_codes():
    spec = pack_specs(grid_rows([dict(level="L1", column=4, terrain="x")]), label_prefix="p", namespace=2)[0]
    with pytest.raises(ValueError):
        validate_spec(dict(spec, columns=None))
    bad = dict(spec, codes=dict(spec["codes"], u=[[5.0, 0.0]] * 120))
    with pytest.raises(ValueError):
        validate_spec(bad)
```

`tests/ea_lpl/test_headroom.py`：

```python
from rl_training.ea_lpl.headroom import grid_codes


def test_grid_has_one_zero_and_24_codes_per_basis():
    codes = grid_codes()
    assert len(codes) == 73 and codes[0]["code_name"] == "zero"
    for name in ("sag", "roll", "hipx"):
        assert sum(c["basis"] == name and c["code_name"] != "zero" for c in codes) == 24
    assert len({c["code_name"] for c in codes}) == 73
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_specs.py tests/ea_lpl/test_headroom.py -q`
Expected: FAIL。

- [ ] **Step 3: 规格校验支持残差码**

`specs.py`：顶部加 `import math` 和 `from .basis import BASIS_NAMES`，把 `if spec["codes"] is not None: raise ValueError("residual codes are added in P1")` 换成：

```python
    codes = spec["codes"]
    if codes is not None:
        if set(codes) != {"basis", "u", "code_name", "record_replay"}:
            raise ValueError("codes must define basis, u, code_name and record_replay")
        if spec["columns"] is None:
            raise ValueError("residual codes require fixed columns")
        if len(codes["basis"]) != NUM_ENVS or any(b not in BASIS_NAMES for b in codes["basis"]):
            raise ValueError("codes.basis must name one registered basis per environment")
        if len(codes["u"]) != NUM_ENVS or any(
                len(u) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3 for x in u)
                for u in codes["u"]):
            raise ValueError("codes.u must hold 120 finite pairs with |u| <= 3")
        if len(codes["code_name"]) != NUM_ENVS or type(codes["record_replay"]) is not bool:
            raise ValueError("codes.code_name must have 120 entries and record_replay must be boolean")
```

- [ ] **Step 4: 网格与打包**

`headroom.py`（第一部分）：

```python
"""P1 code grid, run packing and the registered headroom rules."""
from __future__ import annotations

import numpy as np

from .basis import BASIS_NAMES
from .specs import NUM_ENVS, SPEC_SCHEMA

GRID_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0)
P1_TABLE_SEED = 20261106
GRID_NAMESPACE = 2
CONFIRM_NAMESPACE = 3


def grid_codes():
    codes = [dict(code_name="zero", basis="sag", u=[0.0, 0.0])]
    for name in BASIS_NAMES:
        for a in GRID_VALUES:
            for b in GRID_VALUES:
                if (a, b) != (0.0, 0.0):
                    codes.append(dict(code_name=f"{name}:{a:+.0f},{b:+.0f}", basis=name, u=[a, b]))
    return codes


def grid_rows(targets, *, starts=8):
    return [dict(level=t["level"], column=t["column"], terrain=t["terrain"], slot=slot, **code)
            for t in targets for code in grid_codes() for slot in range(starts)]


def pack_specs(rows, *, label_prefix, namespace, wall_limit_s=900):
    """One run per 120 rows of the same level; padding rows run the zero code on fresh slots and are ignored."""
    specs = []
    for level in sorted({r["level"] for r in rows}):
        group = [r for r in rows if r["level"] == level]
        for start in range(0, len(group), NUM_ENVS):
            chunk = group[start:start + NUM_ENVS]
            chunk = chunk + [dict(chunk[0], code_name="pad", basis="sag", u=[0.0, 0.0], slot=1000 + i)
                             for i in range(NUM_ENVS - len(chunk))]
            specs.append(dict(
                schema=SPEC_SCHEMA, label=f"{label_prefix}_{level.lower()}_{start // NUM_ENVS:02d}", mode="fixed",
                level=level, heading_hold=True, columns=[r["column"] for r in chunk], reset_table_seed=P1_TABLE_SEED,
                namespace=namespace, slots=[r["slot"] for r in chunk], episodes_per_env=1, max_episode_steps=2000,
                codes=dict(basis=[r["basis"] for r in chunk], u=[list(map(float, r["u"])) for r in chunk],
                           code_name=[r["code_name"] for r in chunk], record_replay=True),
                wall_limit_s=wall_limit_s))
    return specs
```

`ea_lpl_make_specs.py`：`phase` 选项改为 `("p0", "p1-grid", "p1-zero-check")`，并加参数 `--inputs`：

```python
    elif args.phase == "p1-grid":
        targets = json.loads(args.inputs.read_text())["targets"]
        specs = pack_specs(grid_rows(targets), label_prefix="p1_grid", namespace=GRID_NAMESPACE)
    elif args.phase == "p1-zero-check":
        base = next(s for s in p0_specs() if s["label"] == "p0_e1p_l1")
        specs = [dict(base, label="p1_zero_check_l1", codes=dict(
            basis=["sag"] * 120, u=[[0.0, 0.0]] * 120, code_name=["zero"] * 120, record_replay=False))]
```

（P0 分支保持 `specs = p0_specs()`，最后统一校验并写出。）

- [ ] **Step 5: 运行脚本的残差码分支**

在 `collect_fixed` 中，`check_initial_state(base, spec)` 之后加入：

```python
    request = replay = None
    if spec["codes"] is not None:
        from rl_training.ea_lpl.basis import BASIS_NAMES, residual_request, stacked_bases
        index = torch.tensor([BASIS_NAMES.index(b) for b in spec["codes"]["basis"]], device=base.device)
        codes_u = torch.tensor(spec["codes"]["u"], device=base.device, dtype=torch.float32)
        request = residual_request(index, codes_u, stacked_bases(device=base.device))
        if spec["codes"]["record_replay"]:
            replay = {key: [] for key in ("base_action", "executed_action", "gravity_xy", "base_height",
                                          "joint_vel", "nonfoot_contact", "control_timed_out")}
```

循环内把 `observations, reward, done, extras = wrapped.step(action)` 换成：

```python
                if request is None:
                    observations, reward, done, extras = wrapped.step(action)
                else:
                    observations, reward, done, extras = wrapped.step_residual(action, request)
                trace = wrapped.last_trace
                if replay is not None:
                    state_now = trace.safety_state
                    for key, value in (("base_action", trace.base_action), ("executed_action", trace.executed_action),
                                       ("gravity_xy", state_now.projected_gravity_xy),
                                       ("base_height", state_now.base_height), ("joint_vel", state_now.joint_vel),
                                       ("nonfoot_contact", state_now.nonfoot_contact),
                                       ("control_timed_out", state_now.control_timed_out)):
                        replay[key].append(value.cpu().numpy().copy())
                    if "joint_lower" not in replay:
                        replay.update(joint_lower=state_now.joint_lower.cpu().numpy(),
                                      joint_upper=state_now.joint_upper.cpu().numpy(),
                                      default_joint_pos=state_now.default_joint_pos.cpu().numpy(),
                                      action_scale=state_now.action_scale.cpu().numpy())
```

`metrics` 的第 5 列在有残差码时改为执行动作偏离请求的幅度，没有残差码时保持 P0 的含义：

```python
                    (trace.executed_action - (action if request is None else trace.base_action + request)).abs().amax(-1)
```

返回前（`records` 计算之后）写回放文件，并把码放进结果：

```python
    if replay is not None:
        first_end = np.array([next(r["steps"] for r in records if r["env_id"] == i and r["episode_index"] == 0)
                              for i in range(n)])
        arrays_out = {k: np.asarray(v) for k, v in replay.items()
                      if k not in ("joint_lower", "joint_upper", "default_joint_pos", "action_scale")}
        np.savez_compressed(folder / f"{spec['label']}_replay.npz", **arrays_out,
                            joint_lower=replay["joint_lower"], joint_upper=replay["joint_upper"],
                            default_joint_pos=replay["default_joint_pos"], action_scale=replay["action_scale"],
                            request=request.cpu().numpy(), basis_index=index.cpu().numpy(),
                            u=codes_u.cpu().numpy(), episode_end=first_end)
```

`return dict(...)` 里加 `codes=spec["codes"]`。`child` 里 `report["case"] = collect_fixed(...)` 之后加一行 `report.update(level=spec["level"], slots=spec["slots"])`，Task 7 的 `collect_rows` 要用这两个字段，所以必须在网格运行之前加上。

- [ ] **Step 6: 测试、提交、零码复核运行**

```bash
PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/ea_lpl -q
$PY scripts/tools/ea_lpl_make_specs.py p1-zero-check --out docs/lite3-ea-lpl-p1-20261109/specs
git add source/rl_training/rl_training/ea_lpl scripts/tools tests/ea_lpl docs/lite3-ea-lpl-p1-20261109/specs
git commit -m "feat(ea-lpl): residual code runs with filter replay recording"
git push
EVID1=$(cd "$(git rev-parse --git-common-dir)" && pwd)/ea-lpl-evidence/p1
$PY scripts/tools/ea_lpl_run.py --spec docs/lite3-ea-lpl-p1-20261109/specs/p1_zero_check_l1.json --output $EVID1/p1_zero_check_l1 --gpu auto
```

复核：比较 `p1_zero_check_l1` 与 P0 `p0_e1p_l1` 的记录，每个环境的 `completed`、`steps`、`forward_progress_m` 必须完全相同：

```bash
$PY - "$EVID1/p1_zero_check_l1/result.json" "$(dirname $EVID1)/p0/p0_e1p_l1/result.json" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1]))["case"]["records"]
b = json.load(open(sys.argv[2]))["case"]["records"]
key = lambda r: (r["env_id"], r["completed"], r["steps"], r["forward_progress_m"])
diff = [x for x, y in zip(map(key, a), map(key, b)) if x != y]
print("identical" if not diff and len(a) == len(b) == 120 else f"DIFFERENT: {diff[:5]}")
EOF
```

Expected: `identical`。不一致说明残差路径改变了基础行为，停下排查（这是代码问题，不进入网格运行）。

---

### Task 5: 筛选网格运行（预计 1.5 小时，以运行时间为主）

- [ ] **Step 1: 生成规格并提交**

```bash
$PY scripts/tools/ea_lpl_make_specs.py p1-grid --inputs docs/lite3-ea-lpl-p0-20261106/p1_inputs.json --out docs/lite3-ea-lpl-p1-20261109/specs
git add docs/lite3-ea-lpl-p1-20261109/specs && git commit -m "experiment(ea-lpl): P1 grid specifications" && git push
```

目标条件为 3 个时共 15 个规格（每个条件 584 行，按等级打包，最后一个规格有补位行）。

- [ ] **Step 2: 两张卡并行运行**

规格按奇偶分到两张卡，避免两个监督进程同时选中同一张卡：

```bash
D=docs/lite3-ea-lpl-p1-20261109
ls $D/specs/p1_grid_*.json > /tmp/p1_grid_specs.txt
awk 'NR%2==1' /tmp/p1_grid_specs.txt | while read s; do
  $PY scripts/tools/ea_lpl_run.py --spec $s --output $EVID1/$(basename $s .json) --gpu 0; done &
awk 'NR%2==0' /tmp/p1_grid_specs.txt | while read s; do
  $PY scripts/tools/ea_lpl_run.py --spec $s --output $EVID1/$(basename $s .json) --gpu 1; done &
wait
```

某张卡显存不满足准入时，监督进程直接报错；把这些规格改到另一张卡依次重跑。基础设施失败的运行，输出目录加 `_retry1` 重跑一次。

Expected: 所有规格 `status` 为 `passed`。

---

### Task 6: 回放诊断（预计 2 小时）

**Files:**
- Create: `source/rl_training/rl_training/ea_lpl/replay.py`
- Test: `tests/ea_lpl/test_replay.py`

**Interfaces:**
- Consumes: Task 1 的 `stacked_bases/residual_request`，Task 3 的影子函数，Task 4 的回放文件键。
- Produces: `state_at(rec, t) -> SafetyState`、`replay_fidelity(rec, limits) -> float`（最大绝对误差）、`fixed_code_execution_stats(rec, *, ramp_steps=10, tilt_guard=0.55) -> dict(p_proj, p_guard, samples)`、`switching_diagnostics(rec, v, limits, *, sigma0=0.6, interval=10, seed=0, threshold=0.3) -> dict(decisions, informative_fraction_single, informative_fraction_macro, median_prediction_error, projection_activation)`。

- [ ] **Step 1: 写失败的测试**

```python
import numpy as np
import torch

from rl_training.ea_lpl.basis import basis
from rl_training.ea_lpl.replay import fixed_code_execution_stats, replay_fidelity, switching_diagnostics
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits

LIMITS = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4))


def synthetic_recording(steps=40, n=3, contact_rows=()):
    rec = dict(base_action=np.zeros((steps, n, 12), np.float32), gravity_xy=np.zeros((steps, n, 2), np.float32),
               base_height=np.full((steps, n, 1), 0.35, np.float32), joint_vel=np.zeros((steps, n, 12), np.float32),
               nonfoot_contact=np.zeros((steps, n, 1), bool), control_timed_out=np.zeros((steps, n, 1), bool),
               joint_lower=np.full((n, 12), -10.0, np.float32), joint_upper=np.full((n, 12), 10.0, np.float32),
               default_joint_pos=np.zeros((n, 12), np.float32), action_scale=np.full((n, 12), 0.25, np.float32),
               basis_index=np.zeros(n, np.int64), u=np.tile(np.array([[1.0, 0.0]], np.float32), (n, 1)),
               episode_end=np.full(n, steps))
    for row in contact_rows:
        rec["nonfoot_contact"][:, row] = True
    request = torch.einsum("njk,nk->nj", basis("sag").expand(n, 12, 2), torch.tanh(torch.as_tensor(rec["u"])))
    rec["request"] = request.numpy()
    live = SafetyFilter(LIMITS)
    from rl_training.ea_lpl.replay import state_at
    rec["executed_action"] = np.stack([live.project(torch.zeros(n, 12), request, state_at(rec, t)).action.numpy()
                                       for t in range(steps)])
    return rec


def test_fidelity_and_fixed_code_statistics():
    rec = synthetic_recording(contact_rows=(2,))
    assert replay_fidelity(rec, LIMITS) <= 1e-6
    stats = fixed_code_execution_stats(rec)
    assert abs(stats["p_guard"] - 1 / 3) < 1e-9 and abs(stats["p_proj"] - 1 / 3) < 1e-9


def test_switching_diagnostics_detect_rate_limits():
    result = switching_diagnostics(synthetic_recording(steps=60), basis("sag"), LIMITS, seed=1)
    assert result["decisions"] > 0
    assert 0.0 <= result["informative_fraction_macro"] <= 1.0
    assert result["projection_activation"] > 0.0
```

说明：第三行一直处于非足端接触，残差被清零，所以它在斜坡后的全部步数都算作保护清零和执行不一致；另外两行在前 10 步之后执行残差等于请求。

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_replay.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现**

```python
"""Open-loop replay of recorded safety states for execution-response diagnostics."""
from __future__ import annotations

import numpy as np
import torch

from rl_training.online_adaptation.safety import SafetyFilter
from rl_training.online_adaptation.types import SafetyState

from .shadow import (finite_difference_jacobian, parameterization_jacobian, relative_deviation,
                     request_from_codes, rollout_mean_action, sensitivity)


def state_at(rec, t):
    n = rec["base_action"].shape[1]
    return SafetyState(
        projected_gravity_xy=torch.as_tensor(rec["gravity_xy"][t]), base_height=torch.as_tensor(rec["base_height"][t]),
        joint_pos=torch.zeros(n, 12), joint_vel=torch.as_tensor(rec["joint_vel"][t]),
        joint_lower=torch.as_tensor(rec["joint_lower"]), joint_upper=torch.as_tensor(rec["joint_upper"]),
        default_joint_pos=torch.as_tensor(rec["default_joint_pos"]), action_scale=torch.as_tensor(rec["action_scale"]),
        nonfoot_contact=torch.as_tensor(rec["nonfoot_contact"][t]),
        control_timed_out=torch.as_tensor(rec["control_timed_out"][t]))


def replay_fidelity(rec, limits):
    """Replay the recorded constant request from zero memory; compare with the executed actions."""
    request = torch.as_tensor(rec["request"])
    end = np.asarray(rec["episode_end"])
    live = SafetyFilter(limits)
    worst = 0.0
    for t in range(int(end.max())):
        action = live.project(torch.as_tensor(rec["base_action"][t]), request, state_at(rec, t)).action
        alive = torch.as_tensor(t < end)
        diff = (action - torch.as_tensor(rec["executed_action"][t])).abs().amax(-1)[alive]
        if len(diff):
            worst = max(worst, float(diff.max()))
    return worst


def fixed_code_execution_stats(rec, *, ramp_steps=10, tilt_guard=0.55):
    base, executed, request = rec["base_action"], rec["executed_action"], rec["request"]
    steps = base.shape[0]
    window = (np.arange(steps)[:, None] >= ramp_steps) & (np.arange(steps)[:, None] < np.asarray(rec["episode_end"])[None, :])
    mismatch = np.abs(executed - base - request[None]).max(-1) > 1e-6
    guard = rec["nonfoot_contact"][..., 0] | (np.linalg.norm(rec["gravity_xy"], axis=-1) > tilt_guard)
    return dict(p_proj=float(mismatch[window].mean()), p_guard=float(guard[window].mean()),
                samples=int(window.sum()))


def switching_diagnostics(rec, v, limits, *, sigma0=0.6, interval=10, seed=0, threshold=0.3):
    steps, n = rec["base_action"].shape[:2]
    end = torch.as_tensor(np.asarray(rec["episode_end"]))
    vb = v.to(torch.float32).expand(n, 12, 2)
    generator = torch.Generator().manual_seed(seed)
    live = SafetyFilter(limits)
    u_prev = torch.zeros(n, 2)
    single, macro, errors = [], [], []
    activations = samples = decisions = 0
    for k in range(steps // interval):
        t0 = k * interval
        alive = (t0 + interval) <= end
        if not bool(alive.any()):
            break
        times = range(t0, t0 + interval)
        states = [state_at(rec, t) for t in times]
        bases = [torch.as_tensor(rec["base_action"][t]) for t in times]
        snapshot = live.temporal_snapshot(bases[0])
        u_k = sigma0 * torch.randn(n, 2, generator=generator)
        if k > 0:
            zero = torch.zeros(n, 2)
            m_a = sensitivity(finite_difference_jacobian(limits, snapshot, bases[:1], states[:1], vb, zero, 0.05))
            m_a0 = sensitivity(parameterization_jacobian(vb, zero, 0.05))
            j_b = finite_difference_jacobian(limits, snapshot, bases, states, vb, u_prev, sigma0)
            m_b0 = sensitivity(parameterization_jacobian(vb, u_prev, sigma0))
            actual = (rollout_mean_action(limits, snapshot, bases, states, request_from_codes(vb, u_k))
                      - rollout_mean_action(limits, snapshot, bases, states, request_from_codes(vb, u_prev)))
            predicted = torch.einsum("njk,nk->nj", j_b, u_k - u_prev)
            error = (actual - predicted).norm(dim=-1) / actual.norm(dim=-1).clamp_min(1e-9)
            single += relative_deviation(m_a, m_a0)[alive].tolist()
            macro += relative_deviation(sensitivity(j_b), m_b0)[alive].tolist()
            errors += error[alive].tolist()
            decisions += int(alive.sum())
        request = request_from_codes(vb, u_k)
        for base, state in zip(bases, states):
            accepted = live.project(base, request, state).action - base
            activations += int(((accepted - request).abs().amax(-1) > 1e-6)[alive].sum())
            samples += int(alive.sum())
        u_prev = u_k
    return dict(decisions=decisions,
                informative_fraction_single=float(np.mean(np.asarray(single) >= threshold)) if single else 0.0,
                informative_fraction_macro=float(np.mean(np.asarray(macro) >= threshold)) if macro else 0.0,
                median_prediction_error=float(np.median(errors)) if errors else float("nan"),
                projection_activation=activations / max(samples, 1))
```

- [ ] **Step 4: 运行测试，确认通过；Commit**

```bash
git add source/rl_training/rl_training/ea_lpl/replay.py tests/ea_lpl/test_replay.py
git commit -m "feat(ea-lpl): open-loop filter replay diagnostics"
git push
```

---

### Task 7: 筛选、确认、选择、EA 判定与报告（预计 3 小时，含确认运行）

**Files:**
- Modify: `source/rl_training/rl_training/ea_lpl/headroom.py`（加入判定函数）
- Create: `scripts/tools/ea_lpl_p1_analysis.py`
- Modify: `scripts/tools/ea_lpl_make_specs.py`（`p1-confirm` 分支）
- Test: `tests/ea_lpl/test_headroom.py`
- Create: `docs/lite3-ea-lpl-p1-20261109/README.md`, `screen.csv`, `confirm.csv`, `ea_diagnostics.json`, `p2_inputs.json`, `SHA256SUMS`

**Interfaces:**
- Produces: `collect_rows(results) -> list[dict]`（每行含 `level, column, terrain, code_name, basis, u, slot, completed, safety_ending, forward_progress_m`）、`screen_table(rows)`、`confirmation_candidates(table, *, min_gain=2, top=3)`、`confirmation_rows(candidates, targets, *, starts=16)`、`confirmation_result(table, *, min_gain=4)`、`choose_basis(confirmed, *, starts=16, max_targets=2) -> dict | None`、`ea_decision(c1a, c1b, *, basis_name) -> dict`；`p2_inputs.json` = `{"schema": "ea_lpl_p2_inputs_v1", "basis", "targets": [{"level", "column", "terrain", "oracle_u", "oracle_code", "confirmed_gain"}], "ea": {"c1a_go", "c1b_go", "c1a", "c1b"}}`。

- [ ] **Step 1: 写失败的测试**

`tests/ea_lpl/test_headroom.py` 追加：

```python
from rl_training.ea_lpl.headroom import (choose_basis, confirmation_candidates, confirmation_result, ea_decision,
                                         screen_table)


def row(code_name, basis, completed, progress=5.0, safety=False, column=4, slot=0):
    return dict(level="L1", column=column, terrain="slope_cross_slope_left", code_name=code_name, basis=basis,
                u=[1.0, 0.0], slot=slot, completed=completed, safety_ending=safety, forward_progress_m=progress)


def test_screen_and_candidates():
    rows = [row("zero", "sag", s < 1, slot=s) for s in range(8)]
    rows += [row("roll:+1,+0", "roll", s < 4, slot=s) for s in range(8)]
    rows += [row("roll:+2,+0", "roll", s < 2, slot=s) for s in range(8)]
    rows += [row("pad", "sag", True, slot=1000)]
    table = screen_table(rows)
    assert {r["code_name"]: r["completed"] for r in table} == {"zero": 1, "roll:+1,+0": 4, "roll:+2,+0": 2}
    assert [c["code_name"] for c in confirmation_candidates(table)] == ["roll:+1,+0"]


def test_confirmation_and_basis_choice():
    table = [dict(level="L1", column=4, terrain="t", code_name="zero", basis="sag", u=[0.0, 0.0], n=16,
                  completed=3, safety=0, progress_mean_m=5.0),
             dict(level="L1", column=4, terrain="t", code_name="roll:+1,+0", basis="roll", u=[1.0, 0.0], n=16,
                  completed=8, safety=1, progress_mean_m=7.0),
             dict(level="L1", column=4, terrain="t", code_name="sag:+1,+0", basis="sag", u=[1.0, 0.0], n=16,
                  completed=6, safety=3, progress_mean_m=6.0)]
    confirmed = confirmation_result(table)
    flags = {r["code_name"]: r["confirmed"] for r in confirmed}
    assert flags == {"roll:+1,+0": True, "sag:+1,+0": False}
    choice = choose_basis(confirmed)
    assert choice["basis"] == "roll" and choice["targets"][0]["oracle_u"] == [1.0, 0.0]
    assert choose_basis([dict(r, confirmed=False) for r in confirmed]) is None


def test_ea_decision_thresholds():
    decision = ea_decision(dict(p_proj=0.01, p_guard=0.02),
                           dict(sag=dict(informative_fraction_macro=0.25, median_prediction_error=0.2)),
                           basis_name="sag")
    assert decision == dict(c1a_go=False, c1b_go=True)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_headroom.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现判定函数（追加到 `headroom.py`）**

```python
def collect_rows(results):
    """results: list of result.json dicts from code runs; returns one row per environment's first episode."""
    rows = []
    for report in results:
        case = report["case"]
        spec_codes = case["codes"]
        for record in case["records"]:
            if record["episode_index"] != 0:
                continue
            i = record["env_id"]
            rows.append(dict(level=report["level"], column=record["column"], terrain=record["terrain"],
                             code_name=spec_codes["code_name"][i], basis=spec_codes["basis"][i],
                             u=spec_codes["u"][i], slot=report["slots"][i], completed=bool(record["completed"]),
                             safety_ending=bool(record["safety_ending"]),
                             forward_progress_m=record["forward_progress_m"]))
    return rows


def screen_table(rows):
    groups = {}
    for r in rows:
        if r["code_name"] != "pad":
            groups.setdefault((r["level"], r["column"], r["code_name"]), []).append(r)
    table = []
    for (level, column, name), group in sorted(groups.items()):
        table.append(dict(level=level, column=column, terrain=group[0]["terrain"], code_name=name,
                          basis=group[0]["basis"], u=group[0]["u"], n=len(group),
                          completed=sum(g["completed"] for g in group), safety=sum(g["safety_ending"] for g in group),
                          progress_mean_m=float(np.mean([g["forward_progress_m"] for g in group]))))
    return table


def confirmation_candidates(table, *, min_gain=2, top=3):
    zero = {(r["level"], r["column"]): r for r in table if r["code_name"] == "zero"}
    picks = []
    for key, z in sorted(zero.items()):
        for name in BASIS_NAMES:
            rows = [r for r in table if (r["level"], r["column"]) == key and r["basis"] == name
                    and r["code_name"] != "zero" and r["completed"] - z["completed"] >= min_gain]
            rows.sort(key=lambda r: (-r["completed"], -r["progress_mean_m"], r["code_name"]))
            picks += rows[:top]
    return picks


def confirmation_rows(candidates, targets, *, starts=16):
    rows = [dict(level=t["level"], column=t["column"], terrain=t["terrain"], code_name="zero", basis="sag",
                 u=[0.0, 0.0], slot=s) for t in targets for s in range(starts)]
    rows += [dict(level=c["level"], column=c["column"], terrain=c["terrain"], code_name=c["code_name"],
                  basis=c["basis"], u=c["u"], slot=s) for c in candidates for s in range(starts)]
    return rows


def confirmation_result(table, *, min_gain=4):
    zero = {(r["level"], r["column"]): r for r in table if r["code_name"] == "zero"}
    result = []
    for r in table:
        if r["code_name"] == "zero":
            continue
        z = zero[(r["level"], r["column"])]
        gain = r["completed"] - z["completed"]
        result.append(dict(r, zero_completed=z["completed"], gain=gain,
                           confirmed=gain >= min_gain and r["safety"] <= z["safety"] + 1))
    return result


def choose_basis(confirmed, *, starts=16, max_targets=2):
    best = {}
    for r in confirmed:
        if not r["confirmed"]:
            continue
        key = (r["basis"], r["level"], r["column"])
        if key not in best or (r["gain"], r["progress_mean_m"]) > (best[key]["gain"], best[key]["progress_mean_m"]):
            best[key] = r
    scores = {name: sum(r["gain"] / starts for (b, _, _), r in best.items() if b == name) for name in BASIS_NAMES}
    top = max(scores.values())
    if top <= 0:
        return None
    chosen = "sag" if scores["sag"] == top else next(name for name in BASIS_NAMES if scores[name] == top)
    rows = sorted((r for (b, _, _), r in best.items() if b == chosen), key=lambda r: (-r["gain"], -r["progress_mean_m"]))
    return dict(basis=chosen, scores=scores, targets=[
        dict(level=r["level"], column=r["column"], terrain=r["terrain"], oracle_u=r["u"], oracle_code=r["code_name"],
             confirmed_gain=r["gain"] / starts) for r in rows[:max_targets]])


def ea_decision(c1a, c1b, *, basis_name):
    stats = c1b[basis_name]
    return dict(c1a_go=c1a["p_proj"] >= 0.05 or c1a["p_guard"] >= 0.05,
                c1b_go=stats["informative_fraction_macro"] >= 0.20 and stats["median_prediction_error"] <= 0.30)
```

`collect_rows` 读取的 `report["level"]`、`report["slots"]` 已在 Task 4 Step 5 写入结果文件。

- [ ] **Step 4: 分析脚本**

`scripts/tools/ea_lpl_p1_analysis.py` 有两个子命令：

1. `screen --evidence <dir> --inputs p1_inputs.json --out <docs dir>`：读取所有 `p1_grid_*` 的 `result.json`，`collect_rows` → `screen_table` → 写 `screen.csv`；`confirmation_candidates` → `confirmation_rows(candidates, targets)` → `pack_specs(..., label_prefix="p1_confirm", namespace=CONFIRM_NAMESPACE)` → 写到 `<docs dir>/specs/`；对每个 `p1_grid_*_replay.npz` 计算 `replay_fidelity`、`fixed_code_execution_stats`，以及三个动作基的 `switching_diagnostics`（种子 0），合并后写 `ea_diagnostics.json`。合并方式：`p_proj`、`p_guard` 按 `samples` 加权平均；`informative_fraction_*` 按 `decisions` 加权平均；`median_prediction_error` 取各文件中位数的中位数。任一文件 `replay_fidelity > 1e-6` 时报错退出。
2. `final --evidence <dir> --out <docs dir>`：读取所有 `p1_confirm_*` 的 `result.json`，`collect_rows` → `screen_table` → `confirmation_result` → 写 `confirm.csv`；`choose_basis`；读 `ea_diagnostics.json` 调 `ea_decision`；写 `p2_inputs.json`。`choose_basis` 返回 None 时写 `p2_inputs.json` 的 `basis` 为 null 并打印 `DECISION POINT D-1`。

`ea_lpl_make_specs.py` 不需要 `p1-confirm` 分支：确认规格由 `screen` 子命令直接写出。

- [ ] **Step 5: 运行筛选分析、确认运行和最终分析**

```bash
D=docs/lite3-ea-lpl-p1-20261109
$PY scripts/tools/ea_lpl_p1_analysis.py screen --evidence $EVID1 --inputs docs/lite3-ea-lpl-p0-20261106/p1_inputs.json --out $D
git add $D && git commit -m "experiment(ea-lpl): P1 screen results and confirmation specifications" && git push
ls $D/specs/p1_confirm_*.json > /tmp/p1_confirm_specs.txt
awk 'NR%2==1' /tmp/p1_confirm_specs.txt | while read s; do
  $PY scripts/tools/ea_lpl_run.py --spec $s --output $EVID1/$(basename $s .json) --gpu 0; done &
awk 'NR%2==0' /tmp/p1_confirm_specs.txt | while read s; do
  $PY scripts/tools/ea_lpl_run.py --spec $s --output $EVID1/$(basename $s .json) --gpu 1; done &
wait
$PY scripts/tools/ea_lpl_p1_analysis.py final --evidence $EVID1 --out $D
(cd $EVID1 && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $D/SHA256SUMS
```

筛选没有任何入围码时，`confirmation_rows` 只含零码，确认运行照常进行（用于记录零码的基线），最终分析返回 D-1。

- [ ] **Step 6: README、进度日志、提交**

主要结果表固定列：`条件 | 动作基 | 最佳筛选码 | 筛选 完成/8（零码 完成/8） | 确认 完成/16（零码 完成/16） | 安全终止 | 是否确认`。另列：动作基得分与选择、P2 目标条件和预言码、零码复核结果、回放保真最大误差、EA 判定（p_proj、p_guard、两种定义的偏差比例、预测误差中位数）、每个条件每个动作基的 5×5 完成数热图（`headroom_heatmaps.png`）。

```bash
git add scripts/tools/ea_lpl_p1_analysis.py source/rl_training/rl_training/ea_lpl tests/ea_lpl $D docs/lite3-ea-lpl-progress-log.md
git commit -m "results(ea-lpl): P1 headroom, basis selection and EA prerequisites"
git push && git ls-remote --heads origin codex/lite3-ea-lpl-20261106
```

- [ ] **Step 7: 衔接**

`p2_inputs.json` 的 `basis` 非空：直接开始 P2。为空：进入决策点 D-1，停下等待用户。EA 判定只影响 P3、P5 是否执行，不影响 P2。
