# EA-LPL P2：C1-a 并行 session 在线学习 Implementation Plan

> **已取代（2026-11-09）：** 本计划没有执行。用户要求先让基础策略掌握六种 Source 地形，再在线学习更难的地形，见 [研究方案 v2](../../lite3-ea-lpl-research-plan-v2-20261109.md)。C1-a session 的设计并入 v2 的阶段三。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 P1 选出的动作基和目标条件上，比较 Frozen（A）与 Updated（B）两组各 120 个并行 session 的 C1-a 在线学习，用独立评估起点上的冻结快照判断在线权重更新是否带来收益。

**Architecture:** 每个并行环境是一个 session，只学一个二维偏置 b（μψ = b，D7）。回合开始时采样 u ~ N(b, σ0²I)，整回合保持 r = V·tanh(u)；每攒满 4 个训练回合做一次留一法基线的 REINFORCE 更新（Adam）。A、B 两组共用同一张探索噪声表和同一批初始状态键。训练结束后用 P0/P1 已有的固定码模式评估各预算节点的快照。

**Tech Stack:** 同 P0。

**Spec:** `docs/lite3-ea-lpl-research-plan-v1-20261105.md`（D3、D7、D9，第 5 节 P2 规则）；输入为 P1 的 `docs/lite3-ea-lpl-p1-20261109/p2_inputs.json`。

## Global Constraints

- P0、P1 的全部约束继续有效。
- 预登记数值（不得修改）：σ0 = 0.6；Adam 学习率 0.3、betas (0.9, 0.999)、eps 1e-8；b 限制在 [−2.5, 2.5]；批大小 4；训练预算 60000 控制步/session，快照节点 0、5000、10000、20000、40000、60000；主判定用 20000 节点；回报 R = clip(前进距离, 0, 9)/9 + 完成 1 − 安全终止 1。
- 噪声表种子 20261115，A、B 相同；初始状态命名空间：训练 4，评估 5，平地保持 6；槽位等于 session 编号。
- 评估：每个 session 4 个评估起点（命名空间 5，回合序号 0 到 3）；平地保持 2 个起点（命名空间 6）。
- 判定：至少一个目标条件平均收益 ≥ 0.10 且 95% bootstrap 区间下限 > 0；没有条件的区间上限 < 0；评估安全终止比例 B ≤ A + 0.02；平地完成率 B ≥ 零码 − 0.05；平地速度均方根误差比 ≤ 1.10。bootstrap 10000 次，种子 0。

## Review Focus

1. A 组被意外更新：`BiasAdapterBank(update=False)` 必须保持 b 和版本号不变，Task 1 测试；分析脚本再核对 A 组最终 b 全为 0。
2. 一个批次混入不同版本采样的回合：`update` 拒绝混合版本，Task 1 测试。
3. 初始状态键与 session 计划错位：每次复位后核对 `(命名空间, 槽位, 回合序号)`，Task 4 运行时检查。
4. 预算节点快照取在更新之前：规则是跨过节点的那个回合结束后、先做它触发的更新，再存快照。Task 2 测试与 Task 4 的调用顺序。
5. 评估复用训练起点：评估规格必须使用命名空间 5 和 6，Task 6 的规格测试。

## File Structure

| 文件 | 操作 | 职责 |
|---|---|---|
| `source/rl_training/rl_training/ea_lpl/adapter.py` | 新建 | 回报、留一法优势、`BiasAdapterBank` |
| `source/rl_training/rl_training/ea_lpl/sessions.py` | 新建 | 噪声表、`SessionBook`（预算、批次、节点） |
| `source/rl_training/rl_training/ea_lpl/stats.py` | 新建 | session 收益、bootstrap、P2 判定 |
| `source/rl_training/rl_training/ea_lpl/specs.py` | 修改 | 训练模式规格；P2 训练与评估规格生成 |
| `scripts/tools/ea_lpl_run.py` | 修改 | `collect_train` |
| `scripts/tools/ea_lpl_make_specs.py` | 修改 | `p2-train`、`p2-smoke`、`p2-eval` |
| `scripts/tools/ea_lpl_p2_analysis.py` | 新建 | P2 汇总、判定、图 |

---

### Task 1: 适配器（预计 1.5 小时）

**Files:**
- Create: `source/rl_training/rl_training/ea_lpl/adapter.py`
- Test: `tests/ea_lpl/test_adapter.py`

**Interfaces:**
- Produces: `episode_return(progress_m, completed, safety_ending) -> float`、`loo_advantages(returns) -> Tensor`、`BiasAdapterBank(num_sessions, *, sigma0=0.6, lr=0.3, betas=(0.9, 0.999), eps=1e-8, bound=2.5, update=True)`，属性 `b[n,2]`、`version[n]`；`sample(session, noise) -> {"u", "mean", "log_prob", "version"}`；`gradient(batch) -> (grad, advantages)`；`update(session, batch) -> dict`。批次元素是 `sample` 的结果加 `"return"` 键。

- [ ] **Step 1: 写失败的测试**

```python
import math

import pytest
import torch

from rl_training.ea_lpl.adapter import BiasAdapterBank, episode_return, loo_advantages


def entry(u, ret, version=0):
    return {"u": torch.tensor(u, dtype=torch.float64), "mean": torch.zeros(2, dtype=torch.float64),
            "log_prob": 0.0, "version": version, "return": ret}


def test_episode_return_registration():
    assert episode_return(9.3, True, False) == 2.0
    assert episode_return(4.5, False, False) == 0.5
    assert math.isclose(episode_return(3.0, False, True), 3.0 / 9.0 - 1.0)


def test_leave_one_out_advantages_sum_to_zero():
    adv = loo_advantages(torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float64))
    assert math.isclose(float(adv.sum()), 0.0, abs_tol=1e-12)
    assert math.isclose(float(adv[0]), 1.0 - 11.0 / 3.0)


def test_gradient_matches_autograd():
    bank = BiasAdapterBank(1)
    generator = torch.Generator().manual_seed(0)
    batch = []
    for ret in (0.2, 1.5, 0.9, 2.0):
        sample = bank.sample(0, torch.randn(2, generator=generator, dtype=torch.float64))
        batch.append(dict(sample, **{"return": ret}))
    grad, adv = bank.gradient(batch)
    mean = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    u = torch.stack([e["u"] for e in batch])
    loss = -(adv * (-0.5 * (((u - mean) / 0.6) ** 2).sum(-1))).mean()
    loss.backward()
    assert torch.allclose(grad, mean.grad)


def test_first_update_moves_toward_the_better_code_by_lr():
    bank = BiasAdapterBank(2)
    info = bank.update(1, [entry([1.0, -1.0], 2.0), entry([-1.0, 1.0], 0.0)])
    assert torch.allclose(bank.b[1], torch.tensor([0.3, -0.3], dtype=torch.float64), atol=1e-6)
    assert torch.equal(bank.b[0], torch.zeros(2, dtype=torch.float64))
    assert info["applied"] and info["version_after"] == 1


def test_frozen_bank_never_changes_and_mixed_versions_are_rejected():
    frozen = BiasAdapterBank(1, update=False)
    info = frozen.update(0, [entry([1.0, -1.0], 2.0), entry([-1.0, 1.0], 0.0)])
    assert not info["applied"] and torch.equal(frozen.b, torch.zeros(1, 2, dtype=torch.float64))
    assert int(frozen.version[0]) == 0
    with pytest.raises(RuntimeError):
        BiasAdapterBank(1).update(0, [entry([1.0, 0.0], 1.0, version=0), entry([0.0, 1.0], 0.0, version=1)])
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_adapter.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现**

```python
"""C1-a: one two-dimensional mean code per session, fixed Gaussian exploration, REINFORCE + Adam.

Sampled codes are data (no gradient through u). The leave-one-out baseline
uses only the other episodes of the same batch, so it does not depend on
the episode's own action.
"""
from __future__ import annotations

import math

import torch


def episode_return(progress_m, completed, safety_ending):
    return (min(max(float(progress_m), 0.0), 9.0) / 9.0 + (1.0 if completed else 0.0)
            - (1.0 if safety_ending else 0.0))


def loo_advantages(returns):
    n = returns.shape[0]
    if n < 2:
        raise ValueError("leave-one-out baselines need at least two episodes")
    return returns - (returns.sum() - returns) / (n - 1)


class BiasAdapterBank:
    def __init__(self, num_sessions, *, sigma0=0.6, lr=0.3, betas=(0.9, 0.999), eps=1e-8, bound=2.5, update=True):
        self.sigma0, self.lr, self.betas, self.eps, self.bound = sigma0, lr, betas, eps, bound
        self.update_enabled = update
        self.b = torch.zeros(num_sessions, 2, dtype=torch.float64)
        self.m = torch.zeros_like(self.b)
        self.v = torch.zeros_like(self.b)
        self.steps = torch.zeros(num_sessions, dtype=torch.long)
        self.version = torch.zeros(num_sessions, dtype=torch.long)

    def sample(self, session, noise):
        mean = self.b[session].clone()
        u = mean + self.sigma0 * noise.to(torch.float64)
        log_prob = float(-0.5 * (((u - mean) / self.sigma0) ** 2).sum()
                         - 2 * math.log(self.sigma0) - math.log(2 * math.pi))
        return {"u": u, "mean": mean, "log_prob": log_prob, "version": int(self.version[session])}

    def gradient(self, batch):
        u = torch.stack([e["u"] for e in batch])
        mean = torch.stack([e["mean"] for e in batch])
        adv = loo_advantages(torch.tensor([e["return"] for e in batch], dtype=torch.float64))
        return -(adv[:, None] * (u - mean)).mean(0) / self.sigma0 ** 2, adv

    def update(self, session, batch):
        versions = {e["version"] for e in batch}
        if versions != {int(self.version[session])}:
            raise RuntimeError("a batch must contain only episodes sampled under the current version")
        grad, adv = self.gradient(batch)
        before = self.b[session].clone()
        if self.update_enabled:
            beta1, beta2 = self.betas
            self.steps[session] += 1
            t = int(self.steps[session])
            self.m[session] = beta1 * self.m[session] + (1 - beta1) * grad
            self.v[session] = beta2 * self.v[session] + (1 - beta2) * grad ** 2
            step = self.lr * (self.m[session] / (1 - beta1 ** t)) / ((self.v[session] / (1 - beta2 ** t)).sqrt() + self.eps)
            self.b[session] = (self.b[session] - step).clamp(-self.bound, self.bound)
            self.version[session] += 1
        return dict(session=session, applied=self.update_enabled, version_before=min(versions),
                    version_after=int(self.version[session]), gradient=grad.tolist(), advantages=adv.tolist(),
                    b_before=before.tolist(), b_after=self.b[session].tolist())
```

- [ ] **Step 4: 运行测试，确认通过；Commit**

```bash
git add source/rl_training/rl_training/ea_lpl/adapter.py tests/ea_lpl/test_adapter.py
git commit -m "feat(ea-lpl): C1-a bias adapter with leave-one-out REINFORCE"
git push
```

---

### Task 2: session 账本与噪声表（预计 1 小时）

**Files:**
- Create: `source/rl_training/rl_training/ea_lpl/sessions.py`
- Test: `tests/ea_lpl/test_sessions.py`

**Interfaces:**
- Produces: `noise_table(seed, num_sessions, max_episodes=400) -> (Tensor[n, max_episodes, 2], sha256)`；`SessionBook(num_sessions, *, train_budget, marks, batch_size)`，方法 `is_training_next(s)`、`start_episode(s, sample) -> "train" | "filler"`、`end_episode(s, steps, episode_return) -> {"kind", "index", "batch", "crossed_marks"}`、`finished(s)`、`all_finished()`、`ledger() -> list[dict]`；属性 `consumed`、`filler_steps`、`episode_index`、`current`、`buffers`。

- [ ] **Step 1: 写失败的测试**

```python
import torch

from rl_training.ea_lpl.sessions import SessionBook, noise_table


def test_budget_batches_marks_and_filler():
    book = SessionBook(2, train_budget=1000, marks=(0, 500, 1000), batch_size=2)
    assert book.start_episode(0, {"u": 1}) == "train"
    out = book.end_episode(0, 300, 0.5)
    assert out["batch"] is None and out["crossed_marks"] == []
    book.start_episode(0, {"u": 2})
    out = book.end_episode(0, 300, 1.0)
    assert [e["return"] for e in out["batch"]] == [0.5, 1.0] and out["crossed_marks"] == [500]
    book.start_episode(0, {"u": 3})
    out = book.end_episode(0, 500, 0.0)
    assert out["crossed_marks"] == [1000] and book.consumed[0] == 1100 and book.finished(0)
    assert book.start_episode(0, None) == "filler"
    out = book.end_episode(0, 50, 0.0)
    assert out["kind"] == "filler" and book.filler_steps[0] == 50 and not book.all_finished()
    assert book.ledger()[0]["discarded_tail_episodes"] == 1


def test_noise_table_is_reproducible():
    a, hash_a = noise_table(5, 3)
    b, hash_b = noise_table(5, 3)
    assert a.shape == (3, 400, 2) and torch.equal(a, b) and hash_a == hash_b
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_sessions.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现**

```python
"""Per-session training budget, fresh batches and snapshot marks (C1-a)."""
from __future__ import annotations

import hashlib

import torch


def noise_table(seed, num_sessions, max_episodes=400):
    generator = torch.Generator().manual_seed(int(seed))
    table = torch.randn((num_sessions, max_episodes, 2), generator=generator, dtype=torch.float64)
    return table, hashlib.sha256(table.numpy().tobytes()).hexdigest()


class SessionBook:
    """A training episode starts only while consumed < budget; the episode that crosses it still counts."""

    def __init__(self, num_sessions, *, train_budget, marks, batch_size):
        marks = tuple(int(m) for m in marks)
        if marks[0] != 0 or list(marks) != sorted(set(marks)) or marks[-1] != train_budget or batch_size < 2:
            raise ValueError("marks must start at 0, increase and end at the budget; batch_size >= 2")
        self.train_budget, self.marks, self.batch_size = train_budget, marks, batch_size
        self.consumed = [0] * num_sessions
        self.filler_steps = [0] * num_sessions
        self.training_episodes = [0] * num_sessions
        self.episode_index = [0] * num_sessions
        self.current = [None] * num_sessions
        self.buffers = [[] for _ in range(num_sessions)]
        self.next_mark = [1] * num_sessions

    def is_training_next(self, session):
        return self.consumed[session] < self.train_budget

    def start_episode(self, session, sample):
        if self.current[session] is not None:
            raise RuntimeError("previous episode has not ended")
        kind = "train" if self.is_training_next(session) else "filler"
        if kind == "train" and sample is None:
            raise ValueError("a training episode needs a sampled code")
        self.current[session] = dict(kind=kind, sample=sample if kind == "train" else None,
                                     index=self.episode_index[session])
        return kind

    def end_episode(self, session, steps, episode_return):
        current = self.current[session]
        if current is None:
            raise RuntimeError("episode ended before it started")
        self.current[session] = None
        self.episode_index[session] += 1
        batch, crossed = None, []
        if current["kind"] == "train":
            self.consumed[session] += int(steps)
            self.training_episodes[session] += 1
            self.buffers[session].append(dict(current["sample"], **{"return": float(episode_return)}))
            if len(self.buffers[session]) == self.batch_size:
                batch, self.buffers[session] = self.buffers[session], []
            while self.next_mark[session] < len(self.marks) and self.consumed[session] >= self.marks[self.next_mark[session]]:
                crossed.append(self.marks[self.next_mark[session]])
                self.next_mark[session] += 1
        else:
            self.filler_steps[session] += int(steps)
        return dict(kind=current["kind"], index=current["index"], batch=batch, crossed_marks=crossed)

    def finished(self, session):
        return self.consumed[session] >= self.train_budget and self.next_mark[session] == len(self.marks)

    def all_finished(self):
        return all(self.finished(s) for s in range(len(self.consumed)))

    def ledger(self):
        return [dict(session=s, train_steps=self.consumed[s], filler_steps=self.filler_steps[s],
                     training_episodes=self.training_episodes[s], discarded_tail_episodes=len(self.buffers[s]))
                for s in range(len(self.consumed))]
```

- [ ] **Step 4: 运行测试，确认通过；Commit**

```bash
git add source/rl_training/rl_training/ea_lpl/sessions.py tests/ea_lpl/test_sessions.py
git commit -m "feat(ea-lpl): session budgets, batches and snapshot marks"
git push
```

---

### Task 3: 统计与判定（预计 1 小时）

**Files:**
- Create: `source/rl_training/rl_training/ea_lpl/stats.py`
- Test: `tests/ea_lpl/test_stats.py`

**Interfaces:**
- Produces: `session_successes(records, *, episodes) -> {env_id: count}`、`session_gains(records_new, records_ref, *, episodes=4) -> {env_id: gain}`、`bootstrap_mean_ci(values, *, resamples=10000, seed=0, alpha=0.05) -> (mean, low, high)`、`p2_gate(per_target, *, safety_rate_new, safety_rate_ref, flat_success_new, flat_success_ref, flat_velocity_ratio) -> dict(passed, checks, improved_targets, harmed_targets)`。

- [ ] **Step 1: 写失败的测试**

```python
from rl_training.ea_lpl.stats import bootstrap_mean_ci, p2_gate, session_gains


def rec(env, index, completed):
    return dict(env_id=env, episode_index=index, completed=completed)


def test_session_gains_use_the_first_n_episodes():
    new = [rec(0, i, i < 3) for i in range(5)] + [rec(1, i, False) for i in range(4)]
    ref = [rec(0, i, i < 1) for i in range(4)] + [rec(1, i, i < 2) for i in range(4)]
    assert session_gains(new, ref, episodes=4) == {0: 0.5, 1: -0.5}


def test_bootstrap_of_constant_values_is_degenerate():
    assert bootstrap_mean_ci([0.25] * 10) == (0.25, 0.25, 0.25)


def test_gate_requires_every_check():
    good = dict(safety_rate_new=0.01, safety_rate_ref=0.0, flat_success_new=0.95, flat_success_ref=0.97,
                flat_velocity_ratio=1.05)
    assert p2_gate({"cross_left": (0.15, 0.05, 0.25)}, **good)["passed"]
    assert not p2_gate({"cross_left": (0.15, -0.01, 0.30)}, **good)["passed"]
    assert not p2_gate({"a": (0.15, 0.05, 0.25), "b": (-0.2, -0.3, -0.1)}, **good)["passed"]
    assert not p2_gate({"a": (0.15, 0.05, 0.25)}, **dict(good, flat_velocity_ratio=1.2))["passed"]
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_stats.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现**

```python
"""Session-level statistics and the registered P2 decision."""
from __future__ import annotations

import numpy as np


def session_successes(records, *, episodes):
    counts = {}
    for r in records:
        if r["episode_index"] < episodes:
            counts[r["env_id"]] = counts.get(r["env_id"], 0) + int(bool(r["completed"]))
    return counts


def session_gains(records_new, records_ref, *, episodes=4):
    new = session_successes(records_new, episodes=episodes)
    ref = session_successes(records_ref, episodes=episodes)
    if set(new) != set(ref):
        raise ValueError("runs cover different sessions")
    return {s: (new[s] - ref[s]) / episodes for s in sorted(new)}


def bootstrap_mean_ci(values, *, resamples=10000, seed=0, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("need at least one value")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def p2_gate(per_target, *, safety_rate_new, safety_rate_ref, flat_success_new, flat_success_ref, flat_velocity_ratio):
    improved = [name for name, (mean, low, _) in per_target.items() if mean >= 0.10 and low > 0]
    harmed = [name for name, (_, _, high) in per_target.items() if high < 0]
    checks = dict(improvement=bool(improved), no_harm=not harmed,
                  safety=safety_rate_new <= safety_rate_ref + 0.02,
                  flat_success=flat_success_new >= flat_success_ref - 0.05,
                  flat_velocity=flat_velocity_ratio <= 1.10)
    return dict(passed=all(checks.values()), checks=checks, improved_targets=improved, harmed_targets=harmed)
```

- [ ] **Step 4: 运行测试，确认通过；Commit**

```bash
git add source/rl_training/rl_training/ea_lpl/stats.py tests/ea_lpl/test_stats.py
git commit -m "feat(ea-lpl): session gains, bootstrap intervals and P2 gate"
git push
```

---

### Task 4: 训练模式与冒烟运行（预计 3 小时，含一次运行）

**Files:**
- Modify: `source/rl_training/rl_training/ea_lpl/specs.py`
- Modify: `scripts/tools/ea_lpl_run.py`, `scripts/tools/ea_lpl_make_specs.py`
- Test: `tests/ea_lpl/test_specs.py`

**Interfaces:**
- Consumes: Task 1 到 Task 3；P1 的 `basis`、`step_residual`；`p2_inputs.json`。
- Produces: 规格字段 `mode: "train"` 与 `train = {"arm", "basis", "sigma0", "lr", "batch_size", "train_budget", "marks", "noise_seed"}`；`P2_REGISTERED`（第 Global Constraints 的数值）；`p2_train_specs(p2_inputs) -> list[dict]`、`p2_smoke_spec(p2_inputs) -> dict`、`p2_eval_specs(p2_inputs, train_results) -> list[dict]`（Task 6 使用）。训练运行目录产物：`episodes.jsonl`、`updates.jsonl`、`result.json`（`case` 含 `noise_sha256`、`snapshots`、`ledger`、`final_b`、`final_version`、`columns`）。

- [ ] **Step 1: 写失败的测试（追加到 `tests/ea_lpl/test_specs.py`）**

```python
from rl_training.ea_lpl.specs import P2_REGISTERED, p2_smoke_spec, p2_train_specs

P2_INPUTS = {"schema": "ea_lpl_p2_inputs_v1", "basis": "roll",
             "targets": [dict(level="L1", column=4, terrain="slope_cross_slope_left", oracle_u=[1.0, 0.0],
                              oracle_code="roll:+1,+0", confirmed_gain=0.31),
                         dict(level="L1", column=5, terrain="slope_cross_slope_right", oracle_u=[0.0, 1.0],
                              oracle_code="roll:+0,+1", confirmed_gain=0.25)]}


def test_train_specs_pair_the_arms():
    specs = p2_train_specs(P2_INPUTS)
    assert [s["train"]["arm"] for s in specs] == ["frozen", "updated"]
    for spec in specs:
        validate_spec(spec)
        assert spec["columns"] == [4] * 60 + [5] * 60
        assert spec["namespace"] == 4 and spec["slots"] == list(range(120))
        assert {k: spec["train"][k] for k in P2_REGISTERED} == P2_REGISTERED
    assert specs[0]["train"]["noise_seed"] == specs[1]["train"]["noise_seed"]


def test_targets_on_different_levels_get_separate_runs():
    mixed = dict(P2_INPUTS, targets=[P2_INPUTS["targets"][0], dict(P2_INPUTS["targets"][1], level="L2")])
    specs = p2_train_specs(mixed)
    assert len(specs) == 4 and {s["level"] for s in specs} == {"L1", "L2"}
    assert all(set(s["columns"]) == {4} or set(s["columns"]) == {5} for s in specs)


def test_smoke_spec_is_small_and_valid():
    spec = validate_spec(p2_smoke_spec(P2_INPUTS))
    assert spec["train"]["train_budget"] == 3000 and spec["train"]["batch_size"] == 2
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_specs.py -q`
Expected: FAIL，`ImportError: P2_REGISTERED`。

- [ ] **Step 3: 规格支持训练模式并生成 P2 规格**

`specs.py`：`REQUIRED` 不变；`validate_spec` 中把 `if spec["schema"] != SPEC_SCHEMA or spec["mode"] != "fixed":` 改为 `if spec["schema"] != SPEC_SCHEMA or spec["mode"] not in ("fixed", "train"):`，并在函数末尾 `return spec` 之前加入：

```python
    train = spec.get("train")
    if spec["mode"] == "fixed" and train is not None:
        raise ValueError("fixed runs must not carry training settings")
    if spec["mode"] == "train":
        keys = {"arm", "basis", "sigma0", "lr", "batch_size", "train_budget", "marks", "noise_seed"}
        if not isinstance(train, dict) or set(train) != keys:
            raise ValueError(f"train settings must define {sorted(keys)}")
        if train["arm"] not in ("frozen", "updated") or train["basis"] not in BASIS_NAMES:
            raise ValueError("unknown arm or basis")
        if spec["codes"] is not None or spec["columns"] is None or spec["reset_table_seed"] is None:
            raise ValueError("training runs need fixed columns, a reset table and no fixed codes")
        marks = train["marks"]
        if (type(train["train_budget"]) is not int or not marks or marks[0] != 0 or marks[-1] != train["train_budget"]
                or marks != sorted(set(marks)) or type(train["batch_size"]) is not int or train["batch_size"] < 2):
            raise ValueError("invalid budget, marks or batch size")
        if not (0 < train["sigma0"] <= 1.0 and 0 < train["lr"] <= 1.0) or type(train["noise_seed"]) is not int:
            raise ValueError("invalid sigma0, learning rate or noise seed")
```

再加入生成函数：

```python
P2_REGISTERED = dict(sigma0=0.6, lr=0.3, batch_size=4, train_budget=60000,
                     marks=[0, 5000, 10000, 20000, 40000, 60000], noise_seed=20261115)
P2_TABLE_SEED = 20261106
P2_TRAIN_NAMESPACE, P2_EVAL_NAMESPACE, P2_FLAT_NAMESPACE = 4, 5, 6


def _level_groups(targets):
    """Same-level targets share one run (sessions split evenly); each other level gets its own run."""
    groups = {}
    for target in targets:
        groups.setdefault(target["level"], []).append(target)
    out = []
    for level, items in sorted(groups.items()):
        share = NUM_ENVS // len(items)
        columns = [t["column"] for t in items for _ in range(share)]
        columns += [items[-1]["column"]] * (NUM_ENVS - len(columns))
        out.append((level, columns))
    return out


def p2_train_specs(p2_inputs, *, registered=None, label_prefix="p2_train", wall_limit_s=10800):
    settings = dict(registered or P2_REGISTERED)
    specs = []
    for level, columns in _level_groups(p2_inputs["targets"]):
        for arm in ("frozen", "updated"):
            specs.append(dict(
                schema=SPEC_SCHEMA, label=f"{label_prefix}_{level.lower()}_{arm}", mode="train", level=level,
                heading_hold=True, columns=columns, reset_table_seed=P2_TABLE_SEED, namespace=P2_TRAIN_NAMESPACE,
                slots=list(range(NUM_ENVS)), episodes_per_env=1, max_episode_steps=2000, codes=None,
                wall_limit_s=wall_limit_s, train=dict(settings, arm=arm, basis=p2_inputs["basis"])))
    return specs


def p2_smoke_spec(p2_inputs):
    smoke = dict(P2_REGISTERED, batch_size=2, train_budget=3000, marks=[0, 1000, 3000])
    spec = p2_train_specs(p2_inputs, registered=smoke, label_prefix="p2_smoke", wall_limit_s=1800)[1]
    return spec
```

（`p2_smoke_spec` 取第一组等级的 updated 组。）

- [ ] **Step 4: 运行脚本的训练分支**

`ea_lpl_run.py`：`child` 中 `phase("controls", ...)` 与采集调用改为

```python
        if spec["mode"] == "train":
            phase("controls", spec["wall_limit_s"] - 600)
            report["case"] = collect_train(wrapped, actor, spec, folder, app, report)
        else:
            phase("controls", 600 * spec["episodes_per_env"])
            report["case"] = collect_fixed(wrapped, actor, spec, folder, app, report)
        report.update(level=spec["level"], slots=spec["slots"])
```

新增 `collect_train`：

```python
def collect_train(wrapped, actor, spec, folder, app, report):
    import torch
    from rl_training.ea_lpl.adapter import BiasAdapterBank, episode_return
    from rl_training.ea_lpl.basis import basis as basis_matrix
    from rl_training.ea_lpl.episodes import SAFETY_TERMS
    from rl_training.ea_lpl.sessions import SessionBook, noise_table
    base = wrapped.unwrapped
    n = base.num_envs
    settings = spec["train"]
    identity = environment_identity(base)
    if [a["column"] for a in identity["assignments"]] != spec["columns"]:
        raise ValueError("initial terrain columns differ from the spec")
    check_initial_state(base, spec)
    v = basis_matrix(settings["basis"], device=base.device)
    bank = BiasAdapterBank(n, sigma0=settings["sigma0"], lr=settings["lr"], update=settings["arm"] == "updated")
    book = SessionBook(n, train_budget=settings["train_budget"], marks=settings["marks"],
                       batch_size=settings["batch_size"])
    noise, noise_sha = noise_table(settings["noise_seed"], n)
    snapshots = {s: {"0": [0.0, 0.0]} for s in range(n)}
    u_current = torch.zeros(n, 2, dtype=torch.float64)
    samples = [None] * n

    def begin(session):
        k = book.episode_index[session]
        if k >= noise.shape[1]:
            raise RuntimeError("noise table exhausted")
        sample = bank.sample(session, noise[session, k]) if book.is_training_next(session) else None
        kind = book.start_episode(session, sample)
        samples[session] = sample
        u_current[session] = sample["u"] if kind == "train" else 0.0

    for s in range(n):
        begin(s)
    observations = wrapped.get_observations()
    manager = base.reward_manager
    original_compute = manager.compute
    captured = {}

    def compute(dt):
        value = original_compute(dt)
        relative = base.scene["robot"].data.root_link_pos_w - base.scene.terrain.env_origins
        captured["progress"] = relative[:, 0].clone()
        captured["lateral"] = relative[:, 1].clone()
        return value

    episode_log = (folder / "episodes.jsonl").open("w")
    update_log = (folder / "updates.jsonl").open("w")
    manager.compute = compute
    cap = settings["train_budget"] + 4000
    step = 0
    try:
        with torch.inference_mode():
            request = torch.tanh(u_current).to(device=v.device, dtype=v.dtype) @ v.T
            while not book.all_finished():
                step += 1
                if step > cap or not app.is_running():
                    raise RuntimeError("training step cap reached or simulator stopped")
                action = actor(observations)
                observations, reward, done, extras = wrapped.step_residual(action, request)
                report["transitions"] += n
                terminal = extras["source_terminal"]
                if terminal is not None:
                    ids = terminal["ids"].cpu().tolist()
                    steps = terminal["steps"].cpu().tolist()
                    completed = terminal["completed"].cpu().tolist()
                    terms = {k: x.cpu().tolist() for k, x in terminal["terms"].items()}
                    progress = captured["progress"].cpu().tolist()
                    lateral = captured["lateral"].cpu().tolist()
                    keys = getattr(base, "_ea_last_reset_keys", {})
                    for j, s in enumerate(ids):
                        reasons = [k for k, x in terms.items() if x[j]]
                        safety = bool(SAFETY_TERMS.intersection(reasons))
                        ret = episode_return(progress[s], completed[j], safety)
                        sample = samples[s]
                        out = book.end_episode(s, steps[j], ret)
                        episode_log.write(json.dumps(dict(
                            session=s, episode_index=out["index"], kind=out["kind"], control_step=step,
                            steps=steps[j], completed=bool(completed[j]), safety_ending=safety, reasons=reasons,
                            progress_m=progress[s], lateral_m=lateral[s], episode_return=ret,
                            u=None if sample is None else sample["u"].tolist(),
                            mean=None if sample is None else sample["mean"].tolist(),
                            version=None if sample is None else sample["version"])) + "\n")
                        if out["batch"] is not None:
                            update_log.write(json.dumps(dict(bank.update(s, out["batch"]), control_step=step)) + "\n")
                        for mark in out["crossed_marks"]:
                            snapshots[s][str(mark)] = bank.b[s].tolist()
                        expected = (spec["namespace"], spec["slots"][s], book.episode_index[s])
                        if keys.get(s) != expected:
                            raise RuntimeError(f"reset key {keys.get(s)} differs from the session plan {expected}")
                        begin(s)
                    request[ids] = torch.tanh(u_current[ids]).to(device=v.device, dtype=v.dtype) @ v.T
                check_scans(base)
                if step % 1000 == 0:
                    finished = sum(book.finished(s) for s in range(n))
                    write_json(folder / "progress.json", dict(step=step, transitions=report["transitions"],
                                                              finished_sessions=finished))
                    print(f"[ea-lpl] {spec['label']} step={step} finished={finished}/{n}", flush=True)
    finally:
        manager.compute = original_compute
        episode_log.close()
        update_log.close()
    return dict(label=spec["label"], arm=settings["arm"], basis=settings["basis"], noise_sha256=noise_sha,
                snapshots=snapshots, ledger=book.ledger(), final_b=bank.b.tolist(),
                final_version=bank.version.tolist(), controls=step, columns=spec["columns"], initial=identity)
```

`ea_lpl_make_specs.py` 加 `p2-train`、`p2-smoke` 两个分支，读取 `--inputs` 指向的 `p2_inputs.json`，分别写出 `p2_train_specs(...)` 的全部规格和 `p2_smoke_spec(...)`。

- [ ] **Step 5: 测试、提交、冒烟运行**

```bash
PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/ea_lpl -q
D=docs/lite3-ea-lpl-p2-20261115
$PY scripts/tools/ea_lpl_make_specs.py p2-smoke --inputs docs/lite3-ea-lpl-p1-20261109/p2_inputs.json --out $D/specs
git add source/rl_training/rl_training/ea_lpl scripts/tools tests/ea_lpl $D/specs
git commit -m "feat(ea-lpl): C1-a session training runs"
git push
EVID2=$(cd "$(git rev-parse --git-common-dir)" && pwd)/ea-lpl-evidence/p2
$PY scripts/tools/ea_lpl_run.py --spec $D/specs/$(ls $D/specs | grep smoke) --output $EVID2/p2_smoke --gpu auto
```

Expected: `status` 为 `passed`；`updates.jsonl` 非空且 `applied` 为 true；`episodes.jsonl` 中每个 session 的 `episode_index` 连续；`result.json` 的 `snapshots` 每个 session 含 `"0"`、`"1000"`、`"3000"`。

---

### Task 5: 正式训练（预计 1 小时，以运行时间为主）

- [ ] **Step 1: 生成规格并提交**

```bash
$PY scripts/tools/ea_lpl_make_specs.py p2-train --inputs docs/lite3-ea-lpl-p1-20261109/p2_inputs.json --out $D/specs
git add $D/specs && git commit -m "experiment(ea-lpl): P2 training specifications" && git push
```

- [ ] **Step 2: A 组用 GPU0、B 组用 GPU1 同时运行**

```bash
for level in $(ls $D/specs | sed -n 's/^p2_train_\(l[12]\)_frozen.json$/\1/p'); do
  $PY scripts/tools/ea_lpl_run.py --spec $D/specs/p2_train_${level}_frozen.json --output $EVID2/p2_train_${level}_frozen --gpu 0 &
  $PY scripts/tools/ea_lpl_run.py --spec $D/specs/p2_train_${level}_updated.json --output $EVID2/p2_train_${level}_updated --gpu 1 &
  wait
done
```

Expected: 每个运行 `status` 为 `passed`；`ledger` 中每个 session `train_steps ≥ 60000`；两组 `noise_sha256` 相同；frozen 组 `final_b` 全为 0、`final_version` 全为 0。

---

### Task 6: 快照评估（预计 1.5 小时，以运行时间为主）

**Files:**
- Modify: `source/rl_training/rl_training/ea_lpl/specs.py`（`p2_eval_specs`）
- Modify: `scripts/tools/ea_lpl_make_specs.py`（`p2-eval` 分支，参数 `--train-evidence`）
- Test: `tests/ea_lpl/test_specs.py`

**Interfaces:**
- Produces: `p2_eval_specs(p2_inputs, train_results) -> list[dict]`，标签与含义：

| 标签 | 地形 | 码 | 命名空间 | 每 session 回合 |
|---|---|---|---|---|
| `p2_eval_<lv>_b<mark>`（mark ∈ 5000、10000、20000、40000、60000） | 训练列 | B 组该节点的 b | 5 | 4 |
| `p2_eval_<lv>_base` | 训练列 | 零码（等于 A 组 ψ0） | 5 | 4 |
| `p2_eval_<lv>_oracle` | 训练列 | 每列的 `oracle_u` | 5 | 4 |
| `p2_flat_<lv>_b20000` | 全部平地列 0 | B 组 20000 节点的 b | 6 | 2 |
| `p2_flat_<lv>_base` | 全部平地列 0 | 零码 | 6 | 2 |

- [ ] **Step 1: 写失败的测试（追加到 `tests/ea_lpl/test_specs.py`）**

```python
from rl_training.ea_lpl.specs import p2_eval_specs


def test_eval_specs_use_fresh_namespaces_and_snapshots():
    columns = [4] * 60 + [5] * 60
    snapshots = {str(s): {str(m): [0.1 * (s % 3), -0.2] for m in (0, 5000, 10000, 20000, 40000, 60000)}
                 for s in range(120)}
    train_results = {"L1": dict(case=dict(arm="updated", snapshots=snapshots, columns=columns))}
    specs = {s["label"]: validate_spec(s) for s in p2_eval_specs(P2_INPUTS, train_results)}
    assert {"p2_eval_l1_b20000", "p2_eval_l1_base", "p2_eval_l1_oracle", "p2_flat_l1_b20000",
            "p2_flat_l1_base"} <= set(specs)
    b20 = specs["p2_eval_l1_b20000"]
    assert b20["namespace"] == 5 and b20["episodes_per_env"] == 4 and b20["columns"] == columns
    assert b20["codes"]["u"][4] == [0.1, -0.2] and set(b20["codes"]["basis"]) == {"roll"}
    assert specs["p2_eval_l1_oracle"]["codes"]["u"][0] == [1.0, 0.0]
    assert specs["p2_eval_l1_oracle"]["codes"]["u"][119] == [0.0, 1.0]
    flat = specs["p2_flat_l1_b20000"]
    assert flat["namespace"] == 6 and flat["columns"] == [0] * 120 and flat["episodes_per_env"] == 2
```

（JSON 读出的快照键是字符串，所以 `snapshots` 用字符串键。）

- [ ] **Step 2: 运行测试，确认失败**

Run: `PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/ea_lpl/test_specs.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 `p2_eval_specs`**

```python
def _fixed_spec(label, level, columns, namespace, episodes, basis_name, codes_u, code_name, wall_limit_s=2400):
    return dict(schema=SPEC_SCHEMA, label=label, mode="fixed", level=level, heading_hold=True, columns=columns,
                reset_table_seed=P2_TABLE_SEED, namespace=namespace, slots=list(range(NUM_ENVS)),
                episodes_per_env=episodes, max_episode_steps=2000,
                codes=dict(basis=[basis_name] * NUM_ENVS, u=[[float(a), float(b)] for a, b in codes_u],
                           code_name=[code_name] * NUM_ENVS, record_replay=False),
                wall_limit_s=wall_limit_s)


def p2_eval_specs(p2_inputs, train_results):
    """train_results maps level -> result.json of that level's updated run."""
    name = p2_inputs["basis"]
    oracle = {(t["level"], t["column"]): t["oracle_u"] for t in p2_inputs["targets"]}
    specs = []
    for level, result in sorted(train_results.items()):
        case = result["case"]
        if case["arm"] != "updated":
            raise ValueError("evaluation snapshots must come from the updated arm")
        columns, lv = case["columns"], level.lower()
        snap = case["snapshots"]
        for mark in P2_REGISTERED["marks"][1:]:
            codes_u = [snap[str(s)][str(mark)] for s in range(NUM_ENVS)]
            specs.append(_fixed_spec(f"p2_eval_{lv}_b{mark}", level, columns, P2_EVAL_NAMESPACE, 4, name, codes_u,
                                     f"b{mark}"))
        zero = [[0.0, 0.0]] * NUM_ENVS
        specs.append(_fixed_spec(f"p2_eval_{lv}_base", level, columns, P2_EVAL_NAMESPACE, 4, name, zero, "zero"))
        specs.append(_fixed_spec(f"p2_eval_{lv}_oracle", level, columns, P2_EVAL_NAMESPACE, 4, name,
                                 [oracle[(level, c)] for c in columns], "oracle"))
        b20 = [snap[str(s)]["20000"] for s in range(NUM_ENVS)]
        specs.append(_fixed_spec(f"p2_flat_{lv}_b20000", level, [0] * NUM_ENVS, P2_FLAT_NAMESPACE, 2, name, b20,
                                 "b20000"))
        specs.append(_fixed_spec(f"p2_flat_{lv}_base", level, [0] * NUM_ENVS, P2_FLAT_NAMESPACE, 2, name, zero,
                                 "zero"))
    return specs
```

`ea_lpl_make_specs.py` 的 `p2-eval` 分支：读取 `--train-evidence` 下每个 `p2_train_<lv>_updated/result.json`，组成 `{"L1": ..., "L2": ...}`，调用 `p2_eval_specs` 写出。

- [ ] **Step 4: 测试、提交、运行评估**

```bash
PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/ea_lpl -q
$PY scripts/tools/ea_lpl_make_specs.py p2-eval --inputs docs/lite3-ea-lpl-p1-20261109/p2_inputs.json --train-evidence $EVID2 --out $D/specs
git add source/rl_training/rl_training/ea_lpl scripts/tools tests/ea_lpl $D/specs
git commit -m "experiment(ea-lpl): P2 snapshot evaluation specifications" && git push
ls $D/specs/p2_eval_*.json $D/specs/p2_flat_*.json > /tmp/p2_eval_specs.txt
awk 'NR%2==1' /tmp/p2_eval_specs.txt | while read s; do
  $PY scripts/tools/ea_lpl_run.py --spec $s --output $EVID2/$(basename $s .json) --gpu 0; done &
awk 'NR%2==0' /tmp/p2_eval_specs.txt | while read s; do
  $PY scripts/tools/ea_lpl_run.py --spec $s --output $EVID2/$(basename $s .json) --gpu 1; done &
wait
```

Expected: 每个等级 9 个评估运行全部 `passed`。

---

### Task 7: 分析、判定与报告（预计 2 小时）

**Files:**
- Create: `scripts/tools/ea_lpl_p2_analysis.py`
- Create: `docs/lite3-ea-lpl-p2-20261115/README.md`, `summary.json`, `sessions.csv`, `learning_curve.png`, `SHA256SUMS`

**Interfaces:**
- Consumes: Task 3 的统计函数；训练与评估运行的 `result.json`、`episodes.jsonl`。

- [ ] **Step 1: 写分析脚本**

脚本 `ea_lpl_p2_analysis.py --evidence $EVID2 --inputs p2_inputs.json --out $D` 依次完成：

1. 完整性检查（任何一项失败都报错退出）：每个等级 frozen 与 updated 的 `noise_sha256` 相同；frozen 的 `final_b` 全为 0、`final_version` 全为 0；两组 `ledger` 每个 session `train_steps ≥ 60000`；评估运行的 `codes.u` 与训练快照一致。
2. 主判定：对每个目标条件 `(level, column)`，取训练在该列上的 session，`session_gains(b20000 记录, base 记录, episodes=4)`，`bootstrap_mean_ci`，得到 `per_target[terrain] = (mean, low, high)`。
3. 安全：所有 session 评估回合中 `safety_ending` 的比例，`b20000` 对 `base`。
4. 平地：`p2_flat_*_b20000` 与 `p2_flat_*_base` 的完成率；速度均方根误差比 = b20000 各回合 `velocity_rms_error_mps` 均值 / base 各回合均值。
5. `p2_gate(...)`，写入 `summary.json`。
6. 次要：各节点（5000 到 60000）的平均收益与区间；预言码收益 `oracle − base` 及 B 达到它的比例（`(b20000 − base)/(oracle − base)`，分母 ≤ 0 时记为 null）；训练期每个 session 最后 4 个训练回合的完成率（B 对 A，取自 `episodes.jsonl` 中 `kind == "train"` 的记录）。
7. `sessions.csv`：每个 session 一行，列 `level, column, terrain, b20000_u0, b20000_u1, gain_20000, gain_60000, train_last4_updated, train_last4_frozen`。
8. `learning_curve.png`：横轴训练步节点，纵轴评估完成率（每个目标条件一条 B 曲线，加 base 和 oracle 两条水平线）。

- [ ] **Step 2: 运行分析、写 README、提交**

```bash
$PY scripts/tools/ea_lpl_p2_analysis.py --evidence $EVID2 --inputs docs/lite3-ea-lpl-p1-20261109/p2_inputs.json --out $D
(cd $EVID2 && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $D/SHA256SUMS
git add scripts/tools/ea_lpl_p2_analysis.py $D docs/lite3-ea-lpl-progress-log.md
git commit -m "results(ea-lpl): P2 C1-a frozen versus updated"
git push && git ls-remote --heads origin codex/lite3-ea-lpl-20261106
```

README 主要结果表固定列：`目标条件 | session 数 | base 完成率 | B@20k 完成率 | 平均收益 [95% 区间] | B@60k 平均收益 | oracle 完成率 | 达到 oracle 收益的比例`。另列：判定逐项结果、安全、平地保持、训练期对比、学习曲线图、完整性检查结果。

- [ ] **Step 3: 衔接**

P2 结束后停下汇报（P0 到 P2 的批准范围到此为止）。报告里写清：主判定是否通过；`p2_inputs.json` 中的 EA 判定（决定 P3 是否执行）；建议的下一步（通过时进入 C1-b 任务书编写；未通过时进入决策点 D-2）。
