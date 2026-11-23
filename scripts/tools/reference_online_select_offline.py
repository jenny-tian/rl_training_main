"""加快学习 D 的离线检查第二步：用 C0 评估里每组参数的 240 次尝试，模拟一个会话在库里持续选择。不占 GPU。

用法：python reference_online_select_offline.py
先运行 reference_online_select_attempts.py。一个会话每轮 4 次尝试（4 个并行环境各一次）；某组参数的一次尝试从它的 240 次评估尝试里
有放回地抽取。在第 1、2、4、9 轮结束时（4、8、16、36 次尝试）记录当前领先的一组，它的成功数和姿态终止数取这组参数 240 次评估的结果。

分配方式（前两轮都是每组各试 1 次并轮换，与 c3 的选择阶段相同）：
  once      两轮之后不再比较，4 次都用领先的一组（c3 去掉修正）
  uniform   每轮每组各 1 次
  leader    每轮 2 次给领先的一组，另外 2 次在其余几组之间轮换
  thompson  每次尝试按各组的 Beta(1 + 成功次数, 1 + 未成功次数) 各抽一个数，用抽到最大的一组
得分（领先的一组是平均得分最高的一组）：
  windows   所有窗口反馈合在一起的平均（c3 的做法）
  return    整回合回报
  success   成功记 1；后缀 +w 表示加 0.01 倍的该次尝试窗口平均得分，只用来分出并列；后缀 -p 表示姿态终止再扣 0.5
候选参数的假设检查：第 4 轮结束后加入 3 组假想的候选，成功率是库里最好一组的 0.7、1.0、1.3 倍，分配和得分用 thompson:success+w。
结果写到 docs/lite3-online-select-20261123/OFFLINE.json。
"""
from __future__ import annotations

import json

import numpy as np

from reference_stairs_eval import ROOT, save

DATA = ROOT / 'logs/reference_six/online_select_offline_20261123'
DOCS = ROOT / 'docs/lite3-online-select-20261123'
LIBRARY_LOGS = ROOT / 'logs/reference_six/online_c1b_library_20261122'
SESSIONS, ROUNDS, MARK_ROUNDS, ENTRIES = 20000, 9, (1, 2, 4, 9), 4
RULES = [('once', 'windows'), ('once', 'success+w'), ('uniform', 'windows'), ('uniform', 'success+w'), ('leader', 'windows'), ('leader', 'return'),
         ('leader', 'success+w'), ('thompson', 'windows'), ('thompson', 'return'), ('thompson', 'success'), ('thompson', 'success+w'), ('thompson', 'success+w-p')]
CANDIDATE_RATIOS = ((), (1., 1., 1.), (.7, .7, .7), (.7, 1., 1.3), (1.3, 1.3, 1.3))


def table(target):
    """每组参数 240 次尝试的：窗口得分之和、窗口数、整回合回报、是否成功、是否姿态终止。"""
    attempts = json.loads((DATA / f'attempts_{target}.json').read_text())['attempts']
    rows = [[a for a in attempts if a['entry'] == e] for e in range(ENTRIES)]
    assert all(len(r) == 240 for r in rows)
    pick = lambda f: np.asarray([[f(a) for a in r] for r in rows], float)
    return dict(window_sum=pick(lambda a: sum(a['windows'])), window_count=pick(lambda a: len(a['windows'])), ret=pick(lambda a: a['episode_return']),
                success=pick(lambda a: a['outcome'] == 'success'), posture=pick(lambda a: a['outcome'] == 'posture_failure'))


def simulate(data, allocation, score, rng, candidates=(), sessions=SESSIONS, shrink=0.):
    """candidates：第 4 轮结束后加入的假想候选的成功率；它们只有成功与否，没有窗口得分和姿态终止。
    shrink：比较平均得分时，每组先加上这么多次得分等于本会话全部尝试平均得分的假想尝试，试得少的组不会因为一两次成功就领先。"""
    arms = ENTRIES + len(candidates)
    rate = np.concatenate([data['success'].mean(1), candidates])
    fall = np.concatenate([data['posture'].mean(1), np.zeros(len(candidates))])
    total, weight = np.zeros((sessions, arms)), np.zeros((sessions, arms))
    wins, tries = np.zeros((sessions, arms)), np.zeros((sessions, arms))
    leader, marks, rows = np.full(sessions, ENTRIES - 1), {}, np.arange(sessions)[:, None]
    attempt_success = 0.
    for r in range(ROUNDS):
        live = ENTRIES if r < 4 else arms
        if r < 2 or allocation == 'uniform':
            entries = (np.arange(4)[None] + r) % live * np.ones((sessions, 1), int)
        elif allocation == 'once':
            entries = np.repeat(leader[:, None], 4, 1)
        elif allocation == 'thompson':
            entries = rng.beta(1 + wins[:, :live], 1 + (tries - wins)[:, :live], (4, sessions, live)).argmax(2).T
        else:
            others = (leader[:, None] + 1 + (np.arange(2)[None] + 2 * r) % (live - 1)) % live
            entries = np.concatenate([np.repeat(leader[:, None], 2, 1), others], 1)
        real, draw = entries < ENTRIES, rng.integers(0, 240, entries.shape)
        e = np.minimum(entries, ENTRIES - 1)
        success = np.where(real, data['success'][e, draw], rng.random(entries.shape) < rate[entries])
        posture = np.where(real, data['posture'][e, draw], 0.)
        count = np.where(real, data['window_count'][e, draw], 0.)
        windows = np.where(real, data['window_sum'][e, draw], 0.)
        attempt_success += success.mean()
        np.add.at(wins, (rows, entries), success); np.add.at(tries, (rows, entries), 1.)
        if allocation != 'once' or r < 2:
            if score == 'windows': value, w = windows, count
            elif score == 'return': value, w = np.where(real, data['ret'][e, draw], success * 2.), np.ones(entries.shape)
            else:
                value = success + (.01 * windows / np.maximum(count, 1) if '+w' in score else 0.) - (.5 * posture if '-p' in score else 0.)
                w = np.ones(entries.shape)
            np.add.at(total, (rows, entries), value); np.add.at(weight, (rows, entries), w)
            pooled = total.sum(1, keepdims=True) / weight.sum(1, keepdims=True)
            mean = np.where(weight > 0, (total + shrink * pooled) / np.maximum(weight + shrink, 1e-9), -np.inf)
            leader = np.where(np.isfinite(mean).any(1), mean.argmax(1), ENTRIES - 1)
        if r + 1 in MARK_ROUNDS:
            marks[str(4 * (r + 1))] = dict(chosen=(np.bincount(leader, minlength=arms) / sessions).tolist(),
                expected_success_of_240=float(240 * rate[leader].mean()), expected_posture_of_240=float(240 * fall[leader].mean()),
                success_rate_during_attempts=float(attempt_success / (r + 1)))
    return marks


def main():
    result = {}
    for target in ('stairs_up_07', 'boxes_07'):
        data = table(target)
        actual = json.loads((LIBRARY_LOGS / target / 'train/selection.json').read_text())
        picked = np.bincount([s['entry'] for s in actual['sessions']], minlength=ENTRIES)
        success, posture = data['success'].sum(1), data['posture'].sum(1)
        result[target] = dict(success_by_entry=success.tolist(), posture_by_entry=posture.tolist(), c3_chosen_counts=picked.tolist(),
            c3_chosen_expected_success=float(picked @ success / 60), rules={}, candidates={})
        print(target, 'success', success.tolist(), 'posture', posture.tolist(), 'c3 chose', picked.tolist(), '->', round(float(picked @ success / 60), 1))
        show = lambda name, marks: print(f'  {name:26s}', ' | '.join(f"{a:>2}: {m['expected_success_of_240']:6.1f} posture {m['expected_posture_of_240']:5.1f}" for a, m in marks.items()),
                                         '| chosen at 36', np.round(marks['36']['chosen'], 2).tolist(), '| during', round(100 * marks['36']['success_rate_during_attempts'], 1))
        for allocation, score in RULES:
            marks = simulate(data, allocation, score, np.random.default_rng(0))
            result[target]['rules'][f'{allocation}:{score}'] = marks; show(f'{allocation}:{score}', marks)
        best = data['success'].mean(1).max()
        for ratios in CANDIDATE_RATIOS:
            for shrink in (0., 2., 4.):
                marks = simulate(data, 'thompson', 'success+w', np.random.default_rng(0), candidates=np.minimum(np.asarray(ratios) * best, 1.), shrink=shrink)
                result[target]['candidates'][f"{','.join(map(str, ratios)) or 'none'}:shrink{shrink:g}"] = marks; show(f'cand x{list(ratios)} shrink {shrink:g}', marks)
    save(DOCS / 'OFFLINE.json', result)


if __name__ == '__main__': main()
