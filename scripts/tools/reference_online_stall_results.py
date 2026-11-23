"""汇总一个新地形上 C1-b-R（r1：用摆动受阻比例修正摆动幅度）的结果。判定条件见 docs/lite3-online-stall-20261123/PLAN.md。
对照用已有的评估：C2（w2，从 0 起的扰动学习器）、D1（d1，参数库持续选择）、库里各组参数不学习的结果。

用法：C1A2_VARIANT=r1 C1A2_TARGET=<目标> python reference_online_stall_results.py [--partial]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from reference_online_c1a2 import LIBRARY_SOURCES, MARKS, OUTPUT, PRIMARY, RUNS, TARGET, VARIANT, protocol
from reference_online_c1a_results import paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced, read
from reference_online_select_results import done, evaluation, retention
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-stall-20261123'
LOGS = ROOT / 'logs/reference_six'
TRANSFER = [LOGS / 'online_library_transfer_20261122' / TARGET / f'from_{v}_{t}_{m}_pooled' for v, t, m in LIBRARY_SOURCES]


def learning(marks):
    train = read(OUTPUT, 'train')['records']
    windows = read(OUTPUT, 'train', 'windows.json')
    result = {}
    for mark in marks:
        attempts = np.zeros(60)
        for r in train:
            if r['arm'] == 'Updated' and r['control_step'] <= mark: attempts[r['session']] += 1
        heads = torch.load(OUTPUT / 'train' / f'at_{mark}.pt', map_location='cpu', weights_only=True)['heads'][1::2]
        z = np.tanh(np.asarray([np.asarray(h['w']) for h in heads]))[:, 2:]
        result[str(mark)] = dict(mean_attempts_per_session=float(attempts.mean()), mean_updates_per_session=float(np.mean([h['updates'] for h in heads])),
                                 mean_tanh_swing=z.mean(0).tolist(), std_tanh_swing=z.std(0).tolist(), sessions_at_bound=int((z > .98).all(1).sum()))
    # 每个会话第几次更新后两个摆动幅度第一次都到 tanh 0.75 以上（库参数的水平）
    reach, stall = [], []
    for s in range(60):
        updates = [e for e in windows[f'{s}:Updated'] if e['updated']]
        hit = [k + 1 for k, e in enumerate(updates) if min(np.tanh(e['after'][2:])) >= .75]
        reach.append(hit[0] if hit else None); stall += [e['stall'] for e in updates]
    reached = [r for r in reach if r is not None]
    result.update(sessions_reaching_library_level=len(reached), median_updates_to_library_level=float(np.median(reached)) if reached else None,
                  mean_window_stall=np.mean(stall, 0).tolist(), share_of_windows_above_target=(np.asarray(stall) > .30).mean(0).tolist(),
                  train_success={arm: float(np.mean([r['outcome'] == 'success' for r in train if r['arm'] == arm])) for arm in ('Frozen', 'Updated')})
    return result


def main():
    assert VARIANT == 'r1'
    partial = '--partial' in sys.argv
    marks = [m for m in MARKS[1:] if done(OUTPUT, m)] if partial else list(MARKS[1:])
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    c2 = {m: evaluation(LOGS / RUNS['w2'] / TARGET, m) for m in marks}
    d1 = {m: evaluation(LOGS / RUNS['d1'] / TARGET, m) for m in marks}
    transfer = [evaluation(base, mark) for base, (_, _, mark) in zip(TRANSFER, LIBRARY_SOURCES)]
    result = dict(target=TARGET, primary_mark=PRIMARY, marks=marks, versus_frozen={str(m): paired_summary(rows[m]) for m in marks},
        versus_C2={str(m): against_fixed(rows[m], c2[m]) for m in marks}, versus_D1={str(m): against_fixed(rows[m], d1[m]) for m in marks},
        versus_library={f'{v}:{t}:{k}': {str(m): against_fixed(rows[m], x) for m in marks} for (v, t, k), x in zip(LIBRARY_SOURCES, transfer)},
        frozen_reproduced={str(m): frozen_reproduced(rows[m], c2[m]) for m in marks}, learning=learning(marks))
    if not partial:
        result.update(retention(OUTPUT, protocol(), result['versus_frozen'][str(PRIMARY)]))
        five = result['five_criteria']
        result['registered'] = dict(primary_gain_over_frozen=bool(five['target_gain_at_least_10pp'] and five['ci_lower_above_zero']),
                                    primary_better_than_C2=bool(result['versus_C2'][str(PRIMARY)]['ci95'][0] > 0))
        save(DOCS / f'RESULTS_{TARGET}.json', result)
    count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
    posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
    pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())
    print(TARGET, 'partial' if partial else 'final')
    for m in marks:
        x, l = result['versus_frozen'][str(m)], result['learning'][str(m)]
        print(f" {m:>5}: attempts {l['mean_attempts_per_session']:.1f} updates {l['mean_updates_per_session']:.1f} tanh swing {np.round(l['mean_tanh_swing'], 2).tolist()} at bound {l['sessions_at_bound']}"
              f" | Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)} posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')}"
              f" | vs C2 {result['versus_C2'][str(m)]['B_success']} {pp(result['versus_C2'][str(m)])} | vs D1 {result['versus_D1'][str(m)]['B_success']} {pp(result['versus_D1'][str(m)])}"
              f" | frozen identical {result['frozen_reproduced'][str(m)]['identical']}")
    for name, by_mark in result['versus_library'].items():
        x = by_mark[str(marks[-1])]; print(f"   vs library {name} at {marks[-1]}: {x['D_success']} vs {x['B_success']} {pp(x)}")
    l = result['learning']
    print('   reached library level in', l['sessions_reaching_library_level'], 'sessions, median updates', l['median_updates_to_library_level'], '| mean window stall', np.round(l['mean_window_stall'], 3).tolist(),
          'above target', np.round(l['share_of_windows_above_target'], 3).tolist(), '| train success', l['train_success'])
    if not partial: print('   five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
                          'families', {f: round(100 * x['mean_session_delta'], 1) for f, x in result['retention_families'].items()}, '| registered', result['registered'])


if __name__ == '__main__': main()
