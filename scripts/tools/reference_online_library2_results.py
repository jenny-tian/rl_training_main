"""汇总扩充参数库的两步结果。判定条件见 docs/lite3-online-library2-20261123/PLAN.md。

用法：
  C1A2_VARIANT=s2n|s4n C1A2_TARGET=stairs_up_07|boxes_07 python reference_online_library2_results.py native [--partial]
      第一步：新地形上的本地学习。各快照对 Frozen；最后一个快照对库里各组参数（不学习）和对阶段 D 的持续选择。
  C1A2_VARIANT=d3 C1A2_TARGET=stairs_up_075|boxes_08 python reference_online_library2_results.py harder [--partial]
      第二步：更难的新地形。F0（5 组库参数不学习）、F1（d1，原来的库）、F2（d3，扩充后的库）。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from reference_online_c1a2 import LIBRARY_SOURCES, LIBRARY_SOURCES_6, MARKS, OUTPUT, PRIMARY, RUNS, TARGET, VARIANT, protocol
from reference_online_c1a_results import paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced, read
from reference_online_select_results import done, evaluation, retention
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-library2-20261123'
LOGS = ROOT / 'logs/reference_six'
count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())


def attempts_and_parameters(base, marks):
    train, result = read(base, 'train')['records'], {}
    for mark in marks:
        n = np.zeros(60)
        for r in train:
            if r['arm'] == 'Updated' and r['control_step'] <= mark: n[r['session']] += 1
        heads = torch.load(base / 'train' / f'at_{mark}.pt', map_location='cpu', weights_only=True)['heads'][1::2]
        w = np.asarray([np.asarray(h['w']) for h in heads])
        result[str(mark)] = dict(mean_attempts_per_session=float(n.mean()), mean_w=w.mean(0).tolist(), tanh_of_mean_w=np.tanh(w.mean(0)).tolist(), std_tanh=np.tanh(w).std(0).tolist(),
                                 leader_counts=np.bincount([h['selection']['entry'] for h in heads]).tolist() if heads[0].get('selection') else None)
    return result


def native(partial):
    assert VARIANT in ('s2n', 's4n') and TARGET in ('stairs_up_07', 'boxes_07')
    marks = [m for m in MARKS[1:] if done(OUTPUT, m)] if partial else list(MARKS[1:])
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    transfer = {f'{v}:{t}:{m}': evaluation(LOGS / 'online_library_transfer_20261122' / TARGET / f'from_{v}_{t}_{m}_pooled', m) for v, t, m in LIBRARY_SOURCES}
    d1 = evaluation(LOGS / RUNS['d1'] / TARGET, 10000)
    result = dict(target=TARGET, variant=VARIANT, primary_mark=PRIMARY, marks=marks, versus_frozen={str(m): paired_summary(rows[m]) for m in marks},
        versus_library={name: {str(m): against_fixed(rows[m], x) for m in marks} for name, x in transfer.items()},
        versus_D1_at_36_attempts={str(m): against_fixed(rows[m], d1) for m in marks},
        frozen_reproduced={str(m): frozen_reproduced(rows[m], d1) for m in marks}, learning=attempts_and_parameters(OUTPUT, marks))
    if not partial:
        result.update(retention(OUTPUT, protocol(), result['versus_frozen'][str(PRIMARY)]))
        save(DOCS / f'NATIVE_{TARGET}.json', result)
    print(TARGET, VARIANT, 'partial' if partial else 'final')
    for m in marks:
        x, l = result['versus_frozen'][str(m)], result['learning'][str(m)]
        print(f" {m:>5}: attempts {l['mean_attempts_per_session']:.1f} tanh(mean w) {np.round(l['tanh_of_mean_w'], 2).tolist()} | Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)}"
              f" posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')} | vs D1 {result['versus_D1_at_36_attempts'][str(m)]['B_success']} {pp(result['versus_D1_at_36_attempts'][str(m)])}"
              f" | frozen identical {result['frozen_reproduced'][str(m)]['identical']}")
    for name, by_mark in result['versus_library'].items():
        x = by_mark[str(marks[-1])]; print(f"   vs library {name} at {marks[-1]}: {x['D_success']} vs {x['B_success']} {pp(x)}")
    if not partial: print('   five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
                          'families', {f: round(100 * x['mean_session_delta'], 1) for f, x in result['retention_families'].items()})


def harder(partial):
    assert VARIANT == 'd3' and TARGET in ('stairs_up_075', 'boxes_08')
    base = dict(F1=LOGS / RUNS['d1'] / TARGET, F2=OUTPUT)
    names = [f'{v}:{t}:{m}' for v, t, m in LIBRARY_SOURCES_6]
    transfer = {}
    for name, (v, t, m) in zip(names, LIBRARY_SOURCES_6):
        folder = LOGS / 'online_library2_transfer_20261123' / TARGET / f'from_{v}_{t}_{m}_pooled'
        if all((folder / f'target_{m}_{p:02d}' / 'report.json').exists() for p in range(4)): transfer[name] = evaluation(folder, m)
    assert partial or len(transfer) == len(names)
    result = dict(target=TARGET, primary_mark=PRIMARY, F0={name: paired_summary(x) for name, x in transfer.items()})
    rows = {}
    for key, folder in base.items():
        marks = [m for m in MARKS[1:] if done(folder, m)] if partial else list(MARKS[1:])
        rows[key] = {m: evaluation(folder, m) for m in marks}
        result[key] = dict(marks=marks, versus_frozen={str(m): paired_summary(rows[key][m]) for m in marks},
                           learning=attempts_and_parameters(folder, marks) if (folder / 'train/report.json').exists() else {})
        if not partial: result[key].update(retention(folder, protocol(), result[key]['versus_frozen'][str(PRIMARY)]))
    common = [m for m in result['F2']['marks'] if m in rows['F1']]
    result['F2_versus_F1'] = {str(m): against_fixed(rows['F2'][m], rows['F1'][m]) for m in common}
    result['frozen_reproduced'] = {str(m): frozen_reproduced(rows['F2'][m], rows['F1'][m]) for m in common}
    if transfer and result['F2']['marks']:
        best = max(transfer, key=lambda name: count(result['F0'][name], 'Updated'))
        result['best_F0'] = best
        result['F2_versus_best_F0'] = {str(m): against_fixed(rows['F2'][m], transfer[best]) for m in result['F2']['marks']}
    if not partial:
        five = result['F2']['five_criteria']
        result['registered'] = dict(expansion_helps=bool(result['F2_versus_F1'][str(PRIMARY)]['ci95'][0] > 0),
                                    selection_still_works=bool(five['target_gain_at_least_10pp'] and five['ci_lower_above_zero']))
        save(DOCS / f'HARDER_{TARGET}.json', result)
    print(TARGET, 'partial' if partial else 'final')
    for name, v in result['F0'].items(): print(f" F0 {name:26s} Frozen {count(v, 'Frozen')} -> {count(v, 'Updated')} gain {pp(v)} posture {posture(v, 'Frozen')} -> {posture(v, 'Updated')}")
    for key in ('F1', 'F2'):
        for m in result[key]['marks']:
            x, l = result[key]['versus_frozen'][str(m)], result[key]['learning'].get(str(m), {})
            print(f" {key} {m:>5}: attempts {l.get('mean_attempts_per_session', 0):.1f} leaders {l.get('leader_counts')} | Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)} posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')}")
        if not partial: print('    five:', result[key]['five_criteria'], 'flat ratio', round(result[key]['flat_velocity_rmse_ratio'], 3),
                              'families', {f: round(100 * x['mean_session_delta'], 1) for f, x in result[key]['retention_families'].items()})
    for m, x in result['F2_versus_F1'].items(): print(f" F2 - F1 at {m}: {x['D_success']} vs {x['B_success']} {pp(x)} | frozen identical {result['frozen_reproduced'][m]['identical']}")
    for m, x in result.get('F2_versus_best_F0', {}).items(): print(f" F2 - best F0 ({result['best_F0']}) at {m}: {pp(x)}")
    if not partial: print(' registered:', result['registered'])


if __name__ == '__main__':
    dict(native=native, harder=harder)[sys.argv[1]]('--partial' in sys.argv)
