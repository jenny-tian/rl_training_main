"""汇总 S1 小模型的结果。判定条件见 docs/lite3-online-adapter-20261126/PLAN.md 第 7 节。

用法：python reference_online_adapter_results.py [--partial] [变体 ...]
各目标、各变体用子进程分别汇总；参照是同一目标上只有受阻修正的 r2（评估起点相同）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-adapter-20261126'
LOGS = ROOT / 'logs/reference_six'
TARGETS = ('stairs_up_07', 'slope_rough_up_x11')
VARIANTS = ('mfa', 'mfb', 'mfc', 'mra', 'mrb', 'mrc')
count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())


def one(partial):
    from reference_online_c1a2 import ADAPTER, MARKS, OUTPUT, PRIMARY, RUNS, TARGET, VARIANT, protocol
    from reference_online_c1a_results import paired_summary
    from reference_online_c1aea_results import against_fixed, frozen_reproduced
    from reference_online_select_results import done, evaluation, read, retention
    r2 = LOGS / RUNS['r2'] / TARGET
    marks = [m for m in MARKS[1:] if done(OUTPUT, m)]
    if not marks or (not partial and len(marks) != len(MARKS) - 1): return print(TARGET, VARIANT, 'not finished')
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    result = dict(target=TARGET, method=VARIANT, adapter=ADAPTER, marks=marks, versus_frozen={str(m): paired_summary(rows[m]) for m in marks})
    train = read(OUTPUT, 'train')['records']
    result['training'] = {str(m): {arm: dict(attempts=sum(r['arm'] == arm and r['control_step'] <= m for r in train) / 60,
        success=sum(r['arm'] == arm and r['control_step'] <= m and r['outcome'] == 'success' for r in train)) for arm in ('Frozen', 'Updated')} for m in marks}
    if PRIMARY in marks and 1250 in marks:
        result['primary_minus_first'] = against_fixed(rows[PRIMARY], rows[1250])
    if VARIANT != 'r2':
        shared = [m for m in marks if done(r2, m)]
        result['minus_r2'] = {str(m): against_fixed(rows[m], evaluation(r2, m)) for m in shared}
        result['frozen_reproduced'] = {str(m): frozen_reproduced(rows[m], evaluation(r2, m)) for m in shared}
    retention_mark = protocol()['primary_mark']
    has_retention = (OUTPUT / f'retention_{retention_mark}_19' / 'report.json').exists()
    if has_retention and not partial:
        result.update(retention(OUTPUT, protocol(), result['versus_frozen'][str(retention_mark)]))
    if not partial and ADAPTER:
        v = result['versus_frozen'][str(PRIMARY)]
        five = result['five_criteria']
        result['registered'] = dict(
            effective=bool(v['mean_session_delta'] >= .10 and v['ci95'][0] > 0),
            still_improving=bool(result['primary_minus_first']['ci95'][0] > 0),
            adds_to_stall_correction=bool(str(PRIMARY) in result.get('minus_r2', {}) and result['minus_r2'][str(PRIMARY)]['ci95'][0] > 0),
            retention=bool(five['posture_increase_at_most_2pp'] and five['every_family_loss_at_most_5pp'] and five['flat_velocity_ratio_at_most_1p10']))
    if not partial: save(DOCS / f'RESULTS_{VARIANT}_{TARGET}.json', result)
    print(TARGET, VARIANT)
    for m in marks:
        x, t = result['versus_frozen'][str(m)], result['training'][str(m)]
        line = f" {m:>5}: Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)} posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')} | train success {t['Frozen']['success']}/{t['Updated']['success']} attempts {t['Updated']['attempts']:.1f}"
        if str(m) in result.get('minus_r2', {}):
            d = result['minus_r2'][str(m)]
            line += f" | minus r2 ({d['B_success']}) {pp(d)} frozen identical {result['frozen_reproduced'][str(m)]['identical']}/{result['frozen_reproduced'][str(m)]['n']}"
        print(line)
    if 'primary_minus_first' in result: print('   10000 minus 1250:', pp(result['primary_minus_first']))
    if 'five_criteria' in result: print('   five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
                                        'families', {f: round(100 * x['mean_session_delta'], 1) for f, x in result['retention_families'].items()})
    if 'registered' in result: print('   registered', result['registered'])


def compare():
    """只作描述的配对比较（10000 步，同一目标、相同评估起点）：两种结构之间、三种更新方式之间，以及各快照的成功数。"""
    from reference_online_c1a2 import RUNS
    from reference_online_c1aea_results import against_fixed
    from reference_online_select_results import done, evaluation
    result = {}
    for target in TARGETS:
        base = lambda v: LOGS / RUNS[v] / target
        rows = {v: evaluation(base(v), 10000) for v in VARIANTS}
        pairs = [(f'mf{u}', f'mr{u}') for u in 'abc'] + [(f'm{k}{x}', f'm{k}{y}') for k in 'fr' for x, y in (('b', 'a'), ('c', 'a'), ('c', 'b'))]
        result[target] = dict(pairs={f'{a}-{b}': against_fixed(rows[a], rows[b]) for a, b in pairs}, success={})
        for v in VARIANTS + ('r2',) + tuple(x for x in ('mrcz',) if done(base(x), 10000)):
            marks = [m for m in (1250, 2500, 5000, 10000, 20000) if done(base(v), m)]
            counts = {m: [sum(r['arm'] == arm and r['outcome'] == 'success' for r in evaluation(base(v), m)) for arm in ('Frozen', 'Updated')] for m in marks}
            result[target]['success'][v] = {str(m): c[1] for m, c in counts.items()}
            result[target]['frozen'] = counts[marks[0]][0]
        print(target, 'Frozen', result[target]['frozen'])
        for v, row in result[target]['success'].items(): print(f'  {v:5s}', ' '.join(f'{m}:{c}' for m, c in row.items()))
        for name, v in result[target]['pairs'].items(): print(f"  {name}: {v['D_success']} - {v['B_success']} = {pp(v)}")
    save(DOCS / 'COMPARISONS.json', result)


def main():
    if '--compare' in sys.argv: return compare()
    arguments = [a for a in sys.argv[1:] if not a.startswith('--')]
    if os.environ.get('ADAPTER_CHILD'): return one('--partial' in sys.argv)
    for target in TARGETS:
        for variant in arguments or VARIANTS + ('r2',):
            if variant == 'r2' and target != 'slope_rough_up_x11': continue
            subprocess.run([sys.executable, __file__, *[a for a in sys.argv[1:] if a.startswith('--')]],
                           env=dict(os.environ, ADAPTER_CHILD='1', C1A2_TARGET=target, C1A2_VARIANT=variant), check='--partial' not in sys.argv)


if __name__ == '__main__': main()
