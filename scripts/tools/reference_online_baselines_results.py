"""汇总两个对照：单次触发的抬腿反射和 SARA 式在线残差。判定条件见 docs/lite3-online-baselines-20261125/PLAN.md。

用法：python reference_online_baselines_results.py [--partial]
比较对象是只有受阻修正的 r2（上限 0.04，评估起点相同）。各目标、各对照用子进程分别汇总。
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-baselines-20261125'
LOGS = ROOT / 'logs/reference_six'
TARGETS = ('stairs_up_07', 'boxes_07', 'stairs_up_075', 'boxes_08')
count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())


def one(partial):
    from reference_online_c1a2 import MARKS, OUTPUT, REFLEX, RUNS, SARA, TARGET, VARIANT, protocol
    from reference_online_c1a_results import paired_summary
    from reference_online_c1aea_results import against_fixed, frozen_reproduced
    from reference_online_select_results import done, evaluation, read, retention
    r2 = LOGS / RUNS['r2'] / TARGET
    name = f'reflex_{REFLEX}' if REFLEX else VARIANT
    marks = [0] if REFLEX else [m for m in MARKS[1:] if done(OUTPUT, m) or not partial]
    if not marks or not all(done(OUTPUT, m) for m in marks): return print(TARGET, name, 'not finished')
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    result = dict(target=TARGET, method=name, marks=marks, versus_frozen={str(m): paired_summary(rows[m]) for m in marks})
    # r2 减对照：反射没有快照，分别与 r2 的 1250 步（约 4 次尝试）和 10000 步（约 36 次尝试）比较。
    pairs = [(1250, 0), (10000, 0)] if REFLEX else [(m, m) for m in marks]
    result['r2_minus_method'] = {str(a): against_fixed(evaluation(r2, a), rows[b]) for a, b in pairs}
    result['frozen_reproduced'] = {str(a): frozen_reproduced(rows[b], evaluation(r2, a)) for a, b in pairs}
    if REFLEX:
        result['triggers_per_updated_episode'] = float(np.mean([np.asarray(read(OUTPUT, f'target_0_{p:02d}', 'reflex.json')['triggers'])[1::2].mean() for p in range(4)]))
        result['registered'] = dict(persistent_better_than_reflex=bool(result['r2_minus_method']['10000']['ci95'][0] > 0))
    else:
        train = read(OUTPUT, 'train')['records']
        result['training'] = {str(m): {arm: dict(attempts=sum(r['arm'] == arm and r['control_step'] <= m for r in train) / 60,
            success=sum(r['arm'] == arm and r['control_step'] <= m and r['outcome'] == 'success' for r in train)) for arm in ('Frozen', 'Updated')} for m in marks}
    last = str(marks[-1])
    has_retention = (OUTPUT / f'retention_{marks[-1]}_19' / 'report.json').exists()
    if has_retention:
        result.update(retention(OUTPUT, protocol(), result['versus_frozen'][last]))
    if SARA and not partial:
        v = result['versus_frozen'][last]
        result['registered'] = dict(effective=bool(v['mean_session_delta'] >= .10 and v['ci95'][0] > 0))
    if not partial: save(DOCS / f'RESULTS_{name}_{TARGET}.json', result)
    print(TARGET, name)
    for m in marks:
        x = result['versus_frozen'][str(m)]
        extra = f" | train success Frozen/Updated {result['training'][str(m)]['Frozen']['success']}/{result['training'][str(m)]['Updated']['success']} attempts {result['training'][str(m)]['Updated']['attempts']:.1f}" if SARA else f" | triggers per episode {result['triggers_per_updated_episode']:.0f}"
        print(f" {m:>5}: Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)} posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')}" + extra)
    for a, v in result['r2_minus_method'].items():
        print(f"   r2@{a} {v['D_success']} minus method {v['B_success']}: {pp(v)} | frozen identical {result['frozen_reproduced'][a]['identical']}/{result['frozen_reproduced'][a]['n']}")
    if has_retention: print('   five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
                            'families', {f: round(100 * x['mean_session_delta'], 1) for f, x in result['retention_families'].items()})
    if 'registered' in result: print('   registered', result['registered'])


def main():
    partial = '--partial' in sys.argv
    if os.environ.get('BASELINE_CHILD'): return one(partial)
    for target in TARGETS:
        for extra in (dict(C1A2_VARIANT='s2n', C1A2_REFLEX='same'), dict(C1A2_VARIANT='s2n', C1A2_REFLEX='next'), dict(C1A2_VARIANT='sa1'), dict(C1A2_VARIANT='sa2')):
            subprocess.run([sys.executable, __file__, *sys.argv[1:]], env=dict(os.environ, BASELINE_CHILD='1', C1A2_TARGET=target, **extra), check=not partial)


if __name__ == '__main__': main()
