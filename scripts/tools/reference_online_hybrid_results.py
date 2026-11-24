"""汇总受阻修正加持续选择（C1-b-RD，变体 h1）。判定条件见 docs/lite3-online-hybrid-20261124/PLAN.md。

用法：python reference_online_hybrid_results.py [--partial]
对照是只有受阻修正的 r2 和有参数库的持续选择 d4，都在上限 0.04 下，评估起点相同。各目标用子进程分别汇总。
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-hybrid-20261124'
LOGS = ROOT / 'logs/reference_six'
TARGETS = ('stairs_up_07', 'boxes_07', 'stairs_up_075', 'boxes_08')
count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())


def one_target(partial):
    from reference_online_c1a2 import MARKS, OUTPUT, PRIMARY, RUNS, TARGET, VARIANT, protocol
    from reference_online_c1a_results import paired_summary
    from reference_online_c1aea_results import against_fixed, frozen_reproduced
    from reference_online_library2_results import attempts_and_parameters
    from reference_online_select_results import done, evaluation, retention
    assert VARIANT == 'h1'
    others = {name: LOGS / RUNS[name] / TARGET for name in ('r2', 'd4')}
    ready = lambda m: done(OUTPUT, m) and all(done(base, m) for base in others.values())
    marks = [m for m in MARKS[1:] if ready(m)] if partial else list(MARKS[1:])
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    other = {name: {m: evaluation(base, m) for m in marks} for name, base in others.items()}
    result = dict(target=TARGET, primary_mark=PRIMARY, marks=marks, versus_frozen={str(m): paired_summary(rows[m]) for m in marks},
        learning=attempts_and_parameters(OUTPUT, marks) if (OUTPUT / 'train/report.json').exists() else {})
    for name in others:
        result[f'{name}_versus_frozen'] = {str(m): paired_summary(other[name][m]) for m in marks}
        result[f'versus_{name}'] = {str(m): against_fixed(rows[m], other[name][m]) for m in marks}
        result[f'frozen_reproduced_{name}'] = {str(m): frozen_reproduced(rows[m], other[name][m]) for m in marks}
    if not partial:
        result.update(retention(OUTPUT, protocol(), result['versus_frozen'][str(PRIMARY)]))
        five = result['five_criteria']
        result['registered'] = dict(gain_over_frozen=bool(five['target_gain_at_least_10pp'] and five['ci_lower_above_zero']),
                                    better_than_r2=bool(result['versus_r2'][str(PRIMARY)]['ci95'][0] > 0))
        save(DOCS / f'RESULTS_{TARGET}.json', result)
    print(TARGET, 'partial' if partial else 'final')
    for m in marks:
        x, l = result['versus_frozen'][str(m)], result['learning'].get(str(m), {})
        print(f" {m:>5}: attempts {l.get('mean_attempts_per_session', 0):.1f} leaders {l.get('leader_counts')} tanh(mean w) {np.round(l.get('tanh_of_mean_w', []), 2).tolist()}"
              f" | Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)} posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')}"
              f" | r2 {result['versus_r2'][str(m)]['B_success']} diff {pp(result['versus_r2'][str(m)])} | d4 {result['versus_d4'][str(m)]['B_success']} diff {pp(result['versus_d4'][str(m)])}"
              f" | frozen identical {result['frozen_reproduced_r2'][str(m)]['identical']} {result['frozen_reproduced_d4'][str(m)]['identical']}")
    if not partial: print('   five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
                          'families', {f: round(100 * x['mean_session_delta'], 1) for f, x in result['retention_families'].items()}, '| registered', result['registered'])


def main():
    partial = '--partial' in sys.argv
    if os.environ.get('HYBRID_TARGET'): return one_target(partial)
    for target in TARGETS:
        subprocess.run([sys.executable, __file__, *sys.argv[1:]], env=dict(os.environ, HYBRID_TARGET=target, C1A2_VARIANT='h1', C1A2_TARGET=target), check=not partial)


if __name__ == '__main__': main()
