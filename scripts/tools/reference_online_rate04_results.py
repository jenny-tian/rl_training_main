"""汇总每步变化上限 0.04 下的保持评估（G0）和持续选择（G1）。判定条件见 docs/lite3-online-rate04-20261124/PLAN.md。

用法：python reference_online_rate04_results.py [--partial]
G1 的对照是上限 0.02 下的 d1，评估起点相同。各目标用子进程分别汇总，因为入口按环境变量决定目标。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-rate04-20261124'
LOGS = ROOT / 'logs/reference_six'
TARGETS = ('stairs_up_07', 'boxes_07', 'stairs_up_075', 'boxes_08')
G0 = (('stairs_up_07', 0), ('boxes_07', 2))
count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())


def one_target(partial):
    from reference_online_c1a2 import MARKS, OUTPUT, PRIMARY, RUNS, TARGET, VARIANT, protocol
    from reference_online_c1a_results import paired_summary
    from reference_online_c1aea_results import against_fixed, frozen_reproduced
    from reference_online_library2_results import attempts_and_parameters
    from reference_online_select_results import done, evaluation, retention
    assert VARIANT == 'd4'
    old = LOGS / RUNS['d1'] / TARGET
    marks = [m for m in MARKS[1:] if done(OUTPUT, m)] if partial else list(MARKS[1:])
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    before = {m: evaluation(old, m) for m in marks}
    result = dict(target=TARGET, primary_mark=PRIMARY, marks=marks, versus_frozen={str(m): paired_summary(rows[m]) for m in marks},
        d1_versus_frozen={str(m): paired_summary(before[m]) for m in marks}, versus_d1={str(m): against_fixed(rows[m], before[m]) for m in marks},
        frozen_reproduced={str(m): frozen_reproduced(rows[m], before[m]) for m in marks},
        learning=attempts_and_parameters(OUTPUT, marks) if (OUTPUT / 'train/report.json').exists() else {})
    if not partial:
        result.update(retention(OUTPUT, protocol(), result['versus_frozen'][str(PRIMARY)]))
        five = result['five_criteria']
        result['registered'] = dict(gain_over_frozen=bool(five['target_gain_at_least_10pp'] and five['ci_lower_above_zero']),
                                    better_than_d1=bool(result['versus_d1'][str(PRIMARY)]['ci95'][0] > 0))
        save(DOCS / f'RESULTS_{TARGET}.json', result)
    print(TARGET, 'partial' if partial else 'final')
    for m in marks:
        x, l = result['versus_frozen'][str(m)], result['learning'].get(str(m), {})
        print(f" {m:>5}: attempts {l.get('mean_attempts_per_session', 0):.1f} leaders {l.get('leader_counts')} | Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)} posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')}"
              f" | d1 {result['versus_d1'][str(m)]['B_success']} diff {pp(result['versus_d1'][str(m)])} | frozen identical {result['frozen_reproduced'][str(m)]['identical']}")
    if not partial: print('   five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
                          'families', {f: round(100 * x['mean_session_delta'], 1) for f, x in result['retention_families'].items()}, '| registered', result['registered'])


def fixed_retention():
    from reference_online_c1a_results import FAMILY, paired_summary
    from reference_online_c1aea_results import read
    result = {}
    for target, entry in G0:
        base = LOGS / 'online_rate_20261123' / target / f'rate_0.04_{entry}'
        if not all((base / f'retention_0_{p:02d}' / 'report.json').exists() for p in range(20)): continue
        retained = [r for p in range(20) for r in read(base, f'retention_0_{p:02d}')['records']]
        families = {f: paired_summary([r for r in retained if FAMILY[r['condition']] == f], repeats=2 * sum(v == f for v in FAMILY.values())) for f in dict.fromkeys(FAMILY.values())}
        ratio = families['flat']['Updated']['xy_velocity_rmse'] / families['flat']['Frozen']['xy_velocity_rmse']
        row = dict(library_entry=entry, families=families, flat_velocity_rmse_ratio=ratio,
                   every_family_loss_at_most_5pp=all(v['mean_session_delta'] >= -.05 for v in families.values()), flat_velocity_ratio_at_most_1p10=ratio <= 1.10)
        row['passed'] = bool(row['every_family_loss_at_most_5pp'] and row['flat_velocity_ratio_at_most_1p10'])
        result[f'entry{entry}'] = row
        print('G0 entry', entry, {f: (count(v, 'Frozen'), count(v, 'Updated'), pp(v)[0], posture(v, 'Frozen'), posture(v, 'Updated')) for f, v in families.items()}, 'flat ratio', round(ratio, 3), 'passed', row['passed'])
    return result


def main():
    partial = '--partial' in sys.argv
    if os.environ.get('RATE04_TARGET'): return one_target(partial)
    retained = fixed_retention()
    if not partial:
        assert len(retained) == len(G0)
        save(DOCS / 'RETENTION.json', retained)
    for target in TARGETS:
        subprocess.run([sys.executable, __file__, *sys.argv[1:]], env=dict(os.environ, RATE04_TARGET=target, C1A2_VARIANT='d4', C1A2_TARGET=target), check=not partial)


if __name__ == '__main__': main()
