"""汇总提前起抬的脚本化检查。判定条件见 docs/lite3-online-lead-20261123/PLAN.md。

用法：python reference_online_lead_results.py [--partial]
每条规则与同一目标上不提前的结果（none；7 cm 箱体用阶段 C 的 C0）配对比较，并从轨迹里统计实际执行的膝关节残差。
"""
from __future__ import annotations

import json
import sys

import numpy as np

from reference_online_c1a_results import paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced, read
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-lead-20261123'
LOGS = ROOT / 'logs/reference_six'
C0 = dict(stairs_up_07=LOGS / 'online_library_transfer_20261122/stairs_up_07/from_s2n_stairs_up_06_60000_pooled',
          boxes_07=LOGS / 'online_library_transfer_20261122/boxes_07/from_bs_boxes_06_40000_pooled')
C0_MARK = dict(stairs_up_07=60000, boxes_07=40000)
CASES = dict(stairs_up_06=(0, ('none', 'knee', 'partner', 'age')), stairs_up_07=(0, ('none', 'knee', 'partner', 'age')), boxes_07=(2, ('knee', 'partner', 'age')))
KNEE = np.arange(2, 12, 3)


def evaluation(base, mark=0):
    return [r for p in range(4) for r in read(base, f'target_{mark}_{p:02d}')['records']]


def done(base):
    return all((base / f'target_0_{p:02d}' / 'report.json').exists() for p in range(4))


def executed(base, mark=0):
    """Updated 组在起伏地形上：摆动期和着地期膝关节残差的请求值与实际执行值（按真实的着地状态分）。"""
    sums = np.zeros(6)
    for p in range(4):
        z = np.load(base / f'target_{mark}_{p:02d}' / 'trace_0000.npz')
        on = (z['phase'][:, 1::2] == 1) & (z['features'][:, 1::2, 0] > .5)
        stance = z['stance'][:, 1::2]
        request, effect = z['residual_request'][:, 1::2][:, :, KNEE], z['learned_effect'][:, 1::2][:, :, KNEE]
        swing, down = on[:, :, None] & ~stance, on[:, :, None] & stance
        sums += [request[swing].sum(), effect[swing].sum(), swing.sum(), request[down].sum(), effect[down].sum(), down.sum()]
    return dict(requested_in_swing=float(sums[0] / sums[2]), executed_in_swing=float(sums[1] / sums[2]),
                requested_in_stance=float(sums[3] / sums[5]), executed_in_stance=float(sums[4] / sums[5]), swing_fraction=float(sums[2] / (sums[2] + sums[5])))


def main():
    partial = '--partial' in sys.argv
    result = {}
    for target, (entry, rules) in CASES.items():
        runs = {rule: LOGS / 'online_lead_20261123' / target / f'{rule}_{entry}' for rule in rules}
        rows = {rule: evaluation(base) for rule, base in runs.items() if done(base)}
        assert partial or len(rows) == len(rules)
        if 'none' in rules:
            if 'none' not in rows: continue
            reference, reference_executed = rows['none'], executed(runs['none'])
        else:
            reference, reference_executed = evaluation(C0[target], C0_MARK[target]), executed(C0[target], C0_MARK[target])
        row = dict(library_entry=entry, none=dict(versus_frozen=paired_summary(reference), executed=reference_executed), rules={})
        if target == 'stairs_up_07' and 'none' in rows:
            c0 = evaluation(C0[target], C0_MARK[target])
            same = {(r['session'], r['repeat']): (r['outcome'], r['policy_steps']) for r in c0 if r['arm'] == 'Updated'}
            row['none_reproduces_C0'] = dict(n=len(same), identical=sum(same[r['session'], r['repeat']] == (r['outcome'], r['policy_steps']) for r in rows['none'] if r['arm'] == 'Updated'),
                                             frozen=frozen_reproduced(rows['none'], c0))
        base_posture = row['none']['versus_frozen']['Updated']['posture_rate']
        for rule in rules:
            if rule == 'none' or rule not in rows: continue
            x = dict(versus_frozen=paired_summary(rows[rule]), versus_none=against_fixed(rows[rule], reference), executed=executed(runs[rule]),
                     frozen_reproduced=frozen_reproduced(rows[rule], reference))
            x['posture_change_versus_none'] = x['versus_frozen']['Updated']['posture_rate'] - base_posture
            x['passed'] = bool(x['versus_none']['ci95'][0] > 0 and x['posture_change_versus_none'] <= .02)
            row['rules'][rule] = x
        result[target] = row
        count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
        posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
        pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())
        v, e = row['none']['versus_frozen'], row['none']['executed']
        print(target, 'entry', entry, '| none: Frozen', count(v, 'Frozen'), '->', count(v, 'Updated'), 'posture', posture(v, 'Frozen'), '->', posture(v, 'Updated'),
              f"| knee swing req {e['requested_in_swing']:.3f} exe {e['executed_in_swing']:.3f} stance exe {e['executed_in_stance']:.3f}", row.get('none_reproduces_C0', ''))
        for rule, x in row['rules'].items():
            v, e = x['versus_frozen'], x['executed']
            print(f"   {rule:8s} {count(v, 'Updated')} posture {posture(v, 'Updated')} outcomes {v['Updated']['outcomes']} | vs none {pp(x['versus_none'])} posture change {100 * x['posture_change_versus_none']:+.1f}"
                  f" | knee swing req {e['requested_in_swing']:.3f} exe {e['executed_in_swing']:.3f} stance req {e['requested_in_stance']:.3f} exe {e['executed_in_stance']:.3f} | passed {x['passed']} | frozen identical {x['frozen_reproduced']['identical']}")
    if not partial:
        stairs = ('stairs_up_06', 'stairs_up_07')
        result['rules_passed_on_both_stairs'] = [rule for rule in ('knee', 'partner', 'age') if all(result[t]['rules'][rule]['passed'] for t in stairs)]
        result['check_passed'] = bool(result['rules_passed_on_both_stairs'])
        save(DOCS / 'RESULTS.json', result)
        print('passed on both stairs:', result['rules_passed_on_both_stairs'], 'check passed:', result['check_passed'])


if __name__ == '__main__': main()
