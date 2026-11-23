"""汇总每步变化上限的仿真扫描。判定条件见 docs/lite3-online-rate-20261123/PLAN.md。

用法：python reference_online_rate_results.py
每个上限与同一目标上上限 0.02 的已有结果（同一组参数、同一批起点）配对比较，并从轨迹里统计实际执行的膝关节残差。
"""
from __future__ import annotations

import numpy as np

from reference_online_c1a_results import paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced
from reference_online_lead_results import C0, C0_MARK, evaluation, executed
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-rate-20261123'
LOGS = ROOT / 'logs/reference_six'
RATES = ('0.04', '0.06', '0.10', '0.24')
CASES = dict(stairs_up_06=(0, LOGS / 'online_lead_20261123/stairs_up_06/none_0', 0), stairs_up_07=(0, C0['stairs_up_07'], C0_MARK['stairs_up_07']),
             boxes_07=(2, C0['boxes_07'], C0_MARK['boxes_07']))


def main():
    result = {}
    count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
    posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
    pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())
    for target, (entry, base, mark) in CASES.items():
        reference = evaluation(base, mark)
        row = {'0.02': dict(versus_frozen=paired_summary(reference), executed=executed(base, mark))}
        v, e = row['0.02']['versus_frozen'], row['0.02']['executed']
        print(target, 'entry', entry, '| 0.02: Frozen', count(v, 'Frozen'), '->', count(v, 'Updated'), 'posture', posture(v, 'Frozen'), '->', posture(v, 'Updated'),
              f"| knee swing req {e['requested_in_swing']:.3f} exe {e['executed_in_swing']:.3f} stance exe {e['executed_in_stance']:.3f}")
        for rate in RATES:
            run = LOGS / 'online_rate_20261123' / target / f'rate_{rate}_{entry}'
            rows = evaluation(run)
            x = dict(versus_frozen=paired_summary(rows), versus_current=against_fixed(rows, reference), executed=executed(run), frozen_reproduced=frozen_reproduced(rows, reference))
            x['posture_change_versus_current'] = x['versus_frozen']['Updated']['posture_rate'] - row['0.02']['versus_frozen']['Updated']['posture_rate']
            x['improved'] = bool(x['versus_current']['ci95'][0] > 0)
            row[rate] = x
            v, e = x['versus_frozen'], x['executed']
            print(f"   {rate}: {count(v, 'Updated')} posture {posture(v, 'Updated')} outcomes {v['Updated']['outcomes']} | vs 0.02 {pp(x['versus_current'])} posture change {100 * x['posture_change_versus_current']:+.1f}"
                  f" | knee swing exe {e['executed_in_swing']:.3f} stance exe {e['executed_in_stance']:.3f} | improved {x['improved']} | frozen identical {x['frozen_reproduced']['identical']}")
        result[target] = row
    both = [rate for rate in RATES if all(result[t][rate]['improved'] for t in ('stairs_up_06', 'stairs_up_07'))]
    result['rates_improved_on_both_stairs'] = both
    result['smallest_rate_improved_on_both_stairs'] = both[0] if both else None
    save(DOCS / 'RESULTS.json', result)
    print('improved on both stairs:', both)


if __name__ == '__main__': main()
