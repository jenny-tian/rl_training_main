"""汇总摆动抬脚残差的脚本化检查。方案见 docs/lite3-online-swing-probe-20261118/PLAN.md。

用法：C1A2_VARIANT=bh C1A2_TARGET=stairs_up_06 python reference_online_swing_results.py
"""
from __future__ import annotations

import collections
import json

import numpy as np

from reference_online_c1a2 import OUTPUT, SWING, VARIANT
from reference_stairs_eval import ROOT, save

assert VARIANT == 'bh'
DOCS = ROOT / 'docs/lite3-online-swing-probe-20261118'
SECTIONS = ((0., 1.), (1., 1.65), (1.65, 2.85), (2.85, 9.))


def job(name):
    records = json.loads((OUTPUT / name / 'episodes.json').read_text())['records']
    by = {(r['session'], r['repeat'], r['arm']): r for r in records}
    assert len(by) == len(records) == 720
    index = json.loads((OUTPUT / name / 'trace_index.json').read_text())
    parts = [np.load(OUTPUT / name / c['file']) for c in index['chunks']]
    effect = np.concatenate([p['learned_effect'] for p in parts]); request = np.concatenate([p['residual_request'] for p in parts])
    stance = np.concatenate([p['stance'] for p in parts]); phase = np.concatenate([p['phase'] for p in parts])
    result = dict(settings=json.loads((OUTPUT / name / 'swing.json').read_text()), groups={})
    for repeat, group in enumerate('RQS'):
        zero = [by[s, repeat, 'Frozen'] for s in range(120)]; code = [by[s, repeat, 'Updated'] for s in range(120)]
        assert all(r['swing'] == group for r in code) and all(r['swing'] is None for r in zero)
        gain = np.array([(c['outcome'] == 'success') - (z['outcome'] == 'success') for z, c in zip(zero, code)], float)
        boot = gain[np.random.default_rng(0).integers(0, 120, (10000, 120))].mean(1)
        ids = np.array([r['env_id'] for r in code])
        m = phase[:, ids] == 1
        rear = np.abs(effect[:, ids][:, :, [8, 11]])[m]; asked = np.abs(request[:, ids][:, :, [8, 11]])[m]
        swinging = ~stance[:, ids][:, :, 2:][m]
        posture = lambda rows: sum(r['outcome'] == 'posture_failure' for r in rows)
        result['groups'][group] = dict(zero=collections.Counter(r['outcome'] for r in zero), scripted=collections.Counter(r['outcome'] for r in code),
            success_gain=float(gain.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(), posture_increase=posture(code) - posture(zero),
            gained=int((gain > 0).sum()), lost=int((gain < 0).sum()),
            failures_by_section={f'{a}-{b}': collections.Counter(r['outcome'] for r in code if r['outcome'] != 'success' and a <= r['final_root'][0] < b) for a, b in SECTIONS},
            rear_swing_share=float(swinging.mean()), rear_knee_requested_in_swing=float(asked[swinging].mean()),
            rear_knee_executed_in_swing=float(rear[swinging].mean()), rear_knee_executed_in_stance=float(rear[~swinging].mean()))
        g = result['groups'][group]
        g['meets_condition'] = bool(g['success_gain'] >= .10 and g['posture_increase'] <= 3)
    return result


def main():
    names = [n for n in SWING if (OUTPUT / n / 'report.json').exists()]
    result = {n: job(n) for n in names}
    save(DOCS / 'RESULTS.json', json.loads(json.dumps(result)))
    for n, v in result.items():
        print(n, v['settings']['scale'], v['settings']['residual_rate_limit'])
        for k, g in v['groups'].items():
            print('  ', k, dict(g['zero']), '->', dict(g['scripted']), 'gain', round(g['success_gain'], 3), np.round(g['ci95'], 3).tolist(),
                  'posture', g['posture_increase'], 'meets', g['meets_condition'], '| swing share', round(g['rear_swing_share'], 2),
                  'knee asked/executed in swing', round(g['rear_knee_requested_in_swing'], 3), round(g['rear_knee_executed_in_swing'], 3),
                  'in stance', round(g['rear_knee_executed_in_stance'], 3))


if __name__ == '__main__': main()
