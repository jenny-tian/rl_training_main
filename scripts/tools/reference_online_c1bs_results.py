"""汇总一个目标的 C1-b-S 结果。判定条件见 docs/lite3-online-c1b-swing-20261118/PLAN.md。

用法：C1A2_VARIANT=bs（或 bsf） C1A2_TARGET=<目标> python reference_online_c1bs_results.py
"""
from __future__ import annotations

import json

import numpy as np

from reference_online_c1a2 import OUTPUT, TARGET, MARKS, PRIMARY, TERRAINS, VARIANT, jobs, protocol
from reference_online_c1a_results import FAMILY, paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced, read
from reference_stairs_eval import ROOT, save

assert VARIANT in ('bs', 'bsf')
DOCS = ROOT / 'docs/lite3-online-c1b-swing-20261118'
NAME = f'RESULTS_{TARGET}' + ('_fast' if VARIANT == 'bsf' else '')
M4 = ROOT / 'logs/reference_six/online_c1a_m4_20261118' / TARGET


def learning(train, checkpoints):
    result = dict(episodes=sum(r['arm'] == 'Updated' for r in train), train_success_by_third={arm: [float(np.mean([r['outcome'] == 'success' for r in part]))
        for part in np.array_split(np.asarray(sorted([r for r in train if r['arm'] == arm], key=lambda r: r['control_step']), dtype=object), 3)]
        for arm in ('Frozen', 'Updated')}, marks={})
    for mark, heads in checkpoints.items():
        w = np.asarray([np.asarray(h['w']) for h in heads[1::2]]); z = np.tanh(w)
        result['marks'][str(mark)] = dict(rounds_quantiles_0_50_100=np.quantile([h['updates'] for h in heads[1::2]], [0, .5, 1]).tolist(),
            mean_w=w.mean(0).tolist(), std_w=w.std(0).tolist(), mean_tanh=z.mean(0).tolist(),
            share_above_0p5=(z > .5).mean(0).tolist(), share_below_minus_0p5=(z < -.5).mean(0).tolist(), w=w.tolist())
    return result


def velocity_ratios(retained):
    result = {}
    for condition in TERRAINS:
        rows = [r for r in retained if r['condition'] == condition]
        mean = lambda arm: float(np.mean([r['xy_velocity_rmse'] for r in rows if r['arm'] == arm]))
        result[condition] = mean('Updated') / mean('Frozen')
    return result


def main():
    import torch
    p = protocol()
    schedule = [j for j in jobs()[1:] if not j.get('script')]
    reports = {j['name']: read(OUTPUT, j['name'], 'report.json') for j in schedule}
    assert all(r['status'] == 'passed' for r in reports.values())
    records = {j['name']: read(OUTPUT, j['name'])['records'] for j in schedule}
    m4 = {j['name']: read(M4, j['name'])['records'] for j in schedule if j['kind'] != 'train'}
    target_rows = lambda source, m: [r for j in schedule if j['kind'] == 'target' and j['mark'] == m for r in source[j['name']]]
    curves = {str(m): paired_summary(target_rows(records, m)) for m in MARKS[1:]}
    versus = {str(m): against_fixed(target_rows(records, m), target_rows(m4, m)) for m in MARKS[1:]}
    reproduced = {str(m): frozen_reproduced(target_rows(records, m), target_rows(m4, m)) for m in MARKS[1:]}
    retained = [r for j in schedule if j['kind'] == 'retention' for r in records[j['name']]]
    families = {f: paired_summary([r for r in retained if FAMILY[r['condition']] == f],
                repeats=2 * sum(v == f for v in FAMILY.values())) for f in dict.fromkeys(FAMILY.values())}
    subtypes = {c: paired_summary([r for r in retained if r['condition'] == c], repeats=2) for c in TERRAINS}
    target, flat = curves[str(PRIMARY)], families['flat']
    ratio = flat['Updated']['xy_velocity_rmse'] / flat['Frozen']['xy_velocity_rmse']
    criteria = dict(target_gain_at_least_10pp=target['mean_session_delta'] >= p['success_gain_min'],
        ci_lower_above_zero=target['ci95'][0] > p['ci_lower_min'],
        posture_increase_at_most_2pp=target['posture_delta'] <= p['safety_increase_max'],
        every_family_loss_at_most_5pp=all(v['mean_session_delta'] >= -p['family_loss_max'] for v in families.values()),
        flat_velocity_ratio_at_most_1p10=ratio <= 1.10)
    checkpoints = {m: torch.load(OUTPUT / 'train' / f'at_{m}.pt', map_location='cpu', weights_only=True)['heads'] for m in MARKS[1:]}
    save(DOCS / f'{NAME}.json', dict(target=TARGET, primary_mark=PRIMARY, criteria=criteria,
        all_criteria_met=all(criteria.values()), curves=curves, versus_c1a_m4=versus, frozen_reproduced=reproduced,
        retention_families=families, retention_subtypes=subtypes, flat_velocity_rmse_ratio=ratio,
        velocity_rmse_ratio_by_terrain=velocity_ratios(retained),
        learning=json.loads(json.dumps(learning(records['train'], checkpoints))),
        missing_rays=sum(r.get('missing_rays', 0) for r in reports.values()),
        interactions=sum(r['interactions'] for r in reports.values()), train_episodes=len(records['train'])))
    print(TARGET, {m: (v['Frozen']['outcomes'].get('success', 0), v['Updated']['outcomes'].get('success', 0),
        round(v['mean_session_delta'], 4), np.round(v['ci95'], 4).tolist(), round(v['posture_delta'], 4)) for m, v in curves.items()})
    print(' criteria:', criteria, 'flat ratio', round(ratio, 3))
    print(' versus C1-a-M4:', {m: (v['D_success'], v['B_success'], np.round(v['ci95'], 3).tolist()) for m, v in versus.items()})
    print(' frozen identical:', {m: (v['identical'], v['n']) for m, v in reproduced.items()})


if __name__ == '__main__': main()
