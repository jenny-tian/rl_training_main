"""汇总一个目标的 C1-a-EA 结果。判定条件见 docs/lite3-online-c1a-ea-20261117/PLAN.md。

用法：C1A2_VARIANT=ea C1A2_TARGET=<目标> python reference_online_c1aea_results.py
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np

from reference_online_c1a2 import OUTPUT, TARGET, MARKS, PRIMARY, TERRAINS, VARIANT, jobs, protocol
from reference_online_c1a_results import FAMILY, paired_summary
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-c1a-ea-20261117'
FIXED = ROOT / 'logs/reference_six/online_c1a_heading_20261117' / TARGET


def read(base, name, file='episodes.json'):
    return json.loads((base / name / file).read_text())


def identity(r):
    return r['session'], r['condition'], r['repeat']


def against_fixed(ea, fixed, *, pairs=60, repeats=4):
    """D（EA 的 Updated）对 B（C1-a 的 Updated），相同会话和评估起点。"""
    d = {identity(r): r for r in ea if r['arm'] == 'Updated'}
    b = {identity(r): r for r in fixed if r['arm'] == 'Updated'}
    assert set(d) == set(b) and len(d) == pairs * repeats
    assert all(abs(d[k]['start_x'] - b[k]['start_x']) < 1e-6 for k in d)
    deltas = np.asarray([sum((d[k]['outcome'] == 'success') - (b[k]['outcome'] == 'success') for k in d if k[0] == s) / repeats
                         for s in range(pairs)])
    interval = np.quantile(deltas[np.random.default_rng(0).integers(0, pairs, (10000, pairs))].mean(1), [.025, .975]).tolist()
    count = lambda rows, outcome: sum(r['outcome'] == outcome for r in rows.values())
    return dict(D_success=count(d, 'success'), B_success=count(b, 'success'), n=len(d),
        D_posture=count(d, 'posture_failure'), B_posture=count(b, 'posture_failure'),
        mean_session_delta=float(deltas.mean()), ci95=interval, session_deltas=deltas.tolist(),
        gained=sum(d[k]['outcome'] == 'success' and b[k]['outcome'] != 'success' for k in d),
        lost=sum(b[k]['outcome'] == 'success' and d[k]['outcome'] != 'success' for k in d))


def frozen_reproduced(ea, fixed):
    a = {identity(r): r for r in ea if r['arm'] == 'Frozen'}
    b = {identity(r): r for r in fixed if r['arm'] == 'Frozen'}
    assert set(a) == set(b)
    return dict(n=len(a), identical=sum(a[k]['outcome'] == b[k]['outcome'] and a[k]['policy_steps'] == b[k]['policy_steps'] for k in a),
        max_final_root_difference=max(float(np.abs(np.subtract(a[k]['final_root'], b[k]['final_root'])).max()) for k in a))


def exploration(train):
    """训练中实际使用的探索：每个批次的标准差和回退原因；码偏离均值的实际幅度。"""
    result = {}
    for arm in ('Frozen', 'Updated'):
        rows = [r for r in train if r['arm'] == arm]
        batches = [r for r in rows if 'std_after' in r]
        std = np.asarray([r['std'] for r in rows])
        deviation = np.asarray([np.subtract(r['code'], r['mean']) for r in rows])
        ratio = np.asarray([r['execution_ratio'] for r in batches if r['execution_ratio'][0] is not None])
        result[arm] = dict(episodes=len(rows), batches=len(batches), status=dict(Counter(r['covariance_status'] for r in batches)),
            std_quantiles_10_50_90=np.quantile(std, [.1, .5, .9], axis=0).tolist(), mean_trace=float((std**2).sum(1).mean()),
            code_deviation_rms=np.sqrt((deviation**2).mean(0)).tolist(),
            execution_ratio_quantiles_05_50_95=np.quantile(ratio, [.05, .5, .95], axis=0).tolist(),
            train_success_by_third=[float(np.mean([r['outcome'] == 'success' for r in part]))
                for part in np.array_split(np.asarray(sorted(rows, key=lambda r: r['control_step']), dtype=object), 3)])
    return result


def main():
    assert VARIANT == 'ea'
    p = protocol()
    schedule = jobs()[1:]
    reports = {j['name']: read(OUTPUT, j['name'], 'report.json') for j in schedule}
    assert all(r['status'] == 'passed' for r in reports.values())
    records = {j['name']: read(OUTPUT, j['name'])['records'] for j in schedule}
    fixed = {j['name']: read(FIXED, j['name'])['records'] for j in schedule}
    target_rows = lambda source, m: [r for j in schedule if j['kind'] == 'target' and j['mark'] == m for r in source[j['name']]]
    curves = {str(m): paired_summary(target_rows(records, m)) for m in MARKS[1:]}
    versus = {str(m): against_fixed(target_rows(records, m), target_rows(fixed, m)) for m in MARKS[1:]}
    reproduced = {str(m): frozen_reproduced(target_rows(records, m), target_rows(fixed, m)) for m in MARKS[1:]}
    retained = [r for j in schedule if j['kind'] == 'retention' for r in records[j['name']]]
    families = {f: paired_summary([r for r in retained if FAMILY[r['condition']] == f],
                repeats=2 * sum(v == f for v in FAMILY.values())) for f in dict.fromkeys(FAMILY.values())}
    subtypes = {c: paired_summary([r for r in retained if r['condition'] == c], repeats=2) for c in TERRAINS}
    target, flat = curves[str(PRIMARY)], families['flat']
    ratio = flat['Updated']['xy_velocity_rmse'] / flat['Frozen']['xy_velocity_rmse']
    five = dict(target_gain_at_least_10pp=target['mean_session_delta'] >= p['success_gain_min'],
        ci_lower_above_zero=target['ci95'][0] > p['ci_lower_min'],
        posture_increase_at_most_2pp=target['posture_delta'] <= p['safety_increase_max'],
        every_family_loss_at_most_5pp=all(v['mean_session_delta'] >= -p['family_loss_max'] for v in families.values()),
        flat_velocity_ratio_at_most_1p10=ratio <= 1.10)
    primary = versus[str(PRIMARY)]
    final_mean = {}
    for r in records['train']:
        if r['arm'] == 'Updated': final_mean[r['session']] = r['b_after']
    save(DOCS / f'RESULTS_{TARGET}.json', dict(target=TARGET, primary_mark=PRIMARY, ea_passes=primary['ci95'][0] > 0,
        D_versus_B=versus, frozen_reproduced=reproduced, D_versus_frozen=curves, five_criteria=five,
        all_five_met=all(five.values()), retention_families=families, retention_subtypes=subtypes,
        flat_velocity_rmse_ratio=ratio, exploration=exploration(records['train']),
        fixed_code_deviation_rms={arm: np.sqrt(np.mean([np.subtract(r['code'], r['mean'])**2 for r in fixed['train'] if r['arm'] == arm], 0)).tolist()
                                  for arm in ('Frozen', 'Updated')},
        interactions=sum(r['interactions'] for r in reports.values()), adapter_updates=reports['train']['adapter_updates'],
        session_updates=reports['train']['session_updates'], final_means=[final_mean[s] for s in sorted(final_mean)],
        train_episodes=len(records['train'])))
    print(TARGET, 'D-B', {m: (v['D_success'], v['B_success'], round(v['mean_session_delta'], 4), np.round(v['ci95'], 4).tolist()) for m, v in versus.items()})
    print(' EA passes:', primary['ci95'][0] > 0, 'five:', five)


if __name__ == '__main__': main()
