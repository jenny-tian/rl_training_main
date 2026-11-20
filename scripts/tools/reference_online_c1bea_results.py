"""汇总一个目标的 C1-b-EA 结果。判定条件见 docs/lite3-online-c1b-ea-20261120/PLAN.md。

用法：C1A2_VARIANT=s2ea C1A2_TARGET=<目标> python reference_online_c1bea_results.py
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np

from reference_online_c1a2 import OUTPUT, TARGET, MARKS, PRIMARY, TERRAINS, VARIANT, jobs, protocol
from reference_online_c1a_results import FAMILY, paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced, read
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-c1b-ea-20261120'
FIXED = ROOT / 'logs/reference_six/online_c1b_s2n_20261119' / TARGET  # 做法二：固定标准差的对照
REGISTERED = (20000, 40000, 60000)  # 做法二登记过的快照；5000、10000 是后加的，只作描述


def exploration(train):
    """训练中两个摆动幅度实际使用的探索：标准差、回退原因、执行比例、参数偏离均值的幅度。"""
    result = {}
    for arm in ('Frozen', 'Updated'):
        rows = sorted((r for r in train if r['arm'] == arm), key=lambda r: r['control_step'])
        swing = lambda r, key: np.asarray(r[key])[2:]
        deviation = np.asarray([swing(r, 'code') - swing(r, 'mean') for r in rows])
        entry = dict(episodes=len(rows), code_deviation_rms=np.sqrt((deviation**2).mean(0)).tolist(),
            train_success_by_third=[float(np.mean([r['outcome'] == 'success' for r in part])) for part in np.array_split(np.asarray(rows, dtype=object), 3)])
        if 'std' in rows[0]:
            batches = [r for r in rows if 'std_after' in r]
            std = np.asarray([swing(r, 'std') for r in rows])
            ratio = np.asarray([r['execution_ratio'] for r in batches if r['execution_ratio'][0] is not None])
            entry.update(batches=len(batches), status=dict(Counter(r['covariance_status'] for r in batches)),
                std_quantiles_10_50_90=np.quantile(std, [.1, .5, .9], axis=0).tolist(), mean_trace=float((std**2).sum(1).mean()),
                std_by_third=[np.asarray([swing(r, 'std') for r in part]).mean(0).tolist() for part in np.array_split(np.asarray(rows, dtype=object), 3)],
                execution_ratio_quantiles_05_50_95=np.quantile(ratio, [.05, .5, .95], axis=0).tolist())
        result[arm] = entry
    return result


def parameters(base, marks):
    import torch
    result = {}
    for mark in marks:
        heads = torch.load(base / 'train' / f'at_{mark}.pt', map_location='cpu', weights_only=True)['heads']
        z = np.tanh(np.asarray([np.asarray(h['w']) for h in heads[1::2]])[:, 2:])
        result[str(mark)] = dict(mean_tanh_front_rear=z.mean(0).tolist(), share_above_0p5=(z > .5).mean(0).tolist(),
                                 both_above_0p5=float((z > .5).all(1).mean()))
    return result


def main():
    assert VARIANT == 's2ea'
    p = protocol()
    schedule = [j for j in jobs()[1:] if j['kind'] in ('train', 'target', 'retention')]
    reports = {j['name']: read(OUTPUT, j['name'], 'report.json') for j in schedule}
    assert all(r['status'] == 'passed' for r in reports.values())
    records = {j['name']: read(OUTPUT, j['name'])['records'] for j in schedule}
    evaluations = [j for j in schedule if j['kind'] == 'target']
    fixed = {j['name']: read(FIXED, j['name'])['records'] for j in evaluations}
    fixed['train'] = read(FIXED, 'train')['records']
    rows = lambda source, m: [r for j in evaluations if j['mark'] == m for r in source[j['name']]]
    marks = MARKS[1:]
    curves = {str(m): paired_summary(rows(records, m)) for m in marks}  # D 对本轮 Frozen
    fixed_curves = {str(m): paired_summary(rows(fixed, m)) for m in marks}  # B 对做法二的 Frozen
    versus = {str(m): against_fixed(rows(records, m), rows(fixed, m)) for m in marks}
    reproduced = {str(m): frozen_reproduced(rows(records, m), rows(fixed, m)) for m in marks}
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
    save(DOCS / f'RESULTS_{TARGET}.json', dict(target=TARGET, primary_mark=PRIMARY, registered_marks=list(REGISTERED),
        descriptive_early_marks=[m for m in marks if m not in REGISTERED], ea_passes=primary['ci95'][0] > 0,
        D_versus_B=versus, frozen_reproduced=reproduced, D_versus_frozen=curves, B_versus_frozen=fixed_curves,
        five_criteria=five, all_five_met=all(five.values()), retention_families=families, retention_subtypes=subtypes,
        flat_velocity_rmse_ratio=ratio, exploration=exploration(records['train']), fixed_exploration=exploration(fixed['train']),
        parameters=parameters(OUTPUT, marks), fixed_parameters=parameters(FIXED, marks),
        session_updates=reports['train']['session_updates'], adapter_updates=reports['train']['adapter_updates'],
        interactions=sum(r['interactions'] for r in reports.values()), train_episodes=len(records['train'])))
    count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
    print(TARGET, 'D-B', {m: (v['D_success'], v['B_success'], round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist()) for m, v in versus.items()})
    print(' Frozen -> B, D:', {m: (count(fixed_curves[m], 'Frozen'), count(fixed_curves[m], 'Updated'), count(curves[m], 'Updated')) for m in curves})
    print(' frozen identical:', {m: (v['identical'], v['n']) for m, v in reproduced.items()})
    print(' EA passes:', primary['ci95'][0] > 0, 'five:', five, 'flat ratio', round(ratio, 3))


if __name__ == '__main__': main()
