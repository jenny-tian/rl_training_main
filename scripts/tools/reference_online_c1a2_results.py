"""汇总一个目标的 C1-a（闭环打开）结果。判定条件见 docs/lite3-online-c1a-heading-20261117/PLAN.md。"""
from __future__ import annotations

import json

from reference_online_c1a2 import OUTPUT, TARGET, MARKS, PRIMARY, TERRAINS, jobs, protocol
from reference_online_c1a_results import FAMILY, paired_summary
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-c1a-heading-20261117'


def read(name, file='episodes.json'):
    return json.loads((OUTPUT / name / file).read_text())


def main():
    p = protocol()
    schedule = jobs()[1:]
    reports = {j['name']: read(j['name'], 'report.json') for j in schedule}
    assert all(r['status'] == 'passed' for r in reports.values())
    records = {j['name']: read(j['name'])['records'] for j in schedule}
    curves = {str(m): paired_summary([r for j in schedule if j['kind'] == 'target' and j['mark'] == m for r in records[j['name']]])
              for m in MARKS[1:]}
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
    train = [r for r in records['train'] if r['arm'] == 'Updated']
    final_mean = {}
    for r in train: final_mean[r['session']] = r['b_after']
    save(DOCS / f'RESULTS_{TARGET}.json', dict(target=TARGET, primary_mark=PRIMARY, primary=target, curves=curves,
        retention_families=families, retention_subtypes=subtypes, flat_velocity_rmse_ratio=ratio,
        criteria=criteria, all_criteria_met=all(criteria.values()),
        interactions=sum(r['interactions'] for r in reports.values()),
        adapter_updates=reports['train']['adapter_updates'], session_updates=reports['train']['session_updates'],
        final_means=[final_mean[s] for s in sorted(final_mean)], train_episodes=len(records['train'])))
    print(TARGET, criteria)


if __name__ == '__main__': main()
