"""Registered C1-a summary; 20k primary, remaining snapshots descriptive only."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from reference_online_c1a import OUTPUT,DOCS,MARKS,TERRAINS,completed,jobs,source_manifest
from reference_online_targets import read
from reference_stairs_eval import save,sha

FAMILY = dict(flat='flat',rough='rough',slope_up='slope',slope_down='slope',stairs_up='stairs',stairs_down='stairs',
    friction_lateral='friction',friction_longitudinal='friction',slope_rough_up='slope_rough',slope_rough_down='slope_rough')


def paired_summary(records,*,pairs=60,repeats=4):
    grouped = {}
    for r in records:
        key = r['session'],r['condition'],r['repeat']
        arms = grouped.setdefault(key,{})
        if r['arm'] in arms: raise ValueError('duplicate paired identity')
        arms[r['arm']] = r
    if any(set(v)!={'Frozen','Updated'} for v in grouped.values()): raise ValueError('unpaired outcome')
    if set(k[0] for k in grouped)!=set(range(pairs)): raise ValueError('missing sessions')
    deltas = []
    for session in range(pairs):
        group = [v for k,v in grouped.items() if k[0]==session]
        if len(group)!=repeats: raise ValueError('incorrect starts per session')
        deltas.append(sum((g['Updated']['outcome']=='success')-(g['Frozen']['outcome']=='success') for g in group)/repeats)
    values = np.asarray(deltas)
    rng = np.random.default_rng(0)
    interval = np.quantile(values[rng.integers(0,pairs,(10000,pairs))].mean(1),[.025,.975]).tolist()
    arms = {}
    for arm in ('Frozen','Updated'):
        selected = [r for r in records if r['arm']==arm]
        outcomes = Counter(r['outcome'] for r in selected)
        arms[arm] = dict(n=len(selected),outcomes=dict(outcomes),success_rate=outcomes['success']/len(selected),
            posture_rate=outcomes['posture_failure']/len(selected),
            xy_velocity_rmse=math.sqrt(sum(r['xy_velocity_error_sum'] for r in selected)/sum(r['policy_steps'] for r in selected)))
    return dict(**arms,session_deltas=deltas,mean_session_delta=float(values.mean()),ci95=interval,
        posture_delta=arms['Updated']['posture_rate']-arms['Frozen']['posture_rate'],
        gained=sum(g['Updated']['outcome']=='success' and g['Frozen']['outcome']!='success' for g in grouped.values()),
        lost=sum(g['Frozen']['outcome']=='success' and g['Updated']['outcome']!='success' for g in grouped.values()))


def main():
    assert source_manifest()==read(OUTPUT/'lock.json')
    schedule = jobs(); expected = [j['name'] for j in schedule]
    assert [a['name'] for a in read(OUTPUT/'attempts.json')]==expected
    reports,records = {},{}
    for job in schedule:
        name = job['name']; reports[name] = completed(name)
        audit = read(DOCS/f'AUDIT_{name}.json')
        assert audit['status']=='passed' and audit['report_sha256']==sha(OUTPUT/name/'report.json')
        assert audit['trace_index_sha256']==sha(OUTPUT/name/'trace_index.json')
        records[name] = read(OUTPUT/name/'episodes.json')['records']
    curves = {}
    for mark in MARKS:
        rows = [r for j in schedule if j['kind']=='target' and j['mark']==mark for r in records[j['name']]]
        curves[str(mark)] = paired_summary(rows)
    retained = [r for j in schedule if j['kind']=='retention' for r in records[j['name']]]
    families = {family:paired_summary([r for r in retained if FAMILY[r['condition']]==family],
                repeats=2*sum(v==family for v in FAMILY.values())) for family in dict.fromkeys(FAMILY.values())}
    subtypes = {case:paired_summary([r for r in retained if r['condition']==case],repeats=2) for case in TERRAINS}
    target,flat = curves['20000'],families['flat']
    base_rmse,adapted_rmse = flat['Frozen']['xy_velocity_rmse'],flat['Updated']['xy_velocity_rmse']
    ratio = adapted_rmse/base_rmse if base_rmse>0 else 1. if adapted_rmse==0 else None
    criteria = dict(target_gain_at_least_10pp=target['mean_session_delta']>=.10,ci_lower_above_zero=target['ci95'][0]>0.,
        posture_increase_at_most_2pp=target['posture_delta']<=.02,
        every_family_loss_at_most_5pp=all(v['mean_session_delta']>=-.05 for v in families.values()),
        flat_velocity_ratio_at_most_1p10=ratio is not None and ratio<=1.10)
    result = dict(status='completed_C1a_single_target_stage',primary_mark=20000,primary=target,
        descriptive_curves=curves,retention_families=families,retention_subtypes=subtypes,
        flat_velocity_rmse_ratio=ratio,criteria=criteria,all_criteria_met=all(criteria.values()),
        reserved_interactions=sum(j['reserved_steps'] for j in schedule),
        actual_interactions=sum(v['interactions'] for v in reports.values()),
        adapter_updates=reports['train']['adapter_updates'],base_updates=0,
        caveats=['No confirmed fixed-code headroom; this is a separately registered mechanism study.',
                 'Not the original multi-target qualification; retention uses current ranges.',
                 'Secondary snapshots cannot replace a failed 20k primary endpoint.',
                 'No EA variant or later stage is automatically launched.'])
    save(DOCS/'RESULTS.json',result)
    print('C1-a criteria:',criteria)


if __name__=='__main__': main()
