"""Registered session-level statistics for a completed C1-b pilot."""
from __future__ import annotations

from collections import Counter

import numpy as np

from reference_online_c1b import completed
from reference_online_c1b_reset_api import OUTPUT,DOCS,registration
from reference_online_targets import read
from reference_stairs_eval import save,sha

FAMILY = dict(flat='flat',rough='rough',slope_up='slope',slope_down='slope',
    stairs_up='stairs',stairs_down='stairs',friction_lateral='friction',
    friction_longitudinal='friction',slope_rough_up='slope_rough',slope_rough_down='slope_rough')


def describe(records):
    n=len(records)
    outcomes=Counter(r['outcome'] for r in records)
    return dict(n=n,successes=outcomes['success'],success_rate=outcomes['success']/n,
        posture_failures=outcomes['posture_failure'],posture_rate=outcomes['posture_failure']/n,
        mean_progress=float(np.mean([r['mean_progress'] for r in records])),
        mean_policy_steps=float(np.mean([r['policy_steps'] for r in records])),
        outcomes=dict(outcomes),one_episode_resolution=1/n)


def paired_summary(records):
    result={arm:describe([r for r in records if r['arm']==arm]) for arm in ('Frozen','Updated')}
    result['success_delta']=result['Updated']['success_rate']-result['Frozen']['success_rate']
    result['posture_delta']=result['Updated']['posture_rate']-result['Frozen']['posture_rate']
    paired={}
    for r in records:
        pair=paired.setdefault((r['session'],r['condition'],r['repeat']),{})
        assert r['arm'] not in pair,'duplicate paired evaluation identity'
        pair[r['arm']]=r['outcome']=='success'
    assert all(set(v)=={'Frozen','Updated'} for v in paired.values())
    result['gained_starts']=sum(v['Updated'] and not v['Frozen'] for v in paired.values())
    result['lost_starts']=sum(v['Frozen'] and not v['Updated'] for v in paired.values())
    return result


def main():
    registration()
    reports={name:completed(OUTPUT/name) for name in ('smoke_3','train','target','retention')}
    for name in reports:
        audit=read(DOCS/f'AUDIT_{name}.json')
        assert audit['report']==reports[name] and audit['pairing_verified']
    episodes={name:read(OUTPUT/name/'episodes.json') for name in ('train','target','retention')}
    target=episodes['target']['records']; retention=episodes['retention']['records']
    assert len(target)==256 and len(retention)==320
    sessions=[dict(session=i,**paired_summary([r for r in target if r['session']==i])) for i in range(8)]
    assert all(s[arm]['n']==16 for s in sessions for arm in ('Frozen','Updated'))
    diffs=np.array([s['success_delta'] for s in sessions])
    rng=np.random.default_rng(0)
    interval=np.quantile(diffs[rng.integers(0,8,size=(10000,8))].mean(1),[.025,.975]).tolist()
    target_result=paired_summary(target)
    target_result.update(sessions=sessions,mean_session_delta=float(diffs.mean()),
                         session_bootstrap_ci95=interval,bootstrap_samples=10000,bootstrap_seed=0)
    subtypes={condition:paired_summary([r for r in retention if r['condition']==condition]) for condition in FAMILY}
    assert all(v[arm]['n']==16 for v in subtypes.values() for arm in ('Frozen','Updated'))
    families={family:paired_summary([r for r in retention if FAMILY[r['condition']]==family]) for family in dict.fromkeys(FAMILY.values())}
    criteria=dict(target_gain_at_least_10pp=bool(diffs.mean()>=.10),ci_lower_above_zero=bool(interval[0]>0),
        posture_increase_at_most_2pp=bool(target_result['posture_delta']<=.02),
        each_family_loss_at_most_5pp=all(v['success_delta']>=-.05 for v in families.values()))
    train=[]
    for i in range(8):
        arms={arm:describe([r for r in episodes['train']['records'] if r['session']==i and r['arm']==arm]) for arm in ('Frozen','Updated')}
        train.append(dict(session=i,arms=arms,
            adapter_updates=reports['train']['session_updates'][2*i+1],
            counts={arm:{k:v[2*i+int(arm=='Updated')] for k,v in reports['train']['counts'].items()} for arm in ('Frozen','Updated')}))
    attempts=read(OUTPUT/'attempts.json')
    assert sum(a['reserved_steps'] for a in attempts)==953600
    result=dict(status='completed_mechanism_pilot',target=target_result,retention_families=families,
        retention_subtypes=subtypes,training_sessions=train,criteria=criteria,all_criteria_met=all(criteria.values()),
        interactions={name:report['interactions'] for name,report in reports.items()},
        actual_total_interactions=sum(report['interactions'] for report in reports.values()),
        reserved_total_interactions=953600,failed_prior_slots=19200,
        online_adapter_updates=reports['train']['adapter_updates'],base_optimizer_updates=0,
        audits=[dict(path=str(DOCS/f'AUDIT_{name}.json'),sha256=sha(DOCS/f'AUDIT_{name}.json')) for name in reports])
    save(DOCS/'RESULTS.json',result)
    print('target',target_result)
    print('retention',families)
    print('criteria',criteria)


if __name__=='__main__': main()
