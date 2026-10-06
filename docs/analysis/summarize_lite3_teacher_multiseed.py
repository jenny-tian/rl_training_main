"""Aggregate every preregistered frozen-policy seed after independent audits."""
import argparse
import json
import math
from pathlib import Path
import statistics

from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def wilson(success,total):
    if not 0<=success<=total or total<1:raise ValueError('valid nonempty Bernoulli sample required')
    z=1.959963984540054;p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-half),min(1.,center+half)]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();master=json.loads(Path(args.protocol).read_text())
    all_episodes={m:[] for m in ('previous','teacher')};seeds=[];controls=0;evidence={};model_hashes={m:set() for m in all_episodes}
    for run in master['runs']:
        assert sha256_file(run['protocol'])==run['sha256']
        root=Path(run['output']);report=json.loads((root/'report.json').read_text());audit=json.loads((root/'audit.json').read_text())
        assert report['status']=='complete' and audit['status']=='verified'
        assert report['protocol_sha256']==run['sha256']
        assert report['phases']['course-previous']['initial_sha256']==report['phases']['course-teacher']['initial_sha256']
        assert report['ledger']['uncertain_transitions']==0
        controls+=report['ledger']['confirmed'];row={'seed':run['seed']}
        for mode in all_episodes:
            eps=report['phases'][f'course-{mode}']['episodes'];assert len(eps)==64
            all_episodes[mode].extend(eps);model_hashes[mode].add(report['models'][mode]['actual_sha256'])
            rough=[e for e in eps if e['terrain']!='flat']
            row[mode]=dict(success=sum(e['success'] for e in eps),total=len(eps),
                by_terrain={t:sum(e['success'] for e in eps if e['terrain']==t) for t in ('flat','101','107')},
                rough_mean_trial_rms_m=statistics.mean(e['body_lateral_rms_m'] for e in rough),
                rough_peak_m=max(e['max_body_lateral_m'] for e in rough))
        seeds.append(row)
        for file in (root/'report.json',root/'audit.json',Path(run['protocol'])):evidence[str(file)]=sha256_file(file)
    assert controls<=master['max_controls'] and all(len(v)==1 for v in model_hashes.values())
    results={}
    for mode,eps in all_episodes.items():
        success=sum(e['success'] for e in eps);rough=[e for e in eps if e['terrain']!='flat']
        results[mode]=dict(success=success,total=len(eps),completion_rate=success/len(eps),wilson95=wilson(success,len(eps)),
            by_terrain={t:dict(success=sum(e['success'] for e in eps if e['terrain']==t),total=sum(e['terrain']==t for e in eps)) for t in ('flat','101','107')},
            rough_mean_trial_rms_m=statistics.mean(e['body_lateral_rms_m'] for e in rough),rough_peak_m=max(e['max_body_lateral_m'] for e in rough))
    c=results['teacher'];b=results['previous'];a=master['acceptance']
    gates=dict(total_completion=c['completion_rate']>=a['minimum_total_completion_rate'],
        per_terrain_completion=all(v['success']/v['total']>=a['minimum_per_terrain_completion_rate'] for v in c['by_terrain'].values()),
        completion_vs_previous=all(c['by_terrain'][t]['success']>=b['by_terrain'][t]['success'] for t in c['by_terrain']),
        rough_rms=c['rough_mean_trial_rms_m']<=a['maximum_rough_mean_trial_rms_m'],rough_peak=c['rough_peak_m']<=a['maximum_rough_peak_m'])
    output=dict(status='audited_frozen_verification',frozen_course_gate_passed=all(gates.values()),gates=gates,
        results=results,seeds=seeds,charged=controls,confirmed=controls,uncertain=0,
        model_sha256={m:next(iter(h)) for m,h in model_hashes.items()},evidence=evidence,
        master_protocol_sha256=sha256_file(args.protocol),scope=master['scope'])
    write_json_atomic(args.output,output);print(json.dumps({k:v for k,v in output.items() if k!='evidence'},indent=2))


if __name__=='__main__':main()
