"""Judge autonomous imitation in closed loop without the scripted prefix."""
import argparse
import json
from pathlib import Path
import statistics
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','protocol','run','audit','training-audit','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--baseline-run');parser.add_argument('--baseline-audit')
    args=parser.parse_args();p=json.loads(Path(args.plan).read_text());child=json.loads(Path(args.protocol).read_text());root=Path(args.run);r=json.loads((root/'report.json').read_text());a=json.loads(Path(args.audit).read_text());trained=json.loads(Path(args.training_audit).read_text())
    assert r['status']=='complete' and a['status']==trained['status']=='verified'
    assert 'action_prefix' not in child
    assert child['evaluation_seed']==p['pilot_seed']
    evidence_paths=[Path(args.plan),Path(args.protocol),root/'report.json',Path(args.audit),Path(args.training_audit)]
    base_report=r;base_audit=a
    if args.baseline_run:
        base_path=Path(args.baseline_run)/'report.json';base_report=json.loads(base_path.read_text());base_audit=json.loads(Path(args.baseline_audit).read_text())
        assert set(child['models'])=={'trial'} and base_report['status']=='complete' and base_audit['status']=='verified'
        assert sha256_file(base_path)==child['reused_baseline_report_sha256']==base_audit['evidence'][str(base_path)]
        evidence_paths.extend((base_path,Path(args.baseline_audit)))
    else:assert set(child['models'])=={'base','trial'}
    assert child['recovery_resets']['fixed_train_case']==26 and child['recovery_resets']['split']=='train'
    assert r['protocol_sha256']==sha256_file(args.protocol)
    assert a['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
    assert base_report['models']['base']['actual_sha256']==p['base_sha256'] and r['models']['trial']['actual_sha256']==trained['final_checkpoint']['sha256']
    for path,digest in trained['evidence'].items():assert sha256_file(path)==digest
    for condition in ('course','recovery'):assert base_report['phases'][condition+'-base']['initial_sha256']==r['phases'][condition+'-trial']['initial_sha256']
    results={}
    for model in ('base','trial'):
        selected_report=base_report if model=='base' else r;selected_audit=base_audit if model=='base' else a
        selected_root=Path(args.baseline_run) if model=='base' and args.baseline_run else root
        course=selected_audit['phases']['course-'+model];episodes=[e for e in selected_report['phases']['recovery-'+model]['episodes'] if e['case']==26];groups={}
        delays=json.loads((selected_root/f'recovery-{model}/delays.json').read_text())
        for e in episodes:
            signature=[]
            for joint in ('Hip','Knee'):
                values=[delays[joint][key][e['env_id']] for key in ('efforts_delay_buffer','positions_delay_buffer','velocities_delay_buffer')]
                assert len(set(values))==1;signature.append(str(values[0]))
            item=groups.setdefault('/'.join(signature),dict(success=0,total=0));item['success']+=e['recovery_success'];item['total']+=1
        results[model]=dict(course_by_terrain=course['by_terrain'],course_total=sum(course['by_terrain'].values()),course_rms_m=course['rough_mean_rms_m'],course_peak_m=course['rough_peak_m'],
            case26_recovery=sum(e['recovery_success'] for e in episodes),case26_total=len(episodes),case26_rms_m=statistics.mean(e['body_lateral_rms_m'] for e in episodes),case26_peak_m=max(e['max_body_lateral_m'] for e in episodes),case26_groups=groups)
    b=results['base'];t=results['trial'];c=p['pilot_acceptance']
    assert b['case26_total']==t['case26_total']==24
    assert b['case26_groups'].keys()==t['case26_groups'].keys()
    assert all(b['case26_groups'][key]['total']==value['total'] for key,value in t['case26_groups'].items())
    gates=dict(recovery=t['case26_recovery']>=c['case26_minimum_completed_recoveries'] and t['case26_recovery']>b['case26_recovery'],
        delay_strata=set(t['case26_groups'])=={'0/0','0/1','1/0','1/1'} and all(v['success']/v['total']>=c['minimum_delay_stratum_fraction'] for v in t['case26_groups'].values()),
        recovery_lateral=t['case26_peak_m']<=c['case26_maximum_lateral_m'],ordinary_completion=all(t['course_by_terrain'][key]>=b['course_by_terrain'][key] for key in b['course_by_terrain']),
        ordinary_rms=t['course_rms_m']<=c['ordinary_rms_ratio_maximum']*b['course_rms_m'],ordinary_peak=t['course_peak_m']<=b['course_peak_m'])
    assert r['ledger']['charged']==r['ledger']['confirmed'] and r['ledger']['confirmed']<=min(p['max_pilot_controls'],child['max_controls'])
    write_json_atomic(args.output,dict(status='verified',results=results,gates=gates,pilot_passed=all(gates.values()),controls=r['ledger']['confirmed'],uncertain=0,
        evidence={str(x):sha256_file(x) for x in evidence_paths},
        scope='Autonomous48D mean actor, no prefix/time/case flag/teacher takeover. TRAIN case26 pilot is not original7-source recovery qualification or independent learning replication.'))


if __name__=='__main__':main()
