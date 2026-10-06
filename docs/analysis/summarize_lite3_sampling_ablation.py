"""Compare the dedicated-recovery experiment to its already-paid matched control."""
import argparse
import json
from pathlib import Path
import statistics

from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def metrics(phase):
    episodes=phase['episodes'];rough=[e for e in episodes if e['terrain']!='flat']
    return dict(success=sum(e['success'] for e in episodes),total=len(episodes),
        by_terrain={t:sum(e['success'] for e in episodes if e['terrain']==t) for t in ('flat','101','107')},
        rough_rms=statistics.mean(e['body_lateral_rms_m'] for e in rough),rough_peak=max(e['max_body_lateral_m'] for e in rough),
        recovered=sum(e['recovery_success'] for e in rough),
        recovery_by_terrain={t:sum(e['recovery_success'] for e in rough if e['terrain']==t) for t in ('101','107')},
        recovery_by_case={str(i):sum(e['recovery_success'] for e in rough if e['case']==i) for i in sorted({e['case'] for e in rough})})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','control-audit','candidate-audit','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();p=json.loads(Path(args.protocol).read_text())
    control_path=Path(p['control_run'])/'evaluation/report.json';trial_path=Path(p['candidate_run'])/'evaluation/report.json'
    assert sha256_file(control_path)==p['control_eval_report_sha256']
    control=json.loads(control_path.read_text());trial=json.loads(trial_path.read_text())
    for report,path,audit_file in ((control,control_path,args.control_audit),(trial,trial_path,args.candidate_audit)):
        audit=json.loads(Path(audit_file).read_text())
        assert report['status']=='complete' and audit['status']=='verified'
        assert audit['evidence'][str(path)]==sha256_file(path)
    assert sha256_file(p['control_train_protocol'])==p['control_train_sha256']
    assert sha256_file(p['candidate_train_protocol'])==p['candidate_train_sha256']
    assert sha256_file(p['candidate_eval_protocol'])==p['candidate_eval_sha256']
    for condition in ('course','recovery'):
        assert trial['phases'][condition+'-trial']['initial_sha256']==control['phases'][condition+'-'+p.get('control_model','candidate')]['initial_sha256']
    control_model=p.get('control_model','candidate')
    results={name:metrics(phase) for name,phase in control['phases'].items()}
    for condition in ('course','recovery'):
        results[condition+'-candidate']=metrics(control['phases'][condition+'-'+control_model])
    if p.get('baseline_run'):
        baseline_path=Path(p['baseline_run'])/'evaluation/report.json'
        assert sha256_file(baseline_path)==p['baseline_eval_report_sha256']
        baseline=json.loads(baseline_path.read_text())
        audit=json.loads(Path(p['baseline_audit']).read_text())
        assert audit['status']=='verified' and audit['evidence'][str(baseline_path)]==sha256_file(baseline_path)
        for condition in ('course','recovery'):
            for model in ('base','previous'):
                phase=baseline['phases'][condition+'-'+model]
                assert phase['initial_sha256']==trial['phases'][condition+'-trial']['initial_sha256']
                results[condition+'-'+model]=metrics(phase)
    results.update({name:metrics(phase) for name,phase in trial['phases'].items()})
    a=results['course-trial'];b=results['course-candidate'];r=results['recovery-trial'];rb=results['recovery-candidate']
    criteria=p['acceptance']
    gain=dict(recovery_gain=r['recovered']-rb['recovered']>=criteria['minimum_recovery_gain_trials'],
        course_completion=all(a['by_terrain'][t]>=b['by_terrain'][t] for t in a['by_terrain']),
        course_rms=a['rough_rms']<=criteria['maximum_course_rms_ratio_vs_control']*b['rough_rms'],course_peak=a['rough_peak']<=b['rough_peak'])
    prev=results['course-previous']
    qualification=dict(course_completion=all(a['by_terrain'][t]>=results['course-'+m]['by_terrain'][t] for m in ('base','previous') for t in a['by_terrain']),
        course_rms=a['rough_rms']<=.98*prev['rough_rms'],course_peak=a['rough_peak']<=prev['rough_peak'],
        recovery_noninferior=all(r['recovery_by_terrain'][t]>=results['recovery-'+m]['recovery_by_terrain'][t] for m in ('base','previous') for t in ('101','107')),
        recovery_80percent=r['recovered']>=.8*48,every_case=all(v>0 for v in r['recovery_by_case'].values()))
    train=json.loads((Path(p['candidate_run'])/'train/report.json').read_text())
    controls=train['ledger']['confirmed']+trial['ledger']['confirmed']
    assert controls<=p['max_new_controls']
    out=dict(status='audited_sampling_ablation',paired_initial_states=True,results=results,ablation_gates=gain,
        ablation_gain_passed=all(gain.values()),teacher_gates=qualification,teacher_gate_passed=all(qualification.values()),
        new_controls=controls,control_cost_reused_not_repeated=True,final_checkpoint=train['final_checkpoint'],
        evidence={str(f):sha256_file(f) for f in (control_path,trial_path,Path(args.protocol),Path(args.control_audit),Path(args.candidate_audit))},
        scope='One matched learning-seed sampling ablation; original qualification and independent-pair replication remain separate.')
    write_json_atomic(args.output,out);print(json.dumps({k:v for k,v in out.items() if k!='evidence'},indent=2))


if __name__=='__main__':main()
