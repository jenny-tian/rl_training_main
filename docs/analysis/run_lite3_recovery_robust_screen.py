"""Replay all pinned positive TRAIN prefixes across existing actuator-delay strata."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/reinforcement_learning'))
from run_lite3_recovery_sequence_search import phase_metrics


def strata(folder,rows):
    delays=json.loads((folder/'evaluation/smoke-trial/delays.json').read_text());groups={}
    for row in rows:
        i=row['env_id'];signature=[]
        for joint in ('Hip','Knee'):
            values=[delays[joint][f][i] for f in ('efforts_delay_buffer','positions_delay_buffer','velocities_delay_buffer')]
            assert len(set(values))==1 and values[0] in (0,1)
            signature.append(str(values[0]))
        key='/'.join(signature);entry=groups.setdefault(key,dict(positive=0,total=0));entry['positive']+=row['short_recovery'];entry['total']+=1
    assert set(groups)=={'0/0','0/1','1/0','1/1'}
    worst=min(v['positive']/v['total'] for v in groups.values());total=sum(v['positive'] for v in groups.values())
    return dict(groups=groups,worst_fraction=worst,positive=total,mean_score=float(np.mean([v['score'] for v in rows])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','output','device'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();p=json.loads(Path(args.protocol).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',protocol_sha256=sha256_file(args.protocol),driver_sha256=sha256_file(__file__),phases={},confirmation=[])
    root=Path(__file__).resolve().parents[2]
    def save():
        charged=confirmed=0
        for path in out.glob('*/evaluation/ledger.json'):
            ledger=json.loads(path.read_text());charged+=ledger['charged'];confirmed+=ledger['confirmed']
        assert charged<=p['max_controls']
        report['ledger']=dict(charged=charged,confirmed=confirmed,uncertain=charged-confirmed,max_controls=p['max_controls'])
        write_json_atomic(out/'report.json',report)
    def run(label,seed,code):
        save();assert report['ledger']['charged']+64*p['steps']<=p['max_controls']
        folder=out/label;folder.mkdir()
        child=copy.deepcopy(p['evaluation_template']);child.update(evaluation_seed=seed,steps=p['steps'],max_controls=64*p['steps'],robust_parent_sha256=report['protocol_sha256'])
        child['recovery_resets']['fixed_train_case']=26
        if code is not None:
            prefix=np.zeros((64,p['knots'],12),np.float32);prefix[p['target_env_ids']]=code
            np.save(folder/'prefix.npy',prefix);child['action_prefix']=dict(path=str((folder/'prefix.npy').resolve()),sha256=sha256_file(folder/'prefix.npy'),knots=p['knots'],hold_steps=p['hold_steps'],bound=p['bound'])
        write_json_atomic(folder/'protocol.json',child)
        with (folder/'physics.log').open('w') as log:
            subprocess.run([sys.executable,str(root/'scripts/reinforcement_learning/evaluate_lite3_recovery_teacher.py'),'--protocol',str(folder/'protocol.json'),'--output',str(folder/'evaluation'),'--device',args.device,'--headless'],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (folder/'audit.log').open('w') as log:
            subprocess.run([sys.executable,str(root/'docs/analysis/audit_lite3_recovery_teacher.py'),'--run',str(folder/'evaluation'),'--protocol',str(folder/'protocol.json'),'--output',str(folder/'audit.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads((folder/'audit.json').read_text())['status']=='verified'
        rows,initial=phase_metrics(folder,p);entry=dict(rows=rows,initial_sha256=initial,summary=strata(folder,rows),audit_sha256=sha256_file(folder/'audit.json'))
        report['phases'][label]=entry;save();print(json.dumps(dict(phase=label,summary=entry['summary'],controls=report['ledger']['confirmed'])),flush=True)
        return entry
    save()
    try:
        codes=[]
        for candidate in p['candidates']:
            path=Path(candidate['path']);assert sha256_file(path)==candidate['sha256'];codes.append(np.load(path,allow_pickle=False))
        baseline=Path(p['reused_baseline_report']);assert sha256_file(baseline)==p['reused_baseline_sha256']
        initial=json.loads(baseline.read_text())['phases']['smoke-trial']['initial_sha256']
        for i,code in enumerate(codes):
            entry=run(f'candidate{i:02d}',p['search_seed'],code);assert entry['initial_sha256']==initial
        selected=max(range(len(codes)),key=lambda i:tuple(report['phases'][f'candidate{i:02d}']['summary'][k] for k in ('worst_fraction','positive','mean_score')))
        summary=report['phases'][f'candidate{selected:02d}']['summary'];qualified=summary['worst_fraction']>=p['minimum_stratum_fraction'] and summary['positive']>=p['minimum_positives']
        report.update(selected_candidate=selected,screen_passed=qualified);save()
        if qualified:
            for seed in p['confirmation_seeds']:
                a=run(f'confirm{seed}-base',seed,None);b=run(f'confirm{seed}-prefix',seed,codes[selected]);assert a['initial_sha256']==b['initial_sha256']
                report['confirmation'].append(dict(seed=seed,baseline=a['summary'],prefix=b['summary']));save()
        report['confirmation_passed']=bool(report['confirmation']) and all(v['prefix']['worst_fraction']>=p['minimum_stratum_fraction'] and v['prefix']['positive']>=p['minimum_positives'] and v['prefix']['positive']>v['baseline']['positive'] for v in report['confirmation'])
        assert sha256_file(args.protocol)==report['protocol_sha256'] and sha256_file(__file__)==report['driver_sha256']
        report['status']='complete';save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
