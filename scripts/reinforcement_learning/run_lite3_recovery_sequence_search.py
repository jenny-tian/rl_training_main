"""Preregistered CEM prefix search on TRAIN cases, with paid baseline and confirmation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic
from rl_training.recovery_sequence_search import sample_population,update_distribution,short_recovery_metrics


def phase_metrics(folder, options):
    report=json.loads((folder/'evaluation/report.json').read_text())
    phase=report['phases']['smoke-trial'];traces={}
    for path in sorted((folder/'evaluation/smoke-trial/trajectory').glob('chunk-*.npz')):
        with np.load(path,allow_pickle=False) as d:
            for key in ('post_root_state','foot_contact_force_w','active','done','success'):
                traces.setdefault(key,[]).append(d[key])
    traces={k:np.concatenate(v) for k,v in traces.items()}
    rows=[]
    for env_id in options['target_env_ids']:
        row=short_recovery_metrics(traces['post_root_state'][:,env_id],traces['foot_contact_force_w'][:,env_id],
            traces['active'][:,env_id],traces['done'][:,env_id],handoff_step=options['knots']*options['hold_steps'],
            horizon=options['steps'],success=traces['success'][:,env_id])
        row['env_id']=env_id;rows.append(row)
    return rows,phase['initial_sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',required=True);args=parser.parse_args()
    p=json.loads(Path(args.protocol).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    base=Path(__file__).resolve().parents[2]
    sources={str(f.relative_to(base)):sha256_file(f) for folder in (base/'source/rl_training/rl_training',base/'scripts/reinforcement_learning') for f in folder.rglob('*.py')}
    report=dict(status='running',protocol_sha256=sha256_file(args.protocol),source_files=sources,phases={},cases={})
    def save():
        charged=confirmed=0
        for folder in out.glob('case*/**/evaluation'):
            file=folder/'ledger.json'
            if file.exists():
                state=json.loads(file.read_text());charged+=state['charged'];confirmed+=state['confirmed']
        assert charged<=p['max_controls']
        report['ledger']=dict(charged=charged,confirmed=confirmed,uncertain=charged-confirmed,max_controls=p['max_controls'])
        write_json_atomic(out/'report.json',report)
    def run_phase(case,label,seed,population):
        save();assert report['ledger']['charged']+p['num_envs']*p['steps']<=p['max_controls']
        folder=out/f'case{case}'/label;folder.mkdir(parents=True,exist_ok=False)
        protocol=copy.deepcopy(p['evaluation_template']);protocol.update(evaluation_seed=seed,steps=p['steps'],max_controls=p['num_envs']*p['steps'])
        protocol['recovery_resets']['fixed_train_case']=case
        if population is not None:
            prefix=np.zeros((p['num_envs'],p['knots'],12),np.float32);prefix[p['target_env_ids']]=population
            np.save(folder/'prefix.npy',prefix)
            protocol['action_prefix']=dict(path=str((folder/'prefix.npy').resolve()),sha256=sha256_file(folder/'prefix.npy'),knots=p['knots'],hold_steps=p['hold_steps'],bound=p['bound'])
        protocol['parent_protocol_sha256']=report['protocol_sha256'];protocol['search_phase']=label
        write_json_atomic(folder/'protocol.json',protocol)
        command=[sys.executable,str(base/'scripts/reinforcement_learning/evaluate_lite3_recovery_teacher.py'),'--protocol',str(folder/'protocol.json'),'--output',str(folder/'evaluation'),'--device',args.device,'--headless']
        with (folder/'physics.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        command=[sys.executable,str(base/'docs/analysis/audit_lite3_recovery_teacher.py'),'--run',str(folder/'evaluation'),'--protocol',str(folder/'protocol.json'),'--output',str(folder/'audit.json')]
        with (folder/'audit.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        audit=json.loads((folder/'audit.json').read_text());assert audit['status']=='verified'
        rows,initial=phase_metrics(folder,p);entry=dict(rows=rows,initial_sha256=initial,protocol_sha256=sha256_file(folder/'protocol.json'),audit_sha256=sha256_file(folder/'audit.json'))
        report['phases'][f'case{case}/{label}']=entry;save()
        print(json.dumps(dict(case=case,phase=label,positives=sum(r['short_recovery'] for r in rows),best_score=max(r['score'] for r in rows),controls=report['ledger']['confirmed'])),flush=True)
        return rows,initial
    save()
    try:
        for case in p['training_cases']:
            baseline,initial=run_phase(case,'baseline',p['search_seed'],None)
            rng=np.random.default_rng(p['proposal_seed']+case)
            mean=np.zeros((p['knots'],12),np.float32);std=np.full_like(mean,p['initial_std']);best=np.zeros_like(mean);best_score=-float('inf');best_positive=False
            history=[]
            for iteration in range(p['generations']):
                samples=sample_population(rng,mean,std,best,population=len(p['target_env_ids']),bound=p['bound'])
                label=f'generation{iteration:02d}';rows,paired=run_phase(case,label,p['search_seed'],samples);assert paired==initial
                scores=np.array([r['score'] for r in rows]);index=int(np.argmax(scores))
                if scores[index]>best_score:best=samples[index].copy();best_score=float(scores[index]);best_positive=rows[index]['short_recovery']
                mean,std,elite=update_distribution(samples,scores,elite_count=p['elite_count'],std_floor=p['std_floor'])
                history.append(dict(generation=iteration,elite=elite.tolist(),scores=scores.tolist(),incumbent_score=best_score))
            chosen=out/f'case{case}/selected_prefix.npy';np.save(chosen,best)
            result=dict(search_positive=best_positive,selected_sha256=sha256_file(chosen),history=history,confirmations=[])
            report['cases'][str(case)]=result;save()
            if best_positive:
                for seed in p['confirmation_seeds']:
                    before,initial=run_phase(case,f'confirm{seed}-base',seed,None)
                    after,paired=run_phase(case,f'confirm{seed}-prefix',seed,np.broadcast_to(best,(len(p['target_env_ids']),*best.shape)));assert paired==initial
                    result['confirmations'].append(dict(seed=seed,baseline_positive=sum(r['short_recovery'] for r in before),prefix_positive=sum(r['short_recovery'] for r in after)))
                    save()
                result['confirmed_positive']=all(v['prefix_positive']>=p['minimum_confirmation_positives'] and v['prefix_positive']>v['baseline_positive'] for v in result['confirmations'])
            else:result['confirmed_positive']=False
            save()
        assert all(sha256_file(base/name)==digest for name,digest in sources.items())
        report['status']='complete';save()
    except BaseException as error:
        report.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
