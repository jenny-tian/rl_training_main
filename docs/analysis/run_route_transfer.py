"""Run preregistered new-development frozen transfer after independent replication passes."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
from docs.analysis.audit_sac_huber_coverage import read
from docs.analysis.run_episode_learning import write_locked
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def commands(p, slot, device):
    root=Path(p['output'])
    for index,terrain in enumerate(p['terrain_seeds']):
        if index%2!=slot:continue
        for run_index,run in enumerate(p['policies']):
            arms=('candidate','base') if (index+run_index)%2==0 else ('base','candidate')
            for arm in arms:
                name=f"{arm}-{run['learning_seed']}-{terrain}"
                common=[sys.executable,'-u','scripts/reinforcement_learning/online_adaptation/run_adaptation_session.py',
                    '--session-dir',str(root/name),'--algorithm','sac','--base-checkpoint',run['checkpoint']['path'] if arm=='candidate' else p['base_checkpoint'],
                    '--terrain-seed',str(terrain),'--terrain-split',p['terrain_split'],'--seed',str(run['evaluation_seed']),
                    '--budget-steps',str(p['task_steps']),'--target-warmup-steps','0','--trial-steps',str(p['task_steps']),
                    '--selection-budget','0','--probe-milestones','--evaluation-steps',str(p['task_trials']*p['task_steps']),
                    '--evaluation-trials',str(p['task_trials']),'--retention-trials',str(p['retention_trials']),
                    '--retention-trial-steps',str(p['retention_steps']),'--worker-options',str(root/f'{arm}_options.json'),'--device',device]
                for phase in ('evaluation','retention'):
                    yield name,phase,common+['--phase',phase]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--slot',type=int,choices=(0,1),required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    p=read(args.protocol);assert p['schema']=='lite3_route_transfer_v1'
    assert p['terrain_split']=='registered-development'
    assert len(p['terrain_seeds'])==4 and len(set(p['terrain_seeds']))==4 and len(p['policies'])==2
    assert p['task_steps']==2000 and p['task_trials']==5 and p['retention_trials']==2 and p['retention_steps']==1000
    assert 4*2*2*(5*2000+2*1000)==p['max_controls']==192000
    assert sha256_file(p['seed_registration'])==p['seed_registration_sha256']
    registration=read(p['seed_registration']);assert p['terrain_seeds']==registration['terrain_seeds']
    assert [r['evaluation_seed'] for r in p['policies']]==registration['evaluation_master_seeds']
    gate=read(p['replication_audit']);assert gate['status']=='complete' and gate['independent_route_replication_passed']
    assert [r['protocol_sha256'] for r in gate['runs']]==[r['protocol_sha256'] for r in p['policies']]
    for run,proof in zip(p['policies'],gate['runs']):
        assert run['learning_seed']==proof['seed'] and run['checkpoint']==proof['checkpoint']
        assert sha256_file(run['checkpoint']['path'])==run['checkpoint']['sha256']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    # Fail before creating any physical trial if registered geometry is not enabled yet.
    from rl_training.online_adaptation.registered_development import registered_development_geometry, REGISTRATION_SHA256
    assert REGISTRATION_SHA256==p['seed_registration_sha256']
    for seed in p['terrain_seeds']:registered_development_geometry('rough_4cm',seed)
    root=Path(p['output']);root.mkdir(parents=True,exist_ok=True)
    prior=read(p['prior_program_audit'])
    assert prior['status']=='complete' and prior['protocol_sha256']==p['policies'][-1]['protocol_sha256']
    write_locked(root/'transfer_lock.json',{'protocol_sha256':sha256_file(args.protocol),
        'replication_audit_sha256':sha256_file(p['replication_audit']),
        'prior_program_audit_sha256':sha256_file(p['prior_program_audit']),
        'prior_program_controls':prior['full_policy_controls_including_registered_smokes']})
    options={'replay-capacity':256,'min-replay-size':256,'n-step':1,'initial-alpha':.01,'target-critic-warmup-updates':0,'critic-huber-delta':1.}
    write_locked(root/'base_options.json',options)
    write_locked(root/'candidate_options.json',dict(options,**{'route-observations':'sim-localization-v1'}))
    with (root/f'execution-{args.slot}.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/f'execution-{args.slot}.json'
        if path.exists():
            if not args.resume:raise ValueError('Execution exists; verify processes before explicit resume')
            result=read(path);assert result['protocol_sha256']==sha256_file(args.protocol)
        else:result={'status':'running','protocol_sha256':sha256_file(args.protocol),'stages':{}}
        failed={r['session'] for r in result['stages'].values() if r['status']=='failed'}
        for name,phase,command in commands(p,args.slot,args.device):
            key=f'{name}:{phase}';old=result['stages'].get(key)
            if name in failed or old and old['status']=='complete':continue
            if old:assert old['command']==command
            result['stages'][key]={'session':name,'phase':phase,'status':'running','command':command}
            result['status']='running';write_json_atomic(path,result)
            with (root/f'{name}-{phase}.log').open('a') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            result['stages'][key].update(status='complete' if r.returncode==0 else 'failed',returncode=r.returncode)
            if r.returncode:
                failed.add(name);result['status']='failed'
                write_json_atomic(path,result)
                raise SystemExit(r.returncode)
            write_json_atomic(path,result)
        result['status']='failed' if failed else 'complete';write_json_atomic(path,result)
        if failed:raise SystemExit(1)


if __name__=='__main__':main()
