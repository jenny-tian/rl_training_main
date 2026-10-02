"""Run a registered frozen full-policy comparison with durable phase status and a runner lock."""

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from docs.analysis.audit_sac_huber_coverage import read


def commands(protocol, terrain, device):
    root = Path(protocol['output'])
    for arm in (('candidate','base') if terrain==101 else ('base','candidate')):
        common = [sys.executable, '-u', 'scripts/reinforcement_learning/online_adaptation/run_adaptation_session.py',
            '--session-dir', str(root/f'{arm}-{terrain}'), '--algorithm', 'sac',
            '--base-checkpoint', protocol['_candidate_checkpoint'] if arm=='candidate' else protocol['base_checkpoint'], '--terrain-seed', str(terrain),
            '--seed', str(protocol['evaluation_seed']), '--budget-steps', str(protocol['evaluation_trial_steps']),
            '--target-warmup-steps', '0', '--trial-steps', str(protocol['evaluation_trial_steps']),
            '--selection-budget', '0', '--probe-milestones', '--evaluation-steps', str(protocol['evaluation_trials']*protocol['evaluation_trial_steps']),
            '--evaluation-trials', str(protocol['evaluation_trials']), '--retention-trials', str(protocol['retention_trials']),
            '--retention-trial-steps', str(protocol['retention_steps']), '--worker-options', str(root/('evaluation_candidate_options.json' if arm=='candidate' and protocol.get('route_observations') else 'evaluation_options.json')),
            '--device', device]
        for phase in ('evaluation', 'retention'):
            yield arm, phase, common+['--phase', phase]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--terrain', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    p = read(args.protocol)
    assert p['schema']=='lite3_full_policy_pilot_v1' and args.terrain in p['evaluation_terrains']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    root = Path(p['output'])
    train = read(root/'train/report.json')
    history = read(root/'action_history_audit.json')
    if train['status']!='complete' or not history['emitted_action_history_compatible']:
        raise ValueError('complete compatible training required before evaluation')
    assert history['training_report_sha256']==sha256_file(root/'train/report.json')
    assert train['protocol_sha256']==sha256_file(args.protocol)
    assert train['ppo_updates']==p['iterations'] and train['ledger']['confirmed']==p['training_control_transitions']
    candidate = train['final_checkpoint']
    assert Path(candidate['path']).resolve()==(root/'train/final.pt').resolve()
    assert sha256_file(candidate['path'])==candidate['sha256']
    p['_candidate_checkpoint']=candidate['path']
    from docs.analysis.run_episode_learning import write_locked
    anchor_lock={}
    if p.get('flat_anchor_weight') is not None:
        anchor=read(root/'flat_anchor_audit.json')
        assert anchor['status']=='complete' and anchor['training_report_sha256']==sha256_file(root/'train/report.json')
        assert anchor['protocol_sha256']==sha256_file(args.protocol)
        assert anchor['rows']==p['training_control_transitions'] and anchor['flat_target_rows']==train['flat_anchor_label_rows']
        anchor_lock['flat_anchor_audit_sha256']=sha256_file(root/'flat_anchor_audit.json')
    write_locked(root/'evaluation_lock.json',{'protocol_sha256':sha256_file(args.protocol),
        'training_report_sha256':sha256_file(root/'train/report.json'),
        'history_audit_sha256':sha256_file(root/'action_history_audit.json'),
        'candidate':candidate,'base':{'path':p['base_checkpoint'],'sha256':p['base_sha256']},
        'learning_algorithm':'full policy PPO; child SAC label denotes frozen execution engine only',**anchor_lock})
    write_locked(root/'evaluation_options.json',{'replay-capacity':256,'min-replay-size':256,
        'target-critic-warmup-updates':0,'n-step':1,'initial-alpha':.01,'critic-huber-delta':1.})
    if p.get('route_observations'):
        options=read(root/'evaluation_options.json')
        options['route-observations']=p['route_observations']
        write_locked(root/'evaluation_candidate_options.json',options)
    with (root/f'execution-{args.terrain}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root/f'execution-{args.terrain}.json'
        if path.exists():
            if not args.resume:
                raise ValueError('execution exists; inspect running processes and use explicit resume')
            result = read(path)
            assert result['protocol_sha256']==sha256_file(args.protocol)
        else:
            result = {'status':'running', 'protocol_sha256':sha256_file(args.protocol), 'phases':{}}
        failed_arms = {e['arm'] for e in result['phases'].values() if e['status']=='failed'}
        for arm, phase, command in commands(p, args.terrain, args.device):
            key = f'{arm}:{phase}'
            prior = result['phases'].get(key)
            if prior and prior['status']=='complete':
                continue
            if arm in failed_arms:
                continue
            if prior:
                assert prior['command']==command
            result['phases'][key] = {'arm':arm, 'phase':phase, 'status':'running', 'command':command}
            result['status']='running'
            write_json_atomic(path, result)
            with (root/f'{arm}-{args.terrain}-{phase}.log').open('a') as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            result['phases'][key].update(status='complete' if completed.returncode==0 else 'failed', returncode=completed.returncode)
            write_json_atomic(path, result)
            if completed.returncode:
                failed_arms.add(arm)
        result['status']='failed' if failed_arms else 'complete'
        write_json_atomic(path, result)
        if failed_arms:
            raise SystemExit(1)


if __name__=='__main__':
    main()
