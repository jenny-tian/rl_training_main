"""Run a registered fixed-code engineering validation with durable phase status and a runner lock."""

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from docs.analysis.audit_sac_huber_coverage import read


def commands(protocol, terrain, device):
    root = Path(protocol['output'])
    for arm in protocol['arms']:
        common = [sys.executable, '-u', 'scripts/reinforcement_learning/online_adaptation/run_adaptation_session.py',
            '--session-dir', str(root/f'{arm}-{terrain}'), '--algorithm', 'sac',
            '--base-checkpoint', protocol['base_checkpoint'], '--terrain-seed', str(terrain),
            '--seed', str(protocol['learner_seed']), '--budget-steps', str(protocol['trial_steps']),
            '--target-warmup-steps', '0', '--trial-steps', str(protocol['trial_steps']),
            '--selection-budget', '0', '--probe-milestones', '--evaluation-steps', str(protocol['evaluation_steps']),
            '--evaluation-trials', str(protocol['evaluation_trials']), '--retention-trials', str(protocol['retention_trials']),
            '--retention-trial-steps', str(protocol['retention_steps']), '--worker-options', str(root/f'{arm}_options.json'),
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
    assert p['schema']=='lite3_fixed_code_physical_check_v1' and args.terrain in p['terrains']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    root = Path(p['output'])
    prior = read(p['prior_replication_audit'])
    if prior['status'] != 'complete':
        raise ValueError('prior independent replication audit must finish before physical validation')
    for arm, code in p['arms'].items():
        expected = dict(p['worker_options'])
        if code is not None:
            expected['fixed-code'] = code
        assert read(root/f'{arm}_options.json') == expected
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
