"""Execute the registered per-terrain candidate/base phases, preserving failures."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--terrain', type=int, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if args.terrain not in protocol['terrains'] or sha256_file(args.base_checkpoint) != protocol['base_sha256']:
        raise ValueError('unregistered terrain or base checkpoint')
    root = Path(protocol['output'])
    record_path = root/f'execution-{args.terrain}.json'
    if record_path.exists():
        raise ValueError('execution record exists; reconcile existing sessions instead of replacing attempts')
    result = {'status':'running', 'protocol_sha256':sha256_file(args.protocol), 'phases':[]}
    common = [sys.executable, '-u', 'scripts/reinforcement_learning/online_adaptation/run_adaptation_session.py',
              '--algorithm', 'sac', '--base-checkpoint', args.base_checkpoint, '--terrain-seed', str(args.terrain),
              '--seed', str(protocol['learner_seed']), '--budget-steps', str(protocol['adaptation_steps']),
              '--selection-budget', '0', '--probe-milestones', '--evaluation-steps', str(protocol['evaluation_steps']),
              '--evaluation-trials', str(protocol['evaluation_trials']), '--retention-trials', str(protocol['retention_trials']),
              '--retention-trial-steps', str(protocol['retention_steps']), '--worker-options', str(root/'worker_options.json'),
              '--device', args.device]
    for role, phases in [('candidate', ('adaptation', 'evaluation', 'retention')), ('base', ('evaluation', 'retention'))]:
        for phase in phases:
            command = common + ['--session-dir', str(root/f'{role}-{args.terrain}'), '--phase', phase,
                                '--target-warmup-steps', str(protocol['warmup_steps'] if role == 'candidate' else 0)]
            entry = {'role':role, 'phase':phase, 'command':command, 'status':'running'}
            result['phases'].append(entry)
            write_json_atomic(record_path, result)
            with (root/f'{role}-{args.terrain}-{phase}.log').open('w') as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            entry.update(status='complete' if completed.returncode == 0 else 'failed', returncode=completed.returncode)
            write_json_atomic(record_path, result)
            if completed.returncode:
                # Keep failed learner costs; never evaluate its aborted checkpoint.
                break
    result['status'] = 'complete' if all(e['status']=='complete' for e in result['phases']) else 'failed'
    write_json_atomic(record_path, result)
    if result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
