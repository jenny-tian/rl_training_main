"""Run one arm of a context ablation, then measure physical flat retention.

Launch two instances on separate GPUs with identical arguments except --context,
--gpu and --output. The initial checkpoint must have passed a frozen terrain
probe. Run from an archived code tree to keep a long experiment reproducible.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def retention_checks(baseline: dict, candidate: dict) -> dict[str, bool]:
    """Apply the existing physical retention gate to a matched 2000-step pair."""
    pair = (baseline, candidate)
    checks = {
        'physical': all(not s['fell'] and not s['safety_violation'] and s['reset_count'] == 0
                        and s['flat_retention']['comparison_eligible'] for s in pair),
        'frozen_full_budget': all(s['evaluation_only'] and s['updates'] == 0 and s['steps'] == 2000
                                 and s['flat_retention']['full_sample_budget']
                                 and s['flat_retention']['steady_sample_count'] == 1750 for s in pair),
        'forward': all(s['flat_retention']['steady_net_forward_m'] >= 0.3 * 35.0 * 0.8 for s in pair),
    }
    for metric in ('steady_velocity_x_mae', 'steady_velocity_xy_l2_mean'):
        b = baseline['flat_retention'][metric]
        checks[metric] = candidate['flat_retention'][metric] - b <= max(0.005, 0.05 * b)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--initial-checkpoint', required=True)
    parser.add_argument('--flat-anchor-observations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 3), required=True)
    parser.add_argument('--context', action='store_true')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'arm.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Use the same lock as the other Lite3 multiepisode launchers.
    gpu_lock = Path(f'/tmp/lite3-rocky-multiepisode-gpu-{args.gpu}.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'arm_plan.json').exists():
        raise RuntimeError('Use a new output directory; do not overwrite an existing experiment')
    scripts = Path(__file__).resolve().parent
    train = [sys.executable, '-u', str(scripts / 'run_lora_ppo_multiepisode.py'),
             '--base-checkpoint', args.base_checkpoint,
             '--initial-checkpoint', args.initial_checkpoint,
             '--flat-anchor-observations', args.flat_anchor_observations,
             '--output-dir', str(root / 'training'), '--physical-gpu', str(args.gpu),
             '--single-gpu-renderer', '--terrain-seed', '101', '--simulation-seed', '7',
             '--training-seed-stride', '1', '--max-episodes', '20',
             '--max-steps-per-episode', '2000', '--probe-interval', '5',
             '--probe-failure-policy', 'continue_candidate', '--video-probes',
             '--action-std', '0.03', '--actor-lr', '0.00015', '--critic-lr', '0.0003',
             '--reference-anchor-coef', '100', '--max-reference-mse', '0.0001',
             '--stability-window', '10', '--stability-min-success-rate', '0.8',
             '--stability-min-consecutive-successes', '3']
    if args.context:
        train.append('--proprioceptive-context')
    write_json_atomic(root / 'arm_plan.json', {
        'context': args.context, 'command': train,
        'base_sha256': sha256_file(args.base_checkpoint),
        'initial_sha256': sha256_file(args.initial_checkpoint),
        'flat_retention_seeds': [7, 8, 9],
        'flat_retention_steps': 2000,
        'flat_velocity_error_tolerance': 'max(0.005 m/s, 0.05 * baseline)',
        'note': 'Development pilot; context adds parameters and consumes initialization RNG. '
                'Matching seeds do not imply bitwise identical stochastic trajectories.',
    })
    environment = dict(os.environ)
    environment.pop('DISPLAY', None)

    def run(command: list[str], log: Path) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           env=environment, check=True)

    write_json_atomic(root / 'arm_status.json', {'status': 'training'})
    try:
        run(train, root / 'supervisor.log')
        state = json.loads((root / 'training/summary.json').read_text())
        selected = state['final_checkpoint']
        write_json_atomic(root / 'arm_status.json', {
            'status': 'flat_retention', 'selected_checkpoint': selected,
            'selected_sha256': sha256_file(selected),
        })
        results = []
        for seed in (7, 8, 9):
            pair = {}
            for name in ('baseline', 'candidate'):
                output = root / 'flat_retention' / f'seed-{seed}' / name
                command = [sys.executable, '-u', str(scripts / 'train_lora_ppo.py'),
                           '--base-checkpoint', args.base_checkpoint,
                           '--flat-retention-probe', '--evaluation-only',
                           '--seed', str(seed), '--forward-speed', '0.3',
                           '--max-steps', '2000', '--rank', '2', '--headless',
                           '--device', f'cuda:{args.gpu}', '--residual-rate-limit', '0.03',
                           '--nonfoot-contact-rollback-force', '20', '--output', str(output),
                           f'--kit_args=--/renderer/activeGpu={args.gpu} '
                           '--/renderer/multiGpu/enabled=false '
                           '--/renderer/multiGpu/autoEnable=false '
                           '--/renderer/multiGpu/maxGpuCount=1']
                if name == 'candidate':
                    command += ['--checkpoint-in', selected]
                    if args.context:
                        command.append('--proprioceptive-context')
                write_json_atomic(output / 'command.json', {'command': command})
                run(command, output / 'runtime.log')
                pair[name] = json.loads((output / 'summary.json').read_text())
            baseline, candidate = pair['baseline'], pair['candidate']
            checks = retention_checks(baseline, candidate)
            results.append({'seed': seed, 'checks': checks, 'passed': all(checks.values()),
                            'baseline': baseline['flat_retention'], 'candidate': candidate['flat_retention']})
            write_json_atomic(root / 'flat_retention_results.json', {'pairs': results})
        write_json_atomic(root / 'arm_status.json', {
            'status': 'complete', 'training_status': state['status'],
            'stable_success_episode': state['stable_success_episode'],
            'selected_checkpoint': selected, 'selected_sha256': sha256_file(selected),
            'flat_retention_passed': all(r['passed'] for r in results),
            'final_validation': state['final_validation'],
        })
    except Exception as error:
        write_json_atomic(root / 'arm_status.json', {'status': 'error', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
