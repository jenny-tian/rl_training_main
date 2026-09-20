"""Run a context/terrain-anchor arm, then evaluate learned and selected models.

Launch matched instances on separate GPUs, varying either --context or the
terrain-anchor options. The initial checkpoint must have passed a frozen terrain
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
from scripts.reinforcement_learning.online_adaptation.run_lora_ppo_multiepisode import (
    build_parser as supervisor_parser, build_probe_command, _successful,
    _nonnegative_float, _positive_float, _positive_int,
)


def evaluation_targets(state: dict, initial_checkpoint: str) -> list[dict]:
    """Never substitute the selected recovery model for the latest learned model."""
    initial_sha = sha256_file(initial_checkpoint)
    targets = {}
    for role, key in (("selected", "final_checkpoint"), ("latest_candidate", "latest_checkpoint")):
        checkpoint = str(Path(state[key]).resolve())
        digest = sha256_file(checkpoint)
        target = targets.setdefault(digest, {
            "checkpoint": checkpoint, "sha256": digest, "roles": [],
            "is_initial_checkpoint": digest == initial_sha,
        })
        target["roles"].append(role)
    return list(targets.values())


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
    parser.add_argument('--terrain-anchor-coef', type=_nonnegative_float, default=0.0)
    parser.add_argument('--max-terrain-anchor-mse', type=_positive_float, default=None)
    parser.add_argument('--action-std', type=_positive_float, default=0.03)
    parser.add_argument('--final-action-std', type=_positive_float, default=None)
    parser.add_argument('--actor-lr', type=_positive_float, default=0.00015)
    parser.add_argument('--simulation-seed', type=int, default=7)
    parser.add_argument('--max-episodes', type=_positive_int, default=20)
    parser.add_argument('--wait-for-gpu', action='store_true',
                        help='queue on the shared GPU lock without interrupting an existing arm')
    args = parser.parse_args()
    if args.final_action_std is not None and (args.final_action_std > args.action_std or
            (args.max_episodes < 2 and args.final_action_std != args.action_std)):
        parser.error('noise decay requires final std <= initial std and at least two episodes')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'arm.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'arm_plan.json').exists():
        raise RuntimeError('Use a new output directory; do not overwrite an existing experiment')
    scripts = Path(__file__).resolve().parent
    train = [sys.executable, '-u', str(scripts / 'run_lora_ppo_multiepisode.py'),
             '--base-checkpoint', args.base_checkpoint,
             '--initial-checkpoint', args.initial_checkpoint,
             '--flat-anchor-observations', args.flat_anchor_observations,
             '--output-dir', str(root / 'training'), '--physical-gpu', str(args.gpu),
             '--single-gpu-renderer', '--terrain-seed', '101', '--simulation-seed', str(args.simulation_seed),
             '--training-seed-stride', '1', '--max-episodes', str(args.max_episodes),
             '--max-steps-per-episode', '2000', '--probe-interval', '5',
             '--probe-failure-policy', 'continue_candidate', '--video-probes',
             '--action-std', str(args.action_std), '--actor-lr', str(args.actor_lr), '--critic-lr', '0.0003',
             '--reference-anchor-coef', '100', '--max-reference-mse', '0.0001',
             '--stability-window', '10', '--stability-min-success-rate', '0.8',
             '--stability-min-consecutive-successes', '3']
    if args.final_action_std is not None:
        train += ['--final-action-std', str(args.final_action_std)]
    if args.context:
        train.append('--proprioceptive-context')
    if args.terrain_anchor_coef or args.max_terrain_anchor_mse is not None:
        train += ['--terrain-reference-checkpoint', args.initial_checkpoint,
                  '--terrain-anchor-coef', str(args.terrain_anchor_coef)]
        if args.max_terrain_anchor_mse is not None:
            train += ['--max-terrain-anchor-mse', str(args.max_terrain_anchor_mse)]
    write_json_atomic(root / 'arm_plan.json', {
        'context': args.context, 'command': train,
        'terrain_anchor_coef': args.terrain_anchor_coef,
        'max_terrain_anchor_mse': args.max_terrain_anchor_mse,
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

    try:
        # The output lock prevents duplicate queue entries for this arm. The GPU
        # lock spans both training and all evaluation, just as in prior pilots.
        write_json_atomic(root / 'arm_status.json', {'status': 'waiting_for_gpu', 'gpu': args.gpu})
        gpu_lock = Path(f'/tmp/lite3-rocky-multiepisode-gpu-{args.gpu}.lock').open('a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | (0 if args.wait_for_gpu else fcntl.LOCK_NB))
        write_json_atomic(root / 'arm_status.json', {'status': 'training'})
        run(train, root / 'supervisor.log')
        state = json.loads((root / 'training/summary.json').read_text())
        selected = state['final_checkpoint']
        targets = evaluation_targets(state, args.initial_checkpoint)
        write_json_atomic(root / 'evaluation_targets.json', {'targets': targets})
        # The supervisor's final_validation describes its selected model only.
        # Independently probe the actual last training candidate on all five
        # development layouts, even when all periodic probes failed.
        latest = next(t for t in targets if 'latest_candidate' in t['roles'])
        write_json_atomic(root / 'arm_status.json', {
            'status': 'latest_candidate_validation', 'targets': targets,
        })
        probe_args = supervisor_parser().parse_args(train[3:])
        candidate_results = []
        for seed in (101, 103, 107, 109, 113):
            output = root / 'latest_candidate_validation' / f'seed-{seed}'
            command = build_probe_command(probe_args, output, Path(latest['checkpoint']), terrain_seed=seed)
            write_json_atomic(output / 'command.json', {'command': command})
            run(command, output / 'runtime.log')
            summary = json.loads((output / 'summary.json').read_text())
            candidate_results.append({'seed': seed, 'successful_completion': _successful(summary)})
            write_json_atomic(root / 'latest_candidate_validation.json', {
                **latest, 'count': len(candidate_results),
                'successes': sum(r['successful_completion'] for r in candidate_results),
                'results': candidate_results,
            })
        write_json_atomic(root / 'arm_status.json', {
            'status': 'flat_retention', 'selected_checkpoint': selected,
            'selected_sha256': sha256_file(selected), 'targets': targets,
        })
        results = []
        for seed in (7, 8, 9):
            pair = {}
            jobs = [('baseline', None)] + [(t['roles'][0], t) for t in targets]
            for name, target in jobs:
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
                if target is not None:
                    command += ['--checkpoint-in', target['checkpoint']]
                    if args.context:
                        command.append('--proprioceptive-context')
                write_json_atomic(output / 'command.json', {'command': command})
                run(command, output / 'runtime.log')
                pair[name] = json.loads((output / 'summary.json').read_text())
            for target in targets:
                name = target['roles'][0]
                baseline, candidate = pair['baseline'], pair[name]
                checks = retention_checks(baseline, candidate)
                results.append({'seed': seed, 'target': target, 'checks': checks,
                                'passed': all(checks.values()),
                                'baseline': baseline['flat_retention'], 'candidate': candidate['flat_retention']})
            write_json_atomic(root / 'flat_retention_results.json', {'pairs': results})
        write_json_atomic(root / 'arm_status.json', {
            'status': 'complete', 'training_status': state['status'],
            'stable_success_episode': state['stable_success_episode'],
            'selected_checkpoint': selected, 'selected_sha256': sha256_file(selected),
            'targets': targets,
            'flat_retention_passed': all(r['passed'] for r in results if 'selected' in r['target']['roles']),
            'latest_candidate_flat_retention_passed': all(
                r['passed'] for r in results if 'latest_candidate' in r['target']['roles']),
            'latest_candidate_validation': {**latest, 'results': candidate_results,
                                           'successes': sum(r['successful_completion'] for r in candidate_results)},
            'final_validation': state['final_validation'],
        })
    except Exception as error:
        write_json_atomic(root / 'arm_status.json', {'status': 'error', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
