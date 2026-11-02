"""Run one explicitly approved probe, or validate existing artifacts without simulation."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from rl_training.source_scan.smoke_artifacts import atomic_json, validate_smoke_artifacts


def validate_existing(output, *, returncode, attempt_id, revision, checkpoint_sha256,
                      process_group_empty=True):
    result = validate_smoke_artifacts(output, returncode=returncode, expected_attempt_id=attempt_id,
                                     expected_revision=revision, expected_checkpoint_sha256=checkpoint_sha256)
    result["process_group_empty"] = process_group_empty
    if not process_group_empty:
        result["reasons"].append("child_process_group_remains")
        result["accepted"] = False
    return result


def _group_exists(group):
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def run_approved(args):
    plan = json.loads(args.manifest.read_text())
    required = {"user_approved": True, "num_envs": 2, "shared_bank_rows": 1,
                "max_global_control_steps": 24, "max_diagnostic_transitions": 48,
                "ppo_updates": 0, "device": "cuda:0", "wall_timeout_s": 240,
                "termination_cleanup_grace_s": 15, "retry": False,
                "ppo_seed": 130001, "geometry_seed": 30000, "sampler_seed": 131001,
                "noise_rng_seed": 132001, "gaussian_std": 1.0}
    for key, value in required.items():
        if type(plan.get(key)) is not type(value) or plan[key] != value:
            raise ValueError("unapproved probe contract field: " + key)
    if args.output_dir.exists():
        raise FileExistsError("output directory already exists; no automatic retry")
    output = args.output_dir.resolve() / 'result.json'
    if plan.get('output') != str(output):
        raise ValueError('manifest output differs from requested directory')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if (plan['base_revision'] != revision or plan['remote_revision_verified'] != revision
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('probe requires clean, committed, remote-verified inputs')
    required_inputs = {
        'scripts/tools/source_training_integration_smoke.py',
        'scripts/tools/source_training_integration_supervisor.py',
        'source/rl_training/rl_training/source_scan/smoke_artifacts.py',
        'source/rl_training/rl_training/source_scan/training_env.py',
        'source/rl_training/rl_training/source_scan/training_lifecycle.py',
        'source/rl_training/rl_training/source_scan/bridge.py',
        'source/rl_training/rl_training/online_adaptation/safety.py',
        'source/rl_training/rl_training/source_scan/frozen_actor.py',
        'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_training_env_cfg.py',
    }
    if not required_inputs.issubset(plan['input_sha256']):
        raise ValueError('manifest omits required code hashes')
    for name, digest in plan['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('input hash changed: ' + name)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != plan['checkpoint_sha256']:
        raise ValueError('checkpoint hash changed')
    query = ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu',
             '--format=csv,noheader']
    compute_query = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                     '--format=csv,noheader']
    gpu = subprocess.check_output(query, text=True, timeout=10)
    row = next(line for line in gpu.splitlines() if line.startswith('0,'))
    free = int(row.split(',')[3].strip().split()[0])
    if free < 12288:
        raise RuntimeError('GPU0 resource gate failed; do not switch devices')
    compute = subprocess.check_output(compute_query, text=True, timeout=10)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'gpu_before.csv').write_text(gpu)
    (args.output_dir/'compute_before.csv').write_text(compute)
    atomic_json(args.output_dir/'run_manifest.json', plan)
    command = ['timeout', '--signal=TERM', '--kill-after=15s', '240s', sys.executable,
               str(ROOT/'scripts/tools/source_training_integration_smoke.py'), '--headless', '--device', 'cuda:0',
               '--checkpoint', str(args.checkpoint.resolve()), '--output', str(output),
               '--attempt-id', plan['attempt_id']]
    started = time.monotonic()
    execution = {'command': command, 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    process = None
    errors = []
    try:
        with (args.output_dir/'runtime.log').open('x') as log, (args.output_dir/'gpu_samples.jsonl').open('x') as samples:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT/'source/rl_training')},
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            execution['process_group'] = process.pid
            atomic_json(args.output_dir/'execution.json', execution)
            while process.poll() is None:
                try:
                    sample = subprocess.run(query, capture_output=True, text=True, timeout=10)
                    if sample.returncode:
                        errors.append('GPU sample failed: ' + sample.stderr)
                    samples.write(json.dumps({'elapsed_s': time.monotonic()-started,
                                              'returncode': sample.returncode, 'gpu': sample.stdout})+'\n')
                    samples.flush()
                except subprocess.TimeoutExpired:
                    errors.append('GPU sample timed out')
                time.sleep(5)
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            execution.update(returncode=process.returncode, elapsed_s=time.monotonic()-started,
                             process_group_empty=not _group_exists(process.pid))
        execution['monitor_errors'] = errors
        atomic_json(args.output_dir/'execution.json', execution)
    for name, query_args in (('gpu_after.csv', query), ('compute_after.csv', compute_query)):
        sample = subprocess.run(query_args, capture_output=True, text=True, timeout=10)
        (args.output_dir/name).write_text(sample.stdout)
        if sample.returncode:
            errors.append(name + ' failed: ' + sample.stderr)
    result = validate_existing(output, returncode=execution.get('returncode'), attempt_id=plan['attempt_id'],
                               revision=revision, checkpoint_sha256=plan['checkpoint_sha256'],
                               process_group_empty=execution.get('process_group_empty', False))
    if errors:
        result.update(accepted=False, monitor_errors=errors)
        result['reasons'].append('resource_monitoring_incomplete')
    atomic_json(args.output_dir/'acceptance.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['accepted'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    run = sub.add_parser('run')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--checkpoint', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    check = sub.add_parser('validate')
    check.add_argument('--output', type=Path, required=True)
    check.add_argument('--returncode', type=int, required=True)
    check.add_argument('--attempt-id', required=True)
    check.add_argument('--revision', required=True)
    check.add_argument('--checkpoint-sha256', required=True)
    args = parser.parse_args()
    if args.mode == 'run':
        return run_approved(args)
    result = validate_existing(args.output, returncode=args.returncode, attempt_id=args.attempt_id,
                               revision=args.revision, checkpoint_sha256=args.checkpoint_sha256)
    print(json.dumps(result, indent=2))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
