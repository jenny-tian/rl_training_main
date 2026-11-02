"""Bounded eight-environment probe supervisor. Import/validate modes start no runtime."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/rl_training'))
from rl_training.source_scan.smoke_artifacts import atomic_json
from source_eight_env_moving_validation import validate
from source_parallel_terrain_startup_supervisor import _terminate_process_group


def startup_expired(text, elapsed_s):
    return elapsed_s > 120 and '[eight-env-probe] initial-ready' not in text


def check_dependencies(plan):
    dependencies = plan.get('dependency_sha256')
    if not isinstance(dependencies, dict) or len(dependencies) < 35:
        raise ValueError('manifest must lock the 34 reused dependencies and SimulationApp shutdown source')
    for name, digest in dependencies.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('changed runtime dependency: '+name)


def run(args):
    plan = json.loads(args.manifest.read_text())
    expected = {'user_approved': True, 'num_envs': 8, 'shared_bank_rows': 1, 'control_steps': 96,
                'max_diagnostic_transitions': 768, 'ppo_updates': 0, 'device': 'cuda:0',
                'wall_timeout_s': 240, 'cleanup_grace_s': 15, 'minimum_free_vram_mib': 12288,
                'startup_timeout_s': 120,
                'retry': False, 'geometry_seed': 30000, 'sampler_seed': 131001, 'noise_rng_seed': 132001}
    for key, value in expected.items():
        if type(plan.get(key)) is not type(value) or plan[key] != value:
            raise ValueError('unapproved moving probe field: '+key)
    if args.output_dir.exists():
        raise FileExistsError('attempt directory already exists; no retry')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '0'):
        raise ValueError('GPU0 visibility mapping is not the registered mapping')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if (plan['base_revision'] != revision or plan['remote_revision_verified'] != revision
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('run inputs must be clean, committed and remote verified')
    required = {'scripts/tools/source_eight_env_moving_probe.py', 'scripts/tools/source_eight_env_moving_validation.py',
                'scripts/tools/source_eight_env_moving_supervisor.py',
                'source/rl_training/rl_training/source_scan/moving_diagnostic.py',
                'source/rl_training/rl_training/source_scan/training_env.py',
                'source/rl_training/rl_training/source_scan/smoke_artifacts.py',
                'scripts/tools/source_shared_tensor_contact.py', 'scripts/tools/source_shared_contact_report.py',
                'scripts/tools/source_parallel_reset_lifecycle.py', 'scripts/tools/source_parallel_terrain_startup_supervisor.py'}
    if not required.issubset(plan['input_sha256']):
        raise ValueError('manifest omits required diagnostics')
    for name, digest in plan['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('changed input: '+name)
    check_dependencies(plan)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != plan['checkpoint_sha256']:
        raise ValueError('checkpoint mismatch')
    output = args.output_dir.resolve()/'result.json'
    if plan['output'] != str(output):
        raise ValueError('output differs from registered attempt')
    query = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu', '--format=csv,noheader']
    compute_query = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader']
    before = subprocess.check_output(query, text=True, timeout=10)
    gpu0 = next(line for line in before.splitlines() if line.startswith('0,'))
    if int(gpu0.split(',')[4].strip().split()[0]) < 12288:
        raise RuntimeError('GPU0 free VRAM below gate')
    compute = subprocess.check_output(compute_query, text=True, timeout=10)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'gpu_before.csv').write_text(before)
    (args.output_dir/'compute_before.csv').write_text(compute)
    atomic_json(args.output_dir/'run_manifest.json', plan)
    command = ['timeout', '--signal=TERM', '--kill-after=15s', '240s', sys.executable,
               str(ROOT/'scripts/tools/source_eight_env_moving_probe.py'), '--headless', '--device', 'cuda:0',
               '--output', str(output), '--attempt-id', plan['attempt_id'], '--checkpoint', str(args.checkpoint.resolve())]
    started = time.monotonic()
    execution = {'command': command, 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'monitor_errors': []}
    process = None
    try:
        with (args.output_dir/'runtime.log').open('x') as log, (args.output_dir/'gpu_samples.jsonl').open('x') as samples:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT/'source/rl_training')},
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            execution['process_group'] = process.pid
            atomic_json(args.output_dir/'execution.json', execution)
            while process.poll() is None:
                if startup_expired((args.output_dir/'runtime.log').read_text(errors='replace'), time.monotonic()-started):
                    execution['monitor_errors'].append('120-second startup deadline reached')
                    break
                try:
                    result = subprocess.run(query, capture_output=True, text=True, timeout=10)
                    samples.write(json.dumps({'elapsed_s': time.monotonic()-started, 'returncode': result.returncode,
                                              'gpu': result.stdout})+'\n')
                    samples.flush()
                    if result.returncode:
                        raise RuntimeError('GPU query failed: '+result.stderr)
                except (subprocess.TimeoutExpired, RuntimeError) as error:
                    execution['monitor_errors'].append(str(error))
                    break
                time.sleep(5)
    finally:
        if process is not None:
            if process.poll() is None:
                _terminate_process_group(process, 15)
            execution.update(returncode=process.returncode, elapsed_s=time.monotonic()-started)
            try:
                os.killpg(process.pid, 0)
                execution['process_group_empty'] = False
                _terminate_process_group(process, 15)
            except ProcessLookupError:
                execution['process_group_empty'] = True
        atomic_json(args.output_dir/'execution.json', execution)
    for name, command_args in (('gpu_after.csv', query), ('compute_after.csv', compute_query)):
        try:
            result = subprocess.run(command_args, capture_output=True, text=True, timeout=10)
            (args.output_dir/name).write_text(result.stdout)
            if result.returncode:execution['monitor_errors'].append(name+' query failed')
        except subprocess.TimeoutExpired:execution['monitor_errors'].append(name+' query timed out')
    atomic_json(args.output_dir/'execution.json', execution)
    acceptance = validate(output, returncode=execution.get('returncode'), attempt_id=plan['attempt_id'],
        revision=revision, checkpoint_sha256=plan['checkpoint_sha256'], process_group_empty=execution.get('process_group_empty', False))
    if execution['monitor_errors']:
        acceptance['accepted'] = False;acceptance['reasons'].append('resource monitoring incomplete')
    atomic_json(args.output_dir/'acceptance.json', acceptance)
    print(json.dumps(acceptance, indent=2))
    return 0 if acceptance['accepted'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--manifest', required=True, type=Path)
    execute.add_argument('--output-dir', required=True, type=Path)
    execute.add_argument('--checkpoint', required=True, type=Path)
    check = sub.add_parser('validate')
    check.add_argument('--output', required=True, type=Path)
    check.add_argument('--returncode', required=True, type=int)
    check.add_argument('--attempt-id', required=True)
    check.add_argument('--revision', required=True)
    check.add_argument('--checkpoint-sha256', required=True)
    args = parser.parse_args()
    if args.mode == 'run':return run(args)
    result = validate(args.output, returncode=args.returncode, attempt_id=args.attempt_id,
                      revision=args.revision, checkpoint_sha256=args.checkpoint_sha256)
    print(json.dumps(result, indent=2))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
