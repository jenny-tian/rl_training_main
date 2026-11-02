"""Prepare, run after separate approval, or review one bounded live PPO diagnostic."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'source/rl_training'), str(ROOT / 'scripts/tools')]
from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
from rl_training.source_scan.live_ppo_diagnostic import SCHEMA, TIMEOUT_PROBES
from rl_training.source_scan.smoke_artifacts import atomic_json
from rl_training.source_scan.training_budget_ledger import BudgetLedger, canonical_budget_path
from source_training_entrypoint import _git_state, _input_hashes

CONTRACT = {'schema': SCHEMA, 'purpose': 'bounded_live_ppo_integration', 'run': 1,
    'num_envs': 8, 'shared_bank_rows': 1, 'rollout_length': 24, 'max_updates': 2,
    'max_transitions': 384, 'control_steps': 48, 'physics_substeps': 192,
    'learning_epochs': 5, 'mini_batches': 4, 'optimizer_steps': 40,
    'ppo_seed': 130001, 'geometry_seed': 30000, 'sampler_seed': 131001,
    'reset_state_id': 40000, 'reset_randomized': False, 'action_std': 1.0,
    'timeout_probes': {str(k): v for k, v in TIMEOUT_PROBES.items()},
    'device': 'cuda:0', 'startup_timeout_s': 120, 'wall_timeout_s': 240, 'cleanup_grace_s': 15,
    'minimum_free_vram_mib': 12288, 'minimum_free_disk_bytes': 2147483648,
    'events_byte_limit': 1073741824, 'retry': False, 'resume': False,
    'automatic_scale_increase': False, 'development_or_sealed_feedback_allowed': False,
    'registered_cells': 15, 'retain_six_families': True,
    'budget_policy': 'reserve_full_384_against_run1_no_automatic_refund',
    'checkpoint_sha256': CHECKPOINT_SHA256, 'training_ready': False, 'g3_g4_qualified': False}


def dependencies():
    import torch
    from torch.overrides import TorchFunctionMode
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.storage import RolloutStorage
    from rsl_rl.modules.distribution import GaussianDistribution
    if torch.__version__ != '2.7.0+cu128' or importlib.metadata.version('rsl-rl-lib') != '5.0.1':
        raise RuntimeError('live PPO preparation is audited for Torch2.7.0+cu128 and RSL-RL5.0.1')
    reference = json.loads((ROOT / 'docs/lite3-source-eight-env-moving-runtime-20261102/raw/run_manifest.json').read_text())
    hashes = reference['dependency_sha256'].copy()
    for name, digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('changed previously reviewed Isaac dependency: ' + name)
    paths = {Path(inspect.getfile(obj)) for obj in (
        PPO, MLPModel, OnPolicyRunner, RolloutStorage, GaussianDistribution, torch.optim.Adam, TorchFunctionMode)}
    paths.update([Path(torch.__file__), Path('/home/tjx/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py')])
    hashes.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    return dict(sorted(hashes.items()))


def validate_plan(plan, *, revision, inputs, dependency_hashes, output_dir, ledger, approved=True):
    if not isinstance(plan, dict):
        raise ValueError('live PPO manifest must be a mapping')
    if approved and (plan.get('user_approved') is not True or plan.get('status') != 'approved'):
        raise PermissionError('separate live simulator launch approval is required')
    if not approved and (plan.get('user_approved') is not False or plan.get('status') != 'proposed'):
        raise ValueError('preparation may only write an unapproved proposal')
    expected = CONTRACT | {'base_revision': revision, 'remote_revision_verified': revision,
        'input_sha256': inputs, 'dependency_sha256': dependency_hashes,
        'output_dir': str(Path(output_dir).resolve())}
    for name, value in expected.items():
        if type(plan.get(name)) is not type(value) or plan[name] != value:
            raise ValueError('live PPO contract mismatch: ' + name)
    if not isinstance(plan.get('attempt_id'), str) or not plan['attempt_id']:
        raise ValueError('missing attempt identity')
    ledger.assert_fresh(1, plan.get('budget_ledger'))
    if Path(output_dir).exists():
        raise FileExistsError('diagnostic attempt directory already exists; no retry')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '0'):
        raise ValueError('GPU0 visibility mapping differs from contract')
    return plan


def preflight(args, *, approved):
    plan = json.loads(args.manifest.read_text())
    # Reject absent approval before Git/network, GPU queries or ledger writes.
    if approved and (plan.get('user_approved') is not True or plan.get('status') != 'approved'):
        raise PermissionError('separate live simulator launch approval is required')
    revision, remote, dirty = _git_state()
    if dirty or revision != remote:
        raise ValueError('live PPO requires clean, committed and remote-matching code')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('registered original checkpoint changed')
    ledger = BudgetLedger(canonical_budget_path(ROOT))
    validate_plan(plan, revision=revision, inputs=_input_hashes(), dependency_hashes=dependencies(),
                  output_dir=args.output_dir, ledger=ledger, approved=approved)
    return plan, ledger


def prepare(args):
    if args.manifest.exists() or args.output_dir.exists():
        raise FileExistsError('preparation does not overwrite a proposal or attempt')
    revision, remote, dirty = _git_state()
    if dirty or revision != remote:
        raise ValueError('freeze proposal only after commit, push and clean worktree')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('registered original checkpoint changed')
    ledger = BudgetLedger(canonical_budget_path(ROOT))
    plan = CONTRACT | {'user_approved': False, 'status': 'proposed', 'attempt_id': args.attempt_id,
        'base_revision': revision, 'remote_revision_verified': remote,
        'input_sha256': _input_hashes(), 'dependency_sha256': dependencies(),
        'output_dir': str(args.output_dir.resolve()), 'budget_ledger': ledger.reference()}
    validate_plan(plan, revision=revision, inputs=plan['input_sha256'], dependency_hashes=plan['dependency_sha256'],
                  output_dir=args.output_dir, ledger=ledger, approved=False)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': 'proposal_frozen_not_approved', 'manifest': str(args.manifest),
                      'base_revision': revision, 'inputs': len(plan['input_sha256']),
                      'dependencies': len(plan['dependency_sha256']), 'ledger_created': ledger.path.exists()}))
    return 0


def group_exists(group):
    try: os.killpg(group, 0); return True
    except ProcessLookupError: return False


def deadline_expired(elapsed, text, plan):
    return (elapsed > plan['wall_timeout_s'] or
            elapsed > plan['startup_timeout_s'] and '[live-ppo] initial-ready' not in text)


def terminate_group(process, grace=15):
    import signal
    if group_exists(process.pid):
        os.killpg(process.pid, signal.SIGTERM)
        end = time.monotonic() + grace
        while group_exists(process.pid) and time.monotonic() < end:
            process.poll(); time.sleep(.1)
        if group_exists(process.pid): os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def run(args):
    plan, ledger = preflight(args, approved=True)
    query = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu', '--format=csv,noheader']
    before = subprocess.check_output(query, text=True, timeout=10)
    row = next(line for line in before.splitlines() if line.startswith('0,'))
    if int(row.split(',')[4].strip().split()[0]) < plan['minimum_free_vram_mib']:
        raise RuntimeError('GPU0 free memory below diagnostic gate; no device switch')
    parent = args.output_dir.resolve().parent
    if shutil.disk_usage(parent).free < plan['minimum_free_disk_bytes']:
        raise RuntimeError('insufficient free disk for raw evidence')
    reservation = ledger.reserve(run=1, attempt_id=plan['attempt_id'], transitions=384,
        purpose=plan['purpose'], output_dir=args.output_dir, revision=plan['base_revision'],
        expected_reference=plan['budget_ledger'])
    # Reservation precedes process creation; crashes never grant budget back.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output_dir / 'run_manifest.json', plan)
    atomic_json(args.output_dir / 'budget_reservation.json', reservation)
    (args.output_dir / 'gpu_before.csv').write_text(before)
    command = ['timeout', '--signal=TERM', '--kill-after=15s', '240s', sys.executable,
               str(ROOT / 'scripts/tools/source_live_ppo_probe.py'),
               '--manifest', str((args.output_dir / 'run_manifest.json').resolve()),
               '--checkpoint', str(args.checkpoint.resolve())]
    process = None
    started = time.monotonic()
    execution = {'command': command, 'monitor_errors': [], 'timed_out': False,
                 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        with (args.output_dir / 'runtime.log').open('x') as log, (args.output_dir / 'gpu_samples.jsonl').open('x') as samples:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'source/rl_training')},
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            execution['process_group'] = process.pid
            atomic_json(args.output_dir / 'execution.json', execution)
            next_sample = 0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                text = (args.output_dir / 'runtime.log').read_text(errors='replace')
                if deadline_expired(elapsed, text, plan):
                    execution['timed_out'] = True
                    raise TimeoutError('live PPO startup or wall deadline exceeded')
                if elapsed >= next_sample:
                    sample = subprocess.check_output(query, text=True, timeout=10)
                    samples.write(json.dumps({'elapsed_s': elapsed, 'gpu': sample}) + '\n'); samples.flush()
                    next_sample = elapsed + 5
                time.sleep(.2)
    except BaseException as error:
        execution['monitor_errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        if process is not None:
            if group_exists(process.pid): terminate_group(process, plan['cleanup_grace_s'])
            execution.update(returncode=process.poll(), process_group_empty=not group_exists(process.pid))
        execution['elapsed_s'] = time.monotonic() - started
        try: (args.output_dir / 'gpu_after.csv').write_text(subprocess.check_output(query, text=True, timeout=10))
        except Exception as error: execution['monitor_errors'].append(str(error))
        atomic_json(args.output_dir / 'execution.json', execution)
    from source_live_ppo_validation import validate
    try:
        if plan['input_sha256'] != _input_hashes() or plan['dependency_sha256'] != dependencies():
            execution['monitor_errors'].append('inputs changed during live PPO diagnostic')
    except Exception as error:
        execution['monitor_errors'].append('post-run input audit: ' + str(error))
    atomic_json(args.output_dir / 'execution.json', execution)
    acceptance = validate(args.output_dir, plan, execution, checkpoint=args.checkpoint)
    counts = {}
    try: counts = json.loads((args.output_dir / 'training/run_report.json').read_text())
    except (OSError, ValueError): pass
    receipt = ledger.finish(plan['attempt_id'], accepted=acceptance['accepted'],
        attempted=counts.get('attempted_transition_upper_bound', 0), returned=counts.get('returned_transitions', 0),
        stored=counts.get('stored_transitions', 0), updates=counts.get('completed_updates', 0))
    atomic_json(args.output_dir / 'budget_receipt.json', receipt)
    atomic_json(args.output_dir / 'acceptance.json', acceptance)
    print(json.dumps(acceptance, indent=2))
    return 0 if acceptance['accepted'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('prepare', 'run'):
        item = sub.add_parser(mode)
        for name in ('manifest', 'checkpoint', 'output-dir'): item.add_argument('--' + name, type=Path, required=True)
        if mode == 'prepare': item.add_argument('--attempt-id', required=True)
    check = sub.add_parser('validate')
    check.add_argument('--output-dir', required=True, type=Path)
    check.add_argument('--checkpoint', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare': return prepare(args)
    if args.mode == 'run': return run(args)
    from source_live_ppo_validation import validate
    result = validate(args.output_dir, json.loads((args.output_dir/'run_manifest.json').read_text()),
                      json.loads((args.output_dir/'execution.json').read_text()), checkpoint=args.checkpoint)
    print(json.dumps(result, indent=2))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
