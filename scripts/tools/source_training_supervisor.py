"""Supervise one separately approved cumulative numeric trial; no runtime on import."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/rl_training'), str(ROOT/'scripts/tools')]
from rl_training.source_scan.cumulative_training import SCHEMA,NUMERICS_SCHEMA
from rl_training.source_scan.smoke_artifacts import atomic_json
from rl_training.source_scan.training_budget_ledger import BudgetLedger
from source_training_entrypoint import validate_before_simulation, _input_hashes
from source_live_ppo_supervisor import dependencies, group_exists, terminate_group


def deadline_expired(elapsed, runtime_text):
    return elapsed > 900 or elapsed > 120 and '[source-numerics] initial-ready' not in runtime_text


def validate_result(folder, lock, execution):
    result = {'schema': 'lite3_source_bounded_numerics_acceptance_v1', 'accepted': False,
              'reasons': [], 'formal_pilot_authorized': False, 'training_ready': False}
    try:
        if (execution.get('returncode') != 0 or execution.get('process_group_empty') is not True
                or execution.get('timed_out') or execution.get('monitor_errors')):
            raise ValueError('numeric process or monitoring did not complete')
        folder = Path(folder)
        attempt = json.loads((folder/'attempt.json').read_text())
        report = json.loads((folder/'training/run_report.json').read_text())
        if (attempt.get('status') != 'completed' or attempt.get('environment_close_returned') is not True
                or attempt.get('application_close_state') not in ('closing', 'returned')):
            raise ValueError('numeric cleanup evidence incomplete')
        updates, transitions = lock['max_updates'], lock['max_transitions']
        expected = {'status': 'completed', 'run': lock['run'], 'data_kind': 'source_simulation',
            'num_envs': 120, 'rollout_length': 24, 'completed_updates': updates,
            'attempted_transition_upper_bound': transitions, 'returned_transitions': transitions,
            'stored_transitions': transitions, 'real_source_training_transitions': transitions,
            'real_source_ppo_updates': updates, 'source_backward_checks': updates*20,
            'source_optimizer_steps_started': updates*20, 'source_optimizer_steps_completed': updates*20,
            'checkpoint_sha256': lock['checkpoint_sha256'], 'base_revision': lock['base_revision'],
            'input_sha256': lock['input_sha256'], 'cumulative_budget': lock['cumulative_budget']}
        for key, value in expected.items():
            if type(report.get(key)) is not type(value) or report[key] != value:
                raise ValueError('numeric result mismatch: '+key)
        receipts = report['update_receipts']
        if len(receipts) != updates or [r['update'] for r in receipts] != list(range(1,updates+1)):
            raise ValueError('numeric update receipts incomplete')
        import math
        if any(not {'value','surrogate'} <= set(r['losses'])
               or not all(type(v) in (int,float) and math.isfinite(v) for v in r['losses'].values())
               or r['returned_transitions'] != 2880 or r['stored_transitions'] != 2880 for r in receipts):
            raise ValueError('invalid numeric losses or rollout receipts')
        if report['initial_actor_sha256'] == receipts[-1]['actor_sha256']:
            raise ValueError('numeric actor did not change')
        if report['initial_critic_sha256'] == receipts[-1]['critic_sha256']:
            raise ValueError('numeric critic did not change')
        if sum(v['transitions'] for v in report['exposure'].values()) != transitions:
            raise ValueError('numeric exposure does not match interactions')
        import torch
        from rl_training.source_scan.frozen_actor import FrozenSourceActor
        from source_live_ppo_validation import state_digest
        state = torch.load(folder/f'training/actor_update_{updates}.pt', map_location='cpu', weights_only=True)
        FrozenSourceActor(state)
        if state_digest(state) != receipts[-1]['actor_sha256'] or not all(bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError('numeric checkpoint differs from final actor')
        elapsed = report['elapsed_seconds']
        if type(elapsed) not in (int,float) or not 0 < elapsed < float('inf'):
            raise ValueError('invalid numeric elapsed time')
        rate = transitions/elapsed
        passed = rate >= 600.
        result.update(accepted=passed, real_transitions=transitions, completed_updates=updates,
            cumulative_transitions=transitions+lock['cumulative_budget']['baseline_charged_transitions'],
            measured_end_to_end_transitions_per_second=rate, g4_end_to_end_passed=passed,
            g3_g4_qualified=passed, evidence_scope='bounded numeric trial only')
        if lock.get('stage_revision'):
            result.update(g3_g4_qualified=False,limited_motion_evidence_qualified=passed,
                original_G3_accepted=False)
        if not passed: result['reasons'].append('end-to-end throughput below approved 600 transitions/s target')
    except Exception as error:
        result['reasons'].append(type(error).__name__+': '+str(error))
    return result


def run(args):
    manifest = json.loads(args.authorization.read_text())
    if manifest.get('schema') not in (SCHEMA,NUMERICS_SCHEMA) or manifest.get('user_approved') is not True or manifest.get('status') != 'approved':
        raise PermissionError('separately approved cumulative numeric manifest required')
    lock, revision, hashes = validate_before_simulation(args, manifest)
    if manifest['schema']==NUMERICS_SCHEMA:
        from rl_training.source_scan.delivery_stage import validate_slot
        validate_slot(manifest)
    if lock['dependency_sha256'] != dependencies():
        raise ValueError('numeric manifest omits or changes core runtime dependencies')
    query = ['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu','--format=csv,noheader']
    before = subprocess.check_output(query,text=True,timeout=10)
    row = next(r for r in before.splitlines() if r.startswith('0,')).split(',')
    if int(row[4].strip().split()[0]) < 12288: raise RuntimeError('GPU0 free memory below numeric gate')
    ledger = BudgetLedger(lock['budget_ledger']['path'])
    attempt_id = 'source-cumulative-numerics-'+hashlib.sha256(str(args.output_dir.resolve()).encode()).hexdigest()
    reservation = ledger.reserve_numerics(run=args.run, attempt_id=attempt_id,
        output_dir=args.output_dir, revision=revision, expected_reference=lock['budget_ledger'],
        expected_plan=lock['cumulative_budget'])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output_dir/'run_manifest.json',manifest)
    atomic_json(args.output_dir/'supervision.json',{'schema':'lite3_source_numerics_supervision_v1',
        'authorization':lock,'reservation':reservation})
    (args.output_dir/'gpu_before.csv').write_text(before)
    command = ['timeout','--signal=TERM','--kill-after=15s','900s',sys.executable,
        str(ROOT/'scripts/tools/source_training_entrypoint.py'),'--authorization',str((args.output_dir/'run_manifest.json').resolve()),
        '--checkpoint',str(args.checkpoint.resolve()),'--output-dir',str(args.output_dir.resolve()),
        '--run',str(args.run),'--budget-node','numerics','--device','cuda:0',
        '--supervised-ticket',str((args.output_dir/'supervision.json').resolve())]
    process = None; started = time.monotonic()
    execution = {'command':command,'monitor_errors':[],'timed_out':False}
    try:
        with (args.output_dir/'runtime.log').open('x') as log, (args.output_dir/'gpu_samples.jsonl').open('x') as samples:
            process = subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'source/rl_training')},
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            execution['process_group'] = process.pid
            atomic_json(args.output_dir/'execution.json',execution)
            next_sample = 0
            while process.poll() is None:
                elapsed = time.monotonic()-started
                text = (args.output_dir/'runtime.log').read_text(errors='replace')
                if deadline_expired(elapsed, text):
                    execution['timed_out'] = True; raise TimeoutError('numeric startup or wall timeout')
                if elapsed >= next_sample:
                    samples.write(json.dumps({'elapsed_s':elapsed,'gpu':subprocess.check_output(query,text=True,timeout=10)})+'\n')
                    samples.flush(); next_sample = elapsed+5
                time.sleep(.2)
    except BaseException as error:
        execution['monitor_errors'].append(type(error).__name__+': '+str(error))
    finally:
        if process is not None:
            if group_exists(process.pid): terminate_group(process,15)
            execution.update(returncode=process.poll(),process_group_empty=not group_exists(process.pid))
        execution['elapsed_s'] = time.monotonic()-started
        try:
            (args.output_dir/'gpu_after.csv').write_text(subprocess.check_output(query,text=True,timeout=10))
            if _input_hashes() != hashes or dependencies() != lock['dependency_sha256']:
                execution['monitor_errors'].append('numeric inputs changed during execution')
        except Exception as error: execution['monitor_errors'].append(str(error))
        atomic_json(args.output_dir/'execution.json',execution)
    acceptance = validate_result(args.output_dir,lock,execution)
    counts = {}
    try: counts = json.loads((args.output_dir/'training/run_report.json').read_text())
    except (OSError,ValueError): pass
    receipt = ledger.finish(attempt_id,accepted=acceptance['accepted'],
        attempted=counts.get('attempted_transition_upper_bound',0),returned=counts.get('returned_transitions',0),
        stored=counts.get('stored_transitions',0),updates=counts.get('completed_updates',0))
    atomic_json(args.output_dir/'budget_receipt.json',receipt)
    atomic_json(args.output_dir/'acceptance.json',acceptance)
    print(json.dumps(acceptance,indent=2))
    return 0 if acceptance['accepted'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('authorization','checkpoint','output-dir'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run',type=int,choices=(1,2,3),required=True)
    parser.add_argument('--budget-node',choices=('numerics',),default='numerics')
    parser.add_argument('--device',choices=('cuda:0',),default='cuda:0')
    raise SystemExit(run(parser.parse_args()))
