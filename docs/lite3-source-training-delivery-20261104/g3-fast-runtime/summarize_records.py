"""Summarize saved real control records without modifying gate acceptance."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

STAGE = Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/.git/source-delivery-evidence/lite3-training-delivery-20261104-v1')
OUT = STAGE / 'run/g3_fast'
LEDGER = Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/.git/source-ppo-ledgers/lite3-l1-v1.json')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def main():
    execution = json.loads((OUT / 'execution.json').read_text())
    assert execution.get('process_group_empty') is True, 'Wait for the actual job to finish'
    endings = Counter()
    terminal_batches = 0
    ended_envs = 0
    controls = 0
    partial_bytes = 0
    best_flat = dict(forward_progress_m=float('-inf'))
    completions = []
    with (OUT / 'result.steps.jsonl').open('rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'):
                partial_bytes += len(raw)
                break
            row = json.loads(raw)
            controls += 1
            assert row['step'] == controls
            assert len(row['done']) == len(row['metric_rows']) == len(row['columns_before']) == 120
            for env_id, (metric, column) in enumerate(zip(row['metric_rows'], row['columns_before'])):
                if column == 0 and metric['forward_progress_m'] > best_flat['forward_progress_m']:
                    best_flat = dict(forward_progress_m=metric['forward_progress_m'], step=controls,
                                     env_id=env_id, lifecycle_hold=metric['lifecycle_hold'])
            terminal = row['terminal']
            if terminal:
                terminal_batches += 1
                ended_envs += len(terminal['ids'])
                for j, env_id in enumerate(terminal['ids']):
                    for name, values in terminal['terms'].items():
                        if values[j]:
                            endings[name] += 1
                    if row['columns_before'][env_id] == 0 and terminal['completed'][j]:
                        completions.append(dict(step=controls, env_id=env_id))
    summary = dict(schema='lite3_G3_fast_actual_control_record_summary_v1',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Saved real control records and runtime outcomes; this does not grant G3 acceptance',
        complete_control_records=controls, planned_controls=2001,
        readonly_transitions_with_complete_control_records=120*controls,
        incomplete_final_control_record_bytes=partial_bytes,
        ordinary_terminal_batches=terminal_batches, ended_episodes=ended_envs,
        termination_reason_counts=dict(endings), best_flat_record=best_flat,
        candidate_flat_completions=completions, execution=execution,
        acceptance=json.loads((OUT/'acceptance.json').read_text()),
        report_status=json.loads((OUT/'result.json').read_text()).get('status'),
        production_ledger_sha256_after=digest(LEDGER),
        production_ledger_unchanged=digest(LEDGER)=='b1d224ba20a93c93c99189d700aeb145d506259f56462eb506def64f8f888189',
        new_ppo_updates=0, new_training_transitions=0,
        trained_checkpoints_found=[str(p) for p in OUT.rglob('*.pt')],
        current_source_revision=json.loads((OUT/'run_manifest.json').read_text())['base_revision'])
    target = STAGE/'g3_fast_actual_record_summary.json'
    target.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
