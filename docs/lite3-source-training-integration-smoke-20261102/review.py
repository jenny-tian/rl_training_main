"""Review archived evidence only; never start Isaac or reconstruct missing data."""
from pathlib import Path
import hashlib
import json


def review(folder):
    raw = folder / 'raw'
    manifest = json.loads((raw/'run_manifest.json').read_text())
    execution = json.loads((raw/'execution.json').read_text())
    result = json.loads((raw/'result.json').read_text())
    cleanup = json.loads((folder/'cleanup_check.json').read_text())
    runtime = (raw/'runtime.log').read_text()
    kit = (raw/'kit_runtime.log').read_text()
    marker = '[source-training-smoke] wiring probe passed; PPO updates=0'
    checks = {
        'single_attempt_approved': manifest['user_approved'] and manifest['retry'] is False,
        'revision_matches_initial_snapshot': result['base_revision'] == manifest['base_revision'],
        'process_exit_zero': execution['returncode'] == 0,
        'within_wall_timeout': execution['elapsed_s'] < manifest['wall_timeout_s'],
        'exactly_one_pass_marker': runtime.count(marker) == 1,
        'two_environment_construction_logged': 'Number of environments: 2' in runtime,
        'device_cuda0_logged': 'Environment device    : cuda:0' in runtime,
        'manager_endings_registered': all(term in runtime for term in (
            'source_training_safety', 'source_training_completion', 'source_training_timeout')),
        'no_python_traceback_logged': 'Traceback (most recent call last)' not in runtime,
        'app_shutdown_reached': 'SimulationApp.close: shutting down app and releasing framework' in kit,
        'no_process_group_remaining': not cleanup['remaining_group_processes'],
        'known_processes_exited': not any(cleanup['known_pid_exists'].values()),
    }
    samples = [json.loads(line) for line in (raw/'gpu_samples.jsonl').read_text().splitlines()]
    gpu0 = []
    for sample in samples:
        if sample['returncode'] == 0:
            row = next(line for line in sample['gpu'].splitlines() if line.startswith('0,'))
            values = [value.strip() for value in row.split(',')]
            gpu0.append({'elapsed_s': sample['elapsed_s'], 'used_mib': int(values[2].split()[0]),
                         'free_mib': int(values[3].split()[0])})
    complete = (result.get('status') == 'passed' and result.get('simulation_control_steps') == 24
                and len(result.get('steps', [])) == 24 and 'exposure' in result)
    assert all(checks.values()), checks
    # These assertions classify the actual incomplete artifact; they do not
    # change it or turn the log marker into measured per-step evidence.
    assert result.get('status') == 'starting' and not complete
    assert result.get('simulation_control_steps') == 0 and 'steps' not in result
    return {
        'schema': 'lite3_source_training_smoke_evidence_review_v1',
        'status': 'incomplete_runtime_artifacts', 'full_stage_acceptance': False,
        'checks': checks, 'final_structured_report_complete': complete,
        'raw_report_status': result['status'],
        'raw_report_control_steps_are_stale_placeholder': True,
        'raw_step_records': len(result.get('steps', [])),
        'independently_reviewable_control_steps': None,
        'control_steps_inferred_from_pinned_code_pass_marker': 24,
        'diagnostic_transitions_inferred_from_pass_marker': 48,
        'inference_limit': 'No raw actions, terminal tensors or exposure counts were persisted; do not reconstruct them.',
        'ppo_updates': 0, 'retry_count': 0, 'training_ready': False, 'g3_g4_qualified': False,
        'elapsed_supervisor_s': execution['elapsed_s'],
        'gpu0_samples': gpu0, 'gpu0_sampled_total_peak_used_mib': max(row['used_mib'] for row in gpu0),
        'gpu0_sampled_min_free_mib': min(row['free_mib'] for row in gpu0),
        'gpu_measurement_limit': '5-second samples across all GPU users; not isolated process peak or throughput.',
        'shutdown_diagnosis': 'Final report write follows app.close; Kit shutdown/release is logged, but write did not happen.',
        'next_stage': 'CPU-only report persistence repair, awaiting user approval',
        'raw_sha256': {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(raw.iterdir()) if p.is_file()},
    }


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    output = folder/'review.json'
    with output.open('x') as stream:
        json.dump(review(folder), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Review: incomplete_runtime_artifacts; full stage acceptance=False; reviewer simulation launches=0')
