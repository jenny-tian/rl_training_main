"""Offline review of this completed two-environment run; no simulator imports."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/rl_training'))
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.smoke_artifacts import validate_smoke_artifacts


def review(folder):
    raw = folder/'raw'
    report = json.loads((raw/'result.json').read_text())
    plan = json.loads((raw/'run_manifest.json').read_text())
    execution = json.loads((raw/'execution.json').read_text())
    acceptance = json.loads((raw/'acceptance.json').read_text())
    cleanup = json.loads((raw/'result.cleanup.json').read_text())
    independent_cleanup = json.loads((folder/'cleanup_check.json').read_text())
    rows = [json.loads(line) for line in (raw/'result.steps.jsonl').read_text().splitlines()]
    validated = validate_smoke_artifacts(raw/'result.json', returncode=execution['returncode'],
        expected_attempt_id=plan['attempt_id'], expected_revision=plan['base_revision'],
        expected_checkpoint_sha256=plan['checkpoint_sha256'])
    assert validated['accepted'] and acceptance['accepted']
    assert execution['process_group_empty'] and not execution['monitor_errors']
    assert not independent_cleanup['remaining_group_processes']
    assert not independent_cleanup['known_process_group_leader_exists']
    assert len(rows) == 24 and report['total_transitions'] == 48 and report['ppo_updates'] == 0
    assert plan['user_approved'] and plan['retry'] is False
    assert report['base_revision'] == '1c0d1a6dfab96529ba33262cca24e5c3eb323b6d'
    labels = tuple(SOURCE_RESET_BLOCK_COUNTS)
    transitions = Counter()
    resets = Counter(labels[column] for column in rows[0]['terrain_columns_before'])
    clipped = projected = paused = 0
    for index, row in enumerate(rows, 1):
        assert row['step'] == index and row['attempt_id'] == plan['attempt_id']
        transitions.update(labels[column] for column in row['terrain_columns_before'])
        for env_id in range(2):
            clipped += any(abs(a-b) > 1e-6 for a, b in zip(row['raw_action'][env_id], row['base_action'][env_id]))
            projected += row['projected'][env_id][0]
            paused += row['paused'][env_id][0]
            assert row['requested_action'][env_id] == row['base_action'][env_id]
            for field in ('raw_action', 'base_action', 'requested_action', 'executed_action'):
                assert all(math.isfinite(value) for value in row[field][env_id])
            if row['done'][env_id]:
                resets.update([labels[row['terrain_columns_after'][env_id]]])
    for name, counts in report['exposure'].items():
        assert counts == {'episode_assignments': resets[name], 'transitions': transitions[name]}
    last = rows[-1]
    terminal = last['terminal']
    assert last['done'] == [1, 0] and last['time_outs'] == [True, False]
    assert last['episode_counters'] == [0, 24]
    assert terminal['ids'] == [0] and terminal['steps'] == [2000]
    assert terminal['terminated'] == [False, False] and terminal['truncated'] == [True, False]
    assert terminal['terms']['source_training_timeout'] == [True]
    assert all(values == [False] for name, values in terminal['terms'].items() if name != 'source_training_timeout')
    observations = terminal['observations']
    assert len(observations['policy'][0]) == 45 and len(observations['critic'][0]) == report['critic_width'] == 393
    assert all(math.isfinite(value) for group in observations.values() for row in group for value in row)
    last_action_error = max(abs(a-b) for observation, executed in zip(observations['policy'], last['executed_action'])
                            for a, b in zip(observation[-12:], executed))
    assert last_action_error == 0.0
    samples = [json.loads(line) for line in (raw/'gpu_samples.jsonl').read_text().splitlines()]
    gpu0 = [[part.strip() for part in next(row for row in sample['gpu'].splitlines() if row.startswith('0,')).split(',')]
            for sample in samples]
    return {
        'schema': 'lite3_source_training_wiring_v2_offline_review_v1',
        'status': 'passed_within_two_environment_24_step_scope', 'accepted': True,
        'parent_acceptance_reproduced': validated, 'raw_record_count': len(rows),
        'control_steps_per_environment': 24, 'total_diagnostic_transitions': 48,
        'simulation_duration_per_environment_s': 0.48, 'ppo_updates': 0, 'retry_count': 0,
        'raw_clipped_environment_steps': clipped, 'raw_clip_denominator': 48,
        'raw_clip_fraction': clipped/48, 'additional_filter_projection_environment_steps': projected,
        'safety_paused_environment_steps': paused,
        'pure_timeout_terminal_env_ids': terminal['ids'], 'final_episode_counters': last['episode_counters'],
        'terminal_policy_width': 45, 'terminal_critic_width': 393,
        'terminal_policy_last_action_max_error': last_action_error,
        'exposure_nonzero': {name: counts for name, counts in report['exposure'].items() if any(counts.values())},
        'exposure_attribution_matches_independent_counts': True,
        'exposure_limit': 'Allocated cell labels only; this 0.48-second probe does not establish terrain skill or contact coverage.',
        'environment_close_returned': cleanup['environment'] == 'returned',
        'application_close_requested_process_exited': cleanup['application'] == 'closing' and execution['process_group_empty'],
        'application_close_returned': False, 'persistence_errors': cleanup['persistence_errors'],
        'supervisor_elapsed_s': execution['elapsed_s'],
        'gpu0_sampled_total_peak_used_mib': max(int(row[2].split()[0]) for row in gpu0),
        'gpu0_sampled_min_free_mib': min(int(row[3].split()[0]) for row in gpu0),
        'resource_limit': '5-second card-level samples, not isolated process peak or training throughput.',
        'g3_g4_qualified': False, 'training_ready': False,
        'raw_sha256': {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(raw.iterdir()) if path.is_file()},
    }


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    with (folder/'review.json').open('x') as stream:
        json.dump(review(folder), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Offline review passed: 24 rows, 48 diagnostic transitions, one pure timeout, no new simulation.')
