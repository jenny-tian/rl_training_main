"""Streaming CPU review of motion/contact/reset evidence; no Kit imports."""
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
from pathlib import Path

import numpy as np
import torch

from rl_training.source_scan.moving_diagnostic import (
    SCHEMA, CONTROL_STEPS, TIMEOUT_PROBES, EXPECTED_TERMS, restore_tensor,
    check_scan, check_reset_boundary, motion_sample, digest_file,
)
from rl_training.source_scan.smoke_artifacts import artifact_paths, _strict_json, _matrix
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.shared_terrain_bank import SharedBankTerrainAllocator
from source_parallel_reset_lifecycle import check_transition
from source_shared_tensor_contact import validate_topology, tensor_summary_passed, FORCE_ATOL, FORCE_RTOL
from source_shared_contact_report import classify_contact


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scan_record(record, root, origins, name):
    check_scan(root, restore_tensor(record['hits']), restore_tensor(record['hit_ids']),
               restore_tensor(record['mesh_ids']), record['ground_id'], origins,
               age=restore_tensor(record['age_s']), max_age=1e-6 if name=='height_scanner_base' else .02,
               base=name=='height_scanner_base')


def open_events(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()


def events_digest(path):
    digest = hashlib.sha256()
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def review_events(path, report, controls, *, control_steps=CONTROL_STEPS):
    start = report['start_sim_step']
    grid = restore_tensor(report['terrain_origins'])
    require(grid.shape == (1, 15, 3), 'wrong terrain origin grid')
    oracle = SharedBankTerrainAllocator(8, SOURCE_RESET_BLOCK_COUNTS, 131001, bank_rows=1)
    require([asdict(t) for t in oracle.reset(list(range(8)))] == report['initial_assignments'], 'initial ticket replay')
    previous = (restore_tensor(report['initial_positions_w']), restore_tensor(report['initial_velocities_w']))
    motion_count = contact_count = scan_count = reset_count = 0
    pending, reset_pending = set(), None
    topology = cross_mask = None
    details = []
    ground = Counter()
    summaries = []
    for line in open_events(path):
        event = _strict_json(line)
        phase = event.get('phase')
        if phase == 'tensor_contact_topology':
            require(topology is None, 'duplicate contact topology')
            topology = event
            validate_topology(topology['sensor_paths'], topology['filter_paths'], topology['sensor_paths'],
                              topology['ground_colliders'], num_envs=8)
            require(len(topology['sensor_paths']) == 136 and topology['capacity'] == 65536, 'body/capacity inventory')
            require(Counter(path.split('/')[3] for path in topology['sensor_paths']) ==
                    Counter({f'env_{i}': 17 for i in range(8)}), 'per-environment body inventory')
            cross_mask = np.array([[classify_contact(a,b)[0] == 'cross_environment' for b in targets]
                                   for a, targets in zip(topology['sensor_paths'], topology['filter_paths'])])
        elif phase == 'tensor_contact':
            details.append(event)
        elif phase == 'tensor_contact_sample':
            require(topology is not None and event['sim_step'] == start+contact_count+1, 'contact sequence')
            sensors, filters = topology['sensor_paths'], topology['filter_paths']
            counts = np.array(event['counts'])
            matrix = np.array(event['pair_forces_N'], dtype=np.float64)
            net = np.array(event['net_forces_N'], dtype=np.float64)
            require(counts.shape == (136, len(filters[0])) and matrix.shape == (*counts.shape, 3)
                    and net.shape == (136, 3) and counts.dtype.kind in 'iu' and np.all(counts >= 0), 'contact dimensions')
            require(np.isfinite(matrix).all() and np.isfinite(net).all(), 'nonfinite contact matrix')
            require(event['used'] == int(counts.sum()) < 65536, 'contact saturation/count')
            reconstructed = np.zeros_like(matrix)
            seen = set()
            for detail in details:
                require(detail['sim_step'] == event['sim_step'], 'contact points at wrong substep')
                a, b = detail['actors']; i = sensors.index(a); j = filters[i].index(b)
                require((i, j) not in seen, 'duplicate contact pair')
                seen.add((i, j))
                force, normal = np.array(detail['forces_N']), np.array(detail['normals'])
                count = detail['count']
                require(count == counts[i, j] and count > 0 and force.shape == (count, 1)
                        and normal.shape == (count, 3), 'contact point counts')
                for name, width in (('forces_N', 1), ('normals', 3), ('points_m', 3), ('separations_m', 1)):
                    array = np.array(detail[name])
                    require(array.shape == (count, width) and np.isfinite(array).all(), 'invalid contact point')
                require(np.allclose(np.linalg.norm(normal, axis=1), 1, atol=1e-3, rtol=0), 'contact normal')
                reconstructed[i, j] = (force*normal).sum(axis=0)
                kind, owner = classify_contact(a, b)
                require(kind != 'cross_environment', 'cross-environment points')
                if kind == 'ground' and np.any(np.abs(force) > FORCE_ATOL):
                    ground[owner] += count
            require(seen == set(map(tuple, np.argwhere(counts > 0))), 'missing detailed contact pair')
            require(not counts[cross_mask].any() and not matrix[cross_mask].any(), 'cross-environment matrix')
            require(np.allclose(reconstructed, matrix, atol=FORCE_ATOL, rtol=FORCE_RTOL), 'contact force reconstruction')
            require(np.allclose(matrix.sum(axis=1), net, atol=FORCE_ATOL, rtol=FORCE_RTOL), 'contact net mismatch')
            details.clear(); contact_count += 1
        elif phase == 'motion_substep':
            require(event['sim_step'] == start+motion_count+1 and contact_count == motion_count+1,
                    'motion must follow same-step contact polling')
            positions, velocities, quats = [restore_tensor(event[name]) for name in (
                'body_positions_w', 'body_link_velocities_w', 'body_link_quaternions_wxyz')]
            columns = torch.tensor([t.column for t in oracle.assignments])
            levels = torch.tensor([t.row for t in oracle.assignments])
            require(event['terrain_columns'] == columns.tolist() and event['terrain_levels'] == levels.tolist(), 'motion ownership')
            expected = motion_sample(positions, velocities, quats, grid, levels, columns, previous=previous)
            for name, atol in (('displacements_m', 2e-6), ('finite_difference_acceleration_mps2', 1e-4)):
                require(torch.allclose(restore_tensor(event[name]), restore_tensor(expected[name]),
                                       atol=atol, rtol=2e-6), 'motion derivative or reset rebase mismatch')
            require(event['safe_lower_xy_w'] == expected['safe_lower_xy_w'] and event['safe_upper_xy_w'] == expected['safe_upper_xy_w']
                    and event['outside_prediction'] == expected['outside_prediction'], 'guard evidence mismatch')
            origins, root = restore_tensor(event['origins_w']), restore_tensor(event['root_positions_w'])
            require(torch.equal(origins, grid[levels, columns]) and torch.equal(root, positions[:, 0]), 'root/origin mismatch')
            require(event['recovery_ids'] == sorted(pending), 'missing recovery-first-step evidence')
            require(set(event['scanners']) == {'height_scanner_base'}, 'base scan sample missing')
            scan_record(event['scanners']['height_scanner_base'], root, origins, 'height_scanner_base')
            pending.clear(); previous = (positions, velocities); motion_count += 1
        elif phase == 'reset_boundary':
            require(reset_pending is None and event['batch'] == reset_count+1, 'reset boundary sequence')
            step = (event['sim_step']-start)//4
            require(event['sim_step'] == start+step*4 and motion_count == step*4, 'reset boundary timing')
            control = controls[step-1]
            require(event['ids'] == control['terminal']['ids'], 'reset IDs versus terminal IDs')
            before = {name: restore_tensor(value) for name, value in event['before'].items()}
            after = {name: restore_tensor(value) for name, value in event['after'].items()}
            ids = torch.tensor(event['ids'])
            check_reset_boundary(before, after, ids); check_transition(before, after, event['ids'])
            require(event['assignments_before'] == [asdict(t) for t in oracle.assignments], 'before tickets')
            oracle.reset(event['ids'])
            require(event['assignments_after'] == [asdict(t) for t in oracle.assignments], 'after tickets')
            target = grid[torch.tensor([t.row for t in oracle.assignments]), torch.tensor([t.column for t in oracle.assignments])]
            root = restore_tensor(event['root_link_positions_after'])
            default = restore_tensor(event['default_root_positions'])
            require(torch.equal(after['origin'], target) and torch.allclose(root[ids], default[ids]+target[ids],
                    atol=1e-5, rtol=0), 'selected reset spawn origin/pose incorrect')
            reset_pending = event; reset_count += 1
        elif phase == 'reset_checked':
            require(reset_pending is not None and event['batch'] == reset_pending['batch']
                    and event['sim_step'] == reset_pending['sim_step'] and event['ids'] == reset_pending['ids'], 'reset checked mismatch')
            rebased = (restore_tensor(event['rebase_positions_w']), restore_tensor(event['rebase_velocities_w']))
            other = [i for i in range(8) if i not in event['ids']]
            require(all(torch.equal(a[other], b[other]) for a, b in zip(previous, rebased)), 'unaffected rebase mutation')
            previous = rebased; pending.update(event['ids']); reset_pending = None
        elif phase == 'control_scans':
            require(event['step'] == scan_count+1 and event['sim_step'] == start+4*(scan_count+1)
                    and motion_count == 4*(scan_count+1) and reset_pending is None, 'control scan timing')
            origins = torch.tensor(event['origins_w'], dtype=grid.dtype)
            columns = torch.tensor([t.column for t in oracle.assignments]); levels = torch.zeros(8, dtype=torch.long)
            require(torch.equal(origins, grid[levels, columns]), 'post-reset scan ownership')
            require(set(event['scanners']) == {'height_scanner', 'height_scanner_base'}, 'control scanner inventory')
            for name, record in event['scanners'].items():
                scan_record(record, torch.tensor(event['root_positions_w'], dtype=grid.dtype), origins, name)
            scan_count += 1
        elif phase == 'contact_summary':
            summaries.append(event)
        elif phase in ('contact_error', 'cross_environment_contact', 'motion_error'):
            raise ValueError('integrity error in event journal')
    require(motion_count == contact_count == control_steps*4 and scan_count == control_steps and not details and reset_pending is None,
            'incomplete motion/contact/reset evidence')
    require(report['motion_samples'] == control_steps*4 and report['reset_batches'] == reset_count
            and report['pending_recovery_ids'] == sorted(pending), 'event summary mismatch')
    require(len(summaries) == 1, 'missing or duplicate contact summary')
    contact = {key: value for key, value in summaries[0].items() if key != 'phase'}
    require(contact == report['contact_capture'] and tensor_summary_passed(contact, dict(ground), control_steps=control_steps),
            'contact positive controls/summary invalid')
    return {'motion_samples': motion_count, 'contact_samples': contact_count, 'reset_batches': reset_count,
            'control_scans': scan_count, 'ground_contact_points': dict(ground), 'pending_final_recovery_ids': sorted(pending)}


def validate(output, *, returncode, attempt_id, revision, checkpoint_sha256, process_group_empty=True):
    result = {'schema': 'lite3_eight_env_acceptance_v1', 'accepted': False, 'reasons': [], 'g3_g4_qualified': False}
    try:
        require(returncode == 0 and process_group_empty, 'exit failure or process-group residue')
        paths = artifact_paths(output)
        report = _strict_json(paths['result'].read_text())
        cleanup = _strict_json(paths['cleanup'].read_text())
        rows = [_strict_json(line) for line in paths['steps'].read_text().splitlines()]
        expected = {'schema': SCHEMA, 'attempt_id': attempt_id, 'base_revision': revision,
            'checkpoint_sha256': checkpoint_sha256, 'status': 'passed', 'num_envs': 8,
            'shared_bank_rows': 1, 'steps_written': 96, 'simulation_control_steps': 96,
            'control_steps_limit': 96, 'total_transitions': 768, 'ppo_updates': 0,
            'diagnostic_only': True, 'training_ready': False, 'g3_g4_qualified': False,
            'phase': 'result_before_shutdown', 'critic_width': 393,
            'geometry_seed': 30000, 'sampler_seed': 131001, 'ppo_seed': 130001,
            'noise_rng_seed': 132001, 'reset_state_id': 40000, 'reset_randomized': False, 'stochastic_std': 1.0}
        require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items()), 'report contract mismatch')
        require(len(report['body_names']) == len(set(report['body_names'])) == 17, 'body name inventory')
        require(len(rows) == 96 and report['steps_file'] == paths['steps'].name
                and report['steps_sha256'] == digest_file(paths['steps']), 'incomplete or altered control journal')
        events = Path(output).with_suffix('.events.jsonl')
        expected_name = events.name
        if not events.exists():
            events = events.with_suffix(events.suffix+'.gz')
        require(report['events_file'] == expected_name and report['events_sha256'] == events_digest(events), 'missing/altered event journal')
        require(cleanup['attempt_id'] == attempt_id and cleanup['environment'] == 'returned'
                and cleanup['application'] in ('closing', 'returned') and not cleanup['persistence_errors'], 'cleanup failure')
        labels = tuple(SOURCE_RESET_BLOCK_COUNTS)
        exposure = {name: {'episode_assignments': 0, 'transitions': 0} for name in labels}
        counters = [0]*8; prior_columns = None; ordinary_failures = 0; total_endings = 0
        for step, row in enumerate(rows, 1):
            require(row['step'] == step and row['attempt_id'] == attempt_id and row['forced_ids'] == TIMEOUT_PROBES.get(step, []), 'control identity/probe')
            for field in ('raw_action', 'base_action', 'requested_action', 'executed_action'):
                require(_matrix(row[field], 8, 12), 'invalid/nonfinite action journal')
            require(row['paused'] == [[False]]*8 and _matrix(row['projected'], 8, 1, boolean=True), 'paused or malformed filter output')
            require(row['requested_action'] == row['base_action'], 'request/base action mismatch')
            require(len(row['time_outs']) == 8 and all(type(value) is bool for value in row['time_outs']), 'timeout mask dtype')
            before, after = row['terrain_columns_before'], row['terrain_columns_after']
            require(all(isinstance(columns, list) and len(columns)==8
                        and all(type(col) is int and 0<=col<15 for col in columns) for columns in (before, after)), 'invalid terrain columns')
            require(prior_columns is None or before == prior_columns, 'terrain column sequence')
            require(len(row['done']) == 8 and all(type(v) is int and v in (0,1) for v in row['done']), 'invalid done mask')
            ended = [i for i in range(8) if row['done'][i]]
            require(set(row['forced_ids']).issubset(ended), 'forced timeout did not end')
            term = row['terminal']
            if ended:
                require(term['ids'] == ended and len(term['steps']) == len(ended) and set(term['terms']) == EXPECTED_TERMS, 'terminal IDs/terms')
                require(all(isinstance(values, list) and len(values) == len(ended)
                            and all(type(value) is bool for value in values) for values in term['terms'].values()), 'terminal term shape')
                require(_matrix(term['observations']['policy'], 8, 45) and _matrix(term['observations']['critic'], 8, 393), 'terminal observations')
                require(len(term['terminated']) == len(term['truncated']) == 8
                        and all(type(x) is bool for x in term['terminated']+term['truncated']), 'terminal masks')
                require(all(not term['terminated'][i] and not term['truncated'][i] for i in range(8) if i not in ended),
                        'terminal flags affect non-ending row')
                for j, i in enumerate(ended):
                    active = {name for name, values in term['terms'].items() if values[j]}
                    require(active and term['terminated'][i] == bool(active-{'source_training_timeout'})
                            and term['truncated'][i] == ('source_training_timeout' in active), 'terminal flags contradict terms')
                    if i in row['forced_ids']:
                        require(term['truncated'][i], 'injected timeout term missing')
                    ordinary_failures += bool(active-{'source_training_timeout', 'source_training_completion'})
                total_endings += len(ended)
                require(row['time_outs'] == [t and not f for t, f in zip(term['truncated'], term['terminated'])], 'bootstrap timeout mask')
            else:
                require(term is None and row['time_outs'] == [False]*8, 'unexpected terminal snapshot')
            for i in range(8):
                length = (1999 if i in row['forced_ids'] else counters[i])+1
                if step==1:exposure[labels[before[i]]]['episode_assignments'] += 1
                exposure[labels[before[i]]]['transitions'] += 1
                if i in ended:
                    require(term['steps'][ended.index(i)] == length, 'terminal episode counter')
                    exposure[labels[after[i]]]['episode_assignments'] += 1
                    counters[i] = 0
                else:
                    require(before[i] == after[i], 'unreset terrain column changed')
                    counters[i] = length
            require(row['episode_counters'] == counters, 'post-reset counters')
            prior_columns = after
        require(set(report['exposure']) == set(labels) and all(set(value) == {'episode_assignments', 'transitions'}
                    and all(type(count) is int and count >= 0 for count in value.values())
                    for value in report['exposure'].values()) and exposure == report['exposure'],
                'reset versus transition exposure mismatch')
        event_result = review_events(events, report, rows)
        require(event_result['reset_batches'] == sum(row['terminal'] is not None for row in rows), 'reset-boundary count')
        result.update(accepted=True, control_steps=96, diagnostic_transitions=768, ordinary_failure_episodes=ordinary_failures,
                      total_episode_endings=total_endings, event_review=event_result,
                      evidence_scope='bounded eight-environment motion/reset integrity only')
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, AttributeError, AssertionError, EOFError) as error:
        result['reasons'].append(type(error).__name__+':'+str(error))
    return result
