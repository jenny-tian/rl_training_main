"""G2 reset-boundary audit; importable without Isaac Sim. No training."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json

import torch

from rl_training.source_scan.parallel_terrain_allocation import (
    ReservedRowTerrainAllocator, SOURCE_RESET_BLOCK_COUNTS,
)


def reset_batches():
    """300 direct resets, with single, scattered, and large rotating batches."""
    return [(0,), (59, 7, 41, 1, 19)] + [
        tuple((offset + i) % 60 for i in range(45)) for offset in range(0, 60, 10)
    ] + [tuple(range(24))]


def snapshot(env):
    terrain = env.scene.terrain
    robot = env.scene['robot'].data
    values = {
        'row': terrain.terrain_levels, 'column': terrain.terrain_types,
        'origin': terrain.env_origins, 'root': robot.root_state_w,
        'joint_pos': robot.joint_pos, 'joint_vel': robot.joint_vel,
        'episode': env.episode_length_buf,
        'action': env.action_manager.action, 'previous_action': env.action_manager.prev_action,
    }
    for name in env.command_manager.active_terms:
        values['command/' + name] = env.command_manager.get_command(name)
    for group, terms in env.observation_manager._group_obs_term_history_buffer.items():
        for name, buffer in terms.items():
            if buffer._buffer is None:
                raise AssertionError(f'uninitialized observation history: {group}/{name}')
            values[f'history/{group}/{name}'] = buffer._buffer.transpose(0, 1)
            values[f'pushes/{group}/{name}'] = buffer._num_pushes
    for name, sensor in env.scene.sensors.items():
        for attr in ('_timestamp', '_timestamp_last_update', '_is_outdated'):
            values[f'sensor/{name}/{attr}'] = getattr(sensor, attr)
        # Reading raw caches must not trigger lazy sensor recomputation.
        for attr, value in vars(sensor._data).items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == env.num_envs:
                values[f'cache/{name}/{attr}'] = value
    return {name: value.detach().clone() for name, value in values.items()}


def check_transition(before, after, ids):
    """Compare only the reset boundary, never a physics integration interval."""
    if before.keys() != after.keys():
        raise AssertionError('snapshot buffer inventory changed')
    selected = torch.as_tensor(sorted(ids), device=before['row'].device, dtype=torch.long)
    other = torch.ones(len(before['row']), device=selected.device, dtype=torch.bool)
    other[selected] = False
    for name in before:
        if not torch.equal(before[name][other], after[name][other]):
            raise AssertionError(f'reset changed untouched buffer: {name}')
        value = after[name][selected]
        if name in ('root', 'joint_pos', 'joint_vel', 'origin') or name.startswith('command/'):
            if not torch.isfinite(value).all():
                raise AssertionError(f'nonfinite reset state: {name}')
        contact_clear = name.startswith('cache/contact_forces/') and name.rsplit('/', 1)[-1] in (
            'net_forces_w', 'net_forces_w_history', 'force_matrix_w', 'force_matrix_w_history',
            'current_air_time', 'last_air_time', 'current_contact_time', 'last_contact_time',
        )
        if contact_clear or name.startswith(('history/', 'pushes/')) or name in ('episode', 'action', 'previous_action'):
            if torch.count_nonzero(value):
                raise AssertionError(f'reset did not clear {name}')
        if name.startswith('sensor/'):
            if name.endswith('/_is_outdated'):
                if not value.all():
                    raise AssertionError(f'reset did not invalidate {name}')
            elif torch.count_nonzero(value):
                raise AssertionError(f'reset did not clear {name}')


def summarize_tickets(tickets):
    if [t.assignment_index for t in tickets] != list(range(len(tickets))):
        raise AssertionError('missing, duplicate, or reordered assignment tickets')
    blocks = []
    for offset in range(0, len(tickets), 120):
        block = tickets[offset:offset + 120]
        if any(t.block_index != offset // 120 or t.index_in_block != i for i, t in enumerate(block)):
            raise AssertionError('invalid block position')
        counts = dict(Counter(t.subterrain for t in block))
        if len(block) == 120 and counts != SOURCE_RESET_BLOCK_COUNTS:
            raise AssertionError('incorrect full-block terrain counts')
        blocks.append({'block_index': offset // 120, 'size': len(block), 'counts': counts,
                       'complete': len(block) == 120})
    return {'assignment_count': len(tickets), 'blocks': blocks,
            'complete_blocks': sum(b['complete'] for b in blocks)}


def _digest(value):
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


def run_lifecycle(env, seed, journal_path, trace, *, shared_bank=False):
    num_envs = int(env.num_envs) if shared_bank else 60
    if env.num_envs != num_envs or (shared_bank and seed != 130196):
        raise ValueError('G2 requires 60 reserved-row envs or two/eight shared envs with seed 130196')
    direct_batches = None
    if shared_bank:
        from rl_training.source_scan.shared_terrain_bank import SharedBankTerrainAllocator
        from source_shared_terrain_diagnostic import check_scanners
        if num_envs not in (2, 8) or seed != 130196:
            raise ValueError('shared lifecycle supports two or eight environments with seed 130196')
        if tuple(env.scene.terrain.terrain_origins.shape) != (1, 15, 3):
            raise AssertionError('shared lifecycle requires one bank row')
        oracle = SharedBankTerrainAllocator(num_envs, SOURCE_RESET_BLOCK_COUNTS, seed, bank_rows=1)
        if num_envs == 2:
            batches = [(0,), (1,), (1, 0)] * 30
            auto_ids = [0]
        else:
            # Eight fixed permutations cover singleton, scattered, unordered,
            # and all-environment batches while completing one 120-ticket block.
            pattern = [(7,), (6, 1, 3), (4, 2, 0), tuple(range(7, -1, -1))]
            batches = pattern * 8
            auto_ids = [7]
        direct_batches = [tuple(ids) for ids in batches]
    else:
        oracle = ReservedRowTerrainAllocator(60, SOURCE_RESET_BLOCK_COUNTS, seed)
        batches = reset_batches()
        auto_ids = [2, 23, 58]
    tickets = list(oracle.reset(list(range(num_envs))))
    if tuple(tickets) != env._source_parallel_terrain_allocator.assignments:
        raise AssertionError('initial assignments differ from deterministic replay')
    original_reset = env._reset_idx
    records = []
    mode = 'direct'
    with journal_path.open('x', encoding='utf-8') as journal:
        journal.write(json.dumps({'initial_assignments': [asdict(t) for t in tickets]}) + '\n')
        journal.flush()

        def audited_reset(env_ids):
            input_ids = env_ids.cpu().tolist()
            ids = sorted(input_ids)
            before = snapshot(env)
            prior = env._source_parallel_terrain_allocator.assignments
            journal.write(json.dumps({'phase': 'before_reset', 'mode': mode, 'ids': ids,
                                      'input_ids': input_ids,
                                      'prior_assignments': [asdict(t) for t in prior]}) + '\n')
            journal.flush()
            try:
                original_reset(env_ids)
                # The transactional adapter replaces the allocator after committing origins.
                # Re-read it after every reset; the pre-reset object is now a stale snapshot.
                current = env._source_parallel_terrain_allocator.assignments
                expected = oracle.reset(ids)
                actual = tuple(current[i] for i in ids)
                journal.write(json.dumps({
                    'phase': 'assignment_comparison', 'mode': mode, 'ids': ids,
                    'actual_assignments': [asdict(t) if t is not None else None for t in actual],
                    'expected_assignments': [asdict(t) for t in expected],
                    'current_assignments': [asdict(t) if t is not None else None for t in current],
                }, allow_nan=False) + '\n')
                journal.flush()
                after = snapshot(env)
                check_transition(before, after, ids)
                if actual != expected:
                    raise AssertionError('runtime assignment differs from deterministic replay')
                for i in set(range(num_envs)) - set(ids):
                    if current[i] != prior[i]:
                        raise AssertionError('untouched assignment changed')
                terrain = env.scene.terrain
                index = torch.tensor(ids, device=env.device, dtype=torch.long)
                origins = terrain.terrain_origins[terrain.terrain_levels[index], terrain.terrain_types[index]]
                if not torch.equal(origins, terrain.env_origins[index]):
                    raise AssertionError('reset origin does not match assigned tile')
                root = env.scene['robot'].data
                if not torch.allclose(root.root_pos_w[index], root.default_root_state[index, :3] + origins, atol=1e-5, rtol=0):
                    raise AssertionError('reset root does not match assigned origin')
                tickets.extend(actual)
                record = {'before_buffer_sha256': {k: _digest(v) for k, v in before.items()},
                          'phase': 'reset_checked', 'mode': mode, 'ids': ids,
                          'assignments': [asdict(t) for t in actual],
                          'before': {k: {'sha256': _digest(v), 'selected': v[index].cpu().tolist()}
                                     for k, v in before.items() if k in ('root', 'origin', 'row', 'column', 'episode') or k.startswith('command/')},
                          'after': {k: {'sha256': _digest(v)} for k, v in after.items()},
                          'history_buffers': [k for k in after if k.startswith('history/')],
                          'untouched_buffers_equal': True}
                records.append(record)
                journal.write(json.dumps(record, allow_nan=False) + '\n')
                journal.flush()
                trace(f'lifecycle-reset-checked: batch={len(records)} count={len(ids)} mode={mode}')
            except Exception as error:
                journal.write(json.dumps({
                    'phase': 'reset_failed', 'mode': mode, 'ids': ids,
                    'error_type': type(error).__name__, 'error': str(error),
                }) + '\n')
                journal.flush()
                raise

        env._reset_idx = audited_reset
        try:
            for ids in batches:
                env._reset_idx(torch.tensor(ids, device=env.device, dtype=torch.long))
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(0.0)
                for name in ('height_scanner', 'height_scanner_base'):
                    sensor = env.scene.sensors[name]
                    if not torch.isfinite(sensor.data.ray_hits_w).all():
                        raise AssertionError(f'nonfinite refreshed rays: {name}')
                    if sensor._is_outdated[list(ids)].any():
                        raise AssertionError(f'sensor did not refresh: {name}')
                if shared_bank:
                    journal.write(json.dumps({'phase': 'shared_reset_scanners', 'batch': len(records),
                                              'scanners': check_scanners(env, flat_entry=True)}) + '\n')
                    journal.flush()
                obs = env.observation_manager.compute(update_history=True)
                if not all(torch.isfinite(v).all() for v in obs.values()):
                    raise AssertionError('nonfinite refreshed observations')
            mode = 'automatic_timeout'
            env.episode_length_buf[auto_ids] = env.max_episode_length - 1
            count_before = len(records)
            action = torch.zeros_like(env.action_manager.action)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated.any() or truncated.nonzero().flatten().cpu().tolist() != auto_ids:
                raise AssertionError('unexpected termination or missing diagnostic timeout')
            if len(records) != count_before + 1 or records[-1]['ids'] != auto_ids:
                raise AssertionError('env.step did not execute exactly the expected reset batch')
            if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                raise AssertionError('nonfinite automatic reset output')
            for name in ('height_scanner', 'height_scanner_base'):
                if not torch.isfinite(env.scene.sensors[name].data.ray_hits_w).all():
                    raise AssertionError(f'nonfinite automatic reset rays: {name}')
            if shared_bank:
                journal.write(json.dumps({'phase': 'automatic_reset_output',
                                          'terminated': terminated.cpu().tolist(),
                                          'truncated': truncated.cpu().tolist(),
                                          'scanners': check_scanners(env)}) + '\n')
                journal.flush()
                mode = 'post_automatic_reset'
                count_after_auto = len(records)
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated.any() or truncated.any() or len(records) != count_after_auto:
                    raise AssertionError('unexpected reset on first post-reset physics step')
                if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                    raise AssertionError('nonfinite first post-reset physics output')
                journal.write(json.dumps({'phase': 'post_automatic_reset_step',
                                          'scanners': check_scanners(env), 'passed': True}) + '\n')
                journal.flush()
        finally:
            env._reset_idx = original_reset
    summary = summarize_tickets(tickets)
    direct_count = sum(map(len, direct_batches)) if shared_bank else 300
    expected_assignments = (num_envs + direct_count + len(auto_ids)) if shared_bank else 363
    expected_blocks = 1 if shared_bank else 3
    if summary['assignment_count'] != expected_assignments or summary['complete_blocks'] != expected_blocks:
        raise AssertionError('incomplete lifecycle coverage')
    batch_coverage = None
    if shared_bank and num_envs == 8:
        flat_batches = direct_batches
        batch_coverage = {
            'singleton': any(len(batch) == 1 for batch in flat_batches),
            'noncontiguous': any(len(batch) >= 2 and max(batch) - min(batch) + 1 > len(batch)
                                 for batch in flat_batches),
            'unordered_input': any(len(batch) >= 2 and tuple(batch) != tuple(sorted(batch))
                                   for batch in flat_batches),
            'large_batch': any(len(batch) >= 6 for batch in flat_batches),
            'highest_env_id': any(num_envs - 1 in batch for batch in flat_batches),
        }
    return {'status': 'shared_reset_lifecycle_passed' if shared_bank else 'reset_lifecycle_passed',
            **summary, 'direct_reset_count': direct_count if shared_bank else 300,
            'direct_reset_batches': [list(batch) for batch in direct_batches] if shared_bank else None,
            'batch_coverage': batch_coverage,
            'tail_ticket_count': len(tickets) % sum(SOURCE_RESET_BLOCK_COUNTS.values()),
            'bank_rows': 1 if shared_bank else None, 'num_envs': num_envs,
            'physics_control_steps': 2 if shared_bank else 1,
            'post_automatic_reset_step_checked': shared_bank,
            'automatic_reset_ids': auto_ids, 'reset_batch_count': len(records),
            'journal': str(journal_path), 'diagnostic_timeout_injected': True,
            'history_buffers': records[0]['history_buffers'],
            'unchanged_comparison_scope': 'immediately around _reset_idx, excluding physics steps',
            'motion_isolation_qualified': False}
