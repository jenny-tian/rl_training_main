"""Offline checks of G2 coverage and fault detection; no physics evidence."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_training.source_scan.parallel_terrain_allocation import ReservedRowTerrainAllocator, SOURCE_RESET_BLOCK_COUNTS

PATH = Path(__file__).resolve().parents[2] / 'scripts/tools/source_parallel_reset_lifecycle.py'
SPEC = importlib.util.spec_from_file_location('lifecycle_under_test', PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def tickets():
    allocator = ReservedRowTerrainAllocator(60, SOURCE_RESET_BLOCK_COUNTS, 130001)
    result = list(allocator.reset(list(range(60))))
    for ids in m.reset_batches() + [(2, 23, 58)]:
        result.extend(allocator.reset(ids))
    return result


def test_schedule_covers_single_scattered_large_and_three_full_blocks():
    batches = m.reset_batches()
    assert sum(map(len, batches)) == 300
    assert len(batches) == 9
    assert len(batches[0]) == 1
    assert batches[1] != tuple(sorted(batches[1]))
    assert any(len(b) == 45 for b in batches)
    assert set().union(*map(set, batches)) == set(range(60))
    for batch in batches:
        assert len(batch) == len(set(batch))
        assert all(0 <= i < 60 for i in batch)
    report = m.summarize_tickets(tickets())
    assert report['assignment_count'] == 363
    assert report['complete_blocks'] == 3
    assert report['blocks'][-1]['size'] == 3
    assert report['blocks'][-1]['complete'] is False


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'reordered', 'terrain', 'block'])
def test_bad_ticket_stream_is_rejected(fault):
    stream = tickets()
    if fault == 'missing':
        del stream[100]
    elif fault == 'duplicate':
        stream[100] = stream[99]
    elif fault == 'reordered':
        stream[20], stream[21] = stream[21], stream[20]
    elif fault == 'terrain':
        stream[0] = replace(stream[0], subterrain='bad')
    else:
        stream[0] = replace(stream[0], block_index=1)
    with pytest.raises(AssertionError):
        m.summarize_tickets(stream)


def transition():
    before = {name: torch.ones(4, 2) for name in (
        'row', 'root', 'origin', 'column', 'joint_pos', 'joint_vel', 'command/velocity',
        'episode', 'action', 'previous_action', 'history/policy/position', 'pushes/policy/position',
        'sensor/scanner/_timestamp', 'sensor/scanner/_timestamp_last_update', 'cache/scanner/ray_hits_w',
        'cache/contact_forces/net_forces_w_history')}
    before['sensor/scanner/_is_outdated'] = torch.zeros(4, dtype=torch.bool)
    after = {k: v.clone() for k, v in before.items()}
    for name in after:
        if name.startswith(('history/', 'pushes/', 'sensor/', 'cache/contact_forces/')) or name in ('episode', 'action', 'previous_action'):
            after[name][[0, 2]] = name.endswith('_is_outdated')
    return before, after


def test_correct_partial_reset_and_lazy_cache_invalidation_pass():
    before, after = transition()
    m.check_transition(before, after, [2, 0])


@pytest.mark.parametrize('buffer', list(transition()[0]))
def test_any_untouched_buffer_mutation_is_rejected(buffer):
    before, after = transition()
    after[buffer][1] = 9
    with pytest.raises(AssertionError, match='untouched'):
        m.check_transition(before, after, [0, 2])


@pytest.mark.parametrize('buffer', ['episode', 'action', 'previous_action', 'history/policy/position',
                                   'pushes/policy/position', 'sensor/scanner/_timestamp',
                                   'sensor/scanner/_timestamp_last_update', 'sensor/scanner/_is_outdated',
                                   'cache/contact_forces/net_forces_w_history'])
def test_missing_selected_reset_or_invalidation_is_rejected(buffer):
    before, after = transition()
    after[buffer][0] = before[buffer][0]
    with pytest.raises(AssertionError):
        m.check_transition(before, after, [0, 2])


def test_changed_buffer_inventory_fails_closed():
    before, after = transition()
    del after['history/policy/position']
    with pytest.raises(AssertionError, match='inventory'):
        m.check_transition(before, after, [0, 2])


@pytest.mark.parametrize('name', ['root', 'joint_pos', 'joint_vel', 'origin', 'command/velocity'])
def test_nonfinite_selected_state_is_rejected(name):
    before, after = transition()
    after[name][0] = float('nan')
    with pytest.raises(AssertionError, match='nonfinite'):
        m.check_transition(before, after, [0, 2])


def test_snapshot_does_not_refresh_lazy_sensors_and_normalizes_history_axis():
    from types import SimpleNamespace as NS
    class Sensor:
        _timestamp = torch.zeros(4)
        _timestamp_last_update = torch.zeros(4)
        _is_outdated = torch.ones(4, dtype=torch.bool)
        _data = NS(ray_hits_w=torch.ones(4, 3, 3))
        @property
        def data(self):
            raise AssertionError('snapshot must not trigger lazy sensor refresh')
    class Scene(dict):
        terrain = NS(terrain_levels=torch.arange(4), terrain_types=torch.zeros(4),
                     env_origins=torch.zeros(4, 3))
        sensors = {'scanner': Sensor()}
    scene = Scene(robot=NS(data=NS(root_state_w=torch.zeros(4, 13),
                                  joint_pos=torch.zeros(4, 12), joint_vel=torch.zeros(4, 12))))
    history = torch.arange(24.).reshape(3, 4, 2)
    env = NS(scene=scene, num_envs=4, episode_length_buf=torch.zeros(4),
             action_manager=NS(action=torch.zeros(4, 12), prev_action=torch.zeros(4, 12)),
             command_manager=NS(active_terms=['velocity'], get_command=lambda _: torch.ones(4, 3)),
             observation_manager=NS(_group_obs_term_history_buffer={
                 'policy': {'pos': NS(_buffer=history, _num_pushes=torch.ones(4))}}))
    result = m.snapshot(env)
    assert result['history/policy/pos'].shape == (4, 3, 2)
    assert torch.equal(result['history/policy/pos'][2], history[:, 2])
    history.zero_()
    assert result['history/policy/pos'].sum() > 0
