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


def _lifecycle_env(seed=130001, fault=None, fault_batch=1):
    """Synthetic managers, real transactional terrain adapter, no simulator."""
    from types import SimpleNamespace as NS
    from rl_training.source_scan.parallel_terrain_allocation import assign_source_terrain_for_parallel_reset

    helper_spec = importlib.util.spec_from_file_location(
        'allocation_fixtures_for_lifecycle', Path(__file__).with_name('test_parallel_terrain_allocation.py'))
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)
    env = helpers._fake_parallel_env(num_envs=60)
    terrain = env.scene.terrain
    assign_source_terrain_for_parallel_reset(env, list(range(60)), seed)

    class Sensor:
        def __init__(self, contact=False):
            self._timestamp = torch.ones(60)
            self._timestamp_last_update = torch.ones(60)
            self._is_outdated = torch.zeros(60, dtype=torch.bool)
            self._data = NS(net_forces_w_history=torch.ones(60, 3, 2, 3)) if contact else NS(
                ray_hits_w=torch.ones(60, 4, 3))

        @property
        def data(self):
            self._is_outdated[:] = False
            return self._data

    class Scene(dict):
        def write_data_to_sim(self):
            pass

        def update(self, dt):
            assert dt == 0.0

    root = NS(root_state_w=torch.zeros(60, 13), default_root_state=torch.zeros(60, 13),
              joint_pos=torch.ones(60, 12), joint_vel=torch.zeros(60, 12))
    root.default_root_state[:, 2] = 0.5
    root.root_pos_w = root.root_state_w[:, :3]
    root.root_pos_w[:] = root.default_root_state[:, :3] + terrain.env_origins
    env.scene = Scene(robot=NS(data=root))
    env.scene.terrain = terrain
    env.scene.sensors = {name: Sensor(name == 'contact_forces') for name in (
        'height_scanner', 'height_scanner_base', 'contact_forces')}
    env.device = 'cpu'
    env.max_episode_length = 1000
    env.episode_length_buf = torch.ones(60, dtype=torch.long)
    env.action_manager = NS(action=torch.ones(60, 12), prev_action=torch.ones(60, 12))
    command = torch.ones(60, 3)
    env.command_manager = NS(active_terms=['velocity'], get_command=lambda _: command)
    history = NS(_buffer=torch.ones(3, 60, 2), _num_pushes=torch.ones(60, dtype=torch.long))

    def compute(update_history):
        assert update_history
        history._buffer[:] = 1
        history._num_pushes += 1
        return {'policy': torch.ones(60, 2)}

    env.observation_manager = NS(_group_obs_term_history_buffer={'policy': {'pos': history}}, compute=compute)
    env.sim = NS(forward=lambda: None)
    env.replaced_objects = []
    env.reset_calls = []
    env.step_calls = 0

    def reset(ids):
        env.reset_calls.append(ids.tolist())
        previous = env._source_parallel_terrain_allocator
        if fault == 'reset_exception' and len(env.reset_calls) == fault_batch:
            raise RuntimeError('injected reset failure')
        assign_source_terrain_for_parallel_reset(env, ids, seed)
        env.replaced_objects.append(previous)
        assert previous is not env._source_parallel_terrain_allocator
        root.root_pos_w[ids] = root.default_root_state[ids, :3] + terrain.env_origins[ids]
        env.episode_length_buf[ids] = 0
        env.action_manager.action[ids] = 0
        env.action_manager.prev_action[ids] = 0
        history._buffer[:, ids] = 0
        history._num_pushes[ids] = 0
        for sensor in env.scene.sensors.values():
            sensor._timestamp[ids] = 0
            sensor._timestamp_last_update[ids] = 0
            sensor._is_outdated[ids] = True
        env.scene.sensors['contact_forces']._data.net_forces_w_history[ids] = 0
        if len(env.reset_calls) == fault_batch:
            current = env._source_parallel_terrain_allocator
            if fault == 'wrong_selected_ticket':
                i = int(ids[0])
                current._active[i] = replace(current.assignments[i], assignment_index=-1)
            elif fault == 'untouched_ticket':
                i = next(i for i in range(60) if i not in ids.tolist())
                current._active[i] = replace(current.assignments[i], episode_index=999)
            elif fault == 'untouched_state':
                i = next(i for i in range(60) if i not in ids.tolist())
                root.root_state_w[i, 0] += 1

    def step(action):
        env.step_calls += 1
        assert not action.any()
        # Normal movement before the reset boundary must not be called reset cross-talk.
        root.root_pos_w[:, 0] += 0.01
        env.episode_length_buf += 1
        truncated = env.episode_length_buf >= env.max_episode_length
        env._reset_idx(truncated.nonzero().flatten())
        return compute(True), torch.zeros(60), torch.zeros(60, dtype=torch.bool), truncated, {}

    env._reset_idx = reset
    env.step = step
    return env


def _journal(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('seed', [130001, 17])
def test_full_lifecycle_uses_current_allocator_after_each_transaction(tmp_path, seed):
    env = _lifecycle_env(seed)
    original = env._reset_idx
    path = tmp_path / 'resets.jsonl'
    report = m.run_lifecycle(env, seed, path, lambda _: None)
    assert report['assignment_count'] == 363
    assert report['reset_batch_count'] == 10
    assert report['complete_blocks'] == 3
    assert report['history_buffers'] == ['history/policy/pos']
    assert len(env.replaced_objects) == 10
    assert len({id(obj) for obj in env.replaced_objects}) == 10
    assert env.replaced_objects[0].assignments[0].assignment_index == 0
    assert env.step_calls == 1
    assert env.reset_calls[-1] == [2, 23, 58]
    assert env._reset_idx is original
    records = _journal(path)
    comparisons = [r for r in records if r.get('phase') == 'assignment_comparison']
    assert len(comparisons) == 10
    assert comparisons[0]['actual_assignments'][0]['assignment_index'] == 60
    assert comparisons[1]['ids'] == [1, 7, 19, 41, 59]
    for record in comparisons:
        assert record['actual_assignments'] == record['expected_assignments']
    assert sum(r.get('phase') == 'reset_checked' for r in records) == 10
    assert not any(r.get('phase') == 'reset_failed' for r in records)


@pytest.mark.parametrize('batch', [1, 2, 10])
@pytest.mark.parametrize('fault, message', [
    ('wrong_selected_ticket', 'runtime assignment differs'),
    ('untouched_ticket', 'untouched assignment changed'),
    ('untouched_state', 'reset changed untouched buffer'),
    ('reset_exception', 'injected reset failure'),
])
def test_full_hook_failure_records_evidence_stops_and_restores_reset(tmp_path, batch, fault, message):
    env = _lifecycle_env(fault=fault, fault_batch=batch)
    original = env._reset_idx
    path = tmp_path / 'failure.jsonl'
    with pytest.raises((AssertionError, RuntimeError), match=message):
        m.run_lifecycle(env, 130001, path, lambda _: None)
    assert env._reset_idx is original
    assert len(env.reset_calls) == batch
    records = _journal(path)
    assert records[-1]['phase'] == 'reset_failed'
    assert message in records[-1]['error']
    assert sum(r.get('phase') == 'reset_checked' for r in records) == batch - 1
    if fault != 'reset_exception':
        comparison = records[-2]
        assert comparison['phase'] == 'assignment_comparison'
        assert len(comparison['current_assignments']) == 60
        if fault == 'wrong_selected_ticket':
            assert comparison['actual_assignments'] != comparison['expected_assignments']
        elif fault == 'untouched_ticket':
            assert any(t['episode_index'] == 999 for t in comparison['current_assignments'])


def test_assignment_evidence_is_flushed_before_transition_assertions(tmp_path, monkeypatch):
    path = tmp_path / 'flush.jsonl'
    real_check = m.check_transition
    observed_batches = []

    def inspect_then_check(before, after, ids):
        # Read from another handle while run_lifecycle still owns an open journal.
        latest = _journal(path)[-1]
        assert latest['phase'] == 'assignment_comparison'
        assert latest['ids'] == ids
        assert latest['actual_assignments'] == latest['expected_assignments']
        observed_batches.append(ids)
        real_check(before, after, ids)

    monkeypatch.setattr(m, 'check_transition', inspect_then_check)
    m.run_lifecycle(_lifecycle_env(), 130001, path, lambda _: None)
    assert len(observed_batches) == 10
