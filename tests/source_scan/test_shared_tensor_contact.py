"""Pinned tensor frontend + CPU backend fault injection; no physics simulation."""

import copy
from types import SimpleNamespace as NS, MethodType

import numpy as np
import pytest
import torch

from test_shared_contact_report import BODIES, GROUND, pinned_method, stepping_env


@pytest.fixture
def tensor_module(monkeypatch):
    from test_shared_contact_report import ROOT
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/tools'))
    import source_shared_tensor_contact
    return source_shared_tensor_contact


def setup_capture(module, *, fault=None, reorder=False, num_envs=2):
    """Use the installed buffer allocation/getters, with explicitly synthetic data."""
    env, interface, events = stepping_env()
    bodies = [f'/World/envs/env_{i}/Robot/FOOT' for i in range(num_envs)]
    interface.get_contact_report = lambda: pytest.fail('CPU event API must not be used')
    interface.subscribe_contact_report_events = lambda _: pytest.fail('CPU event subscription must not be used')

    class Frontend:
        def create_tensor(self, shape, dtype):
            # Descriptors point directly to numpy storage in this CPU test only.
            value = np.full(shape, np.nan if dtype == np.float32 else 0, dtype=dtype)
            return value, value

    class View:
        pass
    for name in ('sensor_count', 'filter_count', 'max_contact_data_count', 'sensor_paths', 'filter_paths'):
        setattr(View, name, property(pinned_method('/tensors/impl/api.py', 'RigidContactView', name)))
    for name in ('__init__', 'get_net_contact_forces', 'get_contact_force_matrix', 'get_contact_data'):
        setattr(View, name, pinned_method('/tensors/impl/api.py', 'RigidContactView', name,
                                         {'float32': np.float32, 'uint32': np.uint32}))

    class Backend:
        def create_rigid_contact_view(self, patterns, filters, capacity):
            assert patterns == sorted(bodies)
            assert filters == [sorted(bodies) + [GROUND]] * num_envs
            assert capacity == 16
            self.sensor_paths = list(reversed(patterns)) if reorder else patterns
            self.filter_paths = [list(reversed(row)) if reorder else row for row in filters]
            self.sensor_count, self.filter_count, self.max_contact_data_count = num_envs, num_envs + 1, capacity
            if fault == 'missing_filter': self.filter_paths[0][-1] = BODIES[0]
            return self

        def payload(self):
            matrix = np.zeros((num_envs, num_envs + 1, 3), np.float32)
            counts = np.zeros((num_envs, num_envs + 1), np.uint32)
            starts = np.zeros((num_envs, num_envs + 1), np.uint32)
            forces = np.full((16, 1), np.nan, np.float32)
            points = np.full((16, 3), np.nan, np.float32)
            normals = np.full((16, 3), np.nan, np.float32)
            distances = np.full((16, 1), np.nan, np.float32)
            for i, body in enumerate(self.sensor_paths):
                j = self.filter_paths[i].index(GROUND)
                counts[i, j], starts[i, j] = 1, i
                forces[i], points[i], normals[i], distances[i] = 60, [0, 0, 0], [0, 0, 1], -.001
                matrix[i, j, 2] = 60
            net = matrix.sum(axis=1)
            ground_index = self.filter_paths[0].index(GROUND)
            if fault == 'empty': counts[:] = 0; matrix[:] = 0; net[:] = 0
            if fault == 'net_only': counts[:] = 0; matrix[:] = 0
            if fault in ('cross', 'cross_cancelled', 'cross_tiny', 'cross_high'):
                target = self.sensor_paths[-1] if fault == 'cross_high' else self.sensor_paths[1]
                j = self.filter_paths[0].index(target)
                matrix[0, j, 2] = 1e-8 if fault == 'cross_tiny' else 1
                if fault == 'cross_cancelled': matrix[0, j] = 0; counts[0, j] = 2
            if fault == 'saturated': counts[0, ground_index] = 15
            if fault == 'overflow': counts[0, ground_index] = 17
            if fault == 'range': starts[0, ground_index] = 16
            if fault == 'overlap': starts[:] = 0
            if fault == 'truncated': counts[0, ground_index] = 0
            if fault == 'pair_mismatch': matrix[0, ground_index, 2] = 61
            if fault == 'net_mismatch': net[0, 2] += 1
            if fault == 'nan': forces[0] = np.nan
            if fault == 'normal': normals[0] = 0
            return net, matrix, (forces, points, normals, distances, counts, starts)

        def get_net_contact_forces(self, out, dt):
            assert dt == .005 and interface.ready
            events.append('tensor_net')
            out[:] = self.payload()[0]
            return fault != 'backend_failure'

        def get_contact_force_matrix(self, out, dt):
            events.append('tensor_matrix')
            out[:] = self.payload()[1]
            return True

        def get_contact_data(self, force, point, normal, distance, counts, starts, dt):
            events.append('tensor_data')
            for dest, source in zip((force, point, normal, distance, counts, starts), self.payload()[2]):
                dest[:] = source
            return True

    backend = Backend()
    physics = env.sim.physics_sim_view
    physics._frontend, physics._backend = Frontend(), backend
    backend.is_valid = True
    physics.create_rigid_contact_view = MethodType(pinned_method(
        '/tensors/impl/api.py', 'SimulationView', 'create_rigid_contact_view', {'RigidContactView': View}), physics)
    state = dict(suppress_readback_setting=True, readback_suppressed=True, tensor_device='cuda:0')
    if fault == 'readback_mismatch': state['suppress_readback_setting'] = False
    from source_shared_contact_report import ContactRuntime
    runtime = ContactRuntime(env, stage_id_for=lambda stage: stage.stage_id,
                             usd_context_stage_id=lambda: env._usd_context_stage_id, get_disabled=lambda: False,
                             get_backend_state=lambda: state)
    journal = []
    capture = module.TensorContactCapture(physics_view=physics, ground_colliders=[GROUND], physics_dt=.005,
        num_envs=num_envs, capacity=16, stage_id=73, reporting_bodies=bodies, record=journal.append, start_step=12,
        decimation=4, control_steps=20)
    return env, interface, events, runtime, capture, journal, state


def complete(module, **kwargs):
    from source_shared_contact_report import capture_substeps
    env, interface, events, runtime, capture, journal, state = setup_capture(module, **kwargs)
    with capture_substeps(env, interface, capture, runtime):
        for i in range(20):
            env.step(torch.zeros(2, 12))
            capture.check_control(env, i + 1)
    return capture, journal, events


@pytest.mark.parametrize(('num_envs', 'expected'), [(2, 16384), (8, 65536)])
def test_shared_contact_capacity_is_frozen_per_supported_environment_count(tensor_module, num_envs, expected):
    assert tensor_module.contact_capacity_for(num_envs) == expected
    with pytest.raises(ValueError):
        tensor_module.contact_capacity_for(4)


@pytest.mark.parametrize('reorder', [False, True])
def test_pinned_gpu_tensor_contract_captures_all_substeps_without_cpu_reports(tensor_module, reorder):
    capture, journal, events = complete(tensor_module, reorder=reorder)
    assert capture.ground['poll'] == {i: 80 for i in range(2)}
    assert tensor_module.tensor_summary_passed(capture.summary(), {i: 80 for i in range(2)})
    assert events == ['action', 'write', 'simulate', 'fetch', 'tensor_net', 'tensor_matrix', 'tensor_data', 'scene_update'] * 80
    assert len([v for v in journal if v['phase'] == 'tensor_contact_sample']) == 80
    # Live entries copied, unused NaN capacity ignored, later native writes cannot alter the journal.
    assert journal[-1]['samples'] == 80 and capture.peak_used == 2
    assert all(v['forces_N'] == [[60.0]] for v in journal if v['phase'] == 'tensor_contact')


@pytest.mark.parametrize('fault,steps', [
    ('missing_filter', 0), ('readback_mismatch', 0), ('empty', 80),
    *[(f, 1) for f in ('net_only', 'cross', 'cross_cancelled', 'cross_tiny', 'cross_high', 'saturated', 'overflow', 'range',
                       'overlap', 'truncated', 'pair_mismatch', 'net_mismatch', 'nan', 'normal', 'backend_failure')],
])
def test_invalid_or_missing_gpu_evidence_stops_and_restores_hook(tensor_module, fault, steps):
    from source_shared_contact_report import capture_substeps
    env, interface, events, runtime, capture, journal, state = setup_capture(tensor_module, fault=fault)
    with pytest.raises(Exception):
        with capture_substeps(env, interface, capture, runtime):
            for i in range(20):
                env.step(torch.zeros(2, 12))
                capture.check_control(env, i + 1)
    assert interface.simulations == steps
    assert 'record_post_physics_decimation_step' not in vars(env.recorder_manager)
    assert journal[-1]['errors'] and not tensor_module.tensor_summary_passed(capture.summary(), dict(capture.ground['poll']))


@pytest.mark.parametrize('change', ['backend', 'view', 'topology', 'timestep'])
def test_mid_capture_changes_fail_before_next_substep(tensor_module, change):
    from source_shared_contact_report import capture_substeps
    env, interface, events, runtime, capture, journal, state = setup_capture(tensor_module)
    with pytest.raises(AssertionError):
        with capture_substeps(env, interface, capture, runtime):
            env.step(torch.zeros(2, 12))
            capture.check_control(env, 1)
            if change == 'backend': state['tensor_device'] = 'cuda:1'
            elif change == 'view': env._view_manager._physics_sim_view = NS(is_valid=True)
            elif change == 'topology': capture.view._backend.sensor_paths.reverse()
            else: env.physics_dt = .01
            env.step(torch.zeros(2, 12))
    assert interface.simulations == 5



def test_eight_environment_contact_topology_checks_all_actor_pairs(tensor_module):
    capture, journal, events = complete(tensor_module, num_envs=8)
    assert len(capture.topology['sensor_paths']) == 8
    assert capture.topology['num_envs'] == 8
    assert all(len(row) == 9 and len(set(row)) == 9 for row in capture.topology['filter_paths'])
    assert capture.ground['poll'] == {i: 80 for i in range(8)}
    assert tensor_module.tensor_summary_passed(capture.summary(), {i: 80 for i in range(8)})
    assert capture.samples == 80 and capture.cross_contacts == []


def test_eight_environment_cross_contact_with_highest_id_fails_before_next_step(tensor_module):
    from source_shared_contact_report import capture_substeps
    env, interface, events, runtime, capture, journal, state = setup_capture(
        tensor_module, fault='cross_high', num_envs=8)
    with pytest.raises(AssertionError, match='cross-environment'):
        with capture_substeps(env, interface, capture, runtime):
            env.step(torch.zeros(2, 12))
            capture.check_control(env, 1)
    assert interface.simulations == 1
    assert capture.cross_contacts[0]['actors'] == [
        '/World/envs/env_0/Robot/FOOT', '/World/envs/env_7/Robot/FOOT']


def test_contact_topology_rejects_missing_environment_and_pair(tensor_module):
    bodies = [f'/World/envs/env_{i}/Robot/FOOT' for i in range(8)]
    with pytest.raises(AssertionError):
        tensor_module.validate_topology(bodies[:-1], [bodies + [GROUND]] * 7,
                                        bodies[:-1], [GROUND], num_envs=8)
    filters = [bodies + [GROUND] for _ in bodies]
    filters[3] = filters[3][1:]
    with pytest.raises(AssertionError):
        tensor_module.validate_topology(bodies, filters, bodies, [GROUND], num_envs=8)

def test_serialized_summary_rejects_missing_or_old_evidence(tensor_module):
    import json
    capture, _, _ = complete(tensor_module)
    summary = json.loads(json.dumps(capture.summary()))
    assert tensor_module.tensor_summary_passed(summary, {'0': 80, '1': 80})
    required = ('schema', 'evidence', 'tensor_topology', 'runtime_state', 'samples', 'peak_used', 'physics_dt',
                'max_pair_residual_N', 'max_net_residual_N', 'force_atol_N', 'force_rtol',
                'poll_steps', 'ground_contact_points', 'positive_controls_validated')
    for key in required:
        bad = copy.deepcopy(summary)
        del bad[key]
        assert not tensor_module.tensor_summary_passed(bad, {'0': 80, '1': 80}), key
    for key, value in [('schema', 'shared_contact_capture_v2'), ('samples', 79), ('peak_used', 16),
                       ('ground_contact_points', {'0': 80}), ('max_pair_residual_N', float('nan'))]:
        assert not tensor_module.tensor_summary_passed(dict(summary, **{key: value}), {'0': 80, '1': 80})
