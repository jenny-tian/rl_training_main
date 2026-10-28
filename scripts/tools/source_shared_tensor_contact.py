"""Pair-resolved GPU contact evidence for the bounded shared-bank probe.

The installed tensor API supports an explicit filter list per sensor. Wildcard
filters are not a Cartesian product. Never infer isolation from net forces alone.
"""

from collections import Counter
import re

import numpy as np

from source_shared_contact_report import ContactCapture, classify_contact, contact_runtime_passed


CONTACT_CAPACITY = 16384
SHARED_CONTACT_CAPACITIES = {2: CONTACT_CAPACITY, 8: CONTACT_CAPACITY * 4}
FORCE_ATOL = 1e-4
FORCE_RTOL = 1e-4


def contact_capacity_for(num_envs):
    """Return the frozen detailed-contact capacity for a shared-bank size."""
    if type(num_envs) is not int or num_envs not in SHARED_CONTACT_CAPACITIES:
        raise ValueError('shared contact capacity is defined only for two or eight environments')
    return SHARED_CONTACT_CAPACITIES[num_envs]


def _copy(value):
    # The native API reuses borrowed buffers; copy before making another read.
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    return np.array(value, copy=True)


def tensor_runtime_passed(state, stage_id):
    if not contact_runtime_passed(state, stage_id) or state.get('backend_unchanged') is not True:
        return False
    backend = state.get('backend_state')
    return (isinstance(backend, dict)
            and type(backend.get('suppress_readback_setting')) is bool
            and type(backend.get('readback_suppressed')) is bool
            and backend['suppress_readback_setting'] == backend['readback_suppressed']
            and isinstance(backend.get('tensor_device'), str)
            and re.fullmatch(r'cuda(?::\d+)?', backend['tensor_device']) is not None)


def _body_environment(path):
    match = re.fullmatch(r'/World/envs/env_(\d+)/Robot/.+', path)
    return int(match[1]) if match else None


def validate_topology(sensors, filters, bodies, ground, *, num_envs):
    """Require the complete body/ground Cartesian product for every environment."""
    owners = {_body_environment(path) for path in bodies}
    expected_owners = set(range(num_envs)) if type(num_envs) is int and num_envs > 0 else set()
    if (not bodies or len(set(bodies)) != len(bodies) or not ground or len(set(ground)) != len(ground)
            or any(not isinstance(path, str) or not path.startswith('/World/ground/') for path in ground)
            or None in owners or owners != expected_owners or set(bodies) & set(ground)):
        raise AssertionError('invalid contact inventory')
    expected = set(bodies) | set(ground)
    if (len(sensors) != len(bodies) or set(sensors) != set(bodies) or len(filters) != len(sensors)
            or any(len(row) != len(expected) or set(row) != expected for row in filters)):
        raise AssertionError('tensor contact paths do not cover the full Cartesian product')


class TensorContactCapture(ContactCapture):
    """Read every sensor/filter pair after every physics substep, before reset."""

    uses_contact_events = False

    def __init__(self, *, physics_view, ground_colliders, physics_dt, num_envs=None,
                 capacity=CONTACT_CAPACITY, **kwargs):
        super().__init__(decode_path=None, event_types={}, **kwargs)
        if type(capacity) is not int or capacity < 2 or not np.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError('invalid tensor contact capacity or physics timestep')
        self.physics_view, self.physics_dt = physics_view, physics_dt
        owners = {_body_environment(path) for path in self.bodies}
        inferred_envs = max((owner for owner in owners if owner is not None), default=-1) + 1
        self.num_envs = inferred_envs if num_envs is None else num_envs
        if (type(self.num_envs) is not int or self.num_envs < 1
                or owners != set(range(self.num_envs))):
            raise ValueError('contact bodies must cover every configured environment')
        self.capacity = capacity
        self.ground_colliders = sorted(ground_colliders)
        self.view = None
        self.topology = None
        self.samples = 0
        self.peak_used = 0
        self.max_pair_residual = 0.0
        self.max_net_residual = 0.0

    def runtime(self, interface, get_runtime):
        super().runtime(interface, get_runtime)
        if not tensor_runtime_passed(self.runtime_state, self.stage_id):
            raise AssertionError('GPU contact backend unknown, changed, or inconsistent')
        if self.view is None:
            bodies = sorted(self.bodies)
            targets = bodies + self.ground_colliders
            validate_topology(bodies, [targets] * len(bodies), bodies, self.ground_colliders,
                              num_envs=self.num_envs)
            self.view = self.physics_view.create_rigid_contact_view(
                bodies, filter_patterns=[list(targets) for _ in bodies], max_contact_data_count=self.capacity)
            self.topology = self._topology()
            self.record({'phase': 'tensor_contact_topology', **self.topology})
        elif self._topology() != self.topology:
            raise AssertionError('tensor contact topology changed during capture')

    def _topology(self):
        sensors = list(self.view.sensor_paths)
        filters = [list(row) for row in self.view.filter_paths]
        validate_topology(sensors, filters, list(self.bodies), self.ground_colliders,
                          num_envs=self.num_envs)
        if (self.view.sensor_count != len(sensors) or self.view.filter_count != len(filters[0])
                or self.view.max_contact_data_count != self.capacity):
            raise AssertionError('tensor contact counts/capacity do not match resolved paths')
        return {'sensor_paths': sensors, 'filter_paths': filters, 'ground_colliders': self.ground_colliders,
                'sensor_count': len(sensors), 'filter_count': len(filters[0]), 'num_envs': self.num_envs,
                'capacity': self.capacity}

    def poll(self, env, interface, get_runtime):
        step = env._sim_step_counter
        self.record({'phase': 'contact_poll_attempt', 'sim_step': step})
        self.runtime(interface, get_runtime)
        if type(step) is not int or step != self.start_step + len(self.poll_steps) + 1:
            raise AssertionError('nonconsecutive tensor contact poll step')
        if env.physics_dt != self.physics_dt:
            raise AssertionError('physics timestep changed during contact capture')
        net = _copy(self.view.get_net_contact_forces(self.physics_dt))
        matrix = _copy(self.view.get_contact_force_matrix(self.physics_dt))
        data = tuple(_copy(v) for v in self.view.get_contact_data(self.physics_dt))
        self.consume_tensor(net, matrix, data, step)
        self.poll_steps.append(step)

    def consume_tensor(self, net, matrix, data, step):
        n, f = self.topology['sensor_count'], self.topology['filter_count']
        if len(data) != 6:
            raise AssertionError('expected six tensor contact buffers')
        forces, points, normals, distances, counts, starts = data
        shapes = ((n, 3), (n, f, 3), (self.capacity, 1), (self.capacity, 3), (self.capacity, 3),
                  (self.capacity, 1), (n, f), (n, f))
        if any(v.shape != shape for v, shape in zip((net, matrix, *data), shapes)):
            raise AssertionError('invalid tensor contact buffer dimensions')
        if (counts.dtype.kind not in 'iu' or starts.dtype.kind not in 'iu'
                or not np.isfinite(net).all() or not np.isfinite(matrix).all()
                or (counts < 0).any() or (counts > self.capacity).any()):
            raise AssertionError('invalid tensor contact counts/forces')
        used_count = sum(int(v) for v in counts.flat)
        if used_count >= self.capacity:
            raise AssertionError('tensor contact buffer saturated; completeness is unknown')
        used = set()
        reconstructed = np.zeros((n, f, 3), dtype=np.float64)
        positives = Counter()
        records = []
        for i, sensor in enumerate(self.topology['sensor_paths']):
            for j, target in enumerate(self.topology['filter_paths'][i]):
                if classify_contact(sensor, target)[0] == 'cross_environment' and (
                        counts[i, j] > 0 or np.any(matrix[i, j] != 0)):
                    item = {'phase': 'cross_environment_contact', 'sim_step': step,
                            'actors': [sensor, target], 'count': int(counts[i, j]),
                            'force_N': matrix[i, j].tolist()}
                    self.cross_contacts.append(item)
                    self.record(item)
                    raise AssertionError('cross-environment tensor contact reported')
        for i, j in np.argwhere(counts > 0):
            count, start = int(counts[i, j]), int(starts[i, j])
            end = start + count
            if start < 0 or end > self.capacity:
                raise AssertionError('tensor contact range outside capacity')
            indices = set(range(start, end))
            if used & indices:
                raise AssertionError('overlapping tensor contact ranges')
            used.update(indices)
            live = [v[start:end] for v in (forces, points, normals, distances)]
            if any(not np.isfinite(v).all() for v in live):
                raise AssertionError('nonfinite live tensor contact data')
            if not np.allclose(np.linalg.norm(normals[start:end], axis=1), 1.0, atol=1e-3, rtol=0):
                raise AssertionError('invalid contact normals')
            reconstructed[i, j] = np.sum(forces[start:end] * normals[start:end], axis=0, dtype=np.float64)
            paths = [self.topology['sensor_paths'][i], self.topology['filter_paths'][i][j]]
            kind, owner = classify_contact(*paths)
            record = {'phase': 'tensor_contact', 'sim_step': step, 'actors': paths, 'kind': kind,
                      'count': count, 'forces_N': forces[start:end].tolist(), 'points_m': points[start:end].tolist(),
                      'normals': normals[start:end].tolist(), 'separations_m': distances[start:end].tolist()}
            if kind == 'ground' and np.any(np.abs(forces[start:end]) > FORCE_ATOL):
                positives[owner] += count
            records.append(record)
        # Matrix is independent of the detailed-data capacity. Net force also
        # includes non-filter objects; these comparisons reject lost evidence.
        pair_residual = float(np.max(np.abs(reconstructed - matrix)))
        net_residual = float(np.max(np.abs(matrix.sum(axis=1, dtype=np.float64) - net)))
        if not np.allclose(reconstructed, matrix, atol=FORCE_ATOL, rtol=FORCE_RTOL):
            raise AssertionError('contact points disagree with pair force matrix (possible truncation)')
        if not np.allclose(matrix.sum(axis=1, dtype=np.float64), net, atol=FORCE_ATOL, rtol=FORCE_RTOL):
            raise AssertionError('pair force matrix disagrees with net forces (possible missing filter)')
        for value in records:
            self.record(value)
        self.ground['poll'].update(positives)
        self.samples += 1
        self.peak_used = max(self.peak_used, used_count)
        self.max_pair_residual = max(self.max_pair_residual, pair_residual)
        self.max_net_residual = max(self.max_net_residual, net_residual)
        self.record({'phase': 'tensor_contact_sample', 'sim_step': step, 'used': used_count,
                     'capacity': self.capacity, 'counts': counts.tolist(), 'net_forces_N': net.tolist(),
                     'pair_forces_N': matrix.tolist(), 'pair_residual_N': pair_residual,
                     'net_residual_N': net_residual})

    def summary(self):
        result = super().summary()
        result.pop('channels')
        result.pop('callback_ground_contact_points')
        result.update(schema='shared_tensor_contact_capture_v1', evidence='gpu_pair_tensor',
                      tensor_topology=self.topology, samples=self.samples, peak_used=self.peak_used,
                      physics_dt=self.physics_dt, max_pair_residual_N=self.max_pair_residual,
                      max_net_residual_N=self.max_net_residual, force_atol_N=FORCE_ATOL, force_rtol=FORCE_RTOL)
        return result


def tensor_summary_passed(summary, ground_contacts, *, control_steps=20, decimation=4):
    if not isinstance(summary, dict) or not isinstance(ground_contacts, dict):
        return False
    topology = summary.get('tensor_topology')
    try:
        inferred = {_body_environment(path) for path in topology['sensor_paths']}
        num_envs = topology.get('num_envs', max((i for i in inferred if i is not None), default=-1) + 1)
        if topology.get('num_envs', num_envs) != num_envs:
            return False
        validate_topology(topology['sensor_paths'], topology['filter_paths'], topology['sensor_paths'],
                          topology['ground_colliders'], num_envs=num_envs)
        if (topology['sensor_count'] != len(topology['sensor_paths'])
                or topology['filter_count'] != len(topology['filter_paths'][0])):
            return False
    except (TypeError, KeyError, AssertionError):
        return False
    start, points = summary.get('start_sim_step'), summary.get('ground_contact_points')
    if (type(start) is not int or start < 0 or not isinstance(points, dict)
            or type(summary.get('peak_used')) is not int or type(topology.get('capacity')) is not int
            or not 0 < summary['peak_used'] < topology['capacity']):
        return False
    if (summary.get('force_atol_N') != FORCE_ATOL or summary.get('force_rtol') != FORCE_RTOL
            or any(type(summary.get(key)) not in (float, int) or not np.isfinite(summary[key])
                   or summary[key] < 0 for key in ('max_pair_residual_N', 'max_net_residual_N'))):
        return False
    total = control_steps * decimation
    return (summary.get('schema') == 'shared_tensor_contact_capture_v1'
            and summary.get('evidence') == 'gpu_pair_tensor'
            and tensor_runtime_passed(summary.get('runtime_state'), summary.get('stage_id'))
            and summary.get('runtime_checked') is True and summary.get('runtime_check_count') == total + 1
            and summary.get('positive_controls_validated') is True
            and summary.get('physics_dt') == 0.005
            and summary.get('decimation') == decimation and summary.get('control_steps') == control_steps
            and summary.get('poll_steps') == list(range(start + 1, start + total + 1))
            and summary.get('control_poll_counts') == [decimation] * control_steps
            and type(summary.get('samples')) is int and summary['samples'] == total
            and summary.get('errors') == [] and summary.get('cross_environment_contacts') == 0
            and all(type(points.get(str(i), points.get(i))) is int
                    and points.get(str(i), points.get(i, 0)) > 0
                    and points.get(str(i), points.get(i)) == ground_contacts.get(str(i), ground_contacts.get(i))
                    for i in range(num_envs)))
