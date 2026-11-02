"""Eight-environment diagnostic contracts, math and reversible instance hooks.

This module imports Torch but never Isaac/Kit. Simulation is started only by the
separate command. Expected task endings are recorded, not a skill admission gate.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from .shared_terrain_bank import SharedBankTerrainAllocator
from .tile_bounds import source_tile_bounds_terminated

SCHEMA = 'lite3_source_eight_env_motion_v1'
NUM_ENVS = 8
CONTROL_STEPS = 96
DECIMATION = 4
PHYSICS_DT = .005
TIMEOUT_PROBES = {24: [0], 48: [1, 4, 7], 72: [0, 2, 5, 6]}
EVIDENCE_BYTE_LIMIT = 1073741824
EXPECTED_TERMS = {'source_training_safety', 'source_training_completion', 'source_training_timeout',
                  'source_terrain_tile_bounds', 'illegal_contact', 'bad_orientation_2'}


class EventJournal:
    """Batch contact records until the current substep completes, then fsync.

    On any exception the context manager still flushes already encoded records.
    A partial substep never qualifies as complete evidence.
    """
    def __init__(self, path, *, byte_limit=EVIDENCE_BYTE_LIMIT):
        self.path = Path(path)
        self.stream = self.path.open('xb')
        self.pending = []
        self.written = 0
        self.byte_limit = byte_limit

    def record(self, value):
        encoded = (json.dumps(value, allow_nan=False) + '\n').encode()
        if self.written + sum(map(len, self.pending)) + len(encoded) > self.byte_limit:
            raise RuntimeError('diagnostic evidence byte ceiling reached')
        self.pending.append(encoded)

    def flush(self):
        payload = b''.join(self.pending)
        self.stream.write(payload)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.written += len(payload)
        self.pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.flush()
        finally:
            self.stream.close()


def tensor_record(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.is_complex() or not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError('nonfinite or unsupported diagnostic tensor')
    return {'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'values': tensor.detach().cpu().tolist()}


def fault_tensor_record(tensor):
    """Keep NaN/Inf explicit in failure evidence without nonstandard JSON numbers."""
    if tensor.is_complex():
        return {'dtype': str(tensor.dtype), 'unsupported_complex': True}
    def encode(value):
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return 'NaN' if math.isnan(value) else ('+Inf' if value > 0 else '-Inf')
        return value
    return {'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
            'nonfinite_encoding': 'explicit_strings', 'values': encode(tensor.detach().cpu().tolist())}


def restore_tensor(record):
    name = record['dtype'].removeprefix('torch.')
    if name not in ('float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'bool'):
        raise ValueError('unsupported evidence dtype')
    value = torch.tensor(record['values'], dtype=getattr(torch, name))
    if list(value.shape) != record['shape'] or not bool(torch.isfinite(value).all()):
        raise ValueError('malformed diagnostic tensor')
    return value


def check_scan(positions, hits, hit_ids, mesh_ids, ground_id, origins, *, age, max_age, base=False):
    """Check ground-only ownership and each ray's assigned tile (no Kit imports)."""
    n = positions.shape[0]
    if (positions.shape != (n, 3) or origins.shape != (n, 3) or hits.ndim != 3
            or hits.shape[0] != n or hits.shape[1] < 1 or hits.shape[2] != 3
            or hit_ids.shape != (*hits.shape[:2], 1) or hit_ids.dtype != torch.int16
            or mesh_ids.shape != (n, 1) or age.shape != (n,)):
        raise ValueError('scanner evidence shape/dtype mismatch')
    for tensor in (positions, origins, hits, age):
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError('nonfinite scanner evidence')
    if not bool((hit_ids == 0).all()) or not bool((mesh_ids == ground_id).all()):
        raise RuntimeError('scanner target is not the single shared ground')
    if bool(((age < -1e-6) | (age > max_age + 1e-6)).any()):
        raise RuntimeError('scanner cache is stale')
    local = hits[..., :2] - origins[:, None, :2]
    if bool(((local[..., 0] < -1.5 - 1e-5) | (local[..., 0] > 10.5 + 1e-5)
             | (local[..., 1].abs() > 2.0 + 1e-5)).any()):
        raise RuntimeError('ray hit belongs outside its assigned terrain tile')
    if base and bool((torch.linalg.vector_norm(hits[..., :2]-positions[:, None, :2], dim=-1) > .08).any()):
        raise RuntimeError('base scan does not track the current root')


def motion_sample(positions, velocities, quaternions, origins_grid, levels, columns, *, previous=None):
    n, bodies, _ = positions.shape
    if n != 8 or bodies != 17 or velocities.shape != positions.shape or quaternions.shape != (8, 17, 4):
        raise ValueError('motion evidence requires all 17 bodies of eight environments')
    for value in (positions, velocities, quaternions):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('nonfinite body kinematics')
    if not torch.allclose(quaternions.norm(dim=-1), torch.ones((8, 17), device=positions.device), atol=1e-4, rtol=0):
        raise ValueError('invalid body quaternion')
    diagnostics = {}
    source_tile_bounds_terminated(positions, velocities, origins_grid, levels, columns,
        size=(12., 4.), flat_length_m=3., collision_radius_m=.27, numerical_margin_m=.001,
        control_period_s=.02, diagnostics=diagnostics)
    if not bool(diagnostics['valid_indices'].all()) or bool(diagnostics['invalid_state'].any()):
        raise RuntimeError('invalid moving tile assignment/state')
    if bool(diagnostics['outside_now'].any()):
        raise RuntimeError('actual collision envelope left its assigned tile')
    displacement = torch.zeros_like(positions) if previous is None else positions-previous[0]
    acceleration = torch.zeros_like(velocities) if previous is None else (velocities-previous[1])/PHYSICS_DT
    return {'body_positions_w': tensor_record(positions), 'body_link_velocities_w': tensor_record(velocities),
            'body_link_quaternions_wxyz': tensor_record(quaternions), 'displacements_m': tensor_record(displacement),
            'finite_difference_acceleration_mps2': tensor_record(acceleration),
            'terrain_levels': levels.cpu().tolist(), 'terrain_columns': columns.cpu().tolist(),
            'safe_lower_xy_w': diagnostics['safe_lower_xy_w'].cpu().tolist(),
            'safe_upper_xy_w': diagnostics['safe_upper_xy_w'].cpu().tolist(),
            'outside_prediction': diagnostics['outside_prediction'].cpu().tolist(),
            'reference_point': 'link_origin', 'prediction_is_integrity_stop': False}


def check_reset_boundary(before, after, ids):
    if before.keys() != after.keys():
        raise RuntimeError('reset snapshot inventory changed')
    other = torch.ones(NUM_ENVS, dtype=torch.bool, device=next(iter(before.values())).device)
    other[ids] = False
    for name, value in before.items():
        if value.shape[0] != NUM_ENVS or not torch.equal(value[other], after[name][other]):
            raise RuntimeError('reset changed unaffected environment buffer: ' + name)
        if name.startswith(('filter/', 'lifecycle/')) and bool(after[name][ids].any()):
            raise RuntimeError('reset did not clear selected memory: ' + name)


class MovingObserver:
    """Collect after scene.update; compare reset boundaries without physics integration."""
    def __init__(self, env, runtime, journal, snapshot, boundary_checker):
        if env.num_envs != 8 or env.cfg.decimation != 4 or env.physics_dt != PHYSICS_DT:
            raise ValueError('moving probe requires eight environments at 200Hz/50Hz')
        self.env, self.runtime, self.journal, self.snapshot = env, runtime, journal, snapshot
        self.boundary_checker = boundary_checker
        self.oracle = SharedBankTerrainAllocator(8, SOURCE_RESET_BLOCK_COUNTS, 131001, bank_rows=1)
        initial = self.oracle.reset(list(range(8)))
        if initial != env._source_parallel_terrain_allocator.assignments:
            raise RuntimeError('initial training sampler assignments differ from oracle')
        self.initial_assignments = [asdict(item) for item in initial]
        self.start = env._sim_step_counter
        data = env.scene['robot'].data
        self.previous = (data.body_link_pos_w.clone(), data.body_link_lin_vel_w.clone())
        self.pending_recovery = set()
        self.samples, self.resets = 0, 0

    def scans(self, names):
        env = self.env
        result = {}
        positions = env.scene['robot'].data.root_link_pos_w
        origins = env.scene.terrain.env_origins
        for name in names:
            sensor = env.scene.sensors[name]
            data = sensor.data
            mesh_ids = torch.as_tensor(sensor._mesh_ids_wp.numpy(), dtype=torch.int64, device=positions.device)
            age = sensor._timestamp - sensor._timestamp_last_update
            check_scan(positions, data.ray_hits_w, data.ray_mesh_ids, mesh_ids,
                       sensor.meshes['/World/ground'].id, origins, age=age,
                       max_age=1e-6 if name == 'height_scanner_base' else .02,
                       base=name == 'height_scanner_base')
            result[name] = {'hits': tensor_record(data.ray_hits_w), 'hit_ids': tensor_record(data.ray_mesh_ids),
                            'mesh_ids': tensor_record(mesh_ids), 'ground_id': int(sensor.meshes['/World/ground'].id),
                            'age_s': tensor_record(age)}
        return result

    def after_scene_update(self):
        env = self.env
        if env._sim_step_counter != self.start + self.samples + 1:
            raise RuntimeError('motion sample is not consecutive')
        data, terrain = env.scene['robot'].data, env.scene.terrain
        try:
            sample = motion_sample(data.body_link_pos_w, data.body_link_lin_vel_w, data.body_link_quat_w,
                                   terrain.terrain_origins, terrain.terrain_levels, terrain.terrain_types,
                                   previous=self.previous)
            sample.update(phase='motion_substep', sim_step=env._sim_step_counter,
                          recovery_ids=sorted(self.pending_recovery),
                          root_positions_w=tensor_record(data.root_link_pos_w),
                          origins_w=tensor_record(terrain.env_origins), scanners=self.scans(('height_scanner_base',)))
        except Exception as error:
            self.journal.record({'phase': 'motion_error', 'sim_step': env._sim_step_counter,
                'error': repr(error), 'body_positions_w': fault_tensor_record(data.body_link_pos_w),
                'body_link_velocities_w': fault_tensor_record(data.body_link_lin_vel_w),
                'body_link_quaternions_wxyz': fault_tensor_record(data.body_link_quat_w)})
            self.journal.flush()
            raise
        self.journal.record(sample)
        self.journal.flush()
        self.previous = (data.body_link_pos_w.clone(), data.body_link_lin_vel_w.clone())
        self.pending_recovery.clear()
        self.samples += 1

    def memory_snapshot(self):
        result = self.snapshot(self.env)
        action = self.env.action_manager.action
        result.update({'filter/'+name: value for name, value in self.runtime.safety.temporal_snapshot(action).items()})
        result.update({'lifecycle/'+name: getattr(self.runtime.lifecycle, name).clone()
                       for name in ('hold', 'contact_steps', 'completed', 'unsafe', 'timeout')})
        return result

    def reset(self, original, env_ids):
        ids = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        # Refresh lazy sensors before the comparison. The callback's subsequent
        # terminal-observation read must not look like cross-row reset mutation.
        self.env.observation_manager.compute(update_history=False)
        self.scans(('height_scanner', 'height_scanner_base'))
        before = self.memory_snapshot()
        prior = self.env._source_parallel_terrain_allocator.assignments
        original(env_ids)
        after = self.memory_snapshot()
        current = self.env._source_parallel_terrain_allocator.assignments
        data = self.env.scene['robot'].data
        terrain = self.env.scene.terrain
        self.resets += 1
        self.journal.record({'phase': 'reset_boundary', 'batch': self.resets,
            'sim_step': self.env._sim_step_counter, 'ids': sorted(ids.cpu().tolist()),
            'before': {key: tensor_record(value) for key, value in before.items()},
            'after': {key: tensor_record(value) for key, value in after.items()},
            'root_link_positions_after': tensor_record(data.root_link_pos_w),
            'default_root_positions': tensor_record(data.default_root_state[:, :3]),
            'assignments_before': [asdict(item) for item in prior],
            'assignments_after': [asdict(item) for item in current]})
        self.journal.flush()
        check_reset_boundary(before, after, ids)
        self.boundary_checker(before, after, ids.cpu().tolist())
        expected = self.oracle.reset(ids.cpu().tolist())
        if tuple(current[i] for i in sorted(ids.cpu().tolist())) != expected:
            raise RuntimeError('reset tickets differ from independent sampler replay')
        others = set(range(8))-set(ids.cpu().tolist())
        if any(prior[i] != current[i] for i in others):
            raise RuntimeError('reset altered an unaffected ticket')
        expected_origins = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
        if not torch.equal(terrain.env_origins[ids], expected_origins) or not torch.allclose(
                data.root_link_pos_w[ids], data.default_root_state[ids, :3]+expected_origins, atol=1e-5, rtol=0):
            raise RuntimeError('reset selected an incorrect spawn origin or root pose')
        selected = sorted(ids.cpu().tolist())
        self.previous[0][ids] = data.body_link_pos_w[ids]
        self.previous[1][ids] = data.body_link_lin_vel_w[ids]
        self.pending_recovery.update(selected)
        self.journal.record({'phase': 'reset_checked', 'batch': self.resets,
            'sim_step': self.env._sim_step_counter, 'ids': selected,
            'rebase_positions_w': tensor_record(self.previous[0]),
            'rebase_velocities_w': tensor_record(self.previous[1])})
        self.journal.flush()

    @contextmanager
    def installed(self):
        scene = self.env.scene
        originals = [(scene, 'update'), (self.env, '_reset_idx')]
        saved = [(owner, name, name in vars(owner), vars(owner).get(name), getattr(owner, name))
                 for owner, name in originals]
        scene_update, reset_idx = saved[0][-1], saved[1][-1]
        def updated(dt):
            result = scene_update(dt)
            if dt == PHYSICS_DT:
                self.after_scene_update()
            return result
        scene.update = updated
        self.env._reset_idx = lambda ids: self.reset(reset_idx, ids)
        try:
            yield self
        finally:
            for owner, name, present, value, _ in saved:
                if present:
                    setattr(owner, name, value)
                else:
                    delattr(owner, name)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
