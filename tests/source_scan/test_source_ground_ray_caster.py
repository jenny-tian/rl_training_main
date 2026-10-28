"""CPU coverage of real upstream initialization and the Source cache adapter.

Only mesh/ray setup, physics views and Warp raycast are replaced. No Kit/GPU.
"""

import ast
from collections.abc import Sequence
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = 'rl_training.source_scan.source_ground_ray_caster'


def load_adapter(monkeypatch):
    audit = json.loads((ROOT / 'docs/lite3-source-shared-bank-preparation-20261028/dependency_audit.json').read_text())
    pinned = next(r for r in audit['dependencies'] if r['path'].endswith('/multi_mesh_ray_caster.py'))
    path = Path(pinned['path'])
    if not path.is_file():
        pytest.skip('requires the audited local Isaac Lab source, without starting Kit')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned['sha256']
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'MultiMeshRayCaster')
    methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('_initialize_rays_impl', '_update_buffers_impl')]

    class Parent:
        def _initialize_rays_impl(self):
            pass
        def _initialize_warp_meshes(self):
            self.mesh_initializations += 1

    holder = {}
    namespace = dict(Parent=Parent, torch=torch, Sequence=Sequence,
                     raycast_dynamic_meshes=lambda *a, **kw: holder['module'].raycast_dynamic_meshes(*a, **kw))
    extracted = ast.ClassDef(name='Upstream', bases=[ast.Name(id='Parent', ctx=ast.Load())], keywords=[],
                             body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])), str(path), 'exec'), namespace)
    upstream = namespace['Upstream']
    for name in ('isaaclab', 'isaaclab.sensors', 'isaaclab.sensors.ray_caster', 'isaaclab.utils', 'isaaclab.utils.warp'):
        stub = ModuleType(name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)
    sys.modules['isaaclab.sensors.ray_caster'].MultiMeshRayCaster = upstream
    sys.modules['isaaclab.utils.warp'].raycast_dynamic_meshes = lambda *a, **kw: None
    spec = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / 'source/rl_training/rl_training/source_scan/source_ground_ray_caster.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    spec.loader.exec_module(module)
    holder['module'] = module
    return module, upstream


@pytest.fixture
def adapter(monkeypatch):
    return load_adapter(monkeypatch)


def make_sensor(module, rays=345):
    sensor = module.SourceGroundRayCaster()
    sensor._num_envs, sensor.num_rays, sensor.device = 2, rays, 'cpu'
    sensor.cfg = NS(update_mesh_ids=True, max_distance=1000)
    sensor._data = NS(ray_hits_w=torch.full((2, rays, 3), -5.0))
    sensor._raycast_targets_cfg = [NS(prim_expr='/World/ground', merge_prim_meshes=True, track_mesh_transforms=False)]
    sensor._num_meshes_per_env = {'/World/ground': 1}
    sensor._mesh_views = [None]
    sensor.meshes = {'/World/ground': NS(id=42)}
    sensor._mesh_ids_wp = NS(numpy=lambda: np.full((2, 1), 42))
    sensor._mesh_positions_w = torch.arange(6.).reshape(2, 1, 3)
    sensor._mesh_orientations_w = torch.zeros(2, 1, 4)
    sensor._mesh_orientations_w[..., 0] = 1
    sensor._ray_starts_w = torch.arange(2 * rays * 3, dtype=torch.float32).reshape(2, rays, 3)
    sensor._ray_directions_w = torch.zeros(2, rays, 3)
    sensor.mesh_initializations = 0
    sensor.ray_updates = []
    sensor._update_ray_infos = lambda ids: sensor.ray_updates.append(ids)
    sensor._initialize_warp_meshes()
    sensor._initialize_rays_impl()  # Unmodified method from the installed dependency.
    sensor._data.ray_mesh_ids[:] = 7
    return sensor


def raycast_stub(sensor, ids, rank=2):
    expected_starts = sensor._ray_starts_w[ids].clone()
    expected_ids = torch.zeros(expected_starts.shape[:2], dtype=torch.int16)
    expected_ids[..., 0] = -1  # A miss must not be converted into a ground hit.

    def cast(starts, directions, **kwargs):
        assert torch.equal(starts, expected_starts)
        assert torch.equal(directions, sensor._ray_directions_w[ids])
        assert torch.equal(kwargs['mesh_positions_w'], sensor._mesh_positions_w[ids])
        assert torch.equal(kwargs['mesh_orientations_w'], sensor._mesh_orientations_w[ids])
        assert kwargs['mesh_ids_wp'] is sensor._mesh_ids_wp
        assert kwargs['return_mesh_id'] is True
        return starts + 1, None, None, None, expected_ids if rank == 2 else expected_ids.unsqueeze(-1)
    return cast, expected_starts + 1, expected_ids.unsqueeze(-1)


@pytest.mark.parametrize('rays', [345, 9])
@pytest.mark.parametrize('ids', [[0, 1], [1]])
def test_installed_update_reproduces_the_original_failure(adapter, monkeypatch, rays, ids):
    module, upstream = adapter
    sensor = make_sensor(module, rays)
    cast, _, _ = raycast_stub(sensor, ids)
    monkeypatch.setattr(module, 'raycast_dynamic_meshes', cast)
    with pytest.raises(RuntimeError, match='shape mismatch'):
        upstream._update_buffers_impl(sensor, ids)


@pytest.mark.parametrize('rays', [345, 9])
@pytest.mark.parametrize('ids', [[0, 1], [1], [1, 0]])
@pytest.mark.parametrize('rank', [2, 3])
def test_adapter_preserves_contract_values_and_untouched_envs(adapter, monkeypatch, rays, ids, rank):
    module, _ = adapter
    sensor = make_sensor(module, rays)
    cast, expected_hits, expected_ids = raycast_stub(sensor, ids, rank)
    monkeypatch.setattr(module, 'raycast_dynamic_meshes', cast)
    hits_cache, id_cache = sensor._data.ray_hits_w, sensor._data.ray_mesh_ids
    before_hits, before_ids = hits_cache.clone(), id_cache.clone()
    sensor._update_buffers_impl(torch.tensor(ids))
    assert sensor._data.ray_mesh_ids is id_cache and sensor._data.ray_hits_w is hits_cache
    assert id_cache.shape == (2, rays, 1)
    assert torch.equal(id_cache[ids], expected_ids)
    assert torch.equal(hits_cache[ids], expected_hits)
    others = list(set(range(2)) - set(ids))
    assert torch.equal(id_cache[others], before_ids[others])
    assert torch.equal(hits_cache[others], before_hits[others])
    assert len(sensor.ray_updates) == 1


def test_repeated_partial_updates_keep_the_same_public_cache(adapter, monkeypatch):
    module, _ = adapter
    sensor = make_sensor(module, 9)
    original = sensor._data.ray_mesh_ids
    for ids in ([1], [0], [1, 0]):
        cast, _, expected = raycast_stub(sensor, ids)
        monkeypatch.setattr(module, 'raycast_dynamic_meshes', cast)
        sensor._update_buffers_impl(ids)
        assert sensor._data.ray_mesh_ids is original
        assert torch.equal(original[ids], expected)


@pytest.mark.parametrize('fault', ['ids_none', 'ids_float', 'ids_wrong_envs', 'ids_wrong_rays', 'ids_extra_axis',
                                  'hits_wrong_rays', 'cache_rank'])
def test_bad_shapes_fail_before_either_cache_changes(adapter, monkeypatch, fault):
    module, _ = adapter
    sensor = make_sensor(module, 9)
    hits, ids = torch.ones(1, 9, 3), torch.zeros(1, 9, dtype=torch.int16)
    if fault == 'ids_none': ids = None
    elif fault == 'ids_float': ids = ids.float()
    elif fault == 'ids_wrong_envs': ids = torch.zeros(2, 9, dtype=torch.int16)
    elif fault == 'ids_wrong_rays': ids = ids[:, :-1]
    elif fault == 'ids_extra_axis': ids = ids.unsqueeze(-1).expand(1, 9, 2)
    elif fault == 'hits_wrong_rays': hits = hits[:, :-1]
    else: sensor._data.ray_mesh_ids = sensor._data.ray_mesh_ids[..., 0]
    before = (sensor._data.ray_hits_w.clone(), sensor._data.ray_mesh_ids.clone())
    monkeypatch.setattr(module, 'raycast_dynamic_meshes', lambda *a, **kw: (hits, None, None, None, ids))
    with pytest.raises(RuntimeError): sensor._update_buffers_impl([1])
    assert torch.equal(sensor._data.ray_hits_w, before[0])
    assert torch.equal(sensor._data.ray_mesh_ids, before[1])


@pytest.mark.parametrize('fault', ['wrong_target', 'dynamic', 'not_merged', 'disabled_ids',
                                  'extra_target', 'different_env_mesh', 'mesh_view'])
def test_adapter_rejects_targets_outside_the_static_shared_ground_contract(adapter, fault):
    module, _ = adapter
    sensor = make_sensor(module, 9)
    target = sensor._raycast_targets_cfg[0]
    if fault == 'wrong_target': target.prim_expr = '/World/envs/env_.*/Robot'
    elif fault == 'dynamic': target.track_mesh_transforms = True
    elif fault == 'not_merged': target.merge_prim_meshes = False
    elif fault == 'disabled_ids': sensor.cfg.update_mesh_ids = False
    elif fault == 'extra_target': sensor._raycast_targets_cfg.append(copy.copy(target))
    elif fault == 'different_env_mesh': sensor._mesh_ids_wp.numpy = lambda: np.array([[42], [99]])
    else: sensor._mesh_views = [object()]
    with pytest.raises((RuntimeError, ValueError)):
        sensor._initialize_warp_meshes()
