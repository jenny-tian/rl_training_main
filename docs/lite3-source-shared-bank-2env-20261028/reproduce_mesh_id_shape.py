"""CPU reproduction of the installed ray-mesh-ID shape error; no Kit/physics.

Executes the unchanged initializer/update method bodies from the installed
dependency, replacing the parent ray initializer and Warp raycast with stubs.
This is diagnostic evidence only and makes no repair to project/dependency code.
"""
import ast
from collections.abc import Sequence
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    dependency = Path('/home/tjx/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster.py')
    source = dependency.read_text()
    tree = ast.parse(source)
    original_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiMeshRayCaster')
    methods = [copy.deepcopy(n) for n in original_class.body if isinstance(n, ast.FunctionDef)
               and n.name in ('_initialize_rays_impl', '_update_buffers_impl')]
    assert len(methods) == 2
    extracted = ast.ClassDef(name='ExtractedRayCaster', bases=[ast.Name(id='Parent', ctx=ast.Load())],
                             keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))

    class Parent:
        def _initialize_rays_impl(self):
            pass

    results = []
    for rays in (345, 9):
        for ids in ([0, 1], [1]):
            # The failing run established that the real raycast returns rank-2
            # IDs. Preserve that shape while replacing actual ray/mesh work.
            returned_ids = torch.zeros((len(ids), rays), dtype=torch.int16)

            def raycast(*_args, **_kwargs):
                return torch.zeros(len(ids), rays, 3), None, None, None, returned_ids

            namespace = {'torch': torch, 'Sequence': Sequence, 'Parent': Parent,
                         'raycast_dynamic_meshes': raycast}
            exec(compile(module, str(dependency), 'exec'), namespace)
            sensor = namespace['ExtractedRayCaster']()
            sensor._num_envs, sensor.num_rays, sensor.device = 2, rays, 'cpu'
            sensor.cfg = SimpleNamespace(update_mesh_ids=True, max_distance=1000)
            sensor._data = SimpleNamespace(ray_hits_w=torch.zeros(2, rays, 3))
            sensor._mesh_views = [None]
            sensor._raycast_targets_cfg = [SimpleNamespace(track_mesh_transforms=False, prim_expr='/World/ground')]
            sensor._num_meshes_per_env = {'/World/ground': 1}
            sensor._ray_starts_w = torch.zeros(2, rays, 3)
            sensor._ray_directions_w = torch.zeros(2, rays, 3)
            sensor._mesh_positions_w = torch.zeros(2, 1, 3)
            sensor._mesh_orientations_w = torch.zeros(2, 1, 4)
            sensor._mesh_ids_wp = None
            sensor._update_ray_infos = lambda _: None
            sensor._initialize_rays_impl()
            selected = torch.tensor(ids, dtype=torch.long)
            record = {'num_rays': rays, 'env_ids': ids,
                      'allocated_shape': list(sensor._data.ray_mesh_ids.shape),
                      'selected_shape': list(sensor._data.ray_mesh_ids[selected].shape),
                      'returned_ids_shape': list(returned_ids.shape)}
            try:
                sensor._update_buffers_impl(selected)
            except RuntimeError as error:
                if 'shape mismatch' not in str(error):
                    raise
                record.update(reproduced=True, error=str(error))
            else:
                raise AssertionError('expected the installed dependency to reproduce its shape mismatch')
            results.append(record)

    report = {'status': 'installed_mesh_id_shape_error_reproduced',
              'simulation_started': False, 'dependency_path': str(dependency),
              'dependency_sha256': hashlib.sha256(dependency.read_bytes()).hexdigest(),
              'source_method_lines': {n.name: n.lineno for n in methods},
              'stubbed_components': ['parent ray initialization', 'raycast_dynamic_meshes'],
              'cases': results, 'repair_applied': False}
    output = Path(__file__).with_name('offline_diagnosis.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
