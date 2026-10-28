"""CPU contract/fault tests. Fake USD and managers provide no physics evidence."""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from rl_training.source_scan.parallel_terrain_allocation import (
    SOURCE_RESET_BLOCK_COUNTS as COUNTS, ReservedRowTerrainAllocator,
    assign_source_terrain_for_parallel_reset as assign,
    initialize_source_environment_origins,
    set_uniform_subterrain_proportions,
)
from rl_training.source_scan.shared_terrain_bank import (
    SharedBankTerrainAllocator, audit_shared_collision_groups, validate_shared_scene_cfg,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/tools'))
    import source_shared_terrain_diagnostic
    return source_shared_terrain_diagnostic


class Prim:
    def __init__(self, path, *, attributes=None, relations=None, kind='PhysicsCollisionGroup'):
        self.path, self.attributes, self.relations = path, attributes or {}, relations or {}
        self.kind = kind
        self.children = []

    def GetPath(self):
        return self.path

    def GetTypeName(self):
        return self.kind

    def GetAttribute(self, name):
        return NS(Get=lambda: self.attributes.get(name))

    def GetRelationship(self, name):
        return NS(GetTargets=lambda: self.relations.get(name, []))

    def GetChildren(self):
        return self.children


def shared_scene(num_envs=2, rows=1):
    generator = NS(num_rows=rows, num_cols=15, sub_terrains={name: NS(proportion=1/15) for name in COUNTS})
    cfg = NS(terrain_type='generator', use_terrain_origins=True, terrain_generator=generator,
             shared_bank_rows=rows, prim_path='/World/ground', collision_group=-1)
    origins = torch.arange(rows * 45, dtype=torch.float32).reshape(rows, 15, 3)
    terrain = NS(cfg=cfg, terrain_origins=origins, env_origins=torch.zeros(num_envs, 3),
                 terrain_types=torch.zeros(num_envs, dtype=torch.long), terrain_levels=torch.zeros(num_envs, dtype=torch.long))
    target = NS(prim_expr='/World/ground', merge_prim_meshes=True, track_mesh_transforms=False)
    scene_cfg = NS(num_envs=num_envs, terrain=cfg, robot=NS(prim_path='{ENV_REGEX_NS}/Robot', collision_group=0),
                   replicate_physics=False, clone_in_fabric=False, filter_collisions=True,
                   height_scanner=NS(mesh_prim_paths=[target], update_mesh_ids=True),
                   height_scanner_base=NS(mesh_prim_paths=[copy.deepcopy(target)], update_mesh_ids=True))
    paths = [f'/World/envs/env_{i}' for i in range(num_envs)]
    group_paths = [f'/World/collisions/group{i}' for i in range(num_envs)]
    global_group = '/World/collisions/global_group'
    prims = {'/physicsScene': Prim('/physicsScene', attributes={'physxScene:invertCollisionGroupFilter': True}),
             '/World/collisions': Prim('/World/collisions', kind='Scope')}
    for group, includes, allowed in [(global_group, ['/World/ground'], [global_group, *group_paths]),
                                    *((g, [p], [g, global_group]) for g, p in zip(group_paths, paths))]:
        prims[group] = Prim(group, attributes={'collection:colliders:expansionRule': 'expandPrims'},
                            relations={'collection:colliders:includes': includes, 'physics:filteredGroups': allowed})
        prims['/World/collisions'].children.append(prims[group])
    scene = NS(cfg=scene_cfg, terrain=terrain, env_prim_paths=paths, physics_scene_path='/physicsScene',
               _global_prim_paths=['/World/ground'], stage=NS(GetPrimAtPath=lambda p: prims[p]), prims=prims)
    return scene


def test_initial_shared_buffers_are_per_environment_not_aliased_to_bank():
    scene = shared_scene(7, 2)
    terrain = scene.terrain
    original = terrain.terrain_origins.clone()
    initialize_source_environment_origins(terrain, 7, bank_rows=2)
    assert terrain.terrain_levels.tolist() == [0, 1, 0, 1, 0, 1, 0]
    old_other = terrain.env_origins[2].clone()
    terrain.env_origins[0, 0] += 50
    assert torch.equal(terrain.env_origins[2], old_other)
    assert torch.equal(terrain.terrain_origins, original)


@pytest.mark.parametrize('n,rows', [(2, 1), (7, 2), (60, 4)])
def test_real_shared_adapter_preserves_tickets_and_unreset_buffers(n, rows):
    env = NS(num_envs=n, scene=shared_scene(n, rows))
    reference = ReservedRowTerrainAllocator(n, COUNTS, 130196)
    from dataclasses import replace
    for batch in [list(range(n)), [0], [n - 1, 1], [], [1], list(range(n))]:
        # For the two-env case keep the batch unique.
        batch = list(dict.fromkeys(batch))
        prior = env.scene.terrain.env_origins.clone()
        tickets = assign(env, batch, 130196)
        assert tuple(replace(t, row=t.env_id) for t in tickets) == reference.reset(batch)
        assert all(t.row == t.env_id % rows for t in tickets)
        others = list(set(range(n)) - set(batch))
        assert torch.equal(prior[others], env.scene.terrain.env_origins[others])


@pytest.mark.parametrize('fault', ['filter', 'replication', 'fabric', 'robot_global', 'terrain_local',
                                  'ray_robot', 'ray_world', 'ray_dynamic', 'ray_no_ids', 'ray_missing'])
def test_invalid_shared_config_is_rejected_before_spending_a_ticket(fault):
    env = NS(num_envs=2, scene=shared_scene())
    cfg = env.scene.cfg
    if fault == 'filter': cfg.filter_collisions = False
    elif fault == 'replication': cfg.replicate_physics = True
    elif fault == 'fabric': cfg.clone_in_fabric = True
    elif fault == 'robot_global': cfg.robot.collision_group = -1
    elif fault == 'terrain_local': cfg.terrain.collision_group = 0
    elif fault in ('ray_robot', 'ray_world'):
        cfg.height_scanner.mesh_prim_paths[0].prim_expr = '/World/envs/env_.*/Robot' if fault == 'ray_robot' else '/World'
    elif fault == 'ray_dynamic': cfg.height_scanner_base.mesh_prim_paths[0].track_mesh_transforms = True
    elif fault == 'ray_no_ids': cfg.height_scanner.update_mesh_ids = False
    else: cfg.height_scanner.mesh_prim_paths = []
    prior = env.scene.terrain.env_origins.clone()
    with pytest.raises(RuntimeError): assign(env, [0, 1], 130196)
    assert not hasattr(env, '_source_parallel_terrain_allocator')
    assert torch.equal(prior, env.scene.terrain.env_origins)


@pytest.mark.parametrize('fault', ['invert', 'cross_pair', 'global_robot', 'missing_ground', 'wrong_owner',
                                  'excluded_robot', 'expansion', 'include_root', 'extra_group'])
def test_runtime_filter_graph_tampering_is_rejected_atomically(fault):
    env = NS(num_envs=2, scene=shared_scene())
    assign(env, [0, 1], 130196)
    before = env._source_parallel_terrain_allocator
    saved_scene = copy.deepcopy(env.scene)
    p = env.scene.prims['/World/collisions/group0']
    if fault == 'invert': env.scene.prims['/physicsScene'].attributes['physxScene:invertCollisionGroupFilter'] = False
    elif fault == 'cross_pair': p.relations['physics:filteredGroups'].append('/World/collisions/group1')
    elif fault == 'global_robot': env.scene._global_prim_paths.append('/World/envs/env_1/Robot')
    elif fault == 'missing_ground': p.relations['physics:filteredGroups'].remove('/World/collisions/global_group')
    elif fault == 'wrong_owner': p.relations['collection:colliders:includes'] = ['/World/envs/env_1']
    elif fault == 'excluded_robot': p.relations['collection:colliders:excludes'] = ['/World/envs/env_0/Robot']
    elif fault == 'expansion': p.attributes['collection:colliders:expansionRule'] = 'explicitOnly'
    elif fault == 'include_root': p.attributes['collection:colliders:includeRoot'] = True
    else: env.scene.prims['/World/collisions'].children.append(Prim('/World/collisions/extra'))
    with pytest.raises(RuntimeError): assign(env, [0], 130196)
    assert env._source_parallel_terrain_allocator is before
    assert torch.equal(env.scene.terrain.env_origins, saved_scene.terrain.env_origins)


def test_write_failure_rolls_back_shared_origins_and_queue():
    env = NS(num_envs=2, scene=shared_scene())
    assign(env, [0, 1], 130196)
    terrain = env.scene.terrain
    original = terrain.env_origins
    prior = original.clone()
    levels, types = terrain.terrain_levels.clone(), terrain.terrain_types.clone()
    allocator = env._source_parallel_terrain_allocator

    class FailOnce:
        shape = original.shape
        failed = False
        def __getitem__(self, ids): return original[ids]
        def __setitem__(self, ids, values):
            if not self.failed:
                self.failed = True
                original[ids] = values  # also exercise rollback after a partial write
                raise RuntimeError('injected buffer write failure')
            original[ids] = values

    terrain.env_origins = FailOnce()
    with pytest.raises(RuntimeError, match='injected'): assign(env, [0], 130196)
    assert env._source_parallel_terrain_allocator is allocator
    assert torch.equal(original, prior)
    assert torch.equal(terrain.terrain_levels, levels) and torch.equal(terrain.terrain_types, types)
    terrain.env_origins = original
    reference = copy.deepcopy(allocator)
    assert assign(env, [0], 130196) == reference.reset([0])


@pytest.mark.parametrize('rows', [0, -1, True, 1.5, 3])
def test_shared_invalid_rows_rejected(rows):
    env = NS(num_envs=2, scene=shared_scene())
    env.scene.terrain.cfg.shared_bank_rows = rows
    with pytest.raises(RuntimeError): assign(env, [0, 1], 130196)


def test_seed_and_layout_changes_do_not_rewind_shared_queue():
    env = NS(num_envs=2, scene=shared_scene(2, 2))
    assign(env, [0, 1], 130196)
    old = env._source_parallel_terrain_allocator
    with pytest.raises(RuntimeError, match='seed cannot change'): assign(env, [0], 0)
    env.scene.terrain.cfg.shared_bank_rows = None
    with pytest.raises(RuntimeError, match='layout changed'): assign(env, [0], 130196)
    assert env._source_parallel_terrain_allocator is old


def test_candidate_seed_has_real_overlap_without_changing_episode_weights(diagnostic):
    tickets = SharedBankTerrainAllocator(2, COUNTS, diagnostic.SHARED_DIAGNOSTIC_SEED, bank_rows=1).reset([0, 1])
    assert [t.subterrain for t in tickets] == ['flat', 'flat']
    assert sum(map(len, diagnostic.SHARED_RESET_BATCHES)) == 6


@pytest.mark.parametrize('paths,expected', [
    (('/World/envs/env_0/Robot/leg/collision', '/World/envs/env_1/Robot/torso/collision'), ('cross_environment', None)),
    (('/World/ground/terrain/ordinary/mesh', '/World/envs/env_1/Robot/foot/collision'), ('ground', 1)),
    (('/World/envs/env_0/Robot/leg/collision', '/World/envs/env_0/Robot/torso/collision'), ('other', None)),
    (('/World/grounded/mesh', '/World/envs/env_1/Robot/foot/collision'), ('other', None)),
])
def test_contact_classification_uses_exact_environment_and_ground_ownership(diagnostic, paths, expected):
    assert diagnostic.classify_contact(*paths) == expected


def scanner_env():
    sensors = {}
    for name in ('height_scanner', 'height_scanner_base'):
        sensors[name] = NS(data=NS(ray_hits_w=torch.zeros(2, 9, 3), ray_mesh_ids=torch.zeros(2, 9, 1)),
                           _mesh_ids_wp=NS(numpy=lambda: np.full((2, 1), 42)), meshes={'/World/ground': NS(id=42)})
    return NS(scene=NS(sensors=sensors, terrain=NS(env_origins=torch.zeros(2, 3))))


@pytest.mark.parametrize('fault', ['target', 'extra', 'miss', 'nan', 'height'])
def test_runtime_scan_failure_is_detected(diagnostic, fault):
    env = scanner_env()
    sensor = env.scene.sensors['height_scanner']
    if fault == 'target': sensor._mesh_ids_wp.numpy = lambda: np.full((2, 1), 99)
    elif fault == 'extra': sensor._mesh_ids_wp.numpy = lambda: np.full((2, 2), 42)
    elif fault == 'miss': sensor.data.ray_mesh_ids[0, 0] = -1
    elif fault == 'nan': sensor.data.ray_hits_w[0, 0, 0] = float('nan')
    else: sensor.data.ray_hits_w[0, 0, 2] = 0.8
    with pytest.raises(AssertionError): diagnostic.check_scanners(env, flat_entry=True)


def test_runtime_scans_require_ground_height_and_ids(diagnostic):
    assert len(diagnostic.check_scanners(scanner_env(), flat_entry=True)) == 2


def test_shared_report_rejects_missing_coverage(diagnostic):
    report = dict(status='shared_bank_smoke_passed', num_envs=2, bank_rows=1, zero_action_steps=20,
                  overlapping_spawn_checked=True, cross_environment_contacts=0,
                  ground_contact_points={'0': 1, '1': 1}, vertical_occlusion_probe_passed=True,
                  reset_checks=dict(assignment_count=8, reset_batch_count=5, direct_reset_count=6),
                  policy_loaded=False, ppo_updates=0)
    assert diagnostic.shared_report_passed(report)
    for field in report:
        changed = dict(report)
        del changed[field]
        assert not diagnostic.shared_report_passed(changed), field
    for field, value in [('ground_contact_points', {'0': 1}), ('cross_environment_contacts', 1),
                         ('zero_action_steps', 19), ('reset_checks', {'assignment_count': 8}),
                         ('ground_contact_points', None), ('ground_contact_points', {'0': '1', '1': 1}),
                         ('reset_checks', None)]:
        assert not diagnostic.shared_report_passed(dict(report, **{field: value}))


def _construct_scene_config(*, shared=None, enabled=True, num_envs=2):
    """Execute the real scene-construction portion with record-only config types."""
    path = ROOT / ('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/'
                   'config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__post_init__')
    # The remaining method configures rewards/commands/reset events; those are
    # unchanged by shared mode and require full Isaac config types.
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]).startswith('self.commands.'))
    method.body = method.body[:end]
    cls.decorator_list = []
    cls.body = [n for n in cls.body if isinstance(n, ast.AnnAssign)] + [method]

    class Base:
        def __post_init__(self):
            self.scene = NS(robot=NS(prim_path='{ENV_REGEX_NS}/Robot', init_state=NS(pos=(0, 0, 0.4))),
                            replicate_physics=True, filter_collisions=True, clone_in_fabric=False)
            self.sim, self.decimation, self.base_link_name = NS(dt=0.005), 4, 'torso'

    class Scanner(NS):
        RaycastTargetCfg = NS
        OffsetCfg = NS

    namespace = dict(DeeproboticsLite3RoughEnvCfg=Base, DEEPROBOTICS_LITE3_CFG=NS(init_state=NS(pos=(0, 0, 0.4))),
                     SourceTerrainTrackCfg=NS, SourceTerrainImporterCfg=NS, TerrainGeneratorCfg=NS,
                     SourceTerrainGenerator=object, sim_utils=NS(RigidBodyMaterialCfg=NS),
                     MultiMeshRayCasterCfg=Scanner, patterns=NS(GridPatternCfg=NS),
                     set_uniform_subterrain_proportions=set_uniform_subterrain_proportions)
    pure = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))] + [cls]
    exec(compile(ast.Module(body=pure, type_ignores=[]), str(path), 'exec'), namespace)
    config = namespace[cls.name]()
    config.source_shared_bank_rows = shared
    config.source_parallel_sampler_enabled = enabled
    config.source_parallel_sampler_num_envs = num_envs
    config.source_parallel_sampler_seed = 130196
    config.__post_init__()
    return config


def test_actual_config_construction_keeps_shared_opt_in_and_sets_explicit_filters():
    default = _construct_scene_config(enabled=False)
    assert default.source_shared_bank_rows is None and default.scene.num_envs == 1
    reserved = _construct_scene_config()
    assert reserved.scene.terrain.terrain_generator.num_rows == 2
    assert reserved.scene.replicate_physics is True
    shared = _construct_scene_config(shared=1)
    assert shared.scene.terrain.terrain_generator.num_rows == 1
    assert shared.scene.terrain.shared_bank_rows == 1 and shared.scene.num_envs == 2
    validate_shared_scene_cfg(shared.scene)


@pytest.mark.parametrize('shared,enabled', [(1, False), (True, True), (0, True), (3, True)])
def test_actual_config_construction_rejects_implicit_or_invalid_shared_mode(shared, enabled):
    with pytest.raises(ValueError): _construct_scene_config(shared=shared, enabled=enabled)


def test_actual_importer_maps_small_bank_to_independent_environment_buffers(monkeypatch):
    path = Path(__file__).with_name('test_source_terrain_importer_chunking.py')
    spec = importlib.util.spec_from_file_location('shared_importer_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    importer = object.__new__(module.importer_module.SourceTerrainImporter)
    importer.cfg = NS(parallel_reset_sampler_num_envs=7)
    importer._parallel_reset_sampler_enabled = True
    importer._shared_bank_rows = 2
    importer.device = 'cpu'
    origins = np.arange(90, dtype=np.float32).reshape(2, 15, 3)
    actual = importer._compute_env_origins_curriculum(7, origins)
    assert actual.shape == (7, 3)
    assert importer.max_terrain_level == 2
    assert importer.terrain_levels.tolist() == [0, 1, 0, 1, 0, 1, 0]
    np.testing.assert_array_equal(actual.numpy(), origins[[0, 1, 0, 1, 0, 1, 0], 0])


def lifecycle_env(diagnostic, fault=None):
    """Two synthetic managers with the actual shared adapter and USD audit."""
    cfg_scene = shared_scene()
    class Scene(dict):
        def write_data_to_sim(self): pass
        def update(self, _dt): pass
    scene = Scene()
    scene.__dict__.update(vars(cfg_scene))
    env = NS(num_envs=2, scene=scene, device='cpu')
    assign(env, [0, 1], diagnostic.SHARED_DIAGNOSTIC_SEED)
    root = NS(root_state_w=torch.zeros(2, 13), default_root_state=torch.zeros(2, 13),
              joint_pos=torch.ones(2, 12), joint_vel=torch.zeros(2, 12))
    root.default_root_state[:, 2] = 0.4
    root.root_pos_w = root.root_state_w[:, :3]
    root.root_pos_w[:] = root.default_root_state[:, :3] + scene.terrain.env_origins
    scene['robot'] = NS(data=root)
    scene.sensors = scanner_env().scene.sensors
    for sensor in scene.sensors.values():
        sensor._timestamp = torch.ones(2)
        sensor._timestamp_last_update = torch.ones(2)
        sensor._is_outdated = torch.zeros(2, dtype=torch.bool)
        sensor._data = sensor.data
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    env.action_manager = NS(action=torch.ones(2, 12), prev_action=torch.ones(2, 12))
    env.command_manager = NS(active_terms=['velocity'], get_command=lambda _: torch.zeros(2, 3))
    history = NS(_buffer=torch.ones(3, 2, 4), _num_pushes=torch.ones(2, dtype=torch.long))
    def compute(update_history):
        history._buffer[:] = 1
        history._num_pushes += 1
        return {'policy': torch.ones(2, 4)}
    env.observation_manager = NS(_group_obs_term_history_buffer={'policy': {'pos': history}}, compute=compute)
    env.reset_calls = []
    env.old_allocators = []
    def reset(ids):
        env.reset_calls.append(ids.tolist())
        env.old_allocators.append(env._source_parallel_terrain_allocator)
        assign(env, ids, diagnostic.SHARED_DIAGNOSTIC_SEED)
        root.root_pos_w[ids] = root.default_root_state[ids, :3] + scene.terrain.env_origins[ids]
        env.episode_length_buf[ids] = 0
        env.action_manager.action[ids] = 0
        env.action_manager.prev_action[ids] = 0
        history._buffer[:, ids] = 0
        history._num_pushes[ids] = 0
        for sensor in scene.sensors.values():
            sensor._timestamp[ids] = 0
            sensor._timestamp_last_update[ids] = 0
            sensor._is_outdated[ids] = True
        if fault == 'root': root.root_pos_w[1, 0] += 1
        elif fault == 'history': history._buffer[:, 1] += 1
        elif fault == 'selected_root': root.root_pos_w[0, 0] += 1
        elif fault == 'stale_allocator': env._source_parallel_terrain_allocator = env.old_allocators[-1]
    def forward():
        for sensor in scene.sensors.values():
            sensor._data.ray_hits_w[..., 2] = scene.terrain.env_origins[:, 2, None]
    env.sim = NS(forward=forward)
    env._reset_idx = reset
    return env


def test_complete_shared_reset_diagnostic_survives_allocator_replacement(diagnostic):
    env = lifecycle_env(diagnostic)
    records = []
    report = diagnostic.check_shared_resets(env, diagnostic.SHARED_DIAGNOSTIC_SEED, records.append)
    assert report['assignment_count'] == 8 and report['direct_reset_count'] == 6
    assert report['history_buffers'] == ['history/policy/pos']
    assert len(env.old_allocators) == 5
    assert len({id(x) for x in env.old_allocators}) == 5
    assert len([r for r in records if r['phase'] == 'reset_checked']) == 5
    assert env.reset_calls == [list(ids) for ids in diagnostic.SHARED_RESET_BATCHES]


@pytest.mark.parametrize('fault', ['root', 'history', 'selected_root', 'stale_allocator'])
def test_shared_reset_diagnostic_retains_comparison_and_stops_on_fault(diagnostic, fault):
    env = lifecycle_env(diagnostic, fault)
    records = []
    with pytest.raises(AssertionError):
        diagnostic.check_shared_resets(env, diagnostic.SHARED_DIAGNOSTIC_SEED, records.append)
    assert len(env.reset_calls) == 1
    assert [r['phase'] for r in records] == ['before_reset', 'assignment_comparison']
