"""CPU fault injection using pinned RL/PhysX stepping bodies; no simulation launch."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, MethodType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
BODIES = [f'/World/envs/env_{i}/Robot/FOOT' for i in range(2)]
GROUND = '/World/ground/terrain'


@pytest.fixture
def contact(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/tools'))
    import source_shared_contact_report
    return source_shared_contact_report


def batch(owner=0, *, event=0, count=1, other=GROUND):
    actor = BODIES[owner]
    header = NS(actor0=actor, actor1=other, collider0=actor + '/collision', collider1=other,
                stage_id=73, type=event, contact_data_offset=0, num_contact_data=count)
    points = [NS(position=(0., 0., 0.), normal=(0., 0., 1.), impulse=(0., 0., .5), separation=-.001)
              for _ in range(count)]
    return [header], points


def make_capture(contact, *, control_steps=20):
    journal = []
    capture = contact.ContactCapture(stage_id=73, reporting_bodies=BODIES, decode_path=str,
                                     event_types={0: 'found', 1: 'lost', 2: 'persist'}, record=journal.append,
                                     start_step=12, decimation=4, control_steps=control_steps)
    return capture, journal


def pinned_method(suffix, cls, method, namespace=None):
    audit = json.loads((ROOT / 'docs/lite3-source-shared-bank-preparation-20261028/dependency_audit.json').read_text())
    item, = [d for d in audit['dependencies'] if d['path'].endswith(suffix)]
    path = Path(item['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls) if cls else tree
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    # Drop annotations and binding decorators; execute the unchanged function body.
    fn.returns = None
    fn.decorator_list = []
    for arg in fn.args.args:
        arg.annotation = None
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method]


def runtime_reader(contact, env, get_disabled=lambda: False):
    return contact.ContactRuntime(
        env, stage_id_for=lambda stage: 0 if stage is None else stage.stage_id,
        usd_context_stage_id=lambda: env._usd_context_stage_id, get_disabled=get_disabled,
    )


def stepping_env(*, callback_mode='silent', fault=None):
    events = []
    class Recorder:
        active_terms = ()
        record_post_physics_decimation_step = pinned_method(
            '/managers/recorder_manager.py', 'RecorderManager', 'record_post_physics_decimation_step')
        def record_pre_step(self): pass
    env = NS(_sim_step_counter=12, device='cpu', recorder_manager=Recorder(),
             cfg=NS(decimation=4, sim=NS(render_interval=4, create_stage_in_memory=False)), physics_dt=.005, step_dt=.02,
             episode_length_buf=torch.zeros(2), common_step_counter=0, extras={})

    class Subscription:
        def __init__(self, interface): self.interface = interface
        def __del__(self): self.interface.callback = None

    class Interface:
        callback = None
        ready = False
        reports = ([], [])
        simulations = 0
        attached_stage = 0
        def get_attached_stage(self): return self.attached_stage
        def attach_stage(self, stage_id): pytest.fail('diagnostic must not reattach physics')
        def detach_stage(self): pytest.fail('diagnostic must not detach physics')
        def subscribe_contact_report_events(self, callback):
            self.callback = callback
            return Subscription(self)
        def simulate(self, dt, current_time):
            assert dt == .005
            self.ready = False
            self.simulations += 1
            events.append('simulate')
        def fetch_results(self):
            events.append('fetch')
            self.ready = True
            # Transient positives exist only at the first substep of control 1/2.
            self.reports = batch(0) if self.simulations == 1 else batch(1, event=2) if self.simulations == 5 else ([], [])
            if fault == 'poll_cross' and self.simulations == 1:
                self.reports = batch(other=BODIES[1])
            if fault == 'no_contacts': self.reports = ([], [])
            if self.callback is not None and callback_mode != 'silent':
                reports = ([], []) if callback_mode == 'empty' else copy.deepcopy(self.reports)
                if callback_mode == 'malformed': reports = batch(count=0); reports[0][0].num_contact_data = 1
                if callback_mode == 'cross': reports = batch(other=BODIES[1])
                self.callback(*reports)
            if fault == 'counter_jump': env._sim_step_counter += 1
        def get_contact_report(self):
            assert self.ready, 'poll before fetch_results'
            events.append('poll')
            return self.reports
    interface = Interface()
    physics = NS(_physx_sim_interface=interface, get_physics_dt=lambda: .005)
    physics._step = MethodType(pinned_method('/physics_context/physics_context.py', 'PhysicsContext', '_step'), physics)
    # Exercise real simulation-context properties and tensor-view validity rather
    # than pretending explicit attachment is how all Lab stages are initialized.
    class View:
        is_valid = property(pinned_method('/tensors/impl/api.py', 'SimulationView', 'is_valid'))
    view = View()
    view._backend = NS(is_valid=True)
    manager = NS(_physics_sim_view=view)
    manager.get_physics_sim_view = lambda: manager._physics_sim_view
    class Sim:
        stage = property(pinned_method('/simulation_context/simulation_context.py', 'SimulationContext', 'stage'))
        physics_sim_view = property(pinned_method('/simulation_context/simulation_context.py', 'SimulationContext',
                                                  'physics_sim_view', {'SimulationManager': manager}))
        get_initial_stage = pinned_method('/sim/simulation_context.py', 'SimulationContext', 'get_initial_stage')
        def is_playing(self): return self.playing
    env.sim = Sim()
    stage = NS(stage_id=73)
    env.sim._stage = env.sim._initial_stage = stage
    env.sim.cfg = env.cfg.sim
    env.sim.playing = True
    env.sim.has_gui, env.sim.has_rtx_sensors = lambda: False, lambda: False
    env.sim.step = lambda render: physics._step(0.0)
    env._view_manager = manager
    env._usd_context_stage_id = stage.stage_id
    env.action_manager = NS(process_action=lambda a: None, apply_action=lambda: events.append('action'))
    def scene_update(dt):
        assert dt == .005
        events.append('scene_update')
        # PhysX data is borrowed: tests overwrite it before the next substep.
        interface.reports = ([], [])
    env.scene = NS(write_data_to_sim=lambda: events.append('write'), update=scene_update,
                   stage=stage, stage_id=stage.stage_id)
    env.termination_manager = NS(compute=lambda: torch.zeros(2, dtype=torch.bool),
                                 terminated=torch.zeros(2, dtype=torch.bool), time_outs=torch.zeros(2, dtype=torch.bool))
    env.reward_manager = NS(compute=lambda dt: torch.zeros(2))
    env.command_manager = NS(compute=lambda dt: None)
    env.event_manager = NS(available_modes=[])
    env.observation_calls = []
    def observations(update_history):
        env.observation_calls.append(update_history)
        return {'policy': torch.ones(2, 3)}
    env.observation_manager = NS(compute=observations)
    env.step = MethodType(pinned_method('/envs/manager_based_rl_env.py', 'ManagerBasedRLEnv', 'step'), env)
    return env, interface, events


@pytest.mark.parametrize('callback_mode,callback_points', [('silent', {}), ('empty', {}), ('valid', {0: 1, 1: 1})])
def test_actual_decimation_captures_transients_once_and_restores_hook(contact, callback_mode, callback_points):
    env, interface, events = stepping_env(callback_mode=callback_mode)
    other_manager = type(env.recorder_manager)()
    capture, journal = make_capture(contact)
    original = env.recorder_manager.record_post_physics_decimation_step.__func__
    with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)):
        for i in range(20):
            env.step(torch.zeros(2, 12))
            capture.check_control(env, i + 1)
    assert capture.ground['poll'] == {0: 1, 1: 1}
    assert capture.ground['callback'] == callback_points
    assert capture.counts['callback']['calls'] == (0 if callback_mode == 'silent' else 80)
    assert events == ['action', 'write', 'simulate', 'fetch', 'poll', 'scene_update'] * 80
    assert env.observation_calls == [True] * 20
    assert contact.contact_summary_passed(capture.summary(), {0: 1, 1: 1})
    assert 'record_post_physics_decimation_step' not in vars(env.recorder_manager)
    assert other_manager.record_post_physics_decimation_step.__func__ is original
    assert env.recorder_manager.active_terms == () and interface.callback is None
    assert journal[-1]['phase'] == 'contact_summary' and journal[-1]['errors'] == []
    assert len([r for r in journal if r['phase'] == 'contact_batch' and r['channel'] == 'poll']) == 80


@pytest.mark.parametrize('fault,callback_mode,steps', [
    ('poll_cross', 'silent', 1), (None, 'cross', 1), (None, 'malformed', 1),
    ('counter_jump', 'silent', 1), ('no_contacts', 'silent', 80),
])
def test_failures_stop_before_next_substep_and_always_release(contact, fault, callback_mode, steps):
    env, interface, events = stepping_env(callback_mode=callback_mode, fault=fault)
    capture, journal = make_capture(contact)
    original = lambda: None
    env.recorder_manager.record_post_physics_decimation_step = original
    with pytest.raises((AssertionError, ValueError)):
        with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)):
            for i in range(20):
                env.step(torch.zeros(2, 12))
                capture.check_control(env, i + 1)
    assert interface.simulations == steps
    assert env.recorder_manager.record_post_physics_decimation_step is original
    assert interface.callback is None
    assert journal[-1]['phase'] == 'contact_summary' and journal[-1]['errors']
    assert not contact.contact_summary_passed(capture.summary(), {0: 1, 1: 1})


@pytest.mark.parametrize('field,value', [('stage_id', 74), ('contact_data_offset', -1),
    ('contact_data_offset', 1), ('num_contact_data', 2), ('num_contact_data', -1),
    ('num_contact_data', True), ('num_contact_data', 1.5), ('type', 99), ('type', 1),
    ('actor0', '/World/envs/env_2/Robot/FOOT'), ('actor1', '/World/unknown'),
    ('collider0', '/World/envs/env_1/Robot/FOOT/collision')])
def test_malformed_header_never_counts_positive(contact, field, value):
    capture, _ = make_capture(contact)
    headers, data = batch()
    setattr(headers[0], field, value)
    with pytest.raises((AssertionError, ValueError, TypeError)):
        capture.consume('poll', headers, data, 13)
    assert not capture.ground['poll']


@pytest.mark.parametrize('field,value', [('position', (0, float('nan'), 0)), ('normal', (0, 1)),
    ('impulse', (float('inf'), 0, 0)), ('separation', float('nan'))])
def test_nonfinite_or_malformed_point_rejected(contact, field, value):
    capture, _ = make_capture(contact)
    headers, data = batch()
    setattr(data[0], field, value)
    with pytest.raises(ValueError): capture.consume('poll', headers, data, 13)
    assert not capture.ground['poll']


def test_orphan_duplicate_ranges_and_lost_events(contact):
    capture, _ = make_capture(contact)
    headers, data = batch()
    with pytest.raises(ValueError, match='unreferenced'): capture.consume('poll', [], data, 13)
    with pytest.raises(ValueError, match='overlapping'): capture.consume('poll', headers * 2, data, 13)
    capture.consume('poll', *batch(event=1, count=0), 13)
    assert not capture.ground['poll']


@pytest.mark.parametrize('fault', ['disabled', 'unknown_setting', 'stage', 'recorder', 'decimation', 'subscribe'])
def test_bad_runtime_fails_before_simulation(contact, fault):
    env, interface, _ = stepping_env()
    capture, journal = make_capture(contact)
    setting = False
    if fault == 'disabled': setting = True
    if fault == 'unknown_setting': setting = None
    if fault == 'stage': interface.get_attached_stage = lambda: 74
    if fault == 'recorder': env.recorder_manager.active_terms = ['existing']
    if fault == 'decimation': env.cfg.decimation = 3
    if fault == 'subscribe': interface.subscribe_contact_report_events = lambda cb: None
    with pytest.raises(AssertionError):
        with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env, lambda: setting)):
            pytest.fail('must not reach probe')
    assert interface.simulations == 0 and interface.callback is None
    assert journal[-1]['errors'] and not vars(env.recorder_manager).get('record_post_physics_decimation_step')


def test_missing_hook_or_early_exit_cannot_pass(contact):
    env, interface, _ = stepping_env()
    capture, journal = make_capture(contact)
    with pytest.raises(AssertionError, match='incomplete contact polls'):
        with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)):
            env.recorder_manager.record_post_physics_decimation_step = lambda: None
            env.step(torch.zeros(2, 12))
            capture.check_control(env, 1)
    assert interface.simulations == 4 and journal[-1]['errors']
    capture, journal = make_capture(contact)
    with pytest.raises(AssertionError, match='incomplete or failed'):
        with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)): pass
    assert journal[-1]['errors']


def test_setting_change_mid_probe_stops_at_next_hook(contact):
    env, interface, _ = stepping_env()
    capture, journal = make_capture(contact)
    with pytest.raises(AssertionError, match='processing disabled'):
        with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env, lambda: interface.simulations > 0)):
            env.step(torch.zeros(2, 12))
    assert interface.simulations == 1 and journal[-1]['errors']


@pytest.mark.parametrize('fault', [None, 'original_hook', 'poll_api'])
def test_original_instance_hook_is_called_and_restored_even_on_exception(contact, fault):
    env, interface, events = stepping_env()
    capture, journal = make_capture(contact, control_steps=2)
    calls = []
    def original():
        calls.append(True)
        if fault == 'original_hook': raise RuntimeError('original hook failed')
    def broken_poll(): raise RuntimeError('poll API failed')
    env.recorder_manager.record_post_physics_decimation_step = original
    if fault == 'poll_api': interface.get_contact_report = broken_poll
    def probe():
        with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)):
            for i in range(2):
                env.step(torch.zeros(2, 12))
                capture.check_control(env, i + 1)
    if fault:
        with pytest.raises(RuntimeError): probe()
        assert interface.simulations == 1 and journal[-1]['errors']
        assert len(calls) == (1 if fault == 'original_hook' else 0)
    else:
        probe()
        assert len(calls) == 8 and journal[-1]['errors'] == []
    assert env.recorder_manager.record_post_physics_decimation_step is original
    assert interface.callback is None


def test_report_validator_rejects_incomplete_corrupt_or_callback_only_evidence(contact):
    env, interface, _ = stepping_env()
    capture, _ = make_capture(contact)
    with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)):
        for i in range(20):
            env.step(torch.zeros(2, 12)); capture.check_control(env, i + 1)
    summary = json.loads(json.dumps(capture.summary()))
    assert contact.contact_summary_passed(summary, {'0': 1, '1': 1})
    required = ['schema', 'stage_id', 'start_sim_step', 'runtime_checked', 'runtime_check_count', 'runtime_state', 'positive_controls_validated',
                'poll_steps', 'control_poll_counts', 'channels', 'ground_contact_points', 'errors',
                'cross_environment_contacts', 'decimation', 'control_steps']
    for key in required:
        bad = dict(summary); del bad[key]
        assert not contact.contact_summary_passed(bad, {0: 1, 1: 1}), key
    for key, value in [('poll_steps', list(range(80))), ('control_poll_counts', [4] * 19),
                       ('errors', ['bad']), ('cross_environment_contacts', 1), ('start_sim_step', True),
                       ('ground_contact_points', {}), ('ground_contact_points', {'0': 1}),
                       ('channels', {'poll': {}, 'callback': {}})]:
        assert not contact.contact_summary_passed(dict(summary, **{key: value}), {0: 1, 1: 1})
    assert not contact.contact_summary_passed(summary, {0: 2, 1: 1})


@pytest.mark.parametrize('fault', [None, 'no_contacts', 'tensor_cross'])
@pytest.mark.parametrize('shared_g2', [False, True])
def test_diagnostic_integration_gates_resets_and_records_sensor_evidence(contact, monkeypatch, tmp_path, fault, shared_g2):
    import source_shared_terrain_diagnostic as diagnostic
    from test_shared_terrain_runtime import scanner_env
    env, interface, events = stepping_env()
    env.num_envs = 2
    env.cfg.foot_link_name = '.*FOOT'
    env.sim.stage.GetPrimAtPath = lambda path: path
    env.sim.forward = lambda: None
    env.action_manager.action = torch.zeros(2, 12)
    names = ['FOOT', 'B_FOOT', 'C_FOOT', 'D_FOOT']
    roots = torch.zeros(2, 13)
    roots[:, 2] = .375
    data = NS(root_state_w=roots, root_pos_w=roots[:, :3], default_root_state=roots.clone(),
              body_pos_w=torch.zeros(2, 4, 3), body_lin_vel_w=torch.zeros(2, 4, 3))
    def write_pose(pose, env_ids): data.root_state_w[env_ids, :7] = pose
    robot = NS(data=data, num_bodies=4, find_bodies=lambda pattern: (list(range(4)), names),
               write_root_pose_to_sim=write_pose)
    class Scene(dict): pass
    scene = Scene(robot=robot)
    scene.__dict__.update(vars(env.scene))
    scene.update = lambda dt: None
    scene.env_prim_paths = ['/World/envs/env_0', '/World/envs/env_1']
    scene.terrain = NS(terrain_origins=torch.zeros(1, 15, 3), env_origins=torch.zeros(2, 3))
    scene.sensors = scanner_env().scene.sensors
    for sensor in scene.sensors.values(): sensor.update = lambda *a, **k: None
    # Sensor order intentionally differs from articulation order.
    scene.sensors['contact_forces'] = NS(body_names=list(reversed(names)),
                                        data=NS(net_forces_w=torch.arange(24.).reshape(2, 4, 3)))
    env.scene = scene
    tickets = diagnostic.SharedBankTerrainAllocator(2, diagnostic.SOURCE_RESET_BLOCK_COUNTS,
                                                     diagnostic.SHARED_DIAGNOSTIC_SEED, bank_rows=1).reset([0, 1])
    env._source_parallel_terrain_allocator = NS(assignments=tickets)
    env.termination_manager.get_term_cfg = lambda name: NS(params={'asset_cfg': NS(body_ids=slice(None))})
    class Prim:
        def __init__(self, path): self.path = path
        def HasAPI(self, api): return True
        def GetPath(self): return self.path
    class ReportAPI:
        def __init__(self, prim): pass
        def GetThresholdAttr(self): return NS(Get=lambda: 0.)
        def GetCollisionEnabledAttr(self): return NS(Get=lambda: True)
    settings = NS(get_settings=lambda: NS(get=lambda key: key == '/physics/suppressReadback'))
    usd = NS(get_context=lambda: NS(get_stage_id=lambda: env._usd_context_stage_id))
    fake_modules = {
        'carb': NS(settings=settings), 'carb.settings': settings,
        'omni': NS(usd=usd), 'omni.usd': usd,
        'omni.physx': NS(get_physx_simulation_interface=lambda: interface,
                         get_physx_interface=lambda: NS(is_readback_suppressed=lambda: True)),
        'omni.physx.bindings': NS(), 'omni.physx.bindings._physx': NS(
            ContactEventType=NS(CONTACT_FOUND=0, CONTACT_PERSIST=2, CONTACT_LOST=1)),
        'pxr': NS(PhysicsSchemaTools=NS(intToSdfPath=str), PhysxSchema=NS(PhysxContactReportAPI=ReportAPI),
                  Usd=NS(PrimRange=lambda path: [Prim(path + '/' + name) for name in names]),
                  UsdPhysics=NS(RigidBodyAPI=object, CollisionAPI=ReportAPI),
                  UsdUtils=NS(StageCache=NS(Get=lambda: NS(GetId=lambda stage: NS(ToLongInt=lambda: stage.stage_id))))),
    }
    for name, module in fake_modules.items(): monkeypatch.setitem(sys.modules, name, module)
    class TensorView:
        def __init__(self, patterns, filter_patterns, max_contact_data_count):
            self.sensor_paths, self.filter_paths = patterns, filter_patterns
            self.sensor_count, self.filter_count = len(patterns), len(filter_patterns[0])
            self.max_contact_data_count = max_contact_data_count
        def payload(self):
            n, f, c = self.sensor_count, self.filter_count, self.max_contact_data_count
            matrix, counts, starts = torch.zeros(n, f, 3), torch.zeros(n, f, dtype=torch.int32), torch.zeros(n, f, dtype=torch.int32)
            forces, points, normals, distances = torch.zeros(c, 1), torch.zeros(c, 3), torch.zeros(c, 3), torch.zeros(c, 1)
            for i in range(n):
                j = next(j for j, path in enumerate(self.filter_paths[i]) if path.startswith('/World/ground/'))
                if fault == 'tensor_cross' and i == 0:
                    j = next(j for j, path in enumerate(self.filter_paths[i]) if path.startswith('/World/envs/env_1/'))
                if fault != 'no_contacts':
                    counts[i, j], starts[i, j], forces[i, 0], normals[i, 2], matrix[i, j, 2] = 1, i, 60, 1, 60
            return matrix, (forces, points, normals, distances, counts, starts)
        def get_net_contact_forces(self, dt): return self.payload()[0].sum(dim=1)
        def get_contact_force_matrix(self, dt): return self.payload()[0]
        def get_contact_data(self, dt): return self.payload()[1]
    env.sim.physics_sim_view.device = 'cuda:0'
    env.sim.physics_sim_view.create_rigid_contact_view = TensorView
    interface.get_contact_report = lambda: pytest.fail('diagnostic must use tensor contacts')
    interface.subscribe_contact_report_events = lambda _: pytest.fail('diagnostic must use tensor contacts')
    monkeypatch.setattr(diagnostic, 'audit_shared_collision_groups', lambda scene: {'fixture': True})
    reset_calls = []
    def resets(*args):
        reset_calls.append(True)
        return dict(assignment_count=8, reset_batch_count=5, direct_reset_count=6)
    monkeypatch.setattr(diagnostic, 'check_shared_resets', resets)
    import source_parallel_reset_lifecycle as lifecycle
    def g2_resets(actual_env, seed, path, trace, *, shared_bank):
        assert actual_env is env and seed == 130196 and shared_bank
        assert path.name == 'journal.resets.jsonl'
        assert interface.simulations == 80
        reset_calls.append(True)
        return dict(status='shared_reset_lifecycle_passed', num_envs=2, bank_rows=1,
                    assignment_count=123, complete_blocks=1, reset_batch_count=91,
                    direct_reset_count=120, automatic_reset_ids=[0],
                    diagnostic_timeout_injected=True, physics_control_steps=2,
                    post_automatic_reset_step_checked=True)
    monkeypatch.setattr(lifecycle, 'run_lifecycle', g2_resets)
    journal = tmp_path / 'journal.jsonl'
    if fault:
        with pytest.raises(AssertionError):
            diagnostic.run_shared_diagnostic(env, 130196, journal, lambda message: None, reset_lifecycle=shared_g2)
        assert reset_calls == []
        assert interface.simulations == (1 if fault == 'tensor_cross' else 80)
    else:
        report = diagnostic.run_shared_diagnostic(env, 130196, journal, lambda message: None, reset_lifecycle=shared_g2)
        assert diagnostic.shared_report_passed(json.loads(json.dumps(report)), reset_lifecycle=shared_g2)
        assert reset_calls == [True] and interface.simulations == 80
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    summary, = [r for r in entries if r['phase'] == 'contact_summary']
    assert bool(summary['errors']) == bool(fault)
    initial = entries[0]['contact_sensor']
    assert initial['foot_net_contact_forces_w_N'][0][0] == [9., 10., 11.]
    assert initial['foot_body_origin_height_above_tile_m'] == [[0.] * 4] * 2
    assert not initial['collider_clearance_measured'] and not initial['pairwise_isolation_evidence']
    assert interface.callback is None
    assert 'record_post_physics_decimation_step' not in vars(env.recorder_manager)


def test_sensor_telemetry_does_not_accept_missing_feet_or_nonfinite_forces(contact):
    import source_shared_terrain_diagnostic as diagnostic
    robot = NS(find_bodies=lambda pattern: ([0, 1, 2, 3], ['A', 'B', 'C', 'D']),
               data=NS(body_pos_w=torch.zeros(2, 4, 3)))
    class Scene(dict): pass
    scene = Scene(robot=robot)
    scene.sensors = {'contact_forces': NS(body_names=['A', 'B', 'C'], data=NS(net_forces_w=torch.zeros(2, 3, 3)))}
    scene.terrain = NS(env_origins=torch.zeros(2, 3))
    env = NS(scene=scene, cfg=NS(foot_link_name='.*'))
    with pytest.raises(AssertionError, match='four feet'): diagnostic.contact_sensor_snapshot(env)
    scene.sensors['contact_forces'] = NS(body_names=['A', 'B', 'C', 'D'],
                                        data=NS(net_forces_w=torch.full((2, 4, 3), float('nan'))))
    with pytest.raises(AssertionError, match='invalid'): diagnostic.contact_sensor_snapshot(env)


@pytest.mark.parametrize('in_memory,attached,usd_id', [
    (False, 0, 73), (False, 73, 73), (True, 73, 73), (True, 73, 0), (True, 73, 99),
])
def test_stage_modes_keep_all_substep_and_positive_control_requirements(contact, in_memory, attached, usd_id):
    env, interface, _ = stepping_env()
    env.sim.cfg.create_stage_in_memory = in_memory
    env._usd_context_stage_id = usd_id
    interface.attached_stage = attached
    capture, journal = make_capture(contact)
    with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)):
        for i in range(20):
            env.step(torch.zeros(2, 12))
            capture.check_control(env, i + 1)
    assert contact.contact_summary_passed(capture.summary(), {0: 1, 1: 1})
    states = [r for r in journal if r['phase'] == 'contact_runtime']
    assert len(states) == 81 and all(r['attached_stage'] == attached for r in states)
    assert interface.simulations == 80 and interface.callback is None


@pytest.mark.parametrize('fault', [
    'usd_missing', 'usd_other', 'sim_other', 'initial_other', 'scene_other', 'scene_id_other',
    'attached_other', 'attached_negative', 'memory_detached', 'unknown_mode', 'missing_view',
    'invalid_view', 'paused', 'view_replaced', 'mode_changed',
])
def test_zero_attached_stage_never_bypasses_live_identity_or_physics_validity(contact, fault):
    env, interface, _ = stepping_env()
    capture, journal = make_capture(contact)
    runtime = runtime_reader(contact, env)
    if fault == 'usd_missing': env._usd_context_stage_id = 0
    if fault == 'usd_other': env._usd_context_stage_id = 74
    if fault == 'sim_other': env.sim._stage = NS(stage_id=74)
    if fault == 'initial_other': env.sim._initial_stage = NS(stage_id=74)
    if fault == 'scene_other': env.scene.stage = NS(stage_id=74)
    if fault == 'scene_id_other': env.scene.stage_id = 74
    if fault == 'attached_other': interface.attached_stage = 74
    if fault == 'attached_negative': interface.attached_stage = -1
    if fault == 'memory_detached':
        env.sim.cfg.create_stage_in_memory = True
        runtime = runtime_reader(contact, env)
    if fault == 'unknown_mode': env.sim.cfg.create_stage_in_memory = None
    if fault == 'missing_view': env._view_manager._physics_sim_view = None
    if fault == 'invalid_view': env.sim.physics_sim_view._backend.is_valid = False
    if fault == 'paused': env.sim.playing = False
    if fault == 'view_replaced': env._view_manager._physics_sim_view = NS(is_valid=True)
    if fault == 'mode_changed':
        env.sim.cfg.create_stage_in_memory = True
        interface.attached_stage = 73
    with pytest.raises(AssertionError, match='stage/lifecycle mismatch'):
        with contact.capture_substeps(env, interface, capture, runtime):
            pytest.fail('bad lifecycle must fail before subscription or physics')
    assert interface.simulations == 0 and interface.callback is None
    assert journal[-1]['errors'] and not journal[-1]['runtime_checked']
    assert 'record_post_physics_decimation_step' not in vars(env.recorder_manager)


@pytest.mark.parametrize('fault', ['usd_change', 'view_invalidated', 'view_replaced', 'paused'])
def test_lifecycle_change_during_physics_fails_at_first_post_physics_hook(contact, fault):
    env, interface, _ = stepping_env()
    capture, journal = make_capture(contact)
    fetch = interface.fetch_results
    def changed_fetch():
        fetch()
        if fault == 'usd_change': env._usd_context_stage_id = 74
        if fault == 'view_invalidated': env.sim.physics_sim_view._backend.is_valid = False
        if fault == 'view_replaced': env._view_manager._physics_sim_view = NS(is_valid=True)
        if fault == 'paused': env.sim.playing = False
    interface.fetch_results = changed_fetch
    with pytest.raises(AssertionError, match='stage/lifecycle mismatch'):
        with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)):
            env.step(torch.zeros(2, 12))
    assert interface.simulations == 1 and not capture.poll_steps
    assert interface.callback is None and journal[-1]['errors']


def test_archived_zero_attachment_does_not_alias_contact_header_stage(contact):
    archived = ROOT / 'docs/lite3-source-shared-bank-2env-contact-repair-20261028/result.shared.jsonl'
    observed, = [r for r in map(json.loads, archived.read_text().splitlines()) if r['phase'] == 'contact_runtime']
    assert observed == dict(phase='contact_runtime', stage_id=9223001, attached_stage=0,
                            disable_contact_processing=False)
    env, interface, _ = stepping_env()
    env.sim.stage.stage_id = env.scene.stage_id = env._usd_context_stage_id = observed['stage_id']
    capture, journal = make_capture(contact)
    capture.stage_id = observed['stage_id']
    capture.runtime(interface, runtime_reader(contact, env))
    assert capture.runtime_checked and journal[-1]['attached_stage'] == 0
    for wrong_id in (0, observed['stage_id'] + 1):
        headers, data = batch()
        headers[0].stage_id = wrong_id
        with pytest.raises(ValueError, match='header stage mismatch'):
            capture.consume('poll', headers, data, 13)
    assert not capture.ground['poll']
    with pytest.raises(AssertionError, match='incomplete'):
        capture.finish()


def test_result_requires_verified_v2_runtime_evidence(contact):
    env, interface, _ = stepping_env()
    capture, _ = make_capture(contact)
    with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)):
        for i in range(20):
            env.step(torch.zeros(2, 12))
            capture.check_control(env, i + 1)
    summary = capture.summary()
    assert contact.contact_summary_passed(summary, {0: 1, 1: 1})
    for key in summary['runtime_state']:
        bad = copy.deepcopy(summary)
        del bad['runtime_state'][key]
        assert not contact.contact_summary_passed(bad, {0: 1, 1: 1}), key
    for key, value in [('schema', 'shared_contact_capture_v1'), ('runtime_check_count', 80),
                       ('runtime_state', None), ('runtime_check_count', True)]:
        assert not contact.contact_summary_passed(dict(summary, **{key: value}), {0: 1, 1: 1})


@pytest.mark.parametrize('in_memory', [False, True])
def test_pinned_lab_attachment_and_simulation_manager_lifecycle(contact, monkeypatch, in_memory):
    """Actual Python setup bodies with CPU API doubles, not a real PhysX test."""
    env, interface, _ = stepping_env()
    env.sim.cfg.create_stage_in_memory = in_memory
    setup_attaches = []
    def setup_attach(stage_id):
        setup_attaches.append(stage_id)
        interface.attached_stage = stage_id
    interface.attach_stage = setup_attach
    manager = env._view_manager
    usd = NS(get_context=lambda: NS(get_stage=lambda: env.sim.stage))
    physx = NS(get_physx_simulation_interface=lambda: interface)
    settings = NS(get_settings=lambda: NS(get=lambda key: False))
    for name, module in {
        'carb': NS(settings=settings), 'omni': NS(physx=physx, usd=usd), 'omni.physx': physx, 'omni.usd': usd,
        'isaacsim.core.simulation_manager': NS(SimulationManager=manager),
        'isaaclab.sim.simulation_context': NS(SimulationContext=NS()),
    }.items(): monkeypatch.setitem(sys.modules, name, module)
    attach = pinned_method('/sim/utils/stage.py', None, 'attach_stage_to_usd_context', {
        'get_isaac_sim_version': lambda: NS(major=5), 'is_current_stage_in_memory': lambda: in_memory,
        'get_current_stage_id': lambda: env.sim.stage.stage_id,
    })
    attach()
    assert setup_attaches == ([73] if in_memory else [])

    created, updates, dispatched = [], [], []
    def create_view(backend, stage_id):
        created.append((backend, stage_id))
        view = type(env.sim.physics_sim_view)()
        view._backend = NS(is_valid=True)
        view.set_subspace_roots = lambda roots: None
        return view
    manager._backend = 'torch'
    manager.get_physics_sim_device = lambda: 'cpu'
    manager.get_backend = lambda: 'torch'
    manager.get_physics_dt = lambda: .005
    manager._physx_interface = NS(update_simulation=lambda dt, t: updates.append((dt, t)))
    manager._message_bus = NS(dispatch_event=lambda name, payload: dispatched.append(name))
    create = pinned_method('/simulation_manager/impl/simulation_manager.py', 'SimulationManager',
                           '_create_simulation_view', {
        'SimulationManager': manager, 'get_current_stage_id': lambda: env.sim.stage.stage_id,
        'omni': NS(physics=NS(tensors=NS(create_simulation_view=create_view))),
        'IsaacEvents': NS(SIMULATION_VIEW_CREATED=NS(value='view_created'), PHYSICS_READY=NS(value='ready')),
    })
    create(None)
    assert created == [('torch', 73), ('warp', 73)]
    assert updates == [(.005, 0.)] and dispatched == ['view_created', 'ready']
    assert interface.attached_stage == (73 if in_memory else 0)
    # From here on the diagnostic may only observe the established lifecycle.
    interface.attach_stage = lambda stage_id: pytest.fail('diagnostic reattached stage')
    capture, _ = make_capture(contact)
    with contact.capture_substeps(env, interface, capture, runtime_reader(contact, env)):
        for i in range(20):
            env.step(torch.zeros(2, 12))
            capture.check_control(env, i + 1)
    assert contact.contact_summary_passed(capture.summary(), {0: 1, 1: 1})
