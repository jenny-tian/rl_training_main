"""CPU contract tests: installed native method bodies, mocked physics/scene.

No AppLauncher, CUDA device, or physics engine is started. These tests exercise
method lifecycle and the real collector, not robot dynamics or paired physics.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path('/home/tjx/IsaacLab/source/isaaclab/isaaclab')
sys.path.insert(0, str(ROOT/'scripts/tools'))
from reference_online_c1b import rows
from reference_online_c1b_env_repaired import Environment as FailedEnvironment
from reference_online_c1b_env_reset_api import Environment
from reference_online_interface import DEFAULT, SCALE, JOINTS, CurrentIsaacAdapter, observation, wrap_manager_compute


def native_method(file, owner, name):
    """Use installed source unchanged except decorators and deferred annotations."""
    path = NATIVE/file
    tree = ast.parse(path.read_text())
    body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner) if owner else tree.body
    method = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    namespace = {'torch':torch}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace[name]


class NativeAction:
    reset = native_method('managers/action_manager.py','ActionManager','reset')
    process_action = native_method('managers/action_manager.py','ActionManager','process_action')
    apply_action = native_method('managers/action_manager.py','ActionManager','apply_action')
    action = property(lambda self:self._action)
    get_term = lambda self,name:self._terms[name]


class NativeTermination:
    reset = native_method('managers/termination_manager.py','TerminationManager','reset')
    compute = native_method('managers/termination_manager.py','TerminationManager','compute')
    terminated = property(lambda self:self._terminated_buf)
    time_outs = property(lambda self:self._truncated_buf)


class NativeBase:
    reset = native_method('envs/manager_based_env.py','ManagerBasedEnv','reset')
    step = native_method('envs/manager_based_rl_env.py','ManagerBasedRLEnv','step')
    _reset_idx = native_method('envs/manager_based_rl_env.py','ManagerBasedRLEnv','_reset_idx')


@pytest.fixture
def factory(monkeypatch):
    # Replace only the math import needed by reset with its actual CPU body.
    math = types.ModuleType('isaaclab.utils.math')
    math.quat_from_euler_xyz = native_method('utils/math.py',None,'quat_from_euler_xyz')
    monkeypatch.setitem(sys.modules,'isaaclab.utils.math',math)
    package = types.ModuleType('_c1b_reset_contract_buffers')
    package.__path__ = [str(NATIVE/'utils/buffers')]
    monkeypatch.setitem(sys.modules,package.__name__,package)
    Buffer = importlib.import_module(package.__name__+'.delay_buffer').DelayBuffer

    def make(args=None,specs=None):
        env = Environment.__new__(Environment)
        env.args = args or NS(kind='smoke',smoke_index=2)
        env.specs = specs or rows('smoke',2)[:4]
        env.n, env.device = len(env.specs),'cpu'
        n = env.n
        env.policy_age,env.policy_active = np.zeros(n,int),np.zeros(n,bool)
        env.deferred_events = 0
        env.command = NS(is_heading_env=torch.zeros(n,dtype=torch.bool),is_standing_env=torch.zeros(n,dtype=torch.bool),
                         vel_command_b=torch.zeros(n,3))
        root = torch.zeros(n,13); root[:,2] = .35; root[:,3] = 1.
        default = torch.tensor([DEFAULT]*n)
        data = NS(default_root_state=root.clone(),root_state_w=root,root_pos_w=root[:,:3],
            default_joint_pos=default,joint_pos=default.clone(),joint_vel=torch.zeros(n,12),default_joint_vel=torch.zeros(n,12),
            root_ang_vel_b=torch.zeros(n,3),projected_gravity_b=torch.tensor([[0.,0.,-1.]]*n),
            soft_joint_pos_limits=torch.stack((default-2.,default+2.),-1),body_pos_w=torch.zeros(n,5,3))
        env.robot = NS(data=data,joint_names=JOINTS,actuators={'legs':NS(**{key:Buffer(1,n,'cpu') for key in
            ('positions_delay_buffer','velocities_delay_buffer','efforts_delay_buffer')})})
        def write_root(value,*,env_ids): data.root_state_w[env_ids] = value
        def write_joint(pos,vel,*,env_ids):
            data.joint_pos[env_ids] = pos; data.joint_vel[env_ids] = vel
        env.robot.write_root_state_to_sim,env.robot.write_joint_state_to_sim = write_root,write_joint
        forces = torch.zeros(n,5,3); forces[:,:4,2] = 2.
        env.sensor = NS(body_names=['FL_FOOT','FR_FOOT','HL_FOOT','HR_FOOT','TORSO'],data=NS(net_forces_w=forces))
        env.contact_ids = env.foot_ids = list(range(4)); env.torso_id = 4
        env.terrain = NS(env_origins=torch.zeros(n,3))
        hits = torch.zeros(n,9,3)
        env.events = []
        class Scene(dict):
            def reset(self,ids): env.events.append(('scene_reset',ids.tolist()))
            def write_data_to_sim(self): env.events.append(('write',))
            def update(self,dt):
                hits[:,:,:2] = data.root_pos_w[:,None,:2]
        scene = Scene(robot=env.robot,contact_forces=env.sensor,height_scanner_base=NS(data=NS(ray_hits_w=hits)))
        action = NativeAction()
        action.device,action.total_action_dim = 'cpu',12
        action._action,action._prev_action = torch.zeros(n,12),torch.zeros(n,12)
        term = NS(_joint_ids=list(range(12)),_scale=torch.tensor([SCALE]*n),action_dim=12,
                  reset=lambda **kw:None,process_actions=lambda value:None,apply_actions=lambda:None)
        action._terms = {'joint_pos':term}
        no_reset = lambda *a,**kw:{}
        termination = NativeTermination()
        termination._terminated_buf = torch.zeros(n,dtype=torch.bool)
        termination._truncated_buf = torch.zeros(n,dtype=torch.bool)
        termination._term_names,termination._class_term_cfgs = ['forced','timeout'],[]
        termination._term_dones = torch.zeros(n,2,dtype=torch.bool)
        termination._last_episode_dones = torch.zeros(n,2,dtype=torch.bool)
        termination.forced = torch.zeros(n,dtype=torch.bool)
        termination._term_cfgs = [NS(func=lambda base:termination.forced,params={},time_out=False),
            NS(func=lambda base:base.episode_length_buf>=1000,params={},time_out=True)]
        base = NativeBase(); termination._env = base
        base.device,base.num_envs,base.step_dt,base.physics_dt = 'cpu',n,.02,.005
        base.cfg = NS(decimation=4,sim=NS(dt=.005,render_interval=4),num_rerenders_on_reset=0,wait_for_textures=False)
        base.scene,base.action_manager,base.termination_manager = scene,action,termination
        base._sim_step_counter,base.common_step_counter = 0,0
        base.episode_length_buf,base.extras = torch.zeros(n,dtype=torch.long),{}
        base.sim = NS(has_gui=lambda:False,has_rtx_sensors=lambda:False,step=lambda **kw:None,
                      forward=lambda:env.events.append(('forward',)))
        base.recorder_manager = NS(active_terms=[],reset=no_reset,**{key:lambda *a,**kw:None for key in (
            'record_pre_step','record_post_physics_decimation_step','record_pre_reset','record_post_reset')})
        base.event_manager = NS(available_modes=[],reset=no_reset)
        base.curriculum_manager = NS(compute=lambda **kw:None,reset=no_reset)
        base.reward_manager = NS(compute=lambda **kw:torch.zeros(n),reset=no_reset)
        base.command_manager = NS(compute=lambda **kw:None,reset=no_reset,
            get_command=lambda name:env.command.vel_command_b)
        wrap_manager_compute(base.command_manager,lambda dt:env.set_command(next_step=True))
        def obs(**kw):
            return {'policy':observation(env.robot,list(range(12)),env.command.vel_command_b,action.action),
                    'critic':torch.zeros(n,393)}
        base.observation_manager = NS(compute=obs,reset=no_reset)
        env.base = base
        env.native_reset = base._reset_idx
        # First native reset has no step-owned reset_* attributes.
        env.obs,_ = base.reset()
        def defer(ids): env.deferred_events += len(ids)
        base._reset_idx = defer
        env.adapter = CurrentIsaacAdapter(base)
        env.actor = torch.nn.Linear(45,12)
        with torch.no_grad(): env.actor.weight.zero_(); env.actor.bias.zero_()
        env.actor.requires_grad_(False)
        env.geometry = {f'{s["condition"]}:{s["row"]}':dict(fronts=[1.,2.,3.],tops=[.06,.12,.18],goal_x=3.3) for s in env.specs}
        return env
    return make


def test_native_initial_reset_and_old_entry_reproduce_missing_attribute(factory):
    env = factory()
    assert not any(hasattr(env.base,key) for key in ('reset_buf','reset_terminated','reset_time_outs'))
    with pytest.raises(AttributeError,match='reset_buf'):
        FailedEnvironment.reset(env,list(range(env.n)),np.zeros(env.n,int))


def test_candidate_initial_reset_keeps_pairs_and_reaches_refresh_without_step(factory):
    env = factory()
    result = env.reset(list(range(env.n)),np.zeros(env.n,int))
    assert env.base._sim_step_counter == 0
    assert not hasattr(env.base,'reset_buf')
    assert result['keys'][0] == result['keys'][1]
    assert result['offsets'][0] == result['offsets'][1]
    assert all(values[0] == values[1] for values in result['delays'])
    env.adapter.check_observation(env.obs['policy'],torch.zeros(env.n,12))
    assert env.events[-1] == ('forward',)
    assert env.events.count(('write',)) == 1  # Only the fixture's native initial reset.


def test_partial_reset_preserves_terminal_result_other_actions_and_next_compute(factory):
    env = factory(); n = env.n
    env.reset(list(range(n)),np.zeros(n,int))
    env.base.termination_manager.forced[0] = True
    _,_,terminated,_,_ = env.base.step(torch.ones(n,12)*.1)
    captured = terminated.clone()
    assert captured.tolist() == [True,False,False,False]
    before = env.robot.data.root_state_w[2:].clone()
    action = env.base.action_manager.action[2:].clone()
    env.reset([0,1],np.ones(n,int))
    assert torch.equal(terminated,captured)  # Explicit reset does not rewrite a step result.
    assert torch.equal(env.robot.data.root_state_w[2:],before)
    assert torch.equal(env.base.action_manager.action[2:],action)
    assert not env.base.action_manager.action[:2].any()
    assert env.base.episode_length_buf.tolist() == [0,0,1,1]
    env.base.termination_manager.forced[:] = False
    _,_,next_terminated,_,_ = env.base.step(torch.zeros(n,12))
    assert not next_terminated.any() and captured[0]


@pytest.mark.parametrize('scenario',['locked_json_failure','repaired','preparation_failure'])
def test_full_collector_cpu_contract_with_native_methods(factory,monkeypatch,tmp_path,scenario):
    import reference_online_c1b_collect as collector
    from reference_online_c1b_collect_reset_api import collect
    created = []
    def constructor(args,specs):
        env = factory(args,specs); created.append(env)
        if scenario == 'preparation_failure': env.base.termination_manager.forced[0] = True
        return env
    monkeypatch.setitem(sys.modules,'reference_online_c1b_env',NS(Environment=constructor))
    synthetic_checkpoint = tmp_path/'synthetic-base.bin'
    synthetic_checkpoint.write_bytes(b'CPU contract test, not the trained base model')
    monkeypatch.setattr(collector,'MODEL_PATHS',{'B':synthetic_checkpoint})
    monkeypatch.setattr(collector,'HASHES',{'B':hashlib.sha256(synthetic_checkpoint.read_bytes()).hexdigest()})
    args = NS(kind='smoke',smoke_index=2,output=tmp_path)
    if scenario == 'locked_json_failure':
        with pytest.raises(TypeError,match='int64 is not JSON serializable'): collector.collect(args)
        assert created[0].base._sim_step_counter == 600*4
        return
    if scenario == 'preparation_failure':
        with pytest.raises(RuntimeError,match='standing preparation terminated'): collect(args)
        partial = json.loads((tmp_path/'partial_report.json').read_text())
        assert partial['control_steps'] == 1 and not partial['updates']
        clock = json.loads((tmp_path/'failure_clock.json').read_text())
        assert clock['native_physics_counter_delta'] == 4
        return
    collect(args)
    report = json.loads((tmp_path/'report.json').read_text())
    episodes = json.loads((tmp_path/'episodes.json').read_text())
    assert report['control_steps'] == 600 and report['interactions'] == 9600
    assert report['adapter_updates'] == report['base_optimizer_updates'] == 0
    assert report['complete_episodes'] == 48
    assert report['paired_preparations'] == 24 and report['paired_max_error'] == 0.
    assert episodes['episode_index'] == [2]*16
    assert report['counts']['preparation'] == [150]*16
    assert report['counts']['policy'] == [300,450]*8
    assert report['counts']['waiting'] == [150,0]*8
    assert created[0].base._sim_step_counter == 600*4


def test_cpu_train_final_checkpoint_then_frozen_target_and_retention(factory,monkeypatch,tmp_path):
    import reference_online_c1b_collect as collector
    from reference_online_c1b_collect_reset_api import collect
    # Small deterministic synthetic path through real update/checkpoint/eval
    # code. Terminate after ten policy steps; no physical capability is tested.
    def constructor(args,specs):
        env = factory(args,specs)
        env.base.termination_manager._term_cfgs[0].func = lambda base:torch.tensor(
            env.policy_active & (env.policy_age >= 9))
        return env
    monkeypatch.setitem(sys.modules,'reference_online_c1b_env',NS(Environment=constructor))
    monkeypatch.setattr(collector,'rows',lambda kind,index:rows(kind,index)[:2])
    monkeypatch.setattr(collector,'OUTPUT',tmp_path)
    synthetic_checkpoint = tmp_path/'synthetic-base.bin'
    synthetic_checkpoint.write_bytes(b'CPU contract test, not the trained base model')
    monkeypatch.setattr(collector,'MODEL_PATHS',{'B':synthetic_checkpoint})
    monkeypatch.setattr(collector,'HASHES',{'B':hashlib.sha256(synthetic_checkpoint.read_bytes()).hexdigest()})
    for kind in ('train','target','retention'):
        directory = tmp_path/kind; directory.mkdir()
        collect(NS(kind=kind,smoke_index=2,output=directory))
        report = json.loads((directory/'report.json').read_text())
        assert report['base_unchanged'] and report['frozen_unchanged']
        if kind == 'train':
            assert report['session_updates'] == [0,64]
            assert report['control_steps'] == 3840
            checkpoint = torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
            assert checkpoint['heads'][0]['parameter_sha256'] != checkpoint['heads'][1]['parameter_sha256']
        else:
            assert report['adapter_updates'] == 0 and report['complete_episodes'] == 2
            assert report['frozen_zero_equivalence_checked']
            assert report['initial_head_digests'][1] == checkpoint['heads'][1]['parameter_sha256']


def test_json_scalar_conversion_preserves_types_and_strict_invalid_value_rejection(tmp_path):
    from reference_online_c1b_collect_reset_api import json_scalars
    from reference_stairs_eval import save
    value = json_scalars({'ids':[np.int64(3)],'value':np.float32(.5),'ok':np.bool_(True)})
    assert value == {'ids':[3],'value':.5,'ok':True}
    assert type(value['ids'][0]) is int and type(value['ok']) is bool
    with pytest.raises(ValueError): save(tmp_path/'invalid.json',json_scalars({'x':np.float64(np.nan)}))
