"""Real adapter callback and pinned sensor bookkeeping over a CPU fake backend.

The backend supplies tensors, not simulated forces. No Isaac module is imported.
"""
import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace as NS

import torch

from scale_cpu_fixture import BODY_NAMES, ROOT
from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES
from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter


FAILURE = ROOT/'docs/lite3-source-shared-bank-g2-120env-runtime-20261104/raw'


@lru_cache(maxsize=1)
def native_classes():
    plan=json.loads((FAILURE/'run_manifest.json').read_text())
    paths={name:next(Path(p) for p in plan['dependency_sha256'] if p.endswith(suffix))
        for name,suffix in [('sensor','/sensors/contact_sensor/contact_sensor.py'),('base','/sensors/sensor_base.py')]}
    for path in paths.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['dependency_sha256'][str(path)]
    def methods(path,cls,names):
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls)
        return [deepcopy(next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in names]
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    body.append(ast.ClassDef(name='SensorBase',bases=[],keywords=[],decorator_list=[],body=
        methods(paths['base'],'SensorBase',['reset','update','_update_outdated_buffers'])))
    body.append(ast.ClassDef(name='ContactSensor',bases=[ast.Name(id='SensorBase',ctx=ast.Load())],keywords=[],decorator_list=[],body=
        methods(paths['sensor'],'ContactSensor',['data','reset'])))
    env_path=ROOT/'source/rl_training/rl_training/online_adaptation/pre_reset_capture_env.py'
    body.append(ast.ClassDef(name='PreResetCaptureManagerBasedRLEnv',bases=[ast.Name(id='NativeReset',ctx=ast.Load())],
        keywords=[],decorator_list=[],body=methods(env_path,'PreResetCaptureManagerBasedRLEnv',['add_pre_reset_callback','_reset_idx'])))
    class NativeReset:
        def _reset_idx(self,ids):return self.native_reset(ids)
    namespace={'torch':torch,'NativeReset':NativeReset}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(env_path),'exec'),namespace)
    return namespace['ContactSensor'],namespace['PreResetCaptureManagerBasedRLEnv']


def contact_sensor(num_envs,*,net=None,history=None):
    sensor_class,_=native_classes();sensor=sensor_class()
    sensor.cfg=NS(filter_prim_paths_expr=[],track_air_time=False,track_contact_points=False,
        track_friction_forces=False,update_period=0.,history_length=3)
    sensor._is_visualizing=False;sensor.body_names=BODY_NAMES
    sensor._timestamp=torch.zeros(num_envs,dtype=torch.float64)
    sensor._timestamp_last_update=torch.zeros_like(sensor._timestamp)
    sensor._is_outdated=torch.ones(num_envs,dtype=torch.bool)
    if net is None:
        net=torch.zeros(num_envs,17,3);net[:,13:,2]=60.
    if history is None:history=net[:,None].repeat(1,3,1,1)
    sensor._data=NS(net_forces_w=torch.zeros_like(net),net_forces_w_history=torch.zeros_like(history))
    sensor.refresh_calls=[]
    def update(ids):
        sensor.refresh_calls.append(ids.clone())
        sensor._data.net_forces_w[ids]=net[ids];sensor._data.net_forces_w_history[ids]=history[ids]
    sensor._update_buffers_impl=update
    sensor.find_bodies=lambda pattern:([i for i,name in enumerate(BODY_NAMES) if re.fullmatch(pattern,name)],
        [name for name in BODY_NAMES if re.fullmatch(pattern,name)])
    return sensor


def install_real_reset_callback(env,wrapped):
    sensor=contact_sensor(env.num_envs);env.scene.sensors['contact_forces']=sensor
    robot=env.scene['robot'];robot.find_bodies=sensor.find_bodies
    robot.find_joints=lambda names,preserve_order: (list(range(12)),list(LITE3_POLICY_JOINT_NAMES))
    # Remove the old synthetic callback loop; the real environment method below
    # invokes the actual adapter callback before the runtime and native reset.
    env.add_pre_reset_callback.__self__.clear()
    _,capture_class=native_classes();capture=capture_class()
    capture.native_reset=env._reset_idx;capture._pre_reset_callbacks=[]
    env.add_pre_reset_callback=capture.add_pre_reset_callback;env._reset_idx=capture._reset_idx
    adapter=Lite3IsaacLabAdapter(env)
    adapter.safety_state=wrapped.runtime.adapter.safety_state
    wrapped.runtime.adapter=adapter
    env.add_pre_reset_callback(wrapped.runtime.before_reset)
    env._cpu_reset_callback_owner=capture
    return adapter
