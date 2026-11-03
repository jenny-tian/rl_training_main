"""Explicitly synthetic CPU state with the installed Lab step and PhysX getters."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace as NS,MethodType

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/tools'))
from moving_cpu_fixture import environment
from test_shared_contact_report import pinned_method,GROUND

BODY_NAMES=['TORSO','FL_HIP','FR_HIP','HL_HIP','HR_HIP','FL_THIGH','FR_THIGH','HL_THIGH','HR_THIGH',
    'FL_SHANK','FR_SHANK','HL_SHANK','HR_SHANK','FL_FOOT','FR_FOOT','HL_FOOT','HR_FOOT']


def make_environment(num_envs=16):
    env,interface,events,wrapped=environment(num_envs=num_envs)
    robot=env.scene['robot'];robot.body_names=BODY_NAMES
    env.scene.sensors['contact_forces'].body_names=BODY_NAMES
    robot.data.root_lin_vel_b=torch.zeros(num_envs,3)
    robot.data.projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(num_envs,1)
    original_update=env.scene.update
    env.scene.update=lambda dt:original_update(dt) if dt else None
    env.sim.forward=lambda:None
    return env,interface,events,wrapped


def payload(sensors,filters,capacity,*,empty=False):
    n,f=len(sensors),len(filters[0])
    matrix=np.zeros((n,f,3),np.float32);counts=np.zeros((n,f),np.uint32);starts=np.zeros_like(counts)
    forces=np.full((capacity,1),np.nan,np.float32);points=np.full((capacity,3),np.nan,np.float32)
    normals=np.full_like(points,np.nan);distances=np.full_like(forces,np.nan)
    index=0
    for i,sensor in enumerate(sensors):
        if sensor.endswith('/HR_FOOT') and not empty:
            j=filters[i].index(GROUND);matrix[i,j,2]=60;counts[i,j]=1;starts[i,j]=index
            forces[index]=60;points[index]=[0,0,0];normals[index]=[0,0,1];distances[index]=-.001;index+=1
    return matrix.sum(axis=1),matrix,(forces,points,normals,distances,counts,starts)


def install_tensor_backend(env,interface,*,fault=None):
    class Frontend:
        def create_tensor(self,shape,dtype):
            value=np.zeros(shape,dtype=dtype);return value,value
    class View:pass
    for name in ('sensor_count','filter_count','max_contact_data_count','sensor_paths','filter_paths'):
        setattr(View,name,property(pinned_method('/tensors/impl/api.py','RigidContactView',name)))
    for name in ('__init__','get_net_contact_forces','get_contact_force_matrix','get_contact_data'):
        setattr(View,name,pinned_method('/tensors/impl/api.py','RigidContactView',name,{'float32':np.float32,'uint32':np.uint32}))
    class Backend:
        is_valid=True
        def create_rigid_contact_view(self,sensors,filters,capacity):
            self.sensor_paths=sensors;self.filter_paths=filters
            self.sensor_count=len(sensors);self.filter_count=len(filters[0]);self.max_contact_data_count=capacity
            self.step=None;return self
        def current(self):
            if self.step!=interface.simulations:
                self.step=interface.simulations
                self.buffers=payload(self.sensor_paths,self.filter_paths,self.max_contact_data_count,empty=fault=='empty')
                if fault=='cross':
                    i=0;j=self.filter_paths[i].index(next(path for path in self.sensor_paths if '/env_15/' in path))
                    self.buffers[1][i,j,2]=1e-8
                if fault=='nonfinite':self.buffers[1][0,0,0]=np.nan
            return self.buffers
        def get_net_contact_forces(self,out,dt):
            assert interface.ready and dt==.005
            out[:]=self.current()[0];return True
        def get_contact_force_matrix(self,out,dt):out[:]=self.current()[1];return fault!='backend_failure'
        def get_contact_data(self,*args):
            for target,source in zip(args[:6],self.current()[2]):target[:]=source
            return True
    backend=Backend();physics=env.sim.physics_sim_view
    physics._frontend,physics._backend=Frontend(),backend
    physics.create_rigid_contact_view=MethodType(pinned_method('/tensors/impl/api.py','SimulationView','create_rigid_contact_view',{'RigidContactView':View}),physics)
    from source_shared_contact_report import ContactRuntime
    runtime=ContactRuntime(env,stage_id_for=lambda s:s.stage_id,usd_context_stage_id=lambda:73,
        get_disabled=lambda:False,get_backend_state=lambda:dict(suppress_readback_setting=True,readback_suppressed=True,tensor_device='cuda:0'))
    return runtime
