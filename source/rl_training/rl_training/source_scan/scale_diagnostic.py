"""Scale-independent motion, scanner and reset adapters; imports no Isaac/Kit."""
from contextlib import contextmanager
from dataclasses import asdict
import time

import torch

from .moving_diagnostic import check_scan, tensor_record, fault_tensor_record
from .parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from .shared_terrain_bank import SharedBankTerrainAllocator
from .tile_bounds import source_tile_bounds_terminated

SCALES = (16,32,60,120)


def reset_batches(num_envs):
    if num_envs not in SCALES: raise ValueError('unregistered reset scale')
    return [[0],[1,min(17,num_envs-2),num_envs-1],list(range(0,num_envs,2)),list(range(num_envs))]


def kinematic_sample(env):
    data,terrain = env.scene['robot'].data,env.scene.terrain
    position,velocity,quat = data.body_link_pos_w,data.body_link_lin_vel_w,data.body_link_quat_w
    n=env.num_envs
    if position.shape!=(n,17,3) or velocity.shape!=position.shape or quat.shape!=(n,17,4):
        raise ValueError('all 17 link poses and velocities required')
    if any(not bool(torch.isfinite(value).all()) for value in (position,velocity,quat)):
        raise FloatingPointError('nonfinite link kinematics')
    if not torch.allclose(quat.norm(dim=-1),torch.ones((n,17),device=position.device),atol=1e-4,rtol=0):
        raise ValueError('invalid link quaternion')
    diagnostic={}
    source_tile_bounds_terminated(position,velocity,terrain.terrain_origins,terrain.terrain_levels,terrain.terrain_types,
        size=(12.,4.),flat_length_m=3.,collision_radius_m=.27,numerical_margin_m=.001,
        control_period_s=.02,diagnostics=diagnostic)
    if not bool(diagnostic['valid_indices'].all()) or bool(diagnostic['invalid_state'].any()) or bool(diagnostic['outside_now'].any()):
        raise RuntimeError('invalid assignment or actual collision envelope outside tile')
    return dict(body_positions_w=tensor_record(position),body_link_velocities_w=tensor_record(velocity),
        body_link_quaternions_wxyz=tensor_record(quat),root_positions_w=tensor_record(data.root_link_pos_w),
        origins_w=tensor_record(terrain.env_origins),terrain_levels=terrain.terrain_levels.cpu().tolist(),
        terrain_columns=terrain.terrain_types.cpu().tolist(),outside_prediction=diagnostic['outside_prediction'].cpu().tolist())


class ScaleObserver:
    def __init__(self,env,runtime,journal,snapshot,boundary_checker,*,sampler_seed=131001):
        if env.num_envs not in SCALES or env.cfg.decimation!=4 or env.physics_dt!=.005:
            raise ValueError('unregistered scale or physics/control rate')
        self.env,self.runtime,self.journal=env,runtime,journal
        self.snapshot,self.boundary_checker=snapshot,boundary_checker
        self.oracle=SharedBankTerrainAllocator(env.num_envs,SOURCE_RESET_BLOCK_COUNTS,sampler_seed,bank_rows=1)
        self.tickets=list(self.oracle.reset(list(range(env.num_envs))))
        if tuple(self.tickets)!=env._source_parallel_terrain_allocator.assignments:
            raise ValueError('initial scale assignments differ from sampler replay')
        self.start=env._sim_step_counter;self.samples=self.resets=0
        self.pending_recovery=set();self.reset_latency_s=[]

    def scans(self,names=('height_scanner','height_scanner_base')):
        env=self.env;data=env.scene['robot'].data;result={}
        for name in names:
            sensor=env.scene.sensors[name];value=sensor.data
            mesh=torch.as_tensor(sensor._mesh_ids_wp.numpy(),dtype=torch.int64,device=data.root_link_pos_w.device)
            age=sensor._timestamp-sensor._timestamp_last_update
            check_scan(data.root_link_pos_w,value.ray_hits_w,value.ray_mesh_ids,mesh,
                sensor.meshes['/World/ground'].id,env.scene.terrain.env_origins,age=age,
                max_age=1e-6 if name=='height_scanner_base' else .02,base=name=='height_scanner_base')
            result[name]=dict(hits=tensor_record(value.ray_hits_w),hit_ids=tensor_record(value.ray_mesh_ids),
                mesh_ids=tensor_record(mesh),ground_id=int(sensor.meshes['/World/ground'].id),age_s=tensor_record(age))
        return result

    def memory(self):
        state=self.snapshot(self.env)
        state.update({'filter/'+name:value.clone() for name,value in self.runtime.safety.temporal_snapshot(self.env.action_manager.action).items()})
        state.update({'lifecycle/'+name:getattr(self.runtime.lifecycle,name).clone() for name in ('hold','contact_steps','completed','unsafe','timeout')})
        return state

    def reset(self,original,env_ids):
        ids=torch.as_tensor(env_ids,device=self.env.device,dtype=torch.long)
        if not len(ids): return original(env_ids)
        selected=sorted(ids.cpu().tolist())
        if len(set(selected))!=len(selected) or selected[0]<0 or selected[-1]>=self.env.num_envs:
            raise ValueError('invalid selective reset ids')
        self.env.observation_manager.compute(update_history=False);self.scans()
        before=self.memory();prior=self.env._source_parallel_terrain_allocator.assignments
        self.journal.record({'phase':'scale_reset_attempt','ids':selected,'sim_step':self.env._sim_step_counter,
            'before':{key:tensor_record(value) for key,value in before.items()}});self.journal.flush()
        started=time.monotonic();result=original(env_ids);self.reset_latency_s.append(time.monotonic()-started)
        after=self.memory()
        self.journal.record({'phase':'scale_reset_after','ids':selected,'sim_step':self.env._sim_step_counter,
            'after':{key:fault_tensor_record(value) for key,value in after.items()}});self.journal.flush()
        self.boundary_checker(before,after,selected)
        other=torch.ones(self.env.num_envs,dtype=torch.bool,device=ids.device);other[ids]=False
        for key in before:
            if not torch.equal(before[key][other],after[key][other]): raise RuntimeError('reset changed untouched buffer: '+key)
            if key.startswith(('filter/','lifecycle/')) and bool(after[key][ids].any()): raise RuntimeError('reset memory not cleared: '+key)
        current=self.env._source_parallel_terrain_allocator.assignments;expected=self.oracle.reset(selected)
        if tuple(current[i] for i in selected)!=expected or any(current[i]!=prior[i] for i in range(self.env.num_envs) if i not in selected):
            raise RuntimeError('selective reset tickets differ from replay')
        terrain=self.env.scene.terrain;data=self.env.scene['robot'].data
        origins=terrain.terrain_origins[terrain.terrain_levels[ids],terrain.terrain_types[ids]]
        if not torch.equal(origins,terrain.env_origins[ids]) or not torch.allclose(data.root_link_pos_w[ids],data.default_root_state[ids,:3]+origins,atol=1e-5,rtol=0):
            raise RuntimeError('selective reset spawn origin differs')
        self.tickets.extend(expected);self.resets+=1;self.pending_recovery.update(selected)
        self.journal.record({'phase':'scale_reset_checked','ids':selected,'batch':self.resets,
            'sim_step':self.env._sim_step_counter,'assignments':[asdict(t) for t in current],
            'after':{key:tensor_record(value) for key,value in after.items()}});self.journal.flush()
        return result

    def after_scene_update(self):
        if self.env._sim_step_counter!=self.start+self.samples+1: raise RuntimeError('missing scale motion substep')
        try:sample=kinematic_sample(self.env)
        except Exception as error:
            data=self.env.scene['robot'].data
            self.journal.record(dict(phase='scale_motion_fault',sim_step=self.env._sim_step_counter,error=repr(error),
                body_positions_w=fault_tensor_record(data.body_link_pos_w),body_link_velocities_w=fault_tensor_record(data.body_link_lin_vel_w),
                body_link_quaternions_wxyz=fault_tensor_record(data.body_link_quat_w)))
            self.journal.flush();raise
        self.journal.record(dict(phase='scale_motion_substep',sim_step=self.env._sim_step_counter,
            recovery_ids=sorted(self.pending_recovery),scanners=self.scans(('height_scanner_base',)),**sample))
        self.journal.flush();self.pending_recovery.clear();self.samples+=1

    @contextmanager
    def installed(self):
        items=[(self.env.scene,'update'),(self.env,'_reset_idx')]
        saved=[(o,k,k in vars(o),vars(o).get(k),getattr(o,k)) for o,k in items]
        def updated(dt):
            result=saved[0][-1](dt)
            if dt==.005:self.after_scene_update()
            return result
        self.env.scene.update=updated
        self.env._reset_idx=lambda ids:self.reset(saved[1][-1],ids)
        try:yield self
        finally:
            for o,k,present,value,_ in saved:
                if present:setattr(o,k,value)
                else:delattr(o,k)

    def direct_resets(self,batches=None):
        """Caller must install hooks; this adds no physics step or training transition."""
        for ids in reset_batches(self.env.num_envs) if batches is None else batches:
            previous=self.resets
            self.env._reset_idx(torch.tensor(ids,device=self.env.device,dtype=torch.long))
            if self.resets!=previous+1:raise RuntimeError('selective reset observer was not installed')
            self.env.scene.write_data_to_sim();self.env.sim.forward();self.env.scene.update(0.)
            self.scans()
        return dict(reset_batches=self.resets,reset_latency_s=list(self.reset_latency_s),
            recovery_observed=False,pending_recovery_ids=sorted(self.pending_recovery))
