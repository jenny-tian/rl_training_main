"""Explicitly synthetic eight-env CPU environment using the pinned Lab step body."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from test_shared_contact_report import stepping_env
from rl_training.online_adaptation.types import SafetyState
from rl_training.source_scan.shared_terrain_bank import SharedBankTerrainAllocator
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.training_env import SourceTrainingRuntime, SourceTrainingVecEnv
from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter, finalize_and_close
from rl_training.source_scan.moving_diagnostic import SCHEMA, TIMEOUT_PROBES, EventJournal, MovingObserver, tensor_record, digest_file


class Scene(NS):
    def __getitem__(self, name):
        return getattr(self, name)


def environment():
    env, interface, events = stepping_env(num_envs=8)
    env.recorder_manager.record_pre_reset=lambda ids:None
    env.recorder_manager.record_post_reset=lambda ids:None
    env.episode_length_buf = torch.zeros(8, dtype=torch.long)
    old_scene = env.scene
    grid = torch.zeros(1, 15, 3)
    grid[0, :, 1] = torch.arange(15)*4.
    allocator = SharedBankTerrainAllocator(8, SOURCE_RESET_BLOCK_COUNTS, 131001, bank_rows=1)
    initial = allocator.reset(list(range(8)))
    columns = torch.tensor([a.column for a in initial])
    terrain = NS(terrain_origins=grid, terrain_levels=torch.zeros(8, dtype=torch.long),
                 terrain_types=columns, env_origins=grid[0, columns].clone())
    pos = terrain.env_origins[:, None, :].repeat(1, 17, 1);pos[:, :, 2] += .4
    quat = torch.zeros(8, 17, 4);quat[:, :, 0] = 1
    data = NS(body_link_pos_w=pos, body_link_lin_vel_w=torch.full((8, 17, 3), 0.),
              body_link_quat_w=quat, joint_pos=torch.zeros(8, 12), joint_vel=torch.zeros(8, 12),
              root_state_w=torch.zeros(8, 13))
    data.root_link_pos_w = data.body_link_pos_w[:, 0]
    data.root_state_w[:, :3] = data.root_link_pos_w
    data.root_state_w[:, 3] = 1
    data.default_root_state = torch.zeros(8,13)
    data.default_root_state[:,2]=.4;data.default_root_state[:,3]=1.
    env.scene = Scene(terrain=terrain, robot=NS(data=data), sensors={}, stage=old_scene.stage,
                      stage_id=old_scene.stage_id, write_data_to_sim=old_scene.write_data_to_sim)
    env._source_parallel_terrain_allocator = allocator
    env.cfg.scene = NS(terrain=NS(terrain_generator=NS(seed=30000, sub_terrains=dict.fromkeys(SOURCE_RESET_BLOCK_COUNTS))))
    env.cfg.source_training_run=1;env.cfg.source_terrain_level='L1';env.cfg.seed=130001
    env.cfg.source_parallel_sampler_seed=131001
    env.action_manager.action = torch.zeros(8, 12)
    env.action_manager.prev_action = torch.zeros(8, 12)
    def process_action(action):
        env.action_manager.prev_action.copy_(env.action_manager.action)
        env.action_manager.action.copy_(action)
    env.action_manager.process_action = process_action
    command = torch.tensor([[.3, 0., 0.]]).repeat(8, 1)
    env.command_manager.active_terms=['base_velocity']
    env.command_manager.get_command=lambda name: command
    env.observation_manager._group_obs_term_history_buffer={}

    class Sensor:
        def __init__(self, rays, period):
            self.rays=rays;self.period=period
            self._timestamp=torch.zeros(8, dtype=torch.float64)
            self._timestamp_last_update=torch.zeros_like(self._timestamp)
            self._is_outdated=torch.ones(8, dtype=torch.bool)
            if rays:
                self._data=NS(ray_hits_w=torch.zeros(8, rays, 3), ray_mesh_ids=torch.zeros(8, rays, 1, dtype=torch.int16))
                self._mesh_ids_wp=NS(numpy=lambda: np.full((8,1),17,dtype=np.uint64))
                self.meshes={'/World/ground':NS(id=17)}
            else:self._data=NS(net_forces_w=torch.zeros(8,17,3))
        @property
        def data(self):
            ids=self._is_outdated.nonzero().flatten()
            if self.rays:
                self._data.ray_hits_w[ids] = data.root_link_pos_w[ids, None, :]
                self._data.ray_hits_w[ids, :, 2] = 0.
            self._timestamp_last_update[ids]=self._timestamp[ids]
            self._is_outdated[ids]=False
            return self._data
        def reset(self, ids):
            self._timestamp[ids]=0;self._timestamp_last_update[ids]=0;self._is_outdated[ids]=True
            for value in vars(self._data).values():value[ids]=0
        def update(self, dt):
            self._timestamp += dt
            self._is_outdated |= self._timestamp-self._timestamp_last_update >= self.period-1e-9
    env.scene.sensors={name:Sensor(rays,period) for name,rays,period in (
        ('height_scanner',345,.02),('height_scanner_base',9,.005),('contact_forces',0,0.))}
    def update(dt):
        old_scene.update(dt)
        data.body_link_pos_w[:, :, 0] += .1*dt
        data.body_link_lin_vel_w.zero_();data.body_link_lin_vel_w[:, :, 0]=.1
        data.root_state_w[:, :3]=data.root_link_pos_w
        for sensor in env.scene.sensors.values():sensor.update(dt)
        env.scene.sensors['contact_forces']._data.net_forces_w[:,16,2]=60.
    env.scene.update=update
    def observations(update_history=False):
        for name in ('height_scanner','height_scanner_base'):env.scene.sensors[name].data
        policy=torch.zeros(8,45);policy[:,-12:]=env.action_manager.action
        critic=torch.zeros(8,393);critic[:,:3]=data.root_link_pos_w
        return {'policy':policy,'critic':critic}
    env.observation_manager.compute=observations
    callbacks=[]
    env.add_pre_reset_callback=callbacks.append
    def safety_state(timeout):
        return SafetyState(projected_gravity_xy=torch.zeros(8,2), base_height=torch.full((8,1),.4),
            joint_pos=data.joint_pos,joint_vel=data.joint_vel,joint_lower=torch.full((8,12),-10.),
            joint_upper=torch.full((8,12),10.),default_joint_pos=torch.zeros(8,12),
            action_scale=torch.ones(1,12),nonfoot_contact=torch.zeros(8,1,dtype=torch.bool),control_timed_out=timeout)
    runtime=SourceTrainingRuntime(env,NS(safety_state=safety_state,contact_sensor=env.scene.sensors['contact_forces']))
    terms={}
    env.termination_manager.active_terms=['source_training_safety','source_training_completion','source_training_timeout',
                                          'source_terrain_tile_bounds','illegal_contact','bad_orientation_2']
    env.termination_manager.get_term=lambda name: terms[name]
    def compute():
        life=runtime.evaluate()
        terms.update(source_training_safety=life.unsafe.clone(),source_training_completion=life.completed.clone(),
                     source_training_timeout=life.timeout.clone(),source_terrain_tile_bounds=torch.zeros(8,dtype=torch.bool),
                     illegal_contact=torch.zeros(8,dtype=torch.bool),bad_orientation_2=torch.zeros(8,dtype=torch.bool))
        if env.common_step_counter == 13:terms['illegal_contact'][3]=True
        env.termination_manager.terminated=torch.stack([v for k,v in terms.items() if k!='source_training_timeout']).any(dim=0)
        env.termination_manager.time_outs=terms['source_training_timeout'].clone()
        return env.termination_manager.terminated|env.termination_manager.time_outs
    env.termination_manager.compute=compute
    def reset(ids):
        for callback in callbacks:callback(ids)
        current=deepcopy(env._source_parallel_terrain_allocator);current.reset(ids.tolist())
        env._source_parallel_terrain_allocator=current
        terrain.terrain_types[ids]=torch.tensor([current.assignments[i].column for i in ids.tolist()])
        terrain.env_origins[ids]=grid[terrain.terrain_levels[ids],terrain.terrain_types[ids]]
        data.body_link_pos_w[ids]=terrain.env_origins[ids,None,:];data.body_link_pos_w[ids,:,2]+=.4
        data.body_link_lin_vel_w[ids]=0;data.root_state_w[ids,:3]=data.root_link_pos_w[ids]
        data.joint_pos[ids]=0;data.joint_vel[ids]=0
        env.episode_length_buf[ids]=0;env.action_manager.action[ids]=0;env.action_manager.prev_action[ids]=0
        for sensor in env.scene.sensors.values():sensor.reset(ids)
    env._reset_idx=reset
    class Wrapped:
        unwrapped=env;device='cpu';num_envs=8;num_actions=12;clip_actions=None;cfg=env.cfg
        def get_observations(self):
            from tensordict import TensorDict
            return TensorDict(observations(),batch_size=[8])
        def step(self, actions):
            obs,reward,terminated,truncated,extras=env.step(actions)
            from tensordict import TensorDict
            return TensorDict(obs,batch_size=[8]),reward,(terminated|truncated).long(),{**extras,'time_outs':truncated}
    wrapped=SourceTrainingVecEnv(Wrapped(),runtime=runtime)
    return env,interface,events,wrapped


def build_artifacts(folder):
    from source_shared_contact_report import ContactRuntime, capture_substeps
    from source_shared_tensor_contact import TensorContactCapture
    from source_parallel_reset_lifecycle import snapshot, check_transition
    from source_eight_env_moving_probe import serial_terminal
    env,interface,events,wrapped=environment()
    bodies=sorted(f'/World/envs/env_{i}/Robot/B{b:02d}' for i in range(8) for b in range(17))
    ground='/World/ground/terrain'
    filters=[bodies+[ground] for _ in bodies]
    capacity=65536
    matrix=np.zeros((136,137,3),np.float32);counts=np.zeros((136,137),np.uint32);starts=np.zeros_like(counts)
    forces=np.full((capacity,1),np.nan,np.float32);points=np.full((capacity,3),np.nan,np.float32)
    normals=points.copy();distances=forces.copy()
    for i in range(8):
        sensor=bodies.index(f'/World/envs/env_{i}/Robot/B16')
        counts[sensor,-1]=1;starts[sensor,-1]=i;matrix[sensor,-1,2]=60
        forces[i]=60;points[i]=0;normals[i]=[0,0,1];distances[i]=-.001
    def read(name,value):
        events.append(name)
        return value
    view=NS(sensor_paths=bodies,filter_paths=filters,sensor_count=136,filter_count=137,max_contact_data_count=capacity,
            get_net_contact_forces=lambda dt: read('tensor_net',matrix.sum(axis=1)),
            get_contact_force_matrix=lambda dt:read('tensor_matrix',matrix),
            get_contact_data=lambda dt:read('tensor_data',(forces,points,normals,distances,counts,starts)))
    env.sim.physics_sim_view.create_rigid_contact_view=lambda *args,**kw:view
    runtime=ContactRuntime(env,stage_id_for=lambda stage:stage.stage_id,usd_context_stage_id=lambda:73,
        get_disabled=lambda:False,get_backend_state=lambda:{'suppress_readback_setting':True,
            'readback_suppressed':True,'tensor_device':'cuda:0'})
    output=folder/'result.json';event_path=output.with_suffix('.events.jsonl')
    report={'attempt_id':'cpu-eight','base_revision':'a'*40,'checkpoint_sha256':'b'*64,'status':'starting',
        'num_envs':8,'shared_bank_rows':1,'control_steps_limit':96,'simulation_control_steps':0,'ppo_updates':0,
        'training_ready':False,'g3_g4_qualified':False,'diagnostic_only':True,'critic_width':393,
        'geometry_seed':30000,'sampler_seed':131001,'ppo_seed':130001,'noise_rng_seed':132001,
        'reset_state_id':40000,'reset_randomized':False,'stochastic_std':1.0,'body_names':[f'B{i:02d}' for i in range(17)],
        'synthetic_cpu_fixture':True}
    writer=SmokeArtifactWriter(output,report,schema=SCHEMA)
    with EventJournal(event_path) as journal:
        original_record=journal.record
        def record(value):
            if value.get('phase')=='motion_substep':events.append('motion')
            original_record(value)
        journal.record=record
        observer=MovingObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        capture=TensorContactCapture(physics_view=env.sim.physics_sim_view,ground_colliders=[ground],physics_dt=.005,
            num_envs=8,capacity=65536,stage_id=73,reporting_bodies=bodies,record=journal.record,
            start_step=env._sim_step_counter,decimation=4,control_steps=96)
        report.update(initial_assignments=observer.initial_assignments,start_sim_step=env._sim_step_counter,
            terrain_origins=tensor_record(env.scene.terrain.terrain_origins),initial_positions_w=tensor_record(observer.previous[0]),
            initial_velocities_w=tensor_record(observer.previous[1]))
        journal.record({'phase':'initial','synthetic_cpu_fixture':True})
        with capture_substeps(env,interface,capture,runtime),observer.installed():
            for step in range(1,97):
                forced=TIMEOUT_PROBES.get(step,[])
                if forced:env.episode_length_buf[forced]=1999
                before=env.scene.terrain.terrain_types.clone()
                with torch.inference_mode():obs,rew,done,extras=wrapped.step(torch.zeros(8,12))
                capture.check_control(env,step)
                journal.record({'phase':'control_scans','step':step,'sim_step':env._sim_step_counter,
                    'root_positions_w':env.scene['robot'].data.root_link_pos_w.tolist(),
                    'origins_w':env.scene.terrain.env_origins.tolist(),'scanners':observer.scans(('height_scanner','height_scanner_base'))})
                journal.flush()
                writer.append_step({'step':step,'forced_ids':forced,'done':done.tolist(),'time_outs':extras['time_outs'].tolist(),
                    'terminal':serial_terminal(extras['source_terminal']),'episode_counters':env.episode_length_buf.tolist(),
                    'terrain_columns_before':before.tolist(),'terrain_columns_after':env.scene.terrain.terrain_types.tolist(),
                    'raw_action':[[0.]*12]*8,'base_action':[[0.]*12]*8,'requested_action':[[0.]*12]*8,
                    'executed_action':[[0.]*12]*8,'projected':[[False]]*8,'paused':[[False]]*8})
        report.update(motion_samples=observer.samples,reset_batches=observer.resets,
                      pending_recovery_ids=sorted(observer.pending_recovery),contact_capture=capture.summary())
    report.update(status='passed',simulation_control_steps=96,total_transitions=768,exposure=wrapped.exposure.report(),
                  events_file=event_path.name,events_sha256=digest_file(event_path))
    finalize_and_close(writer,report,environment_close=lambda:None,application_close=lambda:None)
    # Lossless archive path is exercised without a second run of the fixture.
    with event_path.open('rb') as source,gzip.open(str(event_path)+'.gz','wb') as target:
        import shutil
        shutil.copyfileobj(source,target)
    event_path.rename(event_path.with_suffix('.source.jsonl'))
    return output,events
