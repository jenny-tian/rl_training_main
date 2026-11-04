"""G3 frozen policy collection shared by real probe and explicit CPU fixtures."""
from dataclasses import asdict
import hashlib
import json
import time

import torch

from rl_training.source_scan.episode_metrics import PreResetMetricCapture,native_metric_rows
from rl_training.source_scan.moving_diagnostic import tensor_record
from rl_training.source_scan.scale_motion import contract_for
from source_eight_env_moving_probe import serial_terminal
from source_shared_contact_report import capture_substeps
from source_shared_contact_report import ContactCapture
from source_sparse_tensor_contact import SparseTensorContactCapture


def actor_digest(actor):
    digest=hashlib.sha256()
    for key,value in sorted(actor.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class MotionContactCapture(SparseTensorContactCapture):
    def __init__(self,*,fast_contacts=False,**kwargs):
        if kwargs.get('num_envs')!=120:raise ValueError('G3 contact requires 120 environments')
        super().__init__(require_ground_positive=True,**kwargs)
        self._checked_assignments=None
        self._immutable_contact_view=None
        self.fast_contacts=fast_contacts
        self.fast_times=dict(native_snapshot=0.,encode=0.,journal_record=0.)

    def runtime(self,interface,get_runtime):
        super().runtime(interface,get_runtime)
        if self._immutable_contact_view is None:self._immutable_contact_view=self.view
        elif self.view is not self._immutable_contact_view:raise RuntimeError('G3 native contact view was replaced')
        if self.fast_contacts:
            from rl_training.source_scan.fast_sparse_contact import FastContactEncoder
            if not isinstance(self.encoder,FastContactEncoder):self.encoder=FastContactEncoder(self.topology)

    def poll(self,env,interface,get_runtime):
        if not self.fast_contacts:return super().poll(env,interface,get_runtime)
        import numpy as np
        import os
        from rl_training.source_scan.fast_sparse_contact import snapshot
        from source_shared_tensor_contact import _copy
        from rl_training.source_scan.sparse_contact import owner
        step=env._sim_step_counter
        self.record(dict(phase='contact_poll_attempt',sim_step=step));self.runtime(interface,get_runtime)
        if type(step) is not int or step!=self.start_step+len(self.poll_steps)+1 or env.physics_dt!=self.physics_dt:
            raise ValueError('nonconsecutive fast contact poll or changed timestep')
        net=matrix=data=None
        try:
            tick=time.monotonic()
            net=snapshot(self.view.get_net_contact_forces(self.physics_dt))
            matrix=snapshot(self.view.get_contact_force_matrix(self.physics_dt))
            data=tuple(snapshot(v) for v in self.view.get_contact_data(self.physics_dt))
            self.fast_times['native_snapshot']+=time.monotonic()-tick;tick=time.monotonic()
            frame=self.encoder.encode_tensors(net,matrix,data,step=step)
            self.fast_times['encode']+=time.monotonic()-tick;tick=time.monotonic()
            self.record(dict(phase='sparse_contact_sample',**frame))
            self.fast_times['journal_record']+=time.monotonic()-tick
            detail=dict(zip(frame['detail_indices'],frame['normal_forces_N']));t=self.topology
            for (i,j),count,start in zip(frame['pair_indices'],frame['pair_counts'],frame['pair_starts']):
                target=t['target_paths'][t['filter_orders'][t['sensor_filter_order'][i]][j]]
                if target in self.ground_colliders:
                    self.ground['poll'][owner(t['sensor_paths'][i])]+=sum(abs(detail[k])>1e-4 for k in range(start,start+count))
            self.peak_used=max(self.peak_used,frame['used_contact_count']);self.samples+=1;self.poll_steps.append(step)
        except Exception as error:
            if not self.fault_written:
                saved={}
                if net is not None:saved['net']=_copy(net)
                if matrix is not None:saved['matrix']=_copy(matrix)
                if data is not None:saved.update({k:_copy(v) for k,v in zip(('forces','points','normals','distances','counts','starts'),data)})
                with self.fault_path.open('xb') as stream:
                    np.savez_compressed(stream,**saved);stream.flush();os.fsync(stream.fileno())
                self.fault_written=True
                self.record(dict(phase='sparse_contact_fault',sim_step=step,error=str(error),file=self.fault_path.name,
                    available_buffers=list(saved),all_native_buffers_read=len(saved)==8,
                    sha256=hashlib.sha256(self.fault_path.read_bytes()).hexdigest()))
            raise

    def check_control(self,env,completed):
        ContactCapture.check_control(self,env,completed)
        # Native contact views have immutable filters. runtime() checks their
        # identity, stage, backend and dimensions on every physics sample.
        # Re-resolve all paths on the first control and after every reset;
        # ordinary motion cannot change the native filter mapping.
        assignments=tuple(env._source_parallel_terrain_allocator.assignments)
        if assignments!=self._checked_assignments:
            if self._topology()!=self.topology:raise ValueError('resolved G3 contact topology changed after reset')
            self._checked_assignments=assignments

    def summary(self):
        result=super().summary()|dict(diagnostic_scope='G3_motion_integrity')
        if self.fast_contacts:result['fast_contact_times_s']=dict(self.fast_times)
        return result


def collect(env,wrapped,actor,observer,capture,journal,writer,report,interface,runtime,*,cpu_controls=None,
        app_running=lambda:True,on_ready=lambda:None):
    contract=report if report.get('delivery_stage') else contract_for()
    total=contract['control_steps'] if cpu_controls is None else cpu_controls
    main=contract['main_control_steps'] if cpu_controls is None else total-1
    if cpu_controls is not None and report.get('synthetic_cpu_fixture') is not True:raise PermissionError('CPU profile is fixture-only')
    if actor.training or any(v.requires_grad for v in actor.parameters()):raise ValueError('G3 actor must be frozen')
    fingerprint=actor_digest(actor);obs=wrapped.get_observations();env.observation_manager.compute(update_history=False)
    report.update(status='running',simulation_control_steps=0,completed_diagnostic_transitions=0,
        initial_assignments=[asdict(t) for t in observer.tickets],start_sim_step=env._sim_step_counter,
        terrain_origins=tensor_record(env.scene.terrain.terrain_origins),default_root_state=tensor_record(env.scene['robot'].data.default_root_state),
        initial_positions_w=tensor_record(observer.previous[0]),initial_velocities_w=tensor_record(observer.previous[1]),
        actor_sha256_before=fingerprint,flat_completion_witnesses=[],reset_snapshot_inventory={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in observer.memory().items()})
    journal.record(dict(phase='g3_initial',scanners=observer.scans()))
    generator=torch.Generator(device=env.device).manual_seed(contract['noise_seed'])
    def metric_rows(base):
        rows=native_metric_rows(base);state=wrapped.runtime.safety_state(wrapped.runtime.zeros)
        stable=~wrapped.runtime.lifecycle.unsafe & ~state.nonfoot_contact[:,0]
        stable=stable.cpu().tolist();hold=wrapped.runtime.lifecycle.hold.cpu().tolist()
        unsafe=wrapped.runtime.lifecycle.unsafe.cpu().tolist();nonfoot=state.nonfoot_contact[:,0].cpu().tolist()
        for i,row in enumerate(rows):row.update(stable_for_completion=stable[i],
            lifecycle_hold=hold[i],unsafe=unsafe[i],nonfoot_contact=nonfoot[i])
        return rows
    metrics=PreResetMetricCapture(env,metric_rows);latencies=[];main_pending=set()
    original_update=observer.after_scene_update;original_reset=observer.reset
    def updated():
        result=original_update();main_pending.clear();return result
    def reset(original,ids):
        result=original_reset(original,ids)
        if observer.control_step<=main:main_pending.update(torch.as_tensor(ids).cpu().tolist())
        return result
    observer.after_scene_update=updated;observer.reset=reset
    try:
        with torch.inference_mode(),capture_substeps(env,interface,capture,runtime),observer.installed(),metrics.installed():
            on_ready()
            collection_started=time.monotonic()
            for step in range(1,total+1):
                if not app_running():raise RuntimeError('simulator exited before G3 completed')
                observer.control_step=step
                commands=env.command_manager.get_command('base_velocity')
                if not torch.equal(commands,commands.new_tensor(contract['command_mps']).expand_as(commands)):
                    raise ValueError('G3 command differs from fixed range')
                columns=env.scene.terrain.terrain_types.cpu().tolist()
                policy=obs['policy'].clone();mu=actor(obs);noise=torch.randn(mu.shape,device=env.device,generator=generator)
                raw=mu+noise;metrics.begin_control();started=time.monotonic()
                obs,reward,done,extras=wrapped.step(raw)
                report['simulation_control_steps']+=1;report['completed_diagnostic_transitions']+=env.num_envs
                latencies.append(time.monotonic()-started);capture.check_control(env,step)
                rows=metrics.collect_control(done);terminal=serial_terminal(extras['source_terminal'])
                if terminal is not None and step<=main:
                    for j,i in enumerate(terminal['ids']):
                        if columns[i]==0 and terminal['completed'][j] and rows[i]['forward_progress_m']>=9. and terminal['hold'][j]>=50:
                            report['flat_completion_witnesses'].append(dict(control_step=step,env_id=i,episode_steps=terminal['steps'][j]))
                journal.record(dict(phase='g3_control_scans',step=step,sim_step=env._sim_step_counter,
                    roots=tensor_record(env.scene['robot'].data.root_link_pos_w),scanners=observer.scans()))
                journal.flush()
                row=dict(step=step,policy_observation=tensor_record(policy),actor_mean=tensor_record(mu),noise=tensor_record(noise),
                    raw_action=raw.cpu().tolist(),executed_action=wrapped.last_trace.executed_action.cpu().tolist(),
                    projected=wrapped.last_trace.decision.projected.cpu().tolist(),paused=wrapped.last_trace.decision.paused.cpu().tolist(),
                    done=done.cpu().tolist(),time_outs=extras['time_outs'].cpu().tolist(),terminal=terminal,columns_before=columns,
                    episode_counters=env.episode_length_buf.cpu().tolist(),metric_rows=rows)
                if writer.paths['steps'].stat().st_size+len(json.dumps(row).encode())+1024>contract['steps_byte_limit']:
                    raise RuntimeError('G3 steps evidence ceiling reached')
                writer.append_step(row)
                if step%32==0:print(f'[scale-g3] controls={step}/{total} collection_s={time.monotonic()-collection_started:.3f}',flush=True)
                if step%32==0 and getattr(capture,'fast_contacts',False):print('[scale-g3] fast_contact_times_s='+json.dumps(capture.fast_times),flush=True)
                if contract.get('delivery_stage') and step==64:
                    projected=(time.monotonic()-collection_started)*total/step
                    report['projected_collection_seconds_at_64']=projected
                    report['projected_total_seconds_at_64']=projected+report.get('startup_elapsed_s',120)+30
                    if report['projected_total_seconds_at_64']>=contract['wall_timeout_s']:
                        raise TimeoutError('G3 prefix speed cannot meet approved wall budget')
    finally:
        observer.after_scene_update=original_update;observer.reset=original_reset
        report.update(contact_capture=capture.summary(),motion_samples=observer.samples,reset_attempts=observer.reset_attempts,
            reset_batches=observer.resets,pending_recovery_ids=sorted(observer.pending_recovery),pending_main_recovery_ids=sorted(main_pending),
            all_assignment_tickets=[asdict(t) for t in observer.tickets],reset_latency_s=list(observer.reset_latency_s),
            recovery_control_latency_s=latencies,actor_sha256_after=actor_digest(actor),
            max_body_speed_mps=observer.max_speed,max_body_displacement_m=observer.max_displacement,
            max_finite_difference_acceleration_mps2=observer.max_acceleration)
    if report['actor_sha256_after']!=fingerprint:raise RuntimeError('G3 actor changed')
    report['status']='collected'
