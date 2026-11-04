"""Bounded lossless G3 journal and scale observer; imports no simulator."""
import gzip
import hashlib
import json
import os
from pathlib import Path

import torch

from .moving_diagnostic import tensor_record,fault_tensor_record
from .scale_diagnostic import ScaleObserver,kinematic_sample
from .smoke_artifacts import SmokeArtifactWriter,atomic_json
from .tensor_codec import pack,unpack


def file_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()


class MotionArtifactWriter(SmokeArtifactWriter):
    def persist(self,report):
        if report.get('attempt_id')!=self.attempt_id:raise ValueError('attempt identity changed')
        atomic_json(self.paths['result'],dict(report,schema=self.schema,steps_written=self.count,
            steps_file=self.paths['steps'].name,steps_sha256=file_digest(self.paths['steps'])))


class CompressedJournal:
    def __init__(self,path,*,byte_limit,uncompressed_limit,record_limit,binary_tensors=False):
        self.path=Path(path);self.stream=self.path.open('xb');self.pending=[];self.pending_bytes=0
        self.written=self.stored_bytes=0
        self.byte_limit,self.uncompressed_limit,self.record_limit=byte_limit,uncompressed_limit,record_limit
        self.binary_tensors=binary_tensors

    def record(self,value):
        encoded=(json.dumps(pack(value) if self.binary_tensors else value,allow_nan=False)+'\n').encode()
        if len(encoded)>self.record_limit or self.written+self.pending_bytes+len(encoded)>self.uncompressed_limit:
            raise RuntimeError('G3 uncompressed evidence ceiling reached')
        self.pending.append(encoded);self.pending_bytes+=len(encoded)

    def flush(self):
        if not self.pending:return
        payload=b''.join(self.pending);member=gzip.compress(payload,compresslevel=1,mtime=0)
        if self.stored_bytes+len(member)>self.byte_limit:raise RuntimeError('G3 stored evidence ceiling reached')
        self.stream.write(member);self.stream.flush();os.fsync(self.stream.fileno())
        self.written+=len(payload);self.stored_bytes+=len(member);self.pending.clear();self.pending_bytes=0

    def __enter__(self):return self
    def __exit__(self,*_):
        try:self.flush()
        finally:self.stream.close()


def event_records(path,*,record_limit=512*1024**2,uncompressed_limit=64*1024**3):
    total=0
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate G3 JSON key')
            result[key]=value
        return result
    with gzip.open(path,'rb') as stream:
        while True:
            line=stream.readline(record_limit+1)
            if not line:break
            total+=len(line)
            if len(line)>record_limit or total>uncompressed_limit or not line.endswith(b'\n'):
                raise ValueError('G3 event stream ceiling or incomplete record')
            yield unpack(json.loads(line,object_pairs_hook=unique,
                parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON'))))


def digest_stream(path,*,byte_limit=64*1024**3):
    digest=hashlib.sha256();size=0
    with gzip.open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            size+=len(chunk)
            if size>byte_limit:raise ValueError('G3 uncompressed digest ceiling reached')
            digest.update(chunk)
    return digest.hexdigest()


class MotionObserver(ScaleObserver):
    def __init__(self,*args,scan_guard_enabled=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.scan_guard_enabled=scan_guard_enabled
        data=self.env.scene['robot'].data
        self.previous=(data.body_link_pos_w.clone(),data.body_link_lin_vel_w.clone())
        self.control_step=0;self.max_speed=self.max_displacement=self.max_acceleration=0.

    def scans(self,names=('height_scanner','height_scanner_base')):
        result={}
        for name in names:
            try:result.update(super().scans((name,)))
            except Exception as error:
                sensor=self.env.scene.sensors[name];data=self.env.scene['robot'].data
                value=getattr(sensor,'_data',None)
                if value is None:value=sensor.data
                fields=dict(root=data.root_link_pos_w,origin=self.env.scene.terrain.env_origins,
                    hits=value.ray_hits_w,hit_ids=value.ray_mesh_ids,
                    age_s=sensor._timestamp-sensor._timestamp_last_update)
                for key,attr in [('ray_starts_w','_ray_starts_w'),('ray_directions_w','_ray_directions_w')]:
                    if hasattr(sensor,attr):fields[key]=getattr(sensor,attr)
                invalid={key:(~torch.isfinite(v)).nonzero()[:32].cpu().tolist()
                         for key,v in fields.items() if not bool(torch.isfinite(v).all())}
                self.journal.record(dict(phase='g3_scanner_fault',sensor=name,error=repr(error),
                    control_step=self.control_step,sim_step=self.env._sim_step_counter,
                    invalid_indices=invalid,fields={k:fault_tensor_record(v) for k,v in fields.items()}))
                self.journal.flush();raise
        return result

    def reset(self,original,env_ids):
        ids=torch.as_tensor(env_ids,device=self.env.device,dtype=torch.long)
        if not len(ids):return original(env_ids)
        self.journal.record(dict(phase='g3_reset_context',control_step=self.control_step,sim_step=self.env._sim_step_counter,
            ids=sorted(ids.cpu().tolist())))
        result=super().reset(original,env_ids)
        data=self.env.scene['robot'].data
        self.previous[0][ids]=data.body_link_pos_w[ids];self.previous[1][ids]=data.body_link_lin_vel_w[ids]
        self.journal.record(dict(phase='g3_reset_rebase',ids=ids.cpu().tolist(),control_step=self.control_step,
            sim_step=self.env._sim_step_counter,positions=tensor_record(data.body_link_pos_w[ids]),
            velocities=tensor_record(data.body_link_lin_vel_w[ids])))
        self.journal.flush();return result

    def after_scene_update(self):
        env=self.env;data=env.scene['robot'].data
        if env._sim_step_counter!=self.start+self.samples+1:raise RuntimeError('G3 missing motion substep')
        try:
            sample=kinematic_sample(env)
            delta=data.body_link_pos_w-self.previous[0]
            acceleration=(data.body_link_lin_vel_w-self.previous[1])/.005
            if not bool(torch.isfinite(acceleration).all()):raise FloatingPointError('G3 invalid acceleration')
            sample.update(displacements_m=tensor_record(delta),finite_difference_acceleration_mps2=tensor_record(acceleration))
            if self.scan_guard_enabled:
                from .scan_footprint import scanner_boundary
                diagnostic={}
                scanner_boundary(data.root_link_pos_w,data.root_link_quat_w,data.root_link_lin_vel_w,
                    data.root_ang_vel_w[:,2],env.scene.terrain.env_origins,diagnostics=diagnostic)
                sample['scanner_guard']=dict(root_linear_velocity_w=tensor_record(data.root_link_lin_vel_w),
                    root_yaw_rate_w=tensor_record(data.root_ang_vel_w[:,2]),
                    current=diagnostic['current'].cpu().tolist(),predicted=diagnostic['predicted'].cpu().tolist())
            self.max_speed=max(self.max_speed,float(data.body_link_lin_vel_w.norm(dim=-1).max()))
            self.max_displacement=max(self.max_displacement,float(delta.norm(dim=-1).max()))
            self.max_acceleration=max(self.max_acceleration,float(acceleration.norm(dim=-1).max()))
            self.journal.record(dict(phase='g3_motion_substep',sim_step=env._sim_step_counter,
                control_step=self.control_step,recovery_ids=sorted(self.pending_recovery),
                scanners=self.scans(('height_scanner_base',)),**sample))
            self.journal.flush()
        except Exception as error:
            self.journal.record(dict(phase='g3_motion_fault',sim_step=env._sim_step_counter,error=repr(error),
                body_positions_w=fault_tensor_record(data.body_link_pos_w),body_link_velocities_w=fault_tensor_record(data.body_link_lin_vel_w)))
            self.journal.flush();raise
        self.previous=(data.body_link_pos_w.clone(),data.body_link_lin_vel_w.clone())
        self.pending_recovery.clear();self.samples+=1
