"""Native metric sampling and normalized Source episodes; runtime qualification pending."""
from contextlib import contextmanager
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import re

import numpy as np
import torch

from .source_selection import family_of, _episode

FOOT_NAMES = ('FL_FOOT','FR_FOOT','HL_FOOT','HR_FOOT')


def geometry_digest(vertices,faces,*,parameters,materials):
    """Canonical local mesh identity, independent of vertex/face ordering and seed IDs."""
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces)
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all()
            or f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.dtype.kind not in 'iu'
            or (f<0).any() or (f>=len(v)).any()):
        raise ValueError('invalid local geometry mesh')
    material=np.asarray(materials,dtype=np.float64)
    if material.shape!=(len(f),3) or not np.isfinite(material).all() or (material<0).any():
        raise ValueError('one static/dynamic friction and restitution triple per face required')
    if not isinstance(parameters,dict) or any('seed' in name or name=='state_id' for name in parameters):
        raise ValueError('geometry identity excludes seed/state bookkeeping')
    unique,inverse=np.unique(v,axis=0,return_inverse=True)
    remapped=inverse[f];rotated=[]
    for triangle,coefficients in zip(remapped,material):
        offset=int(np.argmin(triangle));rotated.append(np.roll(triangle,-offset).tolist()+coefficients.tolist())
    payload=dict(vertices_m=unique.tolist(),faces_with_materials=sorted(rotated),parameters=parameters)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def material_binding_audit(stage,paths,resolve_bound_material):
    """Read actual bound physics materials; resolver is supplied after USD startup."""
    result=[]
    if not paths:raise ValueError('no colliders supplied for material binding audit')
    for path in paths:
        prim=stage.GetPrimAtPath(path)
        material=resolve_bound_material(prim)
        if not prim or not material:raise ValueError('unresolved physics material: '+path)
        values={name:material.GetAttribute('physics:'+name).Get() for name in ('staticFriction','dynamicFriction','restitution')}
        if any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in values.values()):
            raise ValueError('invalid bound physics material parameters')
        result.append(dict(collider=path,material=str(material.GetPath()),**values))
    return dict(status='verified_direct_USD_binding',records=result,effective_contact_friction_measured=False)


def native_metric_rows(env):
    """Copy current state before reset; slide speed is a world-horizontal proxy."""
    robot=env.scene['robot'];data=robot.data;sensor=env.scene.sensors['contact_forces']
    body=list(robot.body_names);sensor_names=list(sensor.body_names)
    if len(set(body))!=len(body) or len(set(sensor_names))!=len(sensor_names):raise ValueError('duplicate body names')
    foot=[body.index(name) for name in FOOT_NAMES];force_ids=[sensor_names.index(name) for name in FOOT_NAMES]
    pos=data.root_link_pos_w;origin=env.scene.terrain.env_origins
    gravity=data.projected_gravity_b;velocity=data.root_lin_vel_b[:,:2]
    command=env.command_manager.get_command('base_velocity')[:,:2]
    feet=data.body_link_pos_w[:,foot];speed=data.body_link_lin_vel_w[:,foot,:2].norm(dim=-1)
    forces=sensor.data.net_forces_w[:,force_ids]
    values=(pos,origin,gravity,velocity,command,feet,speed,forces)
    if any(value.shape[0]!=env.num_envs or not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError('invalid native metric tensors')
    if gravity.shape!=(env.num_envs,3) or bool((gravity.abs()>1+1e-5).any()):raise ValueError('invalid projected gravity')
    error=(velocity-command).norm(dim=-1)
    scalar=torch.stack((pos[:,0]-origin[:,0],pos[:,1]-origin[:,1],error,velocity[:,0],
        torch.asin(gravity[:,1].clamp(-1,1)),torch.asin(gravity[:,0].clamp(-1,1))),dim=1).cpu().tolist()
    ages=env.episode_length_buf.cpu().tolist()
    foot_xy=(feet[:,:,:2]-origin[:,None,:2]).cpu().tolist()
    speeds=speed.cpu().tolist();contacts=(forces.norm(dim=-1)>1.).cpu().tolist()
    horizontal=forces[:,:,:2].norm(dim=-1).cpu().tolist()
    rows=[]
    for i in range(env.num_envs):
        progress,lateral,err,forward,roll,pitch=scalar[i]
        rows.append(dict(step=int(ages[i]),forward_progress_m=progress,lateral_m=lateral,
            velocity_error_mps=err,forward_velocity_mps=forward,roll_rad=roll,pitch_rad=pitch,
            foot_positions_reset_xy_m=foot_xy[i],foot_speeds_mps=speeds[i],foot_contact=contacts[i],
            foot_horizontal_forces_N=horizontal[i]))
    return rows


class PreResetMetricCapture:
    """One begin/collect pair per control step; terminal samples precede reset."""
    def __init__(self,env,sampler=native_metric_rows):
        self.env,self.sampler=env,sampler;self.active=False;self.terminal={}

    def begin_control(self):
        if self.active:raise RuntimeError('previous metric control not collected')
        self.active=True;self.terminal={}

    def collect_control(self,done):
        if not self.active:raise RuntimeError('metric control was not started')
        if done.shape!=(self.env.num_envs,) or not bool(((done==0)|(done==1)).all()):raise ValueError('invalid metric done mask')
        ids=done.nonzero().flatten().cpu().tolist()
        if set(ids)!=set(self.terminal):raise RuntimeError('terminal metric capture differs from done rows')
        current=self.sampler(self.env)
        rows=[self.terminal.get(i,current[i]) for i in range(self.env.num_envs)]
        self.active=False;self.terminal={}
        return rows

    @contextmanager
    def installed(self):
        present='_reset_idx' in vars(self.env);stored=vars(self.env).get('_reset_idx');original=self.env._reset_idx
        def reset(ids):
            if self.active:
                rows=self.sampler(self.env)
                for i in torch.as_tensor(ids).cpu().tolist():
                    if i in self.terminal:raise RuntimeError('duplicate terminal metric row')
                    self.terminal[i]=deepcopy(rows[i])
            return original(ids)
        self.env._reset_idx=reset
        try:yield self
        finally:
            if present:self.env._reset_idx=stored
            else:delattr(self.env,'_reset_idx')


class MetricEpisode:
    """Aggregate raw rows with audited family geometry. No inferred missing metrics."""
    def __init__(self,identity,geometry):
        self.identity=deepcopy(identity);self.geometry=deepcopy(geometry);self.finished=None
        if identity.get('state_randomized') is not False or not re.fullmatch('[0-9a-f]{64}',identity.get('geometry_sha256','')):
            raise ValueError('metric identity requires actual geometry digest and deterministic state ID')
        if geometry.get('geometry_sha256')!=identity['geometry_sha256'] or geometry.get('cell')!=identity['cell']:
            raise ValueError('metric geometry differs from episode identity')
        self.family=family_of(identity['cell']);self.rows=[];self.projected=0;self.hold=0
        self.emergency=self.nonfoot_terminations=0
        self.collisions=0;self.previous_collision=[False]*4;self.windows=deque(maxlen=25);self.stuck=set()
        self.contacts=self.slips=0;self.region_counts={}
        if self.family=='multi_step_stairs':
            if len(geometry.get('riser_progress_m',[]))!=3 or len(geometry.get('tread_intervals_m',[]))!=3:
                raise ValueError('three audited stair risers/treads required')
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in geometry['riser_progress_m']):raise ValueError('invalid riser locations')
            if any(len(pair)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pair) or pair[0]>=pair[1] for pair in geometry['tread_intervals_m']):raise ValueError('invalid tread intervals')
        if self.family in ('slope','slope_plus_rough') and (geometry.get('slope_angle_deg')!=5. or 'exit_transition_progress_m' not in geometry):
            raise ValueError('audited slope angle and exit interval required')
        if self.family in ('multi_scale_rough','slope_plus_rough'):
            if type(geometry.get('audited_rough_peak_valley_m')) not in (int,float) or len(geometry.get('audited_spatial_scales_m',[]))<2:
                raise ValueError('rough mesh and spatial-scale audits required')
            if not math.isfinite(geometry['audited_rough_peak_valley_m']) or geometry['audited_rough_peak_valley_m']<0 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in geometry['audited_spatial_scales_m']):raise ValueError('invalid rough audit values')
        if self.family=='friction_variation':
            binding=geometry.get('material_binding_audit',{})
            if binding.get('status')!='verified_direct_USD_binding' or not binding.get('records') or len(geometry.get('regions',[]))<2:
                raise ValueError('friction region geometry and native material binding audit required')
            if len({str(r['id']) for r in geometry['regions']})!=len(geometry['regions']):raise ValueError('duplicate friction region IDs')
            for region in geometry['regions']:
                bounds=region['bounds_reset_xy_m']
                if len(bounds)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in bounds) or bounds[0]>=bounds[1] or bounds[2]>=bounds[3]:raise ValueError('invalid friction region bounds')

    def add(self,row,*,stable,projected=False,emergency=False,nonfoot_termination=False):
        if self.finished is not None:raise RuntimeError('episode metrics already finalized')
        if any(type(v) is not bool for v in (stable,projected,emergency,nonfoot_termination)):
            raise ValueError('metric flags must be booleans')
        expected=len(self.rows)+1
        if type(row.get('step')) is not int or row['step']!=expected or expected>2000:raise ValueError('episode metric step missing/repeated')
        scalars=('forward_progress_m','lateral_m','velocity_error_mps','forward_velocity_mps','roll_rad','pitch_rad')
        if any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) for k in scalars):raise ValueError('invalid episode metric scalar')
        if row['velocity_error_mps']<0:raise ValueError('negative velocity error')
        for k in ('foot_speeds_mps','foot_horizontal_forces_N'):
            if len(row.get(k,[]))!=4 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in row[k]):raise ValueError('invalid foot metric: '+k)
        if len(row.get('foot_contact',[]))!=4 or any(type(v) is not bool for v in row['foot_contact']):raise ValueError('missing foot contact mask')
        xy=row.get('foot_positions_reset_xy_m')
        if not isinstance(xy,list) or len(xy)!=4 or any(len(p)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in xy):raise ValueError('invalid foot positions')
        # Validate the complete row before mutating episode state.
        region_ids=[]
        if self.family=='friction_variation':
            for point,contact in zip(xy,row['foot_contact']):
                matches=[str(region['id']) for region in self.geometry['regions'] if
                    region['bounds_reset_xy_m'][0]<=point[0]<=region['bounds_reset_xy_m'][1] and
                    region['bounds_reset_xy_m'][2]<=point[1]<=region['bounds_reset_xy_m'][3]]
                if contact and not matches:raise ValueError('contacting foot outside audited friction regions')
                region_ids.append(matches[0] if matches else None)
        self.rows.append(deepcopy(row));self.projected+=projected;self.emergency+=emergency;self.nonfoot_terminations+=nonfoot_termination
        self.hold=self.hold+1 if stable and row['forward_progress_m']>=9. else 0
        for index,(contact,speed) in enumerate(zip(row['foot_contact'],row['foot_speeds_mps'])):
            self.contacts+=contact;self.slips+=contact and speed>.35
            if self.family=='friction_variation' and contact:
                counts=self.region_counts.setdefault(region_ids[index],[0,0]);counts[0]+=1;counts[1]+=speed>.35
        if self.family=='multi_step_stairs':
            flags=[index<2 and row['foot_horizontal_forces_N'][index]>=20. and
                any(abs(xy[index][0]-riser)<=.08 for riser in self.geometry['riser_progress_m']) for index in range(4)]
            self.collisions+=sum(now and not previous for now,previous in zip(flags,self.previous_collision));self.previous_collision=flags
            self.windows.append((row['forward_progress_m'],row['forward_velocity_mps']))
            if len(self.windows)==25:
                positions,velocities=zip(*self.windows)
                for index,(start,end) in enumerate(self.geometry['tread_intervals_m']):
                    if all(start<=p<=end for p in positions) and max(positions)-min(positions)<=.05 and all(abs(v)<=.05 for v in velocities):self.stuck.add(index)

    def finish(self,*,completed):
        if self.finished is not None:
            if completed is not self.finished['metrics']['completed']:raise ValueError('finalized episode outcome cannot change')
            return deepcopy(self.finished)
        if type(completed) is not bool or not self.rows or not self.contacts:raise ValueError('episode/contact evidence incomplete')
        final=self.rows[-1];progress=final['forward_progress_m']
        if completed and (progress<9 or self.hold<50):raise ValueError('completion lacks consecutive stable progress')
        errors=[row['velocity_error_mps'] for row in self.rows];count=len(errors)
        metrics=dict(completed=completed,steps=count,forward_progress_m=progress,stable_hold_steps=self.hold,
            velocity_error_mean_mps=sum(errors)/count,velocity_error_rms_mps=(sum(v*v for v in errors)/count)**.5,
            roll_peak_rad=max(abs(r['roll_rad']) for r in self.rows),pitch_peak_rad=max(abs(r['pitch_rad']) for r in self.rows),
            lateral_peak_m=max(abs(r['lateral_m']) for r in self.rows),slip_contact_fraction=self.slips/self.contacts,
            projection_fraction=self.projected/count,emergency_count=self.emergency,invalid_state_count=0,
            nonfoot_contact_termination_count=self.nonfoot_terminations)
        if self.family=='multi_step_stairs':
            maximum=max(r['forward_progress_m'] for r in self.rows)
            metrics.update(completed_step_index=sum(maximum>=end for start,end in self.geometry['tread_intervals_m']),
                stuck_count=len(self.stuck),front_foot_collision_count=self.collisions)
        if self.family in ('slope','slope_plus_rough'):
            metrics.update(slope_direction=self.identity['cell'].removeprefix('slope_rough_').removeprefix('slope_'),
                slope_angle_deg=self.geometry['slope_angle_deg'],transition_completed=progress>=self.geometry['exit_transition_progress_m'])
        if self.family in ('multi_scale_rough','slope_plus_rough'):
            metrics.update({key:self.geometry[key] for key in ('audited_rough_peak_valley_m','audited_spatial_scales_m')})
        if self.family=='friction_variation':
            metrics.update(region_layout=self.identity['cell'].removeprefix('friction_'),material_binding_verified=True,
                slip_by_region={name:slip/contacts for name,(contacts,slip) in self.region_counts.items()})
        result=dict(self.identity,schema='lite3_source_episode_metrics_v1',metrics=metrics)
        _episode(result,result['cell'],result['split'],result['checkpoint_sha256'])
        self.finished=deepcopy(result)
        return result
