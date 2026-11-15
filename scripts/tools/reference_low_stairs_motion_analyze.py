"""冻结足端/关节时序分析；重算奖励只在离线副本执行。"""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
import re
import shutil
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import numpy as np
import torch

from reference_low_stairs_motion_probe import (OUTPUT,DOCS,LEGS,PROTOCOL,PROTOCOL_SHA,
    identity,exact_replay,read,ROOT,save,sha,PARENT_RUNS,EVALUATION,GUARDED_RUNS)
from reference_reward_code_audit import definitions,REWARD,Entity,Scene
from reference_core500_analyze import write_csv
from reference_low_stairs_regression_analyze import yaw,lateral_components,first_time
from reference_low_stairs_equal100_analyze import validate_trace
from reference_low_stairs_analyze import check_log
from reference_turn_analyze import WINDOWS

DT=.02
CASES=('low_left','low_right','turn_left','turn_right','fast_left','fast_right')
REPAIR=ROOT/'scripts/tools/reference_low_stairs_repair.py'
VARIANTS=ROOT/'scripts/tools/reference_reward_variants.py'


def npz(path):
    with np.load(path) as values:return {k:values[k] for k in values.files}


def inverse_rotation(q,v):
    """单位 wxyz 四元数，将世界向量旋转到机身坐标。"""
    w,u=q[...,:1],q[...,1:]
    return (2*w*w-1)*v-2*w*torch.cross(u,v,dim=-1)+2*u*(u*v).sum(-1,keepdim=True)


def airtime_rates(command,last_air,current_contact,threshold=.5,weight=5.,gait=1.):
    first=(current_contact>0)&(current_contact<DT+1e-8)
    turning=np.abs(command[...,2])>.1
    pure=(np.linalg.norm(command[...,:2],axis=-1)<.1)&turning
    original=((last_air-threshold)*first).sum(-1)*turning*weight*gait
    nonnegative=(np.maximum(last_air-threshold,0)*first).sum(-1)*turning*weight*gait
    return original,np.where(pure,nonnegative,original),first


def phase_function():
    # 离线编译原函数，唯一 AST 改动是将最后返回值扩为 (原奖励, 中间量)。
    # 仿真进程中不执行本函数，不回写原源码。
    ns=dict(torch=torch,math=math,SceneEntityCfg=Entity,gait_level=1.,math_utils=NS(quat_apply_inverse=inverse_rotation))
    definitions(REWARD,['get_gait_level_tensor','_bernstein_torch','_bezier_curve_torch','_bezier_curve_derivative_torch'],ns)
    fn=next(n for n in ast.parse(REWARD.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='phase_foot_trajectory_exp')
    last=fn.body[-1];assert isinstance(last,ast.Return)
    last.value=ast.Tuple(elts=[last.value]+[ast.Name(id=k,ctx=ast.Load()) for k in
        ('S','ref_pos_b','ref_vel_b','foot_pos_b','foot_vel_b')],ctx=ast.Load())
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(REWARD),'exec'),ns)
    return ns['phase_foot_trajectory_exp']


def reconstruct_phase(motion,initial,physics,names,cfg):
    fn=phase_function();params=copy.deepcopy(cfg['cfg']['rewards']['phase_foot_trajectory_exp']['params'])
    order=[names['feet_body_ids'].index(i) for i in names['phase_body_ids']]
    params['asset_cfg']=Entity('robot',body_ids=list(range(4)))
    output={k:[] for k in ('rate','phase','reference_position','reference_velocity','position_body','velocity_body')}
    for start in range(0,len(motion['root']),10):
        end=min(start+10,len(motion['root']));shape=motion['root'][start:end].shape[:2];count=math.prod(shape)
        tensor=lambda a:torch.from_numpy(np.ascontiguousarray(a)).reshape(count,*a.shape[2:])
        root=tensor(motion['root'][start:end])
        feet=motion['feet'][start:end][:,:,order]+physics['origins'][None,:,None]
        data=NS(root_pos_w=root[:,:3],root_quat_w=root[:,3:7],root_lin_vel_w=root[:,7:10],
                body_pos_w=tensor(feet),body_lin_vel_w=tensor(motion['foot_velocity'][start:end][:,:,order]))
        command=tensor(motion['command'][start:end])
        env=NS(num_envs=count,device='cpu',scene={'robot':NS(data=data)},step_dt=DT,
            episode_length_buf=tensor(motion['episode_length'][start:end]),
            command_manager=NS(get_command=lambda _:command),phase_foot_ref_body=torch.tensor(initial['phase_reference']))
        values=fn(env,**params)
        for k,v in zip(output,values):output[k].append(v.numpy().reshape(*shape,*v.shape[1:]))
    output={k:np.concatenate(v) for k,v in output.items()}
    output['rate']*=cfg['cfg']['rewards']['phase_foot_trajectory_exp']['weight']*names['gait_level']
    return output


def selected_mask(trace,env_id,start,end):
    time=np.arange(len(trace['active']))*DT
    mask=trace['active'][:,env_id]&~trace['terminated'][:,env_id]&(time>=start)&(time<end)
    return mask if mask.sum()==round((end-start)/DT) else None


def turn_rows(policy,profile,motion,trace,report,physics,names,phase):
    air,adjusted,first=airtime_rates(trace['command'],motion['last_air_time'],motion['contact_time'],gait=names['gait_level'])
    names_rw=report['reward_names'];idx=names_rw.index('feet_air_time_ang_z_lite3')
    error=float(np.abs(air-trace['reward_rates'][...,idx]).max());assert error<2e-5,error
    phase_error=float(np.abs(phase['rate']-trace['reward_rates'][...,names_rw.index('phase_foot_trajectory_exp')]).max())
    assert phase_error<2e-4,phase_error
    rows,legs,joints=[],[],[]
    for record in report['records']:
        if record['case'] not in CASES:continue
        i=record['env_id']
        for window,(start,end) in WINDOWS.items():
            mask=selected_mask(trace,i,start,end)
            row=dict(policy=policy,profile=profile,case=record['case'],repeat=record['repeat'],env_id=i,
                window=window,complete=mask is not None,outcome=record['outcome'])
            if mask is None:continue
            contact=motion['contact_time'][mask,i]>0
            contact_force=motion['force'][mask,i,:,2]>1.
            landing=first[mask,i];airvalues=motion['last_air_time'][mask,i][landing]
            joint_error=motion['joint_target'][mask,i]-motion['joint_pos'][mask,i]
            torque=np.abs(motion['torque'][mask,i])
            # 本模型显式 PD 的实际执行器限值；物理关节限值另保留供核验。
            limits=np.array([36. if 'Knee' in name else 24. for name in names['joint_names']])
            row.update(yaw_actual=float(trace['velocity'][mask,i,5].mean()),
                yaw_mae=float(np.abs(trace['command'][mask,i,2]-trace['velocity'][mask,i,5]).mean()),
                all_feet_contact_fraction=float(contact.all(-1).mean()),
                mean_support_feet=float(contact_force.sum(-1).mean()),landings=int(landing.sum()),
                air_time_at_landing_mean=float(airvalues.mean()) if len(airvalues) else None,
                air_time_over_threshold=int((airvalues>.5).sum()),
                original_yaw_air_rate=float(air[mask,i].mean()),repaired_yaw_air_rate_common_scale=float(adjusted[mask,i].mean()),
                repaired_yaw_air_rate_training_scale=float(adjusted[mask,i].mean()/names['gait_level']),
                phase_rate_training_scale=float(phase['rate'][mask,i].mean()/names['gait_level']),
                training_stand_rate=float((-.5*np.abs(motion['joint_pos'][mask,i]-physics['default_joint_pos'][i]).sum(-1)
                    *((np.linalg.norm(trace['command'][mask,i,:2],axis=-1)<.1)&(np.abs(trace['command'][mask,i,2])<.1))).mean()),
                joint_error_rms=float(np.sqrt(np.mean(joint_error**2))),
                torque_limit_fraction=float((torque>=.95*limits).mean()),
                phase_position_error_rms=float(np.sqrt(np.mean((phase['position_body'][mask,i]-phase['reference_position'][mask,i])**2))))
            for term in ('track_ang_vel_z_exp','track_lin_vel_xy_exp','feet_gait','phase_foot_trajectory_exp','stand_still',
                         'action_rate_l2','joint_mirror','hipx_joint_pos_penalty','joint_torques_l2'):
                row[f'rate_{term}']=float(trace['reward_rates'][mask,i,names_rw.index(term)].mean())
            rows.append(row)
            for j,leg in enumerate(LEGS):
                c=contact_force[:,j];speed=np.linalg.norm(motion['foot_velocity'][mask,i,j,:2],axis=-1)
                clearance=motion['feet'][mask,i,j,2]-.022
                l=landing[:,j];swing=motion['last_air_time'][mask,i,j][l]
                legs.append(dict(policy=policy,profile=profile,case=record['case'],repeat=record['repeat'],window=window,leg=leg,
                    support_fraction=float(c.mean()),air_fraction=float((motion['air_time'][mask,i,j]>0).mean()),
                    clearance_p95_m=float(np.quantile(clearance,.95)),clearance_max_m=float(clearance.max()),
                    clearance_over_1cm_fraction=float((clearance>.01).mean()),landings=int(l.sum()),
                    landing_air_mean_s=float(swing.mean()) if len(swing) else None,
                    support_slip_mean_m_s=float(speed[c].mean()) if c.any() else None,
                    vertical_force_mean=float(motion['force'][mask,i,j,2].mean())))
            for j,name in enumerate(names['joint_names']):
                q=motion['joint_pos'][mask,i,j];target=motion['joint_target'][mask,i,j]
                joints.append(dict(policy=policy,profile=profile,case=record['case'],repeat=record['repeat'],window=window,joint=name,
                    position_mean=float(q.mean()),position_std=float(q.std()),position_range=float(np.ptp(q)),
                    target_range=float(np.ptp(target)),tracking_rms=float(np.sqrt(np.mean((target-q)**2))),
                    velocity_rms=float(np.sqrt(np.mean(motion['joint_vel'][mask,i,j]**2))),
                    torque_rms=float(np.sqrt(np.mean(motion['torque'][mask,i,j]**2))),
                    saturation=float((torque[:,j]>=.95*limits[j]).mean())))
    return rows,legs,joints,dict(original_yaw_air_max_error=error,phase_reward_max_error=phase_error)


def group_rows(rows,keys,metrics):
    groups=[]
    for identity_ in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected=[r for r in rows if tuple(r[k] for k in keys)==identity_]
        row=dict(zip(keys,identity_),n=len(selected))
        for metric in metrics:
            values=[r[metric] for r in selected if r[metric] is not None]
            row[metric]=float(np.mean(values)) if values else None
        groups.append(row)
    return groups


def paired_groups(rows,extra=()):
    # 只比较父/新策略都完整存活的同一重复编号，所有窗按条件单列 n。
    keys=('profile','case','window',*extra)
    metadata=set(keys)|{'policy','repeat','env_id','complete','outcome'}
    metrics=[k for k in rows[0] if k not in metadata]
    out=[]
    for identity_ in sorted({tuple(r[k] for k in keys) for r in rows}):
        maps={p:{r['repeat']:r for r in rows if r['policy']==p and tuple(r[k] for k in keys)==identity_} for p in ('parent','zero')}
        ids=sorted(set(maps['parent'])&set(maps['zero']))
        row=dict(zip(keys,identity_),starts=20,n=len(ids))
        for p,m in maps.items():
            for k in metrics:
                values=[m[i][k] for i in ids if m[i][k] is not None]
                row[f'{p}_{k}']=float(np.mean(values)) if values else None
        out.append(row)
    return out


def recovery_rows(policy,m,trace,initial,physics,names):
    i=57;n=min(100,int(trace['active'][:,i].sum()));rows=[]
    for k in range(n):
        row=dict(policy=policy,time_s=(k+1)*DT,tilt_deg=float(np.rad2deg(np.arccos(np.clip(-m['gravity'][k,i,2],-1,1)))),
            support_count=int((m['force'][k,i,:,2]>1).sum()),root_z=float(m['root'][k,i,2]),
            yaw_command=float(m['command'][k,i,2]),terminated=bool(m['terminated'][k,i]))
        for j,leg in enumerate(LEGS):
            row[f'{leg}_force_z']=float(m['force'][k,i,j,2]);row[f'{leg}_foot_z']=float(m['feet'][k,i,j,2])
        for j,name in enumerate(names['joint_names']):
            row[f'{name}_target']=float(m['joint_target'][k,i,j]);row[f'{name}_position']=float(m['joint_pos'][k,i,j])
        rows.append(row)
    return rows


def down_rows(policy,root,profile):
    path=root/f'down_{profile}';report,cfg=read(path/'report.json'),read(path/'config.json')
    data,initial=npz(path/'trace.npz'),npz(path/'initial.npz');validate_trace(report,data)
    rows=[]
    for record in report['records']:
        i=record['env_id'];n=int(data['active'][:,i].sum());position=data['root'][:n,i]
        q=data['quat'][:n,i];heading=yaw(q);y0=initial['root'][i,1]-initial['origins'][i,1]
        yaw0=yaw(initial['root'][i,3:7]);change=(heading-yaw0+np.pi)%(2*np.pi)-np.pi
        components=lateral_components(q,data['velocity'][:n,i]).sum(0)*DT
        support=data['forces'][:n,i,:,2]>1.
        row=dict(policy=policy,profile=profile,**record,displacement_y=float(position[-1,1]-y0),
            final_yaw_deg=float(np.rad2deg(heading[-1])),first_yaw10_s=first_time(np.abs(change)>np.deg2rad(10)),
            first_y05_s=first_time(np.abs(position[:,1]-y0)>.05),first_y20_s=first_time(np.abs(position[:,1]-y0)>.2),
            forward_y=float(components[0]),body_lateral_y=float(components[1]),vertical_y=float(components[2]),
            integration_error=float(position[-1,1]-y0-components.sum()),
            support_feet_mean=float(support.sum(-1).mean()),support_left=float(support[:,[0,2]].mean()),
            support_right=float(support[:,[1,3]].mean()))
        for t in (2,4,6):
            k=round(t/DT)-1
            row[f'y_{t}s']=float(position[k,1]-y0) if k<n else None
            row[f'yaw_{t}s']=float(np.rad2deg(change[k])) if k<n else None
        rows.append(row)
    return rows,path


def main():
    complete=read(OUTPUT/'complete.json');assert complete['status']=='passed' and complete['episodes']==1280
    assert sha(PROTOCOL)==PROTOCOL_SHA
    tables={k:[] for k in ('turn_episodes','foot_episodes','joint_episodes','recovery57','down_episodes')}
    checks=[];raw_paths=[];initials={};physics_all={};inputs={}
    for entry in complete['matrix']:
        policy,profile=entry['policy'],entry['profile'];path=Path(entry['path'])
        checks.append(exact_replay(path,Path(entry['baseline'])))
        report,cfg,names=read(path/'report.json'),read(path/'config.json'),read(path/'names.json')
        m,trace,initial,physics=npz(path/'motion.npz'),npz(path/'trace.npz'),npz(path/'initial.npz'),npz(path/'physics_initial.npz')
        assert all(np.isfinite(a).all() for a in m.values());validate_trace(report,trace)
        assert np.array_equal(m['command'],trace['command']) and np.array_equal(m['gravity'],trace['gravity'])
        assert np.array_equal(m['terminated'],trace['terminated']) and np.array_equal(m['truncated'],trace['truncated'])
        assert np.array_equal(m['observation'][0],initial['observation'])
        assert np.array_equal(m['observation'][...,6:9],trace['command'])
        assert all(np.array_equal(m['episode_length'][trace['active'][:,i],i],np.arange(1,trace['active'][:,i].sum()+1)) for i in range(320))
        initials[policy,profile]=initial;physics_all[policy,profile]=physics;inputs[policy,profile]=m['observation'][0].copy()
        phase=reconstruct_phase(m,initial,physics,names,cfg)
        output=OUTPUT/'derived'/path.name;output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'phase.npz',**phase)
        rows,legs,joints,reward_check=turn_rows(policy,profile,m,trace,report,physics,names,phase)
        tables['turn_episodes']+=rows;tables['foot_episodes']+=legs;tables['joint_episodes']+=joints
        if read(path/'execution.json')['status']=='failed':
            log=(path/'runtime.log').read_text()
            assert len(re.findall('Traceback',log))==1
            assert not re.search(r'\[Error\]|switching to software|GPU solver failed|GPU Bp pipeline failed|Contacts have been dropped',log,re.I)
            assert read(path/'motion_report.json')['completion_type']=='capture_completed_verifier_corrected'
            log_check=dict(run=str(path),post_capture_verifier_errors=1,physics_fallback_messages=0)
        else:log_check=check_log(path)
        checks.append(dict(**reward_check,**log_check))
        if profile=='training_randomized':tables['recovery57']+=recovery_rows(policy,m,trace,initial,physics,names)
        raw_paths.append(path);del m,trace,phase
    for profile in ('nominal','training_randomized'):
        a,b=(physics_all[p,profile] for p in ('parent','zero'))
        assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
        assert np.array_equal(inputs['parent',profile],inputs['zero',profile])
    tables['turn_paired']=paired_groups(tables['turn_episodes'])
    tables['feet_paired']=paired_groups(tables['foot_episodes'],('leg',))
    tables['joints_paired']=paired_groups(tables['joint_episodes'],('joint',))
    for policy,root in (('parent',PARENT_RUNS),('guarded',GUARDED_RUNS),('zero',EVALUATION)):
        for profile in ('nominal','training_randomized'):
            rows,path=down_rows(policy,root,profile);tables['down_episodes']+=rows;raw_paths.append(path)
    tables['down_groups']=group_rows(tables['down_episodes'],('policy','profile','height','outcome'),
        ('displacement_y','final_yaw_deg','first_yaw10_s','first_y20_s','forward_y','body_lateral_y','integration_error','support_feet_mean'))
    for name,rows in tables.items():write_csv(DOCS/f'{name}.csv',rows)
    historic={r['path']:r['sha256'] for r in read(ROOT/'docs/lite3-low-stairs-zero-preservation-20261115/raw_manifest.json')}
    reused=0
    for entry in complete['matrix']:raw_paths.append(Path(entry['baseline']))
    for path in raw_paths:
        if OUTPUT in path.parents:continue
        for file in path.iterdir():
            if file.is_file():assert sha(file)==historic[str(file)];reused+=1
    manifest={}
    for path in raw_paths+[OUTPUT]:
        files=path.rglob('*') if path==OUTPUT else path.iterdir()
        for f in files:
            if f.is_file():manifest[str(f)]=dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f))
    for file in OUTPUT.rglob('*.json'):
        target=DOCS/'runs'/file.relative_to(OUTPUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    for policy in ('parent','zero'):
        checkpoint,digest,_,_=identity(policy);assert sha(checkpoint)==digest
        manifest[str(checkpoint)]=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,sha256=digest)
    tests=[]
    for name in ('probe_tests.xml','analysis_tests.xml'):tests+=ET.parse(DOCS/name).getroot().findall('testsuite')
    assert all(int(s.get(k))==0 for s in tests for k in ('errors','failures','skipped'))
    save(DOCS/'summary.json',dict(status='passed',scope='冻结诊断完整性；不代表能力通过',episodes=1280,optimizer_updates=0,
        checks=checks,reused_files_verified=reused,tests_passed=sum(int(s.get('tests')) for s in tests),
        paired_initial_physics_exact=True,turns=tables['turn_paired'],feet=tables['feet_paired'],down_groups=tables['down_groups'],
        implementation_sources=[dict(path=str(p),sha256=sha(p)) for p in (Path(__file__),PROTOCOL,REWARD,REPAIR,VARIANTS,Path(__file__).with_name('reference_low_stairs_motion_probe.py'))]))
    save(DOCS/'raw_manifest.json',list(manifest.values()))
    print('MOTION ANALYSIS COMPLETE',flush=True)


if __name__=='__main__':main()
