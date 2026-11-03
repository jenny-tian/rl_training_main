"""CPU acceptance of one G1 attempt, including independent sparse reconstruction."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from rl_training.source_scan.scale_startup import CONTRACT
from rl_training.source_scan.shared_terrain_bank import SharedBankTerrainAllocator
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.sparse_contact import reconstruct_frame
from rl_training.source_scan.moving_diagnostic import restore_tensor,check_scan
from source_shared_tensor_contact import tensor_runtime_passed


def _require(condition,message):
    if not condition:raise ValueError(message)


def _load(value):
    def invalid(text):raise ValueError('nonfinite JSON number: '+text)
    def floating(text):
        value=float(text)
        if not math.isfinite(value):invalid(text)
        return value
    return json.loads(value,parse_constant=invalid,parse_float=floating)


def _scans(scans,roots,origins,*,only_base=False):
    names=('height_scanner_base',) if only_base else ('height_scanner','height_scanner_base')
    _require(set(scans)==set(names),'missing or unexpected G1 scanner records')
    for name in names:
        value=scans[name]
        check_scan(roots,restore_tensor(value['hits']),restore_tensor(value['hit_ids']),
            restore_tensor(value['mesh_ids']),value['ground_id'],origins,age=restore_tensor(value['age_s']),
            max_age=1e-6 if name=='height_scanner_base' else .02,base=name=='height_scanner_base')


def validate(folder,plan,execution):
    result=dict(schema='lite3_source_shared_scale_G1_acceptance_v1',accepted=False,reasons=[],
        training_ready=False,g3_g4_qualified=False,scope='16_env_startup_one_zero_control_step_only')
    try:
        _require(execution.get('returncode')==0 and execution.get('process_group_empty') is True
            and execution.get('timed_out') is False and execution.get('monitor_errors')==[], 'G1 process/monitoring failed')
        folder=Path(folder)
        report=_load((folder/'result.json').read_text())
        cleanup=_load((folder/'result.cleanup.json').read_text())
        steps_raw=(folder/'result.steps.jsonl').read_bytes();events_raw=(folder/'events.jsonl').read_bytes()
        rows=[_load(line) for line in steps_raw.splitlines()];events=[_load(line) for line in events_raw.splitlines()]
        expected=CONTRACT|dict(status='passed',base_revision=plan['base_revision'],simulation_control_steps=1,
            attempt_id=plan['attempt_id'],phase='result_before_shutdown',steps_written=1,input_sha256=plan['input_sha256'],
            motion_samples=4,reset_batches=0,pending_recovery_ids=[])
        for key,value in expected.items():
            _require(type(report.get(key)) is type(value) and report[key]==value,'G1 report mismatch: '+key)
        _require(report['steps_sha256']==hashlib.sha256(steps_raw).hexdigest() and report['events_sha256']==hashlib.sha256(events_raw).hexdigest(),'G1 journal hash mismatch')
        _require(report['steps_file']=='result.steps.jsonl' and len(rows)==1,'G1 control journal length/path differs')
        _require(cleanup.get('attempt_id')==plan['attempt_id'] and cleanup.get('environment')=='returned'
            and cleanup.get('application') in ('closing','returned') and cleanup.get('persistence_errors')==[],'G1 cleanup incomplete')
        row=rows[0]
        _require(row.get('step')==1 and row.get('attempt_id')==plan['attempt_id'] and row.get('done')==[0]*16
            and row.get('time_outs')==[False]*16,'G1 unexpected terminal or control identity')
        for key in ('raw_action','executed_action'):_require(row.get(key)==[[0.]*12 for _ in range(16)],'G1 action differs from zero')
        for key in ('projected','paused'):_require(row.get(key)==[[False] for _ in range(16)],'G1 action projection/pause')
        _require(len(row['metric_rows'])==16 and all(r.get('step')==1 for r in row['metric_rows']),'G1 native metric sample incomplete')
        for metric in row['metric_rows']:
            for key in ('forward_progress_m','lateral_m','velocity_error_mps','forward_velocity_mps','roll_rad','pitch_rad'):
                _require(type(metric.get(key)) in (int,float) and math.isfinite(metric[key]),'G1 native metric invalid: '+key)
            for key in ('foot_speeds_mps','foot_horizontal_forces_N'):
                _require(len(metric[key])==4 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in metric[key]),'G1 foot metric invalid')
            _require(len(metric['foot_contact'])==4 and all(type(v) is bool for v in metric['foot_contact']),'G1 contact metric mask invalid')
        _require(report['observation_widths'].get('policy')==45 and report['observation_widths'].get('critic')==393,'G1 observation widths differ')
        oracle=SharedBankTerrainAllocator(16,SOURCE_RESET_BLOCK_COUNTS,131001,bank_rows=1)
        assignments=[asdict(t) for t in oracle.reset(list(range(16)))]
        _require(report['initial_assignments']==assignments,'G1 initial assignments differ from replay')
        grid=restore_tensor(report['terrain_origins']);_require(tuple(grid.shape)==(1,15,3),'G1 bank shape differs')
        origins=grid[0,torch.tensor([t['column'] for t in assignments])]
        groups=report['collision_groups'];global_group='/World/collisions/global_group'
        _require(groups.get('inverted') is True and set(groups['groups'])=={global_group,*[f'/World/collisions/group{i}' for i in range(16)]},'G1 collision group inventory differs')
        for path,value in groups['groups'].items():
            own=int(path.rsplit('group',1)[1]) if path!=global_group else None
            includes=['/World/ground'] if own is None else [f'/World/envs/env_{own}']
            allowed=[global_group,*[f'/World/collisions/group{i}' for i in range(16)]] if own is None else [path,global_group]
            _require(set(value['includes'])==set(includes) and set(value['allowed_groups'])==set(allowed),'G1 collision permission differs')
        def phase(name):return [value for value in events if value.get('phase')==name]
        _require(not any(value.get('phase') in ('contact_error','sparse_contact_fault','scale_motion_fault','scale_reset_attempt','scale_reset_after','scale_reset_checked') for value in events),'G1 fault/reset journal entry')
        initial=phase('scale_initial');_require(len(initial)==1,'G1 initial scanner records missing')
        _scans(initial[0]['scanners'],restore_tensor(report['initial_root_positions_w']),origins)
        topology=phase('sparse_contact_topology');_require(len(topology)==1,'G1 compact topology missing/duplicated')
        topology={k:v for k,v in topology[0].items() if k!='phase'}
        _require(topology['num_envs']==16 and topology['sensor_count']==272 and topology['capacity']==131072,'G1 contact scale differs')
        expected_paths={f'/World/envs/env_{i}/Robot/{name}' for i in range(16) for name in report['body_names']}
        _require(len(report['body_names'])==17 and set(topology['sensor_paths'])==expected_paths,'G1 rigid body inventory differs')
        frames=phase('sparse_contact_sample');motion=phase('scale_motion_substep');start=report['start_sim_step']
        for records in (frames,motion):_require([value['sim_step'] for value in records]==list(range(start+1,start+5)),'G1 physics records incomplete')
        ground={str(i):0 for i in range(16)}
        for frame in frames:
            decoded=reconstruct_frame(frame,topology)
            for name,count in decoded['ground_contact_points'].items():ground[name]+=count
        runtime=phase('contact_runtime');stage=report['contact_capture']['stage_id']
        _require(len(runtime)==5 and all(tensor_runtime_passed(value,stage) for value in runtime),'G1 runtime backend evidence incomplete')
        summary=report['contact_capture']
        _require(summary['schema']=='lite3_sparse_tensor_capture_v1' and summary['topology']==topology
            and summary['runtime_checked'] is True and summary['runtime_check_count']==5
            and summary['samples']==4 and summary['poll_steps']==list(range(start+1,start+5))
            and summary['control_poll_counts']==[4] and summary['errors']==[] and summary['fault_written'] is False
            and summary['require_ground_positive'] is False and summary['ground_positive_qualified'] is False,
            'G1 capture summary incomplete or overclaims ground qualification')
        _require(all(summary['ground_contact_points'].get(str(i),0)==ground[str(i)] for i in range(16)),
            'G1 ground counters differ from detailed points')
        roots=None
        for value in motion:
            position=restore_tensor(value['body_positions_w']);velocity=restore_tensor(value['body_link_velocities_w'])
            quat=restore_tensor(value['body_link_quaternions_wxyz']);roots=restore_tensor(value['root_positions_w'])
            _require(position.shape==velocity.shape==(16,17,3) and quat.shape==(16,17,4),'G1 link evidence shape differs')
            _require(torch.allclose(quat.norm(dim=-1),torch.ones(16,17),atol=1e-4,rtol=0),'G1 invalid quaternion')
            _require(torch.equal(restore_tensor(value['origins_w']),origins),'G1 recorded origins differ from assignment grid')
            _require(value['terrain_levels']==[0]*16 and value['terrain_columns']==[t['column'] for t in assignments], 'G1 motion assignment fields changed')
            local=position[:,:,:2]-origins[:,None,:2]
            _require(bool(((local[:,:,0]>=-1.5+.271-1e-6)&(local[:,:,0]<=10.5-.271+1e-6)&(local[:,:,1].abs()<=2-.271+1e-6)).all()),'G1 actual body envelope outside tile')
            _scans(value['scanners'],roots,origins,only_base=True)
        controls=phase('scale_control_scans');_require(len(controls)==1,'G1 final scans missing')
        _scans(controls[0]['scanners'],roots,origins)
        for name in ('application_start_s','scene_build_s','first_step_s','elapsed_seconds_before_shutdown'):
            _require(type(report.get(name)) in (int,float) and 0<report[name]<float('inf'),'invalid G1 timing: '+name)
        _require(not (folder/'contact_fault.npz').exists(),'G1 contact fault snapshot exists')
        result.update(accepted=True,diagnostic_transitions=16,ppo_updates=0,physics_substeps=4,
            ground_contact_points=ground,ground_positive_qualification=False)
    except Exception as error:result['reasons'].append(type(error).__name__+': '+str(error))
    return result
