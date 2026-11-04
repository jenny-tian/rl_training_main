"""CPU acceptance of reset boundaries, replayed tickets and every recovery substep."""
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import torch

from rl_training.source_scan.scale_reset import contract_for,same_value,validate_predecessor
from rl_training.source_scan.shared_terrain_bank import SharedBankTerrainAllocator
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.moving_diagnostic import restore_tensor
from rl_training.source_scan.sparse_contact import reconstruct_frame
from source_parallel_reset_lifecycle import check_transition,summarize_tickets
from source_shared_tensor_contact import tensor_runtime_passed
from source_scale_startup_validation import _require,_load,_scans


def _inventory(value,history_terms,n):
    required={'row':([n],'torch.int64'),'column':([n],'torch.int64'),'origin':([n,3],'torch.float32'),
        'root':([n,13],'torch.float32'),'joint_pos':([n,12],'torch.float32'),'joint_vel':([n,12],'torch.float32'),
        'episode':([n],'torch.int64'),'action':([n,12],'torch.float32'),'previous_action':([n,12],'torch.float32'),
        'command/base_velocity':([n,3],'torch.float32'),'cache/contact_forces/net_forces_w':([n,17,3],'torch.float32'),
        'cache/contact_forces/net_forces_w_history':([n,3,17,3],'torch.float32'),
        'filter/filter_previous_residual_action':([n,12],'torch.float32'),'filter/filter_paused':([n,1],'torch.bool'),
        'filter/filter_stable_steps':([n,1],'torch.int64'),'filter/filter_contact_steps':([n,1],'torch.int64')}
    for name in ('hold','contact_steps','completed','unsafe','timeout'):
        required['lifecycle/'+name]=([n],'torch.int64' if name in ('hold','contact_steps') else 'torch.bool')
    for key,(shape,dtype) in required.items():
        _require(value.get(key)==dict(shape=shape,dtype=dtype),'missing or changed snapshot field: '+key)
    for name in ('height_scanner','height_scanner_base','contact_forces'):
        for attr in ('_timestamp','_timestamp_last_update','_is_outdated'):
            field=value.get('sensor/'+name+'/'+attr,{})
            _require(field.get('shape')==[n] and field.get('dtype') in
                (('torch.bool',) if attr=='_is_outdated' else ('torch.float32','torch.float64')),'sensor timestamp inventory differs')
    for name in ('height_scanner','height_scanner_base'):
        hits=value.get('cache/'+name+'/ray_hits_w',{});ids=value.get('cache/'+name+'/ray_mesh_ids',{})
        shape=hits.get('shape',[])
        _require(len(shape)==3 and shape[0]==n and shape[1]>0 and shape[2]==3
            and hits.get('dtype')=='torch.float32' and ids==dict(shape=[n,shape[1],1],dtype='torch.int16'),
            'scanner cache inventory differs')
    _require(history_terms==sorted(key for key in value if key.startswith('history/')),'history inventory differs')
    _require({key.removeprefix('pushes/') for key in value if key.startswith('pushes/')}==
        {key.removeprefix('history/') for key in history_terms},'history push inventory differs')
    for key in history_terms:
        _require(value['pushes/'+key.removeprefix('history/')]==dict(shape=[n],dtype='torch.int64'), 'history pushes missing')
    for key,field in value.items():
        _require(isinstance(field,dict) and field.get('shape',[None])[0]==n,'snapshot environment dimension differs: '+key)


def _state(records,inventory):
    _require(set(records)==set(inventory),'snapshot field inventory changed')
    result={}
    for key,record in records.items():
        _require({k:record.get(k) for k in ('shape','dtype')}==inventory[key],'snapshot dtype/shape differs: '+key)
        value=restore_tensor(record)
        _require(same_value(record['values'],value.tolist()),'snapshot values are not lossless for dtype: '+key)
        result[key]=value
    return result


def _metrics(rows,ages):
    _require(len(rows)==len(ages),'G2 metric rows missing')
    for i,row in enumerate(rows):
        _require(type(row.get('step')) is int and row['step']==int(ages[i]),'metric episode age differs')
        for key in ('forward_progress_m','lateral_m','velocity_error_mps','forward_velocity_mps','roll_rad','pitch_rad'):
            _require(type(row.get(key)) in (int,float) and math.isfinite(row[key]),'G2 metric invalid: '+key)
        for key in ('foot_speeds_mps','foot_horizontal_forces_N'):
            _require(len(row[key])==4 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in row[key]),'G2 foot metric invalid')
        _require(len(row['foot_contact'])==4 and all(type(v) is bool for v in row['foot_contact']),'G2 foot mask invalid')


def validate(folder,plan,execution):
    result=dict(schema='lite3_source_shared_scale_G2_acceptance_v2',accepted=False,reasons=[],training_ready=False,
        g3_g4_qualified=False,ground_positive_qualification=False,scope='120_env_four_selective_resets_and_first_recovery_step_only')
    try:
        contract=contract_for(plan.get('num_envs'));n=120
        if plan.get('delivery_stage'):
            from rl_training.source_scan.delivery_stage import job_contract
            contract=job_contract('g2',plan)
        _require(plan.get('status')=='approved' and plan.get('user_approved') is True,'G2 plan is not approved')
        for key,value in contract.items():_require(same_value(plan.get(key),value),'G2 plan mismatch: '+key)
        validate_predecessor(plan.get('predecessor_qualification'))
        folder=Path(folder);report=_load((folder/'result.json').read_text())
        # A child may close Isaac normally with exit code zero after an exception.
        # Surface its original failure before secondary incomplete-report checks.
        if report.get('status')=='failed':
            result['probe_error']=report.get('error','')
            _require(False,'G2 probe failed: '+str(report.get('error') or 'missing original error'))
        _require(execution.get('returncode')==0 and execution.get('process_group_empty') is True
            and execution.get('timed_out') is False and execution.get('monitor_errors')==[],'G2 process/monitor failed')
        cleanup=_load((folder/'result.cleanup.json').read_text())
        steps_raw=(folder/'result.steps.jsonl').read_bytes();events_raw=(folder/'events.jsonl').read_bytes()
        _require(len(events_raw)<=contract['events_byte_limit'],'G2 event byte limit exceeded')
        rows=[_load(line) for line in steps_raw.splitlines()];events=[_load(line) for line in events_raw.splitlines()]
        expected=contract|dict(status='passed',base_revision=plan['base_revision'],input_sha256=plan['input_sha256'],
            simulation_control_steps=4,completed_diagnostic_transitions=480,phase='result_before_shutdown',
            steps_written=4,motion_samples=16,reset_attempts=4,reset_batches=4,pending_recovery_ids=[])
        for key,value in expected.items():_require(same_value(report.get(key),value),'G2 report mismatch: '+key)
        _require(report['steps_file']=='result.steps.jsonl' and len(rows)==4
            and report['steps_sha256']==hashlib.sha256(steps_raw).hexdigest()
            and report['events_sha256']==hashlib.sha256(events_raw).hexdigest(),'G2 journal identity/hash differs')
        _require(cleanup.get('attempt_id')==plan['attempt_id'] and cleanup.get('environment')=='returned'
            and cleanup.get('application') in ('closing','returned') and cleanup.get('persistence_errors')==[],'G2 cleanup incomplete')
        _require(report['observation_widths']==dict(policy=45,critic=393),'G2 observation widths differ')
        inventory=report['reset_snapshot_inventory'];_inventory(inventory,report['observation_history_terms'],n)
        oracle=SharedBankTerrainAllocator(n,SOURCE_RESET_BLOCK_COUNTS,131001,bank_rows=1)
        tickets=list(oracle.reset(list(range(n))));_require(report['initial_assignments']==[asdict(t) for t in tickets],'initial G2 assignments differ')
        grid=restore_tensor(report['terrain_origins']);default=restore_tensor(report['default_root_state'])
        _require(grid.shape==(1,15,3) and default.shape==(n,13),'G2 terrain/default root shape differs')
        def origins():return grid[0,torch.tensor([t.column for t in oracle.assignments])]
        groups=report['collision_groups'];world='/World/collisions/global_group'
        _require(groups.get('inverted') is True and set(groups['groups'])=={world,*[f'/World/collisions/group{i}' for i in range(n)]},'G2 collision groups differ')
        for path,value in groups['groups'].items():
            own=None if path==world else int(path.rsplit('group',1)[1])
            _require(set(value['includes'])==set(['/World/ground'] if own is None else [f'/World/envs/env_{own}'])
                and set(value['allowed_groups'])==set([world,*[f'/World/collisions/group{i}' for i in range(n)]] if own is None else [path,world]),'G2 collision permission differs')
        def phase(name):return [v for v in events if v.get('phase')==name]
        _require(not any(v.get('phase') in ('contact_error','sparse_contact_fault','scale_motion_fault') for v in events),'G2 fault journal entry')
        initial=phase('scale_initial');_require(len(initial)==1,'G2 initial scanner record missing')
        _scans(initial[0]['scanners'],restore_tensor(report['initial_root_positions_w']),origins())
        topology=phase('sparse_contact_topology');_require(len(topology)==1,'G2 topology missing or duplicated')
        topology={k:v for k,v in topology[0].items() if k!='phase'}
        _require(topology['num_envs']==n and topology['sensor_count']==2040 and topology['capacity']==1048576,'G2 contact scale differs')
        _require(len(report['body_names'])==17 and set(topology['sensor_paths'])==
            {f'/World/envs/env_{i}/Robot/{name}' for i in range(n) for name in report['body_names']},'G2 rigid body inventory differs')
        frames=phase('sparse_contact_sample');motion=phase('scale_motion_substep');start=report['start_sim_step']
        for records in (frames,motion):_require([v['sim_step'] for v in records]==list(range(start+1,start+17)),'G2 physics records incomplete')
        attempts,afters,checked,reset_scans,controls=(phase(name) for name in
            ('scale_reset_attempt','scale_reset_after','scale_reset_checked','g2_reset_scans','scale_control_scans'))
        _require(all(len(v)==4 for v in (attempts,afters,checked,reset_scans,controls)),'G2 reset/control records incomplete')
        ages=torch.zeros(n,dtype=torch.long);ground={str(i):0 for i in range(n)}
        for batch,ids in enumerate(contract['planned_reset_batches']):
            step=batch+1;tick=start+4*batch;selected=torch.tensor(ids)
            for record in (attempts[batch],afters[batch],checked[batch],reset_scans[batch]):
                _require(same_value(record.get('ids'),ids) and record['sim_step']==tick,'G2 reset schedule differs')
            _require(checked[batch]['batch']==step and reset_scans[batch]['batch']==step,'G2 reset batch identity differs')
            before=_state(attempts[batch]['before'],inventory);after=_state(afters[batch]['after'],inventory)
            _require(not bool(before['sensor/contact_forces/_is_outdated'].any())
                and torch.equal(before['sensor/contact_forces/_timestamp'],before['sensor/contact_forces/_timestamp_last_update']),
                'G2 contact cache baseline was not refreshed')
            audited=_state(checked[batch]['after'],inventory)
            _require(all(torch.equal(after[k],audited[k]) for k in after),'checked reset state differs from actual after-state')
            _require(torch.equal(before['episode'],ages) and before['column'].tolist()==[t.column for t in oracle.assignments]
                and torch.equal(before['origin'],origins()),'reset before-state differs from previous assignment/age')
            check_transition(before,after,ids)
            for key in after:
                if key.startswith(('filter/','lifecycle/')):_require(not bool(after[key][selected].any()),'G2 reset memory not cleared: '+key)
            tickets.extend(oracle.reset(ids));current=[asdict(t) for t in oracle.assignments]
            _require(checked[batch]['assignments']==current,'reset tickets differ from independent replay')
            _require(after['row'].tolist()==[0]*n and after['column'].tolist()==[t.column for t in oracle.assignments]
                and torch.equal(after['origin'],origins()),'reset terrain assignment/origin differs')
            _require(torch.allclose(after['root'][selected,:3],default[selected,:3]+origins()[selected],atol=1e-5,rtol=0),'reset spawn pose differs')
            _scans(reset_scans[batch]['scanners'],restore_tensor(reset_scans[batch]['roots']),origins())
            record_order=[events.index(v) for v in (attempts[batch],afters[batch],checked[batch],reset_scans[batch])]
            _require(record_order==sorted(set(record_order)) and record_order[-1]<min(events.index(frames[4*batch]),events.index(motion[4*batch])),
                'G2 reset/recovery event ordering differs')
            if batch:_require(events.index(controls[batch-1])<record_order[0],'G2 batches overlap')
            for offset in range(4):
                value=motion[4*batch+offset];_require(value['recovery_ids']==(ids if offset==0 else []),'reset first physics step not observed')
                position=restore_tensor(value['body_positions_w']);velocity=restore_tensor(value['body_link_velocities_w'])
                quat=restore_tensor(value['body_link_quaternions_wxyz']);roots=restore_tensor(value['root_positions_w'])
                _require(position.shape==velocity.shape==(n,17,3) and quat.shape==(n,17,4)
                    and torch.allclose(quat.norm(dim=-1),torch.ones(n,17),atol=1e-4,rtol=0),'G2 link evidence invalid')
                _require(torch.equal(restore_tensor(value['origins_w']),origins()) and value['terrain_levels']==[0]*n
                    and value['terrain_columns']==[t.column for t in oracle.assignments],'G2 recovery assignment differs')
                local=position[:,:,:2]-origins()[:,None,:2]
                _require(bool(((local[:,:,0]>=-1.5+.271-1e-6)&(local[:,:,0]<=10.5-.271+1e-6)&(local[:,:,1].abs()<=2-.271+1e-6)).all())
                    and value['outside_prediction']==[False]*n,'G2 recovery envelope outside tile')
                _scans(value['scanners'],roots,origins(),only_base=True)
                decoded=reconstruct_frame(frames[4*batch+offset],topology)
                for name,count in decoded['ground_contact_points'].items():ground[name]+=count
            control=controls[batch];_require(control['step']==step and control['sim_step']==tick+4,'G2 control identity differs')
            _require(events.index(control)>max(events.index(frames[4*batch+3]),events.index(motion[4*batch+3])),'control scans precede physics')
            _scans(control['scanners'],roots,origins())
            row=rows[batch];_require(row['step']==step and row['attempt_id']==plan['attempt_id'] and row['done']==[0]*n
                and row['time_outs']==[False]*n,'G2 unexpected terminal/control identity')
            for key in ('raw_action','executed_action'):_require(row[key]==[[0.]*12 for _ in range(n)],'G2 action differs from zero')
            for key in ('projected','paused'):_require(row[key]==[[False] for _ in range(n)],'G2 action projection/pause')
            ages[selected]=0;ages+=1;_metrics(row['metric_rows'],ages)
        expected_summary=summarize_tickets(tickets)
        _require(report['all_assignment_tickets']==[asdict(t) for t in tickets] and report['ticket_summary']==expected_summary
            and expected_summary['assignment_count']==304 and expected_summary['complete_blocks']==2,'G2 cumulative tickets differ')
        summary=report['contact_capture'];runtime=phase('contact_runtime')
        _require(len(runtime)==17 and all(v.get('stage_id')==summary['stage_id']
            and tensor_runtime_passed(v,summary['stage_id']) for v in runtime),'G2 runtime backend incomplete')
        _require(summary['schema']=='lite3_sparse_tensor_capture_v1' and summary['topology']==topology
            and summary['diagnostic_scope']=='G2_reset_recovery' and summary['runtime_checked'] is True
            and summary['runtime_check_count']==17 and summary['samples']==16 and summary['poll_steps']==list(range(start+1,start+17))
            and summary['control_poll_counts']==[4]*4 and summary['errors']==[] and summary['fault_written'] is False
            and summary['require_ground_positive'] is False and summary['ground_positive_qualified'] is False,'G2 contact summary differs')
        _require(all(summary['ground_contact_points'].get(str(i),0)==ground[str(i)] for i in range(n)),'G2 ground counters differ')
        for key in ('reset_latency_s','recovery_control_latency_s'):
            _require(len(report[key])==4 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in report[key]),'G2 latency samples invalid')
        for key in ('application_start_s','scene_build_s','first_step_s','elapsed_seconds_before_shutdown'):
            _require(type(report[key]) in (int,float) and math.isfinite(report[key]) and report[key]>0,'G2 timing invalid')
        _require(report['first_step_s']==report['recovery_control_latency_s'][0] and not (folder/'contact_fault.npz').exists(),'G2 first step/fault differs')
        result.update(accepted=True,reset_boundary_qualified=True,reset_first_step_qualified=True,diagnostic_transitions=480,
            physics_substeps=16,ppo_updates=0,reset_batches=4,selected_environment_resets=184,assignment_count=304,
            complete_assignment_blocks=2,tail_assignment_count=64,ground_contact_points=ground)
        if plan.get('startup_covered_by_g2') is True:
            # Above checks include all 120 initial assignments/scans, 121
            # collision groups, the complete native body inventory and the
            # first zero-action physics interval, as well as all four resets.
            result['startup_and_first_zero_control_qualified']=True
    except Exception as error:result['reasons'].append(type(error).__name__+': '+str(error))
    return result
