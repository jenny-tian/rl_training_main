"""Streaming CPU acceptance of G3 motion, contacts, resets and frozen actions."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
import torch
from rl_training.source_scan.scale_motion import contract_for,same_value,validate_predecessor
from rl_training.source_scan.scale_motion_evidence import event_records,digest_stream
from rl_training.source_scan.shared_terrain_bank import SharedBankTerrainAllocator
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.moving_diagnostic import restore_tensor,EXPECTED_TERMS
from rl_training.source_scan.tile_bounds import source_tile_bounds_terminated
from rl_training.source_scan.sparse_contact import ContactDecoder
from rl_training.source_scan.frozen_actor import load_frozen_source_actor
from source_scale_motion_collection import actor_digest
from source_scale_reset_validation import _inventory,_state,_metrics
from source_scale_startup_validation import _require,_load,_scans
from source_parallel_reset_lifecycle import check_transition
from source_shared_tensor_contact import tensor_runtime_passed


def audit_metrics(rows,ages,holds,positions,origins):
    _metrics(rows,ages)
    for i,row in enumerate(rows):
        _require(all(type(row[k]) is bool for k in ('stable_for_completion','unsafe','nonfoot_contact'))
            and row['stable_for_completion']==(not row['unsafe'] and not row['nonfoot_contact']),'G3 stable flag differs')
        _require(abs(row['forward_progress_m']-float(positions[i,0,0]-origins[i,0]))<=1e-5
            and abs(row['lateral_m']-float(positions[i,0,1]-origins[i,1]))<=1e-5,'G3 metric progress differs from actual motion')
        holds[i]=holds[i]+1 if row['forward_progress_m']>=9. and row['stable_for_completion'] else 0
        _require(type(row['lifecycle_hold']) is int and row['lifecycle_hold']==holds[i],'G3 stable hold trace differs')


def audit_terminal(terminal,ended,ages,holds,time_outs,columns,step,main):
    _require(len(time_outs)==120 and all(type(v) is bool for v in time_outs),'invalid G3 timeout mask')
    if not ended:
        _require(terminal is None and not any(time_outs),'G3 spurious terminal/timeout')
        return []
    _require(terminal is not None and terminal['ids']==ended and set(terminal['terms'])==EXPECTED_TERMS,'G3 terminal terms/IDs differ')
    m=len(ended)
    for value in terminal['terms'].values():_require(len(value)==m and all(type(v) is bool for v in value),'invalid terminal term')
    for key in ('terminated','truncated'):_require(len(terminal[key])==120 and all(type(v) is bool for v in terminal[key]),'invalid terminal mask')
    _require(terminal['steps']==ages[ended].tolist() and terminal['hold']==[holds[i] for i in ended]
        and len(terminal['completed'])==m and all(type(v) is bool for v in terminal['completed']),'G3 terminal age/hold differs')
    for key,width in [('policy',45),('critic',393)]:
        value=torch.tensor(terminal['observations'][key]);_require(value.shape==(120,width) and torch.isfinite(value).all(),'invalid terminal observations')
    expected_timeout=[False]*120;witnesses=[]
    for j,i in enumerate(ended):
        terms={k:v[j] for k,v in terminal['terms'].items()}
        terminated=any(v for k,v in terms.items() if k!='source_training_timeout');truncated=terms['source_training_timeout']
        _require(terminated or truncated,'unexplained G3 termination')
        _require(terminal['terminated'][i]==terminated and terminal['truncated'][i]==truncated,'G3 terminal reason masks differ')
        expected_timeout[i]=truncated and not terminated
        completed=terms['source_training_completion'] and not any(v for k,v in terms.items() if k not in ('source_training_completion','source_training_timeout'))
        _require(terminal['completed'][j]==completed,'G3 completed flag differs from reasons')
        if completed:
            _require(holds[i]>=50,'G3 completion lacks stable progress')
            if columns[i]==0 and step<=main:witnesses.append(dict(control_step=step,env_id=i,episode_steps=terminal['steps'][j]))
    _require(time_outs==expected_timeout,'G3 timeout versus ordinary failure differs')
    _require(all(not terminal['terminated'][i] and not terminal['truncated'][i] for i in set(range(120))-set(ended)),'G3 terminal mask outside ended IDs')
    return witnesses


def validate(folder,plan,execution,checkpoint,*,cpu_profile=None):
    result=dict(schema='lite3_source_shared_G3_acceptance_v1',accepted=False,reasons=[],training_ready=False,
        g4_qualified=False,continuous_time_safety_claim=False)
    try:
        folder=Path(folder);report=_load((folder/'result.json').read_text());contract=contract_for()
        if report.get('status')=='failed':raise ValueError('G3 probe failed: '+str(report.get('error')))
        cpu=cpu_profile is not None
        if plan.get('delivery_stage'):
            from rl_training.source_scan.delivery_stage import job_contract
            contract=job_contract('g3',plan)
        _require(cpu or not report.get('synthetic_cpu_fixture'),'synthetic CPU data cannot qualify a real G3 run')
        _require(not cpu or report.get('synthetic_cpu_fixture') is True,'CPU profile requires explicit fixture provenance')
        total,main=(contract['control_steps'],contract['main_control_steps']) if not cpu else cpu_profile
        _require(plan.get('status')=='approved' and plan.get('user_approved') is True,'G3 plan is not approved')
        for k,v in contract.items():_require(same_value(plan.get(k),v),'G3 contract differs: '+k)
        validate_predecessor(plan['predecessor_qualification'],verify_current_inputs=not cpu)
        _require(execution.get('returncode')==0 and execution.get('timed_out') is False
            and execution.get('process_group_empty') is True and execution.get('monitor_errors')==[],'G3 execution failed')
        _require(report.get('status')=='collected' and report.get('simulation_control_steps')==total
            and report.get('completed_diagnostic_transitions')==120*total and report.get('steps_written')==total
            and report.get('base_revision')==plan['base_revision'] and report.get('input_sha256')==plan['input_sha256'],
            'G3 report progress/identity differs')
        _require(set(report['termination_terms'])==EXPECTED_TERMS,'termination term inventory differs')
        world='/World/collisions/global_group';groups=report['collision_groups']
        _require(groups['inverted'] is True and set(groups['groups'])=={world,*[f'/World/collisions/group{i}' for i in range(120)]},'G3 collision group inventory differs')
        for path,value in groups['groups'].items():
            own=None if path==world else int(path.rsplit('group',1)[1])
            _require(set(value['includes'])==set(['/World/ground'] if own is None else [f'/World/envs/env_{own}'])
                and set(value['allowed_groups'])==set([world,*[f'/World/collisions/group{i}' for i in range(120)]] if own is None else [path,world]),'G3 collision permission differs')
        cleanup=_load((folder/'result.cleanup.json').read_text())
        _require(cleanup.get('attempt_id')==plan['attempt_id'] and cleanup.get('environment')=='returned'
            and cleanup.get('application') in ('closing','returned') and cleanup.get('persistence_errors')==[],'G3 cleanup incomplete')
        for k,v in contract.items():_require(same_value(report.get(k),v),'G3 report contract differs: '+k)
        event_path=folder/'events.jsonl.gz';step_path=folder/'result.steps.jsonl'
        _require(event_path.stat().st_size<=contract['events_byte_limit'] and step_path.stat().st_size<=contract['steps_byte_limit'],'G3 file budget exceeded')
        # Steps are bounded to 4GiB; hash them without loading the file into memory.
        def file_digest(path):
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1048576),b''):h.update(block)
            return h.hexdigest()
        _require(report['events_sha256']==file_digest(event_path) and report['events_uncompressed_sha256']==digest_stream(event_path)
            and report['steps_sha256']==file_digest(step_path),'G3 evidence hashes differ')
        inventory=report['reset_snapshot_inventory'];_inventory(inventory,sorted(k for k in inventory if k.startswith('history/')),120)
        oracle=SharedBankTerrainAllocator(120,SOURCE_RESET_BLOCK_COUNTS,131001,bank_rows=1)
        tickets=list(oracle.reset(list(range(120))))
        _require(report['initial_assignments']==[asdict(t) for t in tickets],'G3 initial sampler differs')
        grid=restore_tensor(report['terrain_origins']);default=restore_tensor(report['default_root_state'])
        _require(grid.shape==(1,15,3) and default.shape==(120,13),'G3 terrain/default root dimensions differ')
        previous=(restore_tensor(report['initial_positions_w']),restore_tensor(report['initial_velocities_w']))
        actor=load_frozen_source_actor(Path(checkpoint));fingerprint=actor_digest(actor)
        _require(report['actor_sha256_before']==report['actor_sha256_after']==fingerprint,'G3 actor changed or differs')
        top=None;motions=contacts=controls=resets=runtimes=0;phase_counts=Counter();ground=Counter()
        pending=set();pending_main=set();reset_context=before=after=None;checked=False;awaiting_rebase=False
        ages=torch.zeros(120,dtype=torch.long);holds=[0]*120;step_columns=None;ended=[];witnesses=[];exposure=Counter()
        max_speed=max_delta=max_acceleration=0.
        with step_path.open() as steps:
            for event in event_records(event_path,record_limit=contract['event_record_byte_limit'],uncompressed_limit=contract['events_uncompressed_byte_limit']):
                phase=event.get('phase');phase_counts[phase]+=1
                if phase in ('g3_reset_context','scale_reset_attempt','scale_reset_after','scale_reset_checked','g3_reset_rebase'):
                    _require(event['sim_step']==report['start_sim_step']+motions,'G3 reset timestamp differs')
                if phase=='g3_initial':
                    _require(phase_counts[phase]==1 and motions==0,'duplicate/late G3 initial')
                    _scans(event['scanners'],previous[0][:,0],grid[0,torch.tensor([t.column for t in oracle.assignments])])
                elif phase=='contact_runtime':
                    _require(tensor_runtime_passed(event,report['contact_capture']['stage_id']),'G3 native CUDA backend differs');runtimes+=1
                elif phase=='sparse_contact_topology':
                    _require(top is None,'duplicate topology');top={k:v for k,v in event.items() if k!='phase'}
                    _require(top['num_envs']==120 and top['sensor_count']==2040 and top['capacity']==1048576,'G3 topology dimensions differ')
                    _require(len(report['body_names'])==17 and set(top['sensor_paths'])=={f'/World/envs/env_{i}/Robot/{n}' for i in range(120) for n in report['body_names']},'G3 body coverage differs')
                    decoder=ContactDecoder(top)
                elif phase=='contact_poll_attempt':
                    _require(event['sim_step']==report['start_sim_step']+contacts+1,'contact attempt sequence differs')
                elif phase=='sparse_contact_sample':
                    _require(top is not None and event['sim_step']==report['start_sim_step']+contacts+1,'contact sample sequence differs')
                    if plan.get('contact_digest_scheme')=='sha256_blocks_4096_v1':
                        _require(event.get('schema')=='lite3_sparse_contact_frame_v2'
                            and event.get('dense_digest_scheme')=='sha256_blocks_4096_v1'
                            and (cpu or event.get('native_device')=='cuda:0'),'fast native contact scheme/device differs')
                    else:_require(event.get('schema')=='lite3_sparse_contact_frame_v1','contact encoding differs from the approved plan')
                    decoded=decoder.decode(event);ground.update(decoded['ground_contact_points']);contacts+=1
                elif phase=='g3_motion_substep':
                    _require(not awaiting_rebase and contacts==motions+1 and event['sim_step']==report['start_sim_step']+motions+1
                        and event['control_step']==motions//4+1,'G3 motion order differs')
                    if motions%4==0:step_columns=[t.column for t in oracle.assignments];ended=[]
                    pos,vel,quat=[restore_tensor(event[k]) for k in ('body_positions_w','body_link_velocities_w','body_link_quaternions_wxyz')]
                    _require(pos.shape==vel.shape==(120,17,3) and quat.shape==(120,17,4)
                        and torch.allclose(quat.norm(dim=-1),torch.ones(120,17),atol=1e-4,rtol=0),'G3 body tensor invalid')
                    columns=torch.tensor([t.column for t in oracle.assignments]);origins=grid[0,columns]
                    _require(event['terrain_levels']==[0]*120 and event['terrain_columns']==columns.tolist()
                        and torch.equal(restore_tensor(event['origins_w']),origins),'G3 motion assignment differs')
                    diagnostics={};source_tile_bounds_terminated(pos,vel,grid,torch.zeros(120,dtype=torch.long),columns,
                        size=(12.,4.),flat_length_m=3.,collision_radius_m=.27,numerical_margin_m=.001,control_period_s=.02,diagnostics=diagnostics)
                    _require(not diagnostics['outside_now'].any() and not diagnostics['invalid_state'].any()
                        and event['outside_prediction']==diagnostics['outside_prediction'].tolist(),'G3 actual boundary/guard evidence differs')
                    delta=pos-previous[0];acceleration=(vel-previous[1])/.005
                    _require(torch.allclose(restore_tensor(event['displacements_m']),delta,atol=2e-6,rtol=2e-6)
                        and torch.allclose(restore_tensor(event['finite_difference_acceleration_mps2']),acceleration,atol=1e-4,rtol=2e-6),'G3 derivatives/reset rebase differ')
                    roots=restore_tensor(event['root_positions_w']);_require(torch.equal(roots,pos[:,0]),'G3 root/body mapping differs')
                    if plan.get('delivery_stage'):
                        from rl_training.source_scan.scan_footprint import scanner_boundary
                        guard=event['scanner_guard'];scan_diagnostic={}
                        scanner_boundary(roots,quat[:,0],restore_tensor(guard['root_linear_velocity_w']),
                            restore_tensor(guard['root_yaw_rate_w']),origins,diagnostics=scan_diagnostic)
                        _require(guard['current']==scan_diagnostic['current'].tolist()
                            and guard['predicted']==scan_diagnostic['predicted'].tolist(),'scanner footprint guard does not reconstruct')
                    _scans(event['scanners'],roots,origins,only_base=True)
                    _require(event['recovery_ids']==sorted(pending),'G3 reset first physics step missing')
                    pending.clear();pending_main.clear();previous=(pos,vel);motions+=1
                    last_positions=pos.clone();last_origins=origins.clone()
                    max_speed=max(max_speed,float(vel.norm(dim=-1).max()));max_delta=max(max_delta,float(delta.norm(dim=-1).max()))
                    max_acceleration=max(max_acceleration,float(acceleration.norm(dim=-1).max()))
                elif phase=='g3_reset_context':
                    _require(not ended and reset_context is None and not awaiting_rebase and motions==4*(controls+1)
                        and event['control_step']==controls+1,'G3 reset context/order differs')
                    reset_context=event;ended=event['ids']
                elif phase=='scale_reset_attempt':
                    _require(reset_context is not None and before is None and event['ids']==ended,'G3 reset attempt differs')
                    before=_state(event['before'],inventory)
                    _require(torch.equal(before['episode'],ages+1) and not before['sensor/contact_forces/_is_outdated'].any(),'G3 reset baseline not current')
                elif phase=='scale_reset_after':
                    _require(before is not None and after is None and event['ids']==ended,'G3 reset after differs')
                    after=_state(event['after'],inventory);check_transition(before,after,ended)
                elif phase=='scale_reset_checked':
                    _require(after is not None and not checked and event['ids']==ended and event['batch']==resets+1,'G3 reset check differs')
                    audited=_state(event['after'],inventory);_require(all(torch.equal(after[k],audited[k]) for k in after),'G3 checked snapshot changed')
                    for k in after:
                        if k.startswith(('filter/','lifecycle/')):_require(not after[k][ended].any(),'G3 selected memory not cleared')
                    tickets.extend(oracle.reset(ended));_require(event['assignments']==[asdict(t) for t in oracle.assignments],'G3 reset sampler differs')
                    origins=grid[0,torch.tensor([t.column for t in oracle.assignments])]
                    _require(torch.equal(after['origin'],origins) and torch.allclose(after['root'][ended,:3],default[ended,:3]+origins[ended],atol=1e-5,rtol=0),'G3 reset spawn differs')
                    pending.update(ended)
                    if controls+1<=main:pending_main.update(ended)
                    checked=True;awaiting_rebase=True;resets+=1
                elif phase=='g3_reset_rebase':
                    _require(awaiting_rebase and event['ids']==ended,'G3 missing/changed reset rebase')
                    rebased=restore_tensor(event['positions']);velocity=restore_tensor(event['velocities'])
                    _require(rebased.shape==velocity.shape==(len(ended),17,3)
                        and torch.allclose(rebased[:,0],after['root'][ended,:3],atol=1e-5,rtol=0),'G3 rebase pose differs')
                    previous[0][ended]=rebased;previous[1][ended]=velocity
                    reset_context=before=after=None;checked=awaiting_rebase=False
                elif phase=='g3_control_scans':
                    _require(not awaiting_rebase and reset_context is None and motions==4*(controls+1) and event['step']==controls+1
                        and event['sim_step']==report['start_sim_step']+motions,'G3 control order differs')
                    origins=grid[0,torch.tensor([t.column for t in oracle.assignments])]
                    _scans(event['scanners'],restore_tensor(event['roots']),origins)
                    line=steps.readline(contract['event_record_byte_limit']+1)
                    _require(bool(line) and len(line.encode())<=contract['event_record_byte_limit'] and line.endswith('\n'),'G3 step row missing/incomplete/oversized')
                    row=_load(line);step=controls+1
                    _require(row['step']==step and row['attempt_id']==plan['attempt_id'] and row['columns_before']==step_columns,'G3 step identity differs')
                    done=[i for i,v in enumerate(row['done']) if v];_require(len(row['done'])==120 and all(type(v) is int and v in (0,1) for v in row['done']) and done==ended,'G3 done/reset IDs differ')
                    audit_metrics(row['metric_rows'],ages+1,holds,last_positions,last_origins)
                    witnesses.extend(audit_terminal(row['terminal'],ended,ages+1,holds,row['time_outs'],step_columns,step,main))
                    policy=restore_tensor(row['policy_observation']);mu=restore_tensor(row['actor_mean']);noise=restore_tensor(row['noise'])
                    _require(policy.shape==(120,45) and mu.shape==noise.shape==(120,12)
                        and torch.allclose(actor(policy),mu,atol=3e-5,rtol=3e-5),'G3 frozen actor output differs')
                    raw_action=torch.tensor(row['raw_action']);executed=torch.tensor(row['executed_action'])
                    _require(raw_action.shape==executed.shape==(120,12) and torch.isfinite(executed).all()
                        and torch.allclose(raw_action,mu+noise,atol=1e-6,rtol=1e-6),'G3 action replay differs')
                    _require(row['paused']==[[False]]*120 and len(row['projected'])==120
                        and all(len(v)==1 and type(v[0]) is bool for v in row['projected']),'G3 invalid pause/projection trace')
                    ages+=1;ages[ended]=0
                    for i in ended:holds[i]=0
                    _require(row['episode_counters']==ages.tolist(),'G3 post-reset episode ages differ')
                    exposure.update(step_columns);controls+=1
                elif phase=='contact_summary':
                    _require(phase_counts[phase]==1 and event['samples']==contacts,'contact summary differs')
                else:raise ValueError('G3 unexpected/fault event: '+str(phase))
            _require(not steps.readline(),'extra G3 step row')
        _require(controls==total and contacts==motions==4*total and runtimes==4*total+1
            and resets==report['reset_batches']==report['reset_attempts'] and report['motion_samples']==motions,'G3 evidence incomplete')
        _require(all(phase_counts[k]==1 for k in ('g3_initial','sparse_contact_topology','contact_summary'))
            and phase_counts['contact_poll_attempt']==contacts,'G3 initial/summary/attempt evidence incomplete')
        _require(not pending_main and report['pending_main_recovery_ids']==[] and report['pending_recovery_ids']==sorted(pending),'G3 main reset recovery missing')
        _require(report['all_assignment_tickets']==[asdict(t) for t in tickets] and set(exposure)==set(range(15)),'G3 assignment/cell exposure incomplete')
        _require(report['flat_completion_witnesses']==witnesses and (cpu or bool(witnesses)),'G3 flat progress/hold witness missing')
        summary=report['contact_capture']
        _require(summary['errors']==[] and summary['cross_environment_contacts']==0 and summary['ground_positive_qualified'] is True
            and summary['require_ground_positive'] is True and summary['topology']==top and summary['samples']==motions
            and summary['runtime_check_count']==runtimes and summary['poll_steps']==list(range(report['start_sim_step']+1,report['start_sim_step']+motions+1))
            and summary['diagnostic_scope']=='G3_motion_integrity' and not (folder/'contact_fault.npz').exists()
            and summary['control_poll_counts']==[4]*total and all(ground[str(i)]>0 for i in range(120))
            and all(summary['ground_contact_points'].get(str(i),0)==ground[str(i)] for i in range(120)),'G3 all-environment contact positive control incomplete')
        for key,value in [('max_body_speed_mps',max_speed),('max_body_displacement_m',max_delta),('max_finite_difference_acceleration_mps2',max_acceleration)]:
            _require(abs(report[key]-value)<=1e-5*max(1.,value),'G3 motion extrema differ')
        result.update(accepted=not cpu,cpu_evidence_validated=cpu,motion_integrity_qualified=not cpu,
            diagnostic_transitions=120*total,physics_substeps=4*total,reset_batches=resets,
            deferred_final_recovery_ids=sorted(pending),flat_completion_witnesses=witnesses,ground_positive_qualified=not cpu,
            ppo_updates=0,scope='bounded_main_motion_and_eligible_reset_recovery')
    except Exception as error:result['reasons'].append(type(error).__name__+': '+str(error))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);args=parser.parse_args()
    load=lambda n:_load((args.folder/n).read_text())
    result=validate(args.folder,load('run_manifest.json'),load('execution.json'),args.checkpoint)
    from rl_training.source_scan.smoke_artifacts import atomic_json
    atomic_json(args.folder/'acceptance.json',result);print(json.dumps(result,indent=2));raise SystemExit(0 if result['accepted'] else 1)
