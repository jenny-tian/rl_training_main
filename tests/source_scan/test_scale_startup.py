from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from scale_cpu_fixture import make_environment,install_tensor_backend,BODY_NAMES,GROUND,ROOT
from rl_training.source_scan.scale_startup import CONTRACT,proposal,validate_plan
from rl_training.source_scan.scale_diagnostic import ScaleObserver,reset_batches
from rl_training.source_scan.moving_diagnostic import EventJournal,tensor_record
from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter,finalize_and_close
from rl_training.source_scan.episode_metrics import native_metric_rows
from source_parallel_reset_lifecycle import snapshot,check_transition,summarize_tickets
from source_sparse_tensor_contact import SparseTensorContactCapture
from source_shared_contact_report import capture_substeps
from source_scale_startup_validation import validate
import source_scale_startup_supervisor as supervisor


def cpu_plan(output,num_envs=16):
    return proposal(revision='a'*40,remote='a'*40,inputs={str(i):'b'*64 for i in range(238)},
        dependencies={str(i):'c'*64 for i in range(44)},budget_reference={'path':'cpu_fixture','sha256':'d'*64},output_dir=output,num_envs=num_envs)


def build_g1_fixture(folder,*,contact_fault=None,num_envs=16):
    folder.mkdir();plan=cpu_plan(folder,num_envs)|dict(status='approved',user_approved=True)
    env,interface,_,wrapped=make_environment(num_envs);runtime=install_tensor_backend(env,interface,fault=contact_fault)
    report=plan|dict(status='running',simulation_control_steps=0,cpu_fixture=True,
        application_start_s=.1,scene_build_s=.2,first_step_s=.3,elapsed_seconds_before_shutdown=.6)
    writer=SmokeArtifactWriter(folder/'result.json',report,schema=CONTRACT['schema'])
    with EventJournal(folder/'events.jsonl') as journal:
        observer=ScaleObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        sensors=sorted(f'/World/envs/env_{i}/Robot/{name}' for i in range(num_envs) for name in BODY_NAMES)
        capture=SparseTensorContactCapture(physics_view=env.sim.physics_sim_view,ground_colliders=[GROUND],num_envs=num_envs,
            physics_dt=.005,fault_path=folder/'contact_fault.npz',require_ground_positive=False,
            stage_id=73,reporting_bodies=sensors,record=journal.record,start_step=12,decimation=4,control_steps=1)
        global_path='/World/collisions/global_group';groups={global_path:dict(includes=['/World/ground'],allowed_groups=[global_path,*[f'/World/collisions/group{i}' for i in range(num_envs)]])}
        groups.update({f'/World/collisions/group{i}':dict(includes=[f'/World/envs/env_{i}'],allowed_groups=[f'/World/collisions/group{i}',global_path]) for i in range(num_envs)})
        report.update(initial_assignments=[asdict(t) for t in observer.tickets],start_sim_step=12,
            terrain_origins=tensor_record(env.scene.terrain.terrain_origins),body_names=BODY_NAMES,
            initial_root_positions_w=tensor_record(env.scene['robot'].data.root_link_pos_w),
            collision_groups={'inverted':True,'groups':groups},observation_widths={'policy':45,'critic':393})
        journal.record(dict(phase='scale_initial',scanners=observer.scans()))
        with capture_substeps(env,interface,capture,runtime),observer.installed():
            raw=torch.zeros(num_envs,12);_,_,done,extras=wrapped.step(raw);capture.check_control(env,1)
            journal.record(dict(phase='scale_control_scans',sim_step=env._sim_step_counter,scanners=observer.scans()));journal.flush()
            writer.append_step(dict(step=1,done=done.tolist(),time_outs=extras['time_outs'].tolist(),raw_action=raw.tolist(),
                executed_action=wrapped.last_trace.executed_action.tolist(),projected=wrapped.last_trace.decision.projected.tolist(),
                paused=wrapped.last_trace.decision.paused.tolist(),metric_rows=native_metric_rows(env)))
        report.update(status='passed',simulation_control_steps=1,contact_capture=capture.summary(),motion_samples=observer.samples,reset_batches=observer.resets,pending_recovery_ids=[])
    report['events_sha256']=hashlib.sha256((folder/'events.jsonl').read_bytes()).hexdigest()
    finalize_and_close(writer,report,environment_close=lambda:None,application_close=lambda:None)
    return plan,dict(returncode=0,process_group_empty=True,timed_out=False,monitor_errors=[])


@pytest.mark.parametrize('fault',[None,'approval','scale','step','capacity','retry','resume','device','revision','budget'])
def test_single_g1_contract_does_not_allow_expansion_or_reuse(tmp_path,fault):
    output=tmp_path/'new';plan=cpu_plan(output)|dict(status='approved',user_approved=True)
    fields={'approval':('user_approved',False),'scale':('num_envs',32),'step':('control_steps',2),
        'capacity':('contact_capacity',262144),'retry':('retry',True),'resume':('resume',True),
        'device':('device','cuda:1'),'revision':('base_revision','f'*40),'budget':('budget_ledger',{})}
    if fault in fields:
        key,value=fields[fault];plan[key]=value
    kwargs=dict(revision='a'*40,inputs=cpu_plan(output)['input_sha256'],dependencies=cpu_plan(output)['dependency_sha256'],
        budget_reference=cpu_plan(output)['budget_ledger'],output_dir=output)
    if fault is None:assert validate_plan(plan,**kwargs)==plan
    else:
        with pytest.raises((ValueError,PermissionError)):validate_plan(plan,**kwargs)


@pytest.mark.parametrize('target',['parent','child'])
def test_unapproved_manifest_rejected_before_network_gpu_or_simulator_import(tmp_path,monkeypatch,target):
    path=tmp_path/'proposal.json';path.write_text(json.dumps(cpu_plan(tmp_path/'output')))
    def forbidden(*a,**kw):raise AssertionError('backend before approval')
    monkeypatch.setattr(supervisor,'_git_state',forbidden)
    monkeypatch.setattr(subprocess,'check_output',forbidden)
    before=set(sys.modules)
    with pytest.raises(PermissionError):supervisor.preflight(NS(manifest=path),child=target=='child')
    assert not any(name.startswith(('isaaclab','isaacsim','omni.')) for name in set(sys.modules)-before)
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope='module')
def g1_fixture(tmp_path_factory):
    folder=tmp_path_factory.mktemp('G1_cpu')/'attempt';plan,execution=build_g1_fixture(folder)
    return folder,plan,execution


def test_complete_g1_cpu_fixture_is_accepted_only_for_its_startup_scope(g1_fixture):
    folder,plan,execution=g1_fixture;result=validate(folder,plan,execution)
    assert result['accepted'],result
    assert result['ppo_updates']==0 and result['diagnostic_transitions']==16
    assert result['training_ready'] is False and result['g3_g4_qualified'] is False
    assert result['ground_positive_qualification'] is False


@pytest.mark.parametrize('fault',['missing_frame','duplicate_frame','changed_force','missing_motion','outside_tile',
    'stale_scan','initial_scan','foreign_mesh','wrong_origin','collision_permission','action','reset','metrics','metric_nan',
    'runtime_summary','exit','cleanup','empty'])
def test_parent_rejects_incomplete_or_corrupt_evidence_even_with_new_file_digest(tmp_path,g1_fixture,fault):
    import shutil
    original,plan,execution=g1_fixture;execution=deepcopy(execution)
    folder=tmp_path/'changed';shutil.copytree(original,folder)
    report=json.loads((folder/'result.json').read_text())
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    frames=[v for v in events if v.get('phase')=='sparse_contact_sample'];motion=[v for v in events if v.get('phase')=='scale_motion_substep']
    if fault=='missing_frame':events.remove(frames[-1])
    if fault=='duplicate_frame':frames[-1]['sim_step']=frames[0]['sim_step']
    if fault=='changed_force':frames[0]['pair_forces_N'][0][2]+=1
    if fault=='missing_motion':events.remove(motion[-1])
    if fault=='outside_tile':motion[0]['body_positions_w']['values'][15][0][1]+=10
    if fault=='stale_scan':motion[0]['scanners']['height_scanner_base']['age_s']['values'][15]=.005
    if fault=='initial_scan':events=[v for v in events if v.get('phase')!='scale_initial']
    if fault=='foreign_mesh':motion[0]['scanners']['height_scanner_base']['mesh_ids']['values'][15][0]=18
    if fault=='wrong_origin':motion[0]['origins_w']['values'][15][0]+=1
    if fault=='collision_permission':report['collision_groups']['groups']['/World/collisions/group0']['allowed_groups'].append('/World/collisions/group15')
    if fault=='action':
        row=json.loads((folder/'result.steps.jsonl').read_text());row['executed_action'][15][0]=.1
        (folder/'result.steps.jsonl').write_text(json.dumps(row)+'\n')
    if fault=='reset':report['reset_batches']=1
    if fault=='metrics':
        row=json.loads((folder/'result.steps.jsonl').read_text());row['metric_rows'].pop()
        (folder/'result.steps.jsonl').write_text(json.dumps(row)+'\n')
    if fault=='metric_nan':
        row=json.loads((folder/'result.steps.jsonl').read_text());row['metric_rows'][15]['roll_rad']=float('nan')
        (folder/'result.steps.jsonl').write_text(json.dumps(row)+'\n')
    if fault=='runtime_summary':report['contact_capture']['runtime_check_count']=4
    if fault=='exit':execution['returncode']=1
    if fault=='cleanup':(folder/'result.cleanup.json').write_text('{}')
    if fault=='empty':events=[]
    (folder/'events.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in events))
    report['events_sha256']=hashlib.sha256((folder/'events.jsonl').read_bytes()).hexdigest()
    report['steps_sha256']=hashlib.sha256((folder/'result.steps.jsonl').read_bytes()).hexdigest()
    (folder/'result.json').write_text(json.dumps(report))
    result=validate(folder,plan,execution)
    assert not result['accepted'] and result['reasons']


@pytest.mark.parametrize('num_envs',[16,32,60,120])
def test_selective_reset_adapters_cover_scale_ladder_and_observe_recovery(tmp_path,num_envs):
    env,_,_,wrapped=make_environment(num_envs)
    original=env._reset_idx;scene_update=env.scene.update
    with EventJournal(tmp_path/'resets.jsonl') as journal:
        observer=ScaleObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        with observer.installed():
            summary=observer.direct_resets()
            assert summary['pending_recovery_ids']==list(range(num_envs))
            wrapped.step(torch.zeros(num_envs,12))
            assert observer.samples==4 and not observer.pending_recovery
        assert env._reset_idx==original and env.scene.update==scene_update
        tickets=summarize_tickets(observer.tickets)
        assert tickets['assignment_count']==num_envs+sum(map(len,reset_batches(num_envs)))
        if num_envs==120:assert tickets['complete_blocks']==2


def test_reset_fault_is_detected_and_all_instance_hooks_are_restored(tmp_path):
    env,_,_,wrapped=make_environment();original=env._reset_idx
    def bad(ids):
        result=original(ids);env.action_manager.action[15,0]=.1;return result
    env._reset_idx=bad
    with EventJournal(tmp_path/'reset_fault.jsonl') as journal:
        observer=ScaleObserver(env,wrapped.runtime,journal,snapshot,check_transition)
        with pytest.raises((AssertionError,RuntimeError)):
            with observer.installed():env._reset_idx(torch.tensor([0]))
    assert env._reset_idx==bad


@pytest.mark.parametrize('ending',['deadline','zero_exit'])
def test_g1_supervisor_rejects_dummy_process_timeout_and_empty_exit(tmp_path,monkeypatch,ending):
    output=tmp_path/'attempt';plan=cpu_plan(output)|dict(status='approved',user_approved=True)
    monkeypatch.setattr(supervisor,'preflight',lambda *a,**kw:plan)
    monkeypatch.setattr(supervisor,'_input_hashes',lambda:plan['input_sha256'])
    monkeypatch.setattr(supervisor,'dependencies',lambda:plan['dependency_sha256'])
    monkeypatch.setattr(supervisor,'canonical_budget_path',lambda *a:tmp_path/'ledger.json')
    monkeypatch.setattr(supervisor.BudgetLedger,'reference',lambda self:plan['budget_ledger'])
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'0, uuid, CPU fixture, 10 MiB, 24000 MiB, 24576 MiB, 0 %, 35\n')
    original=subprocess.Popen
    def launch(command,**kw):return original([sys.executable,'-c','import time;time.sleep(60)' if ending=='deadline' else 'pass'],**kw)
    monkeypatch.setattr(subprocess,'Popen',launch)
    if ending=='deadline':monkeypatch.setattr(supervisor,'deadline_expired',lambda *a:True)
    assert supervisor.run(NS(output_dir=output,checkpoint=tmp_path/'unused'))==1
    execution=json.loads((output/'execution.json').read_text())
    assert execution['process_group_empty'] and execution['timed_out']==(ending=='deadline')
    assert not (tmp_path/'ledger.json').exists()


def test_startup_deadline_is_separate_from_total_deadline():
    assert not supervisor.deadline_expired(119,'') and supervisor.deadline_expired(121,'')
    assert not supervisor.deadline_expired(121,'[scale-g1] initial-ready')
    assert supervisor.deadline_expired(241,'[scale-g1] initial-ready')
