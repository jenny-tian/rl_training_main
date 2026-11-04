"""G2 approval, independent evidence faults and CPU process supervision."""
from copy import deepcopy
import hashlib
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from scale_reset_cpu_fixture import cpu_plan,build_fixture
from scale_cpu_fixture import ROOT,BODY_NAMES,GROUND,make_environment
from rl_training.source_scan.scale_reset import contract_for,validate_plan,validate_predecessor,PREDECESSOR_120
from rl_training.source_scan.scale_startup import PREDECESSOR_16,PREDECESSOR_32,PREDECESSOR_60
from source_scale_reset_validation import validate,_inventory
from source_sparse_tensor_contact import SparseTensorContactCapture
import source_scale_reset_supervisor as supervisor
from source_scale_startup_validation import validate as validate_g1


@pytest.mark.parametrize('value',[16,32,60,121,120.0,True,None])
def test_g2_requires_exactly_registered_120_environment_scale(value):
    with pytest.raises(ValueError):contract_for(value)


@pytest.mark.parametrize('fault',[None,'approval','CLI_scale','scale','steps','physics','transitions','reset_ids',
    'boolean_ids','reset_order','reset_count','ticket_count','capacity','wall','events','disk','memory','PPO',
    'policy','ground_scope','training','predecessor','output_exists','source_inventory'])
def test_g2_plan_cannot_change_fixed_resets_recovery_or_resource_limits(tmp_path,fault):
    output=tmp_path/'new';plan=cpu_plan(output)|dict(status='approved',user_approved=True)
    fields={'approval':('user_approved',False),'scale':('num_envs',60),'steps':('control_steps',5),
        'physics':('physics_substeps',20),'transitions':('diagnostic_transitions',600),
        'reset_count':('selected_environment_resets',183),'ticket_count':('assignment_count',303),
        'capacity':('contact_capacity',524288),'wall':('wall_timeout_s',481),'events':('events_byte_limit',536870912),
        'disk':('minimum_free_disk_bytes',2147483648),'memory':('minimum_free_vram_mib',1),
        'PPO':('ppo_updates',1),'policy':('policy_loaded',True),'ground_scope':('require_ground_positive',True),
        'training':('training_ready',True),'predecessor':('predecessor_qualification',dict(PREDECESSOR_60))}
    kwargs=dict(revision='a'*40,inputs=plan['input_sha256'],dependencies=plan['dependency_sha256'],budget_reference=plan['budget_ledger'],output_dir=output,num_envs=120)
    if fault in fields:
        key,value=fields[fault];plan[key]=value
    if fault=='CLI_scale':kwargs['num_envs']=60
    if fault=='reset_ids':plan['planned_reset_batches'][1]=[1,17,118]
    if fault=='boolean_ids':plan['planned_reset_batches'][1][0]=True
    if fault=='reset_order':plan['planned_reset_batches'].reverse()
    if fault=='output_exists':output.mkdir()
    if fault=='source_inventory':plan['input_sha256'].pop('242')
    if fault is None:
        assert validate_plan(plan,**kwargs)==plan
        assert [len(v) for v in plan['planned_reset_batches']]==[1,3,60,120]
        assert plan['assignment_count']==304 and plan['complete_assignment_blocks']==2 and plan['tail_assignment_count']==64
    else:
        with pytest.raises((ValueError,PermissionError,FileExistsError)):validate_plan(plan,**kwargs)


@pytest.mark.parametrize('fault',['missing_120','anchor_120','raw_120','CPU_fixture','skip_120',
    'missing_60','raw_60','missing_32','raw_32','missing_16','raw_16'])
def test_complete_g1_predecessor_chain_cannot_be_skipped_or_tampered(tmp_path,fault):
    for ref in (PREDECESSOR_16,PREDECESSOR_32,PREDECESSOR_60,PREDECESSOR_120):
        dst=tmp_path/ref['archive'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ROOT/ref['archive'],dst)
    ref=dict(PREDECESSOR_120)
    for size,prior in [(120,PREDECESSOR_120),(60,PREDECESSOR_60),(32,PREDECESSOR_32),(16,PREDECESSOR_16)]:
        folder=tmp_path/prior['archive']
        if fault==f'missing_{size}':(folder/'review.json').unlink()
        if fault==f'raw_{size}':(folder/'raw/result.steps.jsonl').write_text('{}\n')
    if fault=='anchor_120':(tmp_path/PREDECESSOR_120['archive']/'review.json').write_text('{}')
    if fault=='CPU_fixture':ref['archive']='docs/lite3-source-shared-bank-g1-120env-preparation-20261103/raw/cpu_example_120'
    if fault=='skip_120':ref=dict(PREDECESSOR_60)
    with pytest.raises((OSError,ValueError)):validate_predecessor(ref,source_root=tmp_path)


@pytest.fixture(scope='module')
def g2_fixture(tmp_path_factory):
    folder=tmp_path_factory.mktemp('G2_synthetic_CPU')/'attempt'
    plan,execution=build_fixture(folder)
    assert not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled()
    return folder,plan,execution


def test_all_four_resets_after_inference_controls_are_accepted_without_mode_leak(g2_fixture):
    folder,plan,execution=g2_fixture;result=validate(folder,plan,execution)
    assert result['accepted'],result
    assert result['diagnostic_transitions']==480 and result['physics_substeps']==16
    assert result['reset_boundary_qualified'] and result['reset_first_step_qualified']
    assert result['selected_environment_resets']==184 and result['assignment_count']==304
    assert result['complete_assignment_blocks']==2 and result['tail_assignment_count']==64
    assert result['training_ready'] is False and result['ground_positive_qualification'] is False
    assert not torch.is_inference_mode_enabled() and not torch.cuda.is_initialized()


def test_g2_keeps_no_supporting_force_as_unqualified_scope(tmp_path):
    folder=tmp_path/'no_ground_force';plan,execution=build_fixture(folder,contact_fault='empty')
    result=validate(folder,plan,execution)
    assert result['accepted'],result
    assert not any(result['ground_contact_points'].values()) and result['ground_positive_qualification'] is False


@pytest.mark.parametrize('fault',['batch_missing','batch_ids','batch_tick','reset_order','missing_root_inventory',
    'unselected_root','unselected_origin','unselected_action','unselected_sensor','history_not_cleared',
    'pushes_not_cleared','filter_not_cleared','lifecycle_not_cleared','contact_not_cleared','sensor_not_invalidated',
    'spawn_wrong','checked_after_changed','ticket_column','ticket_tail','metric_age','metric_rows','nonfinite_metric',
    'done','action','pause','frame_missing','frame_duplicate','force_changed','motion_missing','recovery_missing',
    'extra_recovery','last_body','last_scanner','reset_scanner','collision_group','runtime_stage','scope',
    'capture_count','cleanup','pending','plan_unapproved','missing_contact_history','unselected_contact_history',
    'stale_baseline','baseline_clock','completed_transitions','reset_attempt_count','planned_schedule'])
def test_independent_g2_rejects_corrupt_evidence_even_with_new_outer_hashes(tmp_path,g2_fixture,fault):
    original,plan,execution=g2_fixture;folder=tmp_path/'corrupt';shutil.copytree(original,folder);plan=deepcopy(plan)
    report=json.loads((folder/'result.json').read_text());events=[json.loads(l) for l in (folder/'events.jsonl').read_text().splitlines()]
    rows=[json.loads(l) for l in (folder/'result.steps.jsonl').read_text().splitlines()]
    phases=lambda name:[v for v in events if v.get('phase')==name]
    attempts=phases('scale_reset_attempt');afters=phases('scale_reset_after');checked=phases('scale_reset_checked')
    motion=phases('scale_motion_substep');frames=phases('sparse_contact_sample')
    def change_after(key,index,value):
        for data in (afters[0]['after'],checked[0]['after']):
            target=data[key]['values']
            for i in index[:-1]:target=target[i]
            target[index[-1]]=value
    if fault=='batch_missing':events.remove(checked[-1])
    if fault=='batch_ids':attempts[0]['ids']=[1]
    if fault=='batch_tick':attempts[0]['sim_step']+=1
    if fault=='reset_order':events.remove(checked[0]);events.append(checked[0])
    if fault=='missing_root_inventory':
        del report['reset_snapshot_inventory']['root']
        for event in attempts+afters+checked:
            for key in ('before','after'):
                if key in event:del event[key]['root']
    if fault=='missing_contact_history':
        name='cache/contact_forces/net_forces_w_history';del report['reset_snapshot_inventory'][name]
        for event in attempts+afters+checked:
            for key in ('before','after'):
                if key in event:del event[key][name]
    if fault=='unselected_root':change_after('root',[119,0],7.)
    if fault=='unselected_origin':change_after('origin',[119,0],7.)
    if fault=='unselected_action':change_after('action',[119,0],.1)
    if fault=='unselected_sensor':change_after('sensor/height_scanner/_timestamp',[119],1.)
    if fault=='unselected_contact_history':change_after('cache/contact_forces/net_forces_w_history',[119,0,0,0],1.)
    if fault=='stale_baseline':
        attempts[0]['before']['sensor/contact_forces/_is_outdated']['values'][119]=True
        change_after('sensor/contact_forces/_is_outdated',[119],True)
    if fault=='baseline_clock':
        attempts[0]['before']['sensor/contact_forces/_timestamp']['values'][119]=1.
        change_after('sensor/contact_forces/_timestamp',[119],1.)
    if fault=='history_not_cleared':change_after('history/policy/actions',[0,0,0],.25)
    if fault=='pushes_not_cleared':change_after('pushes/policy/actions',[0],1)
    if fault=='filter_not_cleared':change_after('filter/filter_previous_residual_action',[0,0],.1)
    if fault=='lifecycle_not_cleared':change_after('lifecycle/hold',[0],1)
    if fault=='contact_not_cleared':change_after('cache/contact_forces/net_forces_w',[0,0,0],1.)
    if fault=='sensor_not_invalidated':change_after('sensor/height_scanner/_is_outdated',[0],False)
    if fault=='spawn_wrong':change_after('root',[0,0],999.)
    if fault=='checked_after_changed':checked[0]['after']['root']['values'][119][0]+=1
    if fault=='ticket_column':checked[0]['assignments'][119]['column']+=1
    if fault=='ticket_tail':report['all_assignment_tickets'][-1]['index_in_block']+=1
    if fault=='metric_age':rows[0]['metric_rows'][119]['step']=2
    if fault=='metric_rows':rows[0]['metric_rows'].pop()
    if fault=='nonfinite_metric':rows[0]['metric_rows'][119]['roll_rad']=float('nan')
    if fault=='done':rows[0]['done'][119]=1
    if fault=='action':rows[0]['executed_action'][119][0]=.1
    if fault=='pause':rows[0]['paused'][119][0]=True
    if fault=='frame_missing':events.remove(frames[-1])
    if fault=='frame_duplicate':frames[-1]['sim_step']=frames[0]['sim_step']
    if fault=='force_changed':frames[0]['pair_forces_N'][0][2]+=1
    if fault=='motion_missing':events.remove(motion[-1])
    if fault=='recovery_missing':motion[0]['recovery_ids']=[]
    if fault=='extra_recovery':motion[1]['recovery_ids']=[119]
    if fault=='last_body':motion[0]['body_positions_w']['values'][119][16][1]+=10
    if fault=='last_scanner':motion[0]['scanners']['height_scanner_base']['mesh_ids']['values'][119][0]=18
    if fault=='reset_scanner':phases('g2_reset_scans')[0]['scanners']['height_scanner_base']['age_s']['values'][119]=.005
    if fault=='collision_group':report['collision_groups']['groups']['/World/collisions/group119']['allowed_groups'].append('/World/collisions/group0')
    if fault=='runtime_stage':phases('contact_runtime')[0]['stage_id']+=1
    if fault=='scope':report['contact_capture']['diagnostic_scope']='G1_startup'
    if fault=='capture_count':report['contact_capture']['samples']=15
    if fault=='cleanup':(folder/'result.cleanup.json').write_text('{}')
    if fault=='pending':report['pending_recovery_ids']=[119]
    if fault=='completed_transitions':report['completed_diagnostic_transitions']=360
    if fault=='reset_attempt_count':report['reset_attempts']=3
    if fault=='planned_schedule':report['planned_reset_batches'].reverse()
    if fault=='plan_unapproved':plan['user_approved']=False
    (folder/'result.steps.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in rows))
    (folder/'events.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in events))
    for key,name in [('steps_sha256','result.steps.jsonl'),('events_sha256','events.jsonl')]:report[key]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    (folder/'result.json').write_text(json.dumps(report));result=validate(folder,plan,execution)
    assert not result['accepted'] and result['reasons'],fault


def test_absent_observation_history_is_explicit_but_partial_history_is_rejected(g2_fixture):
    folder,_,_=g2_fixture;report=json.loads((folder/'result.json').read_text());inventory=report['reset_snapshot_inventory']
    no_history={k:v for k,v in inventory.items() if not k.startswith(('history/','pushes/'))}
    _inventory(no_history,[],120)
    no_history['pushes/policy/actions']=inventory['pushes/policy/actions']
    with pytest.raises(ValueError):_inventory(no_history,[],120)


@pytest.mark.parametrize('child',[False,True])
@pytest.mark.parametrize('scale',[60,120])
def test_parent_child_scale_and_supervision_ticket_are_checked_before_simulator(tmp_path,monkeypatch,child,scale):
    output=tmp_path/'attempt';plan=cpu_plan(output)|dict(status='approved',user_approved=True)
    path=output/'run_manifest.json' if child else tmp_path/'approved.json'
    if child:output.mkdir()
    path.write_text(json.dumps(plan))
    if child:(output/'supervision.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),revision='a'*40)))
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'CPU fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setattr(supervisor,'CHECKPOINT_SHA256',hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(supervisor,'_git_state',lambda:('a'*40,'a'*40,''));monkeypatch.setattr(supervisor,'_input_hashes',lambda:plan['input_sha256'])
    monkeypatch.setattr(supervisor,'dependencies',lambda:plan['dependency_sha256']);monkeypatch.setattr(supervisor,'canonical_budget_path',lambda *a:tmp_path/'ledger.json')
    monkeypatch.setattr(supervisor.BudgetLedger,'reference',lambda self:plan['budget_ledger'])
    before=set(sys.modules);args=NS(manifest=path,checkpoint=checkpoint,output_dir=output,num_envs=scale)
    if scale==120:assert supervisor.preflight(args,child=child)==plan
    else:
        with pytest.raises(ValueError):supervisor.preflight(args,child=child)
    assert not torch.cuda.is_initialized() and not any(name.startswith(('isaaclab','isaacsim','omni.')) for name in set(sys.modules)-before)


@pytest.mark.parametrize('child',[False,True])
def test_unapproved_g2_proposal_rejected_before_git_gpu_or_backend(tmp_path,monkeypatch,child):
    path=tmp_path/'proposal.json';path.write_text(json.dumps(cpu_plan(tmp_path/'new')))
    def forbidden(*a,**kw):raise AssertionError('backend before approval')
    monkeypatch.setattr(supervisor,'_git_state',forbidden);monkeypatch.setattr(subprocess,'check_output',forbidden)
    with pytest.raises(PermissionError):supervisor.preflight(NS(manifest=path,num_envs=120),child=child)


@pytest.mark.parametrize('ending',['deadline','zero_exit'])
def test_g2_supervisor_limits_dummy_cpu_process_and_cleans_only_its_group(tmp_path,monkeypatch,ending):
    output=tmp_path/'attempt';plan=cpu_plan(output)|dict(status='approved',user_approved=True)
    monkeypatch.setattr(supervisor,'preflight',lambda *a,**kw:plan);monkeypatch.setattr(supervisor,'_input_hashes',lambda:plan['input_sha256'])
    monkeypatch.setattr(supervisor,'dependencies',lambda:plan['dependency_sha256']);monkeypatch.setattr(supervisor,'canonical_budget_path',lambda *a:tmp_path/'ledger.json')
    monkeypatch.setattr(supervisor.BudgetLedger,'reference',lambda self:plan['budget_ledger'])
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'0, uuid, CPU fixture, 10 MiB, 24000 MiB, 24576 MiB, 0 %, 35\n')
    original=subprocess.Popen;commands=[]
    def launch(command,**kw):
        commands.append(command);return original([sys.executable,'-c','import time;time.sleep(60)' if ending=='deadline' else 'pass'],**kw)
    monkeypatch.setattr(subprocess,'Popen',launch)
    if ending=='deadline':monkeypatch.setattr(supervisor,'deadline_expired',lambda *a:True)
    assert supervisor.run(NS(output_dir=output,checkpoint=tmp_path/'unused',num_envs=120))==1
    execution=json.loads((output/'execution.json').read_text());assert execution['process_group_empty'] and execution['timed_out']==(ending=='deadline')
    assert len(commands)==1 and '480s' in commands[0] and commands[0][-2:]==['--num-envs','120']
    assert not (tmp_path/'ledger.json').exists()


def test_g2_readiness_marker_and_deadline_are_separate_from_g1():
    assert supervisor.deadline_expired(121,'[scale-g1] initial-ready')
    assert not supervisor.deadline_expired(479,'[scale-g2] initial-ready')
    assert supervisor.deadline_expired(481,'[scale-g2] initial-ready')


@pytest.mark.parametrize('scope,controls,positive',[('G1_startup',4,False),('G2_reset_recovery',1,False),('G2_reset_recovery',4,True)])
def test_unqualified_contact_exception_cannot_expand_default_startup_scope(tmp_path,scope,controls,positive):
    env,_,_,_=make_environment(120)
    kwargs=dict(physics_view=env.sim.physics_sim_view,ground_colliders=[GROUND],num_envs=120,physics_dt=.005,
        fault_path=tmp_path/'unused.npz',require_ground_positive=positive,diagnostic_scope=scope,stage_id=73,
        reporting_bodies=[f'/World/envs/env_{i}/Robot/{name}' for i in range(120) for name in BODY_NAMES],
        record=lambda value:None,start_step=0,decimation=4,control_steps=controls)
    with pytest.raises(ValueError):
        capture=SparseTensorContactCapture(**kwargs);capture.finish()


@pytest.mark.parametrize('ref',[PREDECESSOR_16,PREDECESSOR_32,PREDECESSOR_60,PREDECESSOR_120])
def test_current_g1_validator_preserves_every_historical_real_result(ref):
    folder=ROOT/ref['archive']/'raw';read=lambda n:json.loads((folder/n).read_text())
    assert validate_g1(folder,read('run_manifest.json'),read('execution.json'))==read('acceptance.json')


@pytest.mark.parametrize('script',['source_scale_reset_supervisor.py','source_scale_reset_probe.py'])
def test_g2_cli_requires_explicit_120_envs_without_simulator(script):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/tools'/script),'--help'],text=True,capture_output=True,timeout=30)
    assert result.returncode==0 and '--num-envs {120}' in result.stdout
