"""60-env G1 authorization and synthetic CPU evidence; no CUDA or simulator."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_scale_startup import cpu_plan,build_g1_fixture
from scale_cpu_fixture import BODY_NAMES,GROUND,payload,ROOT
from rl_training.source_scan.scale_startup import CONTRACT,contract_for,validate_plan,validate_predecessor,PREDECESSOR_16,PREDECESSOR_32
from rl_training.source_scan.sparse_contact import compact_topology,ContactEncoder,reconstruct_frame
from source_scale_startup_validation import validate
import source_scale_startup_supervisor as supervisor


@pytest.mark.parametrize('fault',[None,'approval','CLI_16','CLI_32','manifest_scale','capacity','transitions',
    'step','policy','PPO','training_ready','wall_timeout','predecessor','output_exists'])
def test_60_env_plan_binds_scale_predecessor_and_single_step_limits(tmp_path,fault):
    output=tmp_path/'new';plan=cpu_plan(output,60)|dict(status='approved',user_approved=True)
    kwargs=dict(revision='a'*40,inputs=plan['input_sha256'],dependencies=plan['dependency_sha256'],
        budget_reference=plan['budget_ledger'],output_dir=output,num_envs=60)
    faults={'approval':('user_approved',False),'manifest_scale':('num_envs',32),'capacity':('contact_capacity',262144),
        'transitions':('diagnostic_transitions',32),'step':('control_steps',2),'policy':('policy_loaded',True),
        'PPO':('ppo_updates',1),'training_ready':('training_ready',True),'wall_timeout':('wall_timeout_s',241),
        'predecessor':('predecessor_qualification',dict(PREDECESSOR_16))}
    if fault in faults:
        key,value=faults[fault];plan[key]=value
    if fault in ('CLI_16','CLI_32'):kwargs['num_envs']=int(fault.split('_')[1])
    if fault=='output_exists':output.mkdir()
    if fault is None:
        assert validate_plan(plan,**kwargs)==plan
        assert plan['diagnostic_transitions']==60 and plan['contact_capacity']==524288
        assert plan['predecessor_qualification']==PREDECESSOR_32 and plan['require_ground_positive'] is False
        assert plan['automatic_scale_increase'] is False and contract_for(16)==CONTRACT
    else:
        with pytest.raises((ValueError,PermissionError,FileExistsError)):validate_plan(plan,**kwargs)


@pytest.mark.parametrize('fault',['missing_32','anchor_32','raw_32','CPU_fixture','old_16_reference','missing_16','raw_16'])
def test_predecessor_chain_cannot_be_skipped_substituted_or_changed(tmp_path,fault):
    for reference in (PREDECESSOR_16,PREDECESSOR_32):
        destination=tmp_path/reference['archive'];destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(ROOT/reference['archive'],destination)
    prior32=tmp_path/PREDECESSOR_32['archive'];prior16=tmp_path/PREDECESSOR_16['archive']
    reference=dict(PREDECESSOR_32)
    if fault=='missing_32':(prior32/'review.json').unlink()
    if fault=='anchor_32':(prior32/'review.json').write_text('{}')
    if fault=='raw_32':(prior32/'raw/result.steps.jsonl').write_text('{}\n')
    if fault=='CPU_fixture':reference['archive']='docs/lite3-source-shared-bank-g1-32env-preparation-20261103/raw/cpu_example_32'
    if fault=='old_16_reference':reference=dict(PREDECESSOR_16)
    if fault=='missing_16':(prior16/'review.json').unlink()
    if fault=='raw_16':(prior16/'raw/result.steps.jsonl').write_text('{}\n')
    with pytest.raises((OSError,ValueError)):validate_predecessor(60,reference,source_root=tmp_path)


def test_qualified_32_to_16_chain_is_read_only_and_retains_limited_scope():
    paths=[ROOT/ref['archive']/name for ref in (PREDECESSOR_16,PREDECESSOR_32) for name in ('SHA256SUMS','review.json')]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    validate_predecessor(60,dict(PREDECESSOR_32))
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for ref in (PREDECESSOR_16,PREDECESSOR_32):
        review=json.loads((ROOT/ref['archive']/'review.json').read_text())
        assert review['ground_positive_qualified'] is False and review['training_ready'] is False


@pytest.mark.parametrize('cross_contact',[False,True])
def test_exact_60_env_1020_body_1044_filter_shape_and_highest_id_contact(cross_contact):
    sensors=sorted(f'/World/envs/env_{i}/Robot/{name}' for i in range(60) for name in BODY_NAMES)
    ground=[GROUND]+[f'/World/ground/other_{i}' for i in range(23)]
    filters=[sensors+ground for _ in sensors]
    topology=compact_topology(sensors,filters,ground,num_envs=60,capacity=524288)
    net,matrix,data=payload(sensors,filters,524288)
    if cross_contact:
        target=next(p for p in sensors if '/env_59/' in p);matrix[0,filters[0].index(target),0]=1e-8
        with pytest.raises(ValueError,match='cross-environment'):ContactEncoder(topology).encode(net,matrix,data,step=1)
    else:
        frame=ContactEncoder(topology).encode(net,matrix,data,step=1);decoded=reconstruct_frame(frame,topology)
        assert decoded['matrix'].shape==(1020,1044,3) and all(v==1 for v in decoded['ground_contact_points'].values())
        assert np.array_equal(matrix,decoded['matrix']) and np.array_equal(net,decoded['net'])


@pytest.fixture(scope='module')
def g1_60_fixture(tmp_path_factory):
    folder=tmp_path_factory.mktemp('G1_60_synthetic_CPU')/'attempt'
    plan,execution=build_g1_fixture(folder,num_envs=60)
    return folder,plan,execution


def test_complete_60_env_cpu_fixture_has_only_selected_startup_scope(g1_60_fixture):
    folder,plan,execution=g1_60_fixture;result=validate(folder,plan,execution)
    assert result['accepted'],result
    assert result['scope']=='60_env_startup_one_zero_control_step_only' and result['diagnostic_transitions']==60
    assert result['ground_positive_qualification'] is False and result['training_ready'] is False
    assert result['ppo_updates']==0 and len(result['ground_contact_points'])==60


def test_60_env_first_step_without_supporting_force_stays_within_G1_scope(tmp_path):
    folder=tmp_path/'empty_contact_CPU_fixture'
    plan,execution=build_g1_fixture(folder,num_envs=60,contact_fault='empty')
    result=validate(folder,plan,execution)
    assert result['accepted'] and not any(result['ground_contact_points'].values())
    assert result['ground_positive_qualification'] is False


@pytest.mark.parametrize('fault',['frame_missing','force_hash','last_body','last_scanner','last_metric','last_action',
    'last_group','16_rows','32_rows','capacity','schema_scale','predecessor','plan_unapproved','cleanup',
    'nonfinite_metric','runtime_missing'])
def test_60_env_validator_checks_all_rows_even_with_recomputed_journal_hashes(tmp_path,g1_60_fixture,fault):
    original,plan,execution=g1_60_fixture;plan=deepcopy(plan);folder=tmp_path/'corrupt';shutil.copytree(original,folder)
    report=json.loads((folder/'result.json').read_text())
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    motion=[v for v in events if v.get('phase')=='scale_motion_substep']
    frames=[v for v in events if v.get('phase')=='sparse_contact_sample']
    row=json.loads((folder/'result.steps.jsonl').read_text())
    if fault=='frame_missing':events.remove(frames[-1])
    if fault=='force_hash':frames[-1]['pair_forces_N'][-1][2]+=.01
    if fault=='last_body':motion[-1]['body_positions_w']['values'][59][16][1]+=10
    if fault=='last_scanner':motion[-1]['scanners']['height_scanner_base']['mesh_ids']['values'][59][0]=18
    if fault=='last_metric':row['metric_rows'].pop()
    if fault=='last_action':row['executed_action'][59][0]=.1
    if fault=='last_group':report['collision_groups']['groups']['/World/collisions/group59']['allowed_groups'].append('/World/collisions/group0')
    if fault in ('16_rows','32_rows'):row['done']=row['done'][:int(fault.split('_')[0])]
    if fault=='capacity':report['contact_capture']['topology']['capacity']=262144
    if fault=='schema_scale':report['num_envs']=32
    if fault=='predecessor':plan['predecessor_qualification']=dict(PREDECESSOR_16)
    if fault=='plan_unapproved':plan['user_approved']=False
    if fault=='cleanup':(folder/'result.cleanup.json').write_text('{}')
    if fault=='nonfinite_metric':row['metric_rows'][59]['pitch_rad']=float('nan')
    if fault=='runtime_missing':events.remove(next(v for v in events if v.get('phase')=='contact_runtime'))
    (folder/'result.steps.jsonl').write_text(json.dumps(row)+'\n')
    (folder/'events.jsonl').write_text(''.join(json.dumps(value)+'\n' for value in events))
    for key,name in (('steps_sha256','result.steps.jsonl'),('events_sha256','events.jsonl')):report[key]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    (folder/'result.json').write_text(json.dumps(report))
    result=validate(folder,plan,execution);assert not result['accepted'] and result['reasons']


@pytest.mark.parametrize('child',[False,True])
@pytest.mark.parametrize('CLI_scale',[16,32,60])
def test_parent_and_child_require_matching_explicit_scale_before_simulator_import(tmp_path,monkeypatch,child,CLI_scale):
    output=tmp_path/'attempt';plan=cpu_plan(output,60)|dict(status='approved',user_approved=True)
    path=output/'run_manifest.json' if child else tmp_path/'approved_cpu_fixture.json'
    if child:output.mkdir()
    path.write_text(json.dumps(plan))
    if child:(output/'supervision.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),revision='a'*40)))
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'CPU fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(supervisor,'CHECKPOINT_SHA256',hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(supervisor,'_git_state',lambda:('a'*40,'a'*40,''))
    monkeypatch.setattr(supervisor,'_input_hashes',lambda:plan['input_sha256'])
    monkeypatch.setattr(supervisor,'dependencies',lambda:plan['dependency_sha256'])
    monkeypatch.setattr(supervisor,'canonical_budget_path',lambda *a:tmp_path/'ledger.json')
    monkeypatch.setattr(supervisor.BudgetLedger,'reference',lambda self:plan['budget_ledger'])
    before=set(sys.modules)
    args=NS(manifest=path,checkpoint=checkpoint,output_dir=output,num_envs=CLI_scale)
    if CLI_scale==60:assert supervisor.preflight(args,child=child)==plan
    else:
        with pytest.raises(ValueError):supervisor.preflight(args,child=child)
    assert not torch.cuda.is_initialized() and not (output/'result.json').exists()
    assert not any(name.startswith(('isaaclab','isaacsim','omni.')) for name in set(sys.modules)-before)


@pytest.mark.parametrize('child',[False,True])
def test_60_env_proposal_never_starts_backend_before_launch_approval(tmp_path,monkeypatch,child):
    path=tmp_path/'proposal.json';path.write_text(json.dumps(cpu_plan(tmp_path/'future_run',60)))
    def forbidden(*a,**kw):raise AssertionError('backend called for an unapproved proposal')
    monkeypatch.setattr(supervisor,'_git_state',forbidden);monkeypatch.setattr(subprocess,'check_output',forbidden)
    with pytest.raises(PermissionError):supervisor.preflight(NS(manifest=path,num_envs=60),child=child)


@pytest.mark.parametrize('ending',['deadline','zero_exit'])
def test_60_env_supervisor_forwards_scale_and_cleans_dummy_group(tmp_path,monkeypatch,ending):
    output=tmp_path/'attempt';plan=cpu_plan(output,60)|dict(status='approved',user_approved=True)
    monkeypatch.setattr(supervisor,'preflight',lambda *a,**kw:plan)
    monkeypatch.setattr(supervisor,'_input_hashes',lambda:plan['input_sha256'])
    monkeypatch.setattr(supervisor,'dependencies',lambda:plan['dependency_sha256'])
    monkeypatch.setattr(supervisor,'canonical_budget_path',lambda *a:tmp_path/'ledger.json')
    monkeypatch.setattr(supervisor.BudgetLedger,'reference',lambda self:plan['budget_ledger'])
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'0, uuid, CPU fixture, 10 MiB, 24000 MiB, 24576 MiB, 0 %, 35\n')
    original=subprocess.Popen;commands=[]
    def launch(command,**kw):
        commands.append(command)
        return original([sys.executable,'-c','import time;time.sleep(60)' if ending=='deadline' else 'pass'],**kw)
    monkeypatch.setattr(subprocess,'Popen',launch)
    if ending=='deadline':monkeypatch.setattr(supervisor,'deadline_expired',lambda *a:True)
    assert supervisor.run(NS(output_dir=output,checkpoint=tmp_path/'unused',num_envs=60))==1
    execution=json.loads((output/'execution.json').read_text())
    assert len(commands)==1 and commands[0][-2:]==['--num-envs','60']
    assert execution['process_group_empty'] and execution['timed_out']==(ending=='deadline')
    assert not (tmp_path/'ledger.json').exists()


@pytest.mark.parametrize('reference',[PREDECESSOR_16,PREDECESSOR_32])
def test_new_validator_preserves_both_archived_real_results(reference):
    folder=ROOT/reference['archive']/'raw'
    read=lambda name:json.loads((folder/name).read_text())
    assert validate(folder,read('run_manifest.json'),read('execution.json'))==read('acceptance.json')


@pytest.mark.parametrize('script',['source_scale_startup_supervisor.py','source_scale_startup_probe.py'])
def test_cli_lists_60_env_choice_without_loading_the_simulator(script):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/tools'/script),'--help'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and '--num-envs {16,32,60}' in result.stdout
