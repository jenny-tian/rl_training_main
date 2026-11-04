from copy import deepcopy
import gzip
import hashlib
import json
import subprocess
import sys

import pytest
import torch

from scale_motion_cpu_fixture import ROOT,CHECKPOINT,cpu_plan,build_fixture
from rl_training.source_scan.scale_motion import contract_for,validate_plan,validate_predecessor,G2
from rl_training.source_scan.scale_motion_evidence import CompressedJournal,event_records,digest_stream
from source_scale_motion_validation import validate
import source_scale_motion_supervisor as supervisor


@pytest.mark.parametrize('value',[8,16,32,60,121,120.0,True,None])
def test_G3_scale_is_exact_and_separately_registered(value):
    with pytest.raises(ValueError):contract_for(value)


@pytest.mark.parametrize('field,value',[('num_envs',60),('main_control_steps',1999),('recovery_control_steps',0),
    ('control_steps',2000),('physics_substeps',8000),('diagnostic_transitions',240000),('ppo_updates',1),
    ('training_transitions',1),('policy_loaded',False),('noise_seed',132002),('noise_std',0.),('command_mps',[.4,0.,0.]),('actor_compute','TF32'),
    ('require_ground_positive',False),('flat_completion_required',False),('contact_capacity',524288),
    ('wall_timeout_s',7201),('validation_timeout_s',7201),('events_byte_limit',1),('retry',True),('resume',True)])
def test_G3_fixed_budget_policy_and_scope_cannot_be_changed(tmp_path,field,value):
    output=tmp_path/'new';plan=cpu_plan(output)|dict(status='approved',user_approved=True);plan[field]=value
    with pytest.raises(ValueError):validate_plan(plan,revision='a'*40,inputs=plan['input_sha256'],dependencies=plan['dependency_sha256'],budget_reference=plan['budget_ledger'],output_dir=output)


def test_G2_archive_is_intact_but_changed_source_requires_new_runtime_qualification():
    validate_predecessor(G2,verify_current_inputs=False)
    with pytest.raises(ValueError,match='qualified G2 input changed'):
        validate_predecessor(G2)


@pytest.mark.parametrize('field',['archive','executed_revision','sha256sums_sha256','review_sha256'])
def test_G3_cannot_replace_G2_with_CPU_or_unverified_evidence(field):
    reference=dict(G2);reference[field]='invalid'
    with pytest.raises(ValueError):validate_predecessor(reference)


def test_multimember_gzip_is_lossless_streamed_and_bounded(tmp_path):
    path=tmp_path/'events.gz';values=[dict(phase='one',value=[1.,2.,3.]),dict(phase='two',value=4)]
    with CompressedJournal(path,byte_limit=10000,uncompressed_limit=10000,record_limit=10000) as journal:
        for value in values:journal.record(value);journal.flush()
    assert list(event_records(path))==values
    assert digest_stream(path)==hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()


@pytest.mark.parametrize('limit',['stored','uncompressed','record'])
def test_log_ceilings_keep_already_written_members(tmp_path,limit):
    path=tmp_path/'events.gz';journal=CompressedJournal(path,byte_limit=10000,uncompressed_limit=10000,record_limit=10000)
    journal.record(dict(phase='kept'));journal.flush()
    if limit=='stored':journal.byte_limit=journal.stored_bytes
    if limit=='uncompressed':journal.uncompressed_limit=journal.written
    if limit=='record':journal.record_limit=1
    with pytest.raises(RuntimeError):journal.record(dict(phase='rejected'));journal.flush()
    journal.stream.close();assert list(event_records(path))==[dict(phase='kept')]


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    folder=tmp_path_factory.mktemp('G3_CPU')/'attempt';plan,execution=build_fixture(folder)
    return folder,plan,execution


def test_real_callback_frozen_actor_collection_and_final_recovery_segment(fixture):
    folder,plan,execution=fixture;report=json.loads((folder/'result.json').read_text())
    result=validate(folder,plan,execution,CHECKPOINT,cpu_profile=(5,4))
    assert result.get('cpu_evidence_validated'),result
    assert result['accepted'] is False and result['motion_integrity_qualified'] is False
    assert report['simulation_control_steps']==5 and report['motion_samples']==20
    assert report['reset_batches']==3 and report['pending_main_recovery_ids']==[]
    assert report['pending_recovery_ids']==result['deferred_final_recovery_ids']==[3]
    assert report['actor_sha256_before']==report['actor_sha256_after']
    assert not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled()


def test_short_or_labeled_CPU_fixture_never_qualifies_runtime(fixture):
    folder,plan,execution=fixture;result=validate(folder,plan,execution,CHECKPOINT)
    assert result['accepted'] is False and result['reasons']


@pytest.mark.parametrize('fault',['missing_motion','motion_sequence','velocity','derivative','recovery','scanner',
    'outside','reset_root','reset_history','rebase','actor_mean','noise_action','terminal','age','hold','snapshot_missing','extra_row','summary'])
def test_streaming_review_rejects_corruption_even_after_rehash(tmp_path,fixture,fault):
    import shutil
    original,plan,execution=fixture;folder=tmp_path/'bad';shutil.copytree(original,folder)
    report=json.loads((folder/'result.json').read_text());events=list(event_records(folder/'events.jsonl.gz'))
    rows=[json.loads(line) for line in (folder/'result.steps.jsonl').read_text().splitlines()]
    phase=lambda p:[e for e in events if e['phase']==p]
    motion=phase('g3_motion_substep');afters=phase('scale_reset_after');checked=phase('scale_reset_checked')
    if fault=='missing_motion':events.remove(motion[-1])
    if fault=='motion_sequence':motion[0]['sim_step']+=1
    if fault=='velocity':motion[0]['body_link_velocities_w']['values'][119][16][0]+=1
    if fault=='derivative':motion[0]['finite_difference_acceleration_mps2']['values'][119][16][0]+=1
    if fault=='recovery':motion[8]['recovery_ids']=[]
    if fault=='scanner':motion[0]['scanners']['height_scanner_base']['age_s']['values'][119]=.01
    if fault=='outside':motion[0]['body_positions_w']['values'][119][16][1]+=10
    if fault in ('reset_root','reset_history'):
        key='root' if fault=='reset_root' else 'cache/contact_forces/net_forces_w_history'
        for event in (afters[0],checked[0]):
            if fault=='reset_root':event['after'][key]['values'][119][0]+=1
            else:event['after'][key]['values'][119][0][0][0]=1
    if fault=='rebase':phase('g3_reset_rebase')[0]['positions']['values'][0][16][0]+=1
    if fault=='actor_mean':rows[0]['actor_mean']['values'][119][0]+=1
    if fault=='noise_action':rows[0]['raw_action'][119][0]+=1
    if fault=='terminal':rows[1]['terminal']['terms']={}
    if fault=='age':rows[0]['episode_counters'][119]+=1
    if fault=='hold':rows[0]['metric_rows'][119]['lifecycle_hold']=50
    if fault=='snapshot_missing':del phase('scale_reset_attempt')[0]['before']['root']
    if fault=='extra_row':rows.append(rows[-1])
    if fault=='summary':report['contact_capture']['ground_positive_qualified']=False
    with CompressedJournal(folder/'changed.gz',byte_limit=16*1024**3,uncompressed_limit=64*1024**3,record_limit=512*1024**2) as journal:
        for event in events:journal.record(event);journal.flush()
    (folder/'changed.gz').replace(folder/'events.jsonl.gz')
    (folder/'result.steps.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in rows))
    report.update(events_sha256=hashlib.sha256((folder/'events.jsonl.gz').read_bytes()).hexdigest(),
        events_uncompressed_sha256=digest_stream(folder/'events.jsonl.gz'),steps_sha256=hashlib.sha256((folder/'result.steps.jsonl').read_bytes()).hexdigest())
    (folder/'result.json').write_text(json.dumps(report));result=validate(folder,plan,execution,CHECKPOINT,cpu_profile=(5,4))
    assert result.get('cpu_evidence_validated') is not True and result['accepted'] is False and result['reasons'],fault


@pytest.mark.parametrize('child',[False,True])
def test_unapproved_G3_fails_before_git_gpu_or_simulator(tmp_path,monkeypatch,child):
    from types import SimpleNamespace as NS
    p=tmp_path/'proposal.json';p.write_text(json.dumps(cpu_plan(tmp_path/'out')))
    monkeypatch.setattr(supervisor,'_git_state',lambda:(_ for _ in ()).throw(AssertionError('Git before approval')))
    with pytest.raises(PermissionError):supervisor.preflight(NS(manifest=p,num_envs=120),child=child)


@pytest.mark.parametrize('script',['source_scale_motion_supervisor.py','source_scale_motion_probe.py'])
def test_G3_cli_help_does_not_import_simulator(script):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/tools'/script),'--help'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and '--num-envs' in result.stdout
