from copy import deepcopy
import gzip
import hashlib
import json
import subprocess
from types import SimpleNamespace as NS

import pytest
import torch

from scale_motion_cpu_fixture import cpu_plan,build_fixture
from rl_training.source_scan.scale_motion_evidence import event_records
from rl_training.source_scan.scale_motion_evidence import file_digest,digest_stream,MotionArtifactWriter
from rl_training.source_scan.moving_diagnostic import EXPECTED_TERMS
from rl_training.source_scan.scale_motion import FLAT,validate_policy_prequalification
from source_scale_motion_validation import audit_metrics,audit_terminal
import source_scale_motion_supervisor as supervisor


@pytest.mark.parametrize('field',list(FLAT))
def test_independent_Flat_policy_prerequisite_cannot_be_replaced(field):
    reference=dict(FLAT);reference[field]='unverified'
    with pytest.raises(ValueError):validate_policy_prequalification(reference)


def flat_trace():
    holds=[0]*120;positions=torch.zeros(120,17,3);positions[:,:,0]=9.;origins=torch.zeros(120,3)
    for step in range(1,51):
        rows=[dict(step=step,forward_progress_m=9.,lateral_m=0.,velocity_error_mps=0.,forward_velocity_mps=.3,
            roll_rad=0.,pitch_rad=0.,foot_speeds_mps=[0.]*4,foot_horizontal_forces_N=[0.]*4,foot_contact=[True]*4,
            stable_for_completion=True,unsafe=False,nonfoot_contact=False,lifecycle_hold=step) for _ in range(120)]
        audit_metrics(rows,torch.full((120,),step),holds,positions,origins)
    terminal=dict(ids=[0],steps=[50],hold=[50],completed=[True],terminated=[True]+[False]*119,truncated=[False]*120,
        terms={k:[k=='source_training_completion'] for k in EXPECTED_TERMS},
        observations=dict(policy=[[0.]*45 for _ in range(120)],critic=[[0.]*393 for _ in range(120)]))
    return holds,positions,origins,rows,terminal


def test_flat_full_progress_and_fifty_stable_steps_are_required():
    holds,_,_,_,terminal=flat_trace()
    result=audit_terminal(terminal,[0],torch.full((120,),50),holds,[False]*120,[0]*120,50,2000)
    assert result==[dict(control_step=50,env_id=0,episode_steps=50)]
    assert audit_terminal(terminal,[0],torch.full((120,),50),holds,[False]*120,[1]*120,50,2000)==[]
    assert audit_terminal(terminal,[0],torch.full((120,),50),holds,[False]*120,[0]*120,2001,2000)==[]


@pytest.mark.parametrize('fault',['progress','stable','hold'])
def test_flat_metrics_cannot_be_faked_independently_of_motion_and_hold(fault):
    _,positions,origins,rows,_=flat_trace();holds=[49]*120
    if fault=='progress':rows[119]['forward_progress_m']=10.
    if fault=='stable':rows[119]['unsafe']=True
    if fault=='hold':rows[119]['lifecycle_hold']=49
    with pytest.raises(ValueError):audit_metrics(rows,torch.full((120,),50),holds,positions,origins)


@pytest.mark.parametrize('fault',['completed','hold','steps','reason','mask','timeout','observations'])
def test_terminal_masks_reasons_and_completion_are_independently_checked(fault):
    holds,_,_,_,terminal=flat_trace();timeout=[False]*120
    if fault=='completed':terminal['completed'][0]=False
    if fault=='hold':terminal['hold'][0]=49
    if fault=='steps':terminal['steps'][0]=49
    if fault=='reason':terminal['terms']['source_training_completion'][0]=False
    if fault=='mask':terminal['terminated'][119]=True
    if fault=='timeout':timeout[0]=True
    if fault=='observations':terminal['observations']['critic'][119][0]=float('nan')
    with pytest.raises(ValueError):audit_terminal(terminal,[0],torch.full((120,),50),holds,timeout,[0]*120,50,2000)


def test_ordinary_failure_ends_an_episode_without_a_flat_completion_claim():
    holds,_,_,_,terminal=flat_trace();terminal['completed'][0]=False
    terminal['terms']['source_training_safety'][0]=True
    assert audit_terminal(terminal,[0],torch.full((120,),50),holds,[False]*120,[0]*120,50,2000)==[]


@pytest.mark.parametrize('payload',[b'{"phase":"a","phase":"b"}\n',b'{"phase":"x","v":NaN}\n',b'{"phase":"x"}'])
def test_stream_rejects_duplicate_nonfinite_or_incomplete_records(tmp_path,payload):
    p=tmp_path/'bad.gz';p.write_bytes(gzip.compress(payload))
    with pytest.raises(ValueError):list(event_records(p))


def test_missing_contact_positive_control_cannot_finish_G3(tmp_path):
    with pytest.raises(AssertionError,match='positive control missing'):build_fixture(tmp_path/'no_force',contact_fault='empty')


def test_application_stopped_before_first_control_preserves_zero_progress(tmp_path):
    folder=tmp_path/'stopped'
    with pytest.raises(RuntimeError,match='simulator exited'):build_fixture(folder,app_running=lambda:False)
    report=json.loads((folder/'result.json').read_text())
    assert report['simulation_control_steps']==0 and report['motion_samples']==0 and report['completed_diagnostic_transitions']==0


@pytest.mark.parametrize('child',[False,True])
def test_approved_parent_and_child_check_source_ticket_and_scale_before_simulator(tmp_path,monkeypatch,child):
    out=tmp_path/'attempt';plan=cpu_plan(out)|dict(status='approved',user_approved=True)
    path=out/'run_manifest.json' if child else tmp_path/'approved.json'
    if child:out.mkdir()
    path.write_text(json.dumps(plan))
    if child:(out/'supervision.json').write_text(json.dumps(dict(revision='a'*40,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    checkpoint=tmp_path/'actor';checkpoint.write_bytes(b'CPU fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setattr(supervisor,'CHECKPOINT_SHA256',hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(supervisor,'_git_state',lambda:('a'*40,'a'*40,''));monkeypatch.setattr(supervisor,'_input_hashes',lambda:plan['input_sha256'])
    monkeypatch.setattr(supervisor,'dependencies',lambda:plan['dependency_sha256']);monkeypatch.setattr(supervisor.BudgetLedger,'reference',lambda self:plan['budget_ledger'])
    args=NS(manifest=path,num_envs=120,checkpoint=checkpoint,output_dir=out)
    assert supervisor.preflight(args,child=child)==plan
    args.num_envs=60
    with pytest.raises(ValueError):supervisor.preflight(args,child=child)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('ending',['deadline','zero_exit','log_ceiling'])
def test_G3_supervisor_dummy_CPU_process_cleanup_and_validation_never_accept_empty_output(tmp_path,monkeypatch,ending):
    out=tmp_path/'attempt';plan=cpu_plan(out)|dict(status='approved',user_approved=True)
    if ending=='log_ceiling':plan['runtime_log_byte_limit']=1
    monkeypatch.setattr(supervisor,'preflight',lambda *a,**kw:plan)
    monkeypatch.setattr(supervisor,'_input_hashes',lambda:plan['input_sha256']);monkeypatch.setattr(supervisor,'dependencies',lambda:plan['dependency_sha256'])
    monkeypatch.setattr(supervisor.BudgetLedger,'reference',lambda self:plan['budget_ledger'])
    monkeypatch.setattr(supervisor,'canonical_budget_path',lambda *a:tmp_path/'ledger')
    actor=tmp_path/'actor';actor.write_bytes(b'CPU');monkeypatch.setattr(supervisor,'CHECKPOINT_SHA256',hashlib.sha256(actor.read_bytes()).hexdigest())
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'0, uuid, CPU fixture, 1 MiB, 24000 MiB, 24576 MiB, 0 %, 35\n')
    original=subprocess.Popen;commands=[]
    def launch(command,**kwargs):
        commands.append(command)
        code='pass'
        if any(str(p).endswith('source_scale_motion_probe.py') for p in command) and ending!='zero_exit':
            code='import time;print("dummy",flush=True);time.sleep(60)'
        return original([supervisor.sys.executable,'-c',code],**kwargs)
    monkeypatch.setattr(subprocess,'Popen',launch)
    if ending=='deadline':monkeypatch.setattr(supervisor,'deadline_expired',lambda *a:True)
    assert supervisor.run(NS(output_dir=out,checkpoint=actor,num_envs=120))==1
    execution=json.loads((out/'execution.json').read_text())
    assert execution['process_group_empty'] and execution['validation_process_group_empty']
    assert execution['timed_out']==(ending=='deadline') and len(commands)==2
    assert '7200s' in commands[0] and not (tmp_path/'ledger').exists()


def test_G3_deadlines_and_readiness_marker_are_stage_specific():
    plan=cpu_plan('/tmp/synthetic-G3-unused')
    assert supervisor.deadline_expired(121,'[scale-g2] initial-ready',plan)
    assert not supervisor.deadline_expired(7199,'[scale-g3] initial-ready',plan)
    assert supervisor.deadline_expired(7201,'[scale-g3] initial-ready',plan)


def test_file_digest_reads_bounded_chunks(tmp_path,monkeypatch):
    import io
    from pathlib import Path
    target=tmp_path/'large';payload=b'abc'*(1024**2);sizes=[];original=Path.open
    class Reader(io.BytesIO):
        def read(self,size=-1):
            assert 0<size<=1048576;sizes.append(size);return super().read(size)
    monkeypatch.setattr(Path,'open',lambda self,*a,**kw:Reader(payload) if self==target else original(self,*a,**kw))
    assert file_digest(target)==hashlib.sha256(payload).hexdigest() and len(sizes)>2


def test_step_persistence_never_bulk_loads_the_steps_file(tmp_path,monkeypatch):
    from pathlib import Path
    report=dict(attempt_id='CPU');writer=MotionArtifactWriter(tmp_path/'result.json',report,schema='CPU')
    writer.append_step(dict(step=1,value=2));original=Path.read_bytes
    def read(path):
        if path==writer.paths['steps']:raise AssertionError('bulk steps read')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    writer.persist(report);writer.close()
    assert json.loads((tmp_path/'result.json').read_text())['steps_written']==1


def test_digest_rejects_decompressed_data_over_its_budget(tmp_path):
    path=tmp_path/'events.gz';path.write_bytes(gzip.compress(b'abcdefghij'))
    with pytest.raises(ValueError,match='ceiling'):digest_stream(path,byte_limit=5)
